from moku.instruments import Oscilloscope
from uedinst.delay_stage import XPSController
import os
from os.path import join as pj
import numpy as np
import socket
import tqdm
import matplotlib
import csv
import matplotlib.pylab as plt
import h5py
import argparse
from lmfit.models import GaussianModel, ConstantModel
from pathlib import Path
from datetime import datetime


matplotlib.use("TKAgg")


##### Alignment Variables.
PEAK_POS = 11.213  # mm
# 0.03 mm per 100 fs
RANGE = 0.1
STEP_SIZE = 0.25e-3
TRIGGER_LEVEL = 0.3
#########

parser = argparse.ArgumentParser()

parser.add_argument(
    "file_directory",
    help="Absolute or relative path to the .hdf5 and .pdf files.",
    type=str,
)
# parser.add_argument('--pdf', default= True, action='store_true', help = 'store the scan in a pdf if set to true.')
parser.add_argument('--no-pdf', dest='pdf', action='store_false')
# parser.add_argument('--liveview', default = True, action='store_true')

parser.add_argument('--no-liveview', dest='liveview', action='store_false')
parser.add_argument(
    "-p",
    "--peak_position",
    help="Between 0 and 25 mm which are limits of the linear stage.",
    default=PEAK_POS,
    type=float,
)
parser.add_argument(
    "-r",
    "--range",
    help="Range in each direction the scan should cover. Unit in milimeter.",
    default=RANGE,
    type=float,
)
parser.add_argument(
    "-s",
    "--step_size",
    help="Spacing between scan steps. Unit in milimeter.",
    default=STEP_SIZE,
    type=float,
)
parser.add_argument(
    "-t",
    "--trigger_level",
    help="Trigger level. Unit in volt.",
    default=TRIGGER_LEVEL,
    type=float,
)
parser.add_argument(
    "-n",
    "--num_samples",
    help="Spacing between scan steps. Unit in milimeter.",
    default=1,
    type=int,
)
parser.add_argument(
    "--rolling_mode",
    help="Don't trigger on peak of PD signal, but average on the DC voltage",
    default=False,
    action="store_true",
)

args = parser.parse_args()
####### Extracting from argparser

arg_path = Path(args.file_directory)

if Path(pj(arg_path.parent, arg_path.stem + ".hdf5")).is_file():
    print("hdf5 already exist. Change your output name.")
    exit()

peak_position = args.peak_position
range = args.range
step_size = args.step_size
num_samples = args.num_samples

# #########
moku_address = "172.25.12.13"

## If it says API already connected, close it from fd in the WARNING message. (Probably better ways...)
# socket.socket().close(404)
# socket.socket().close(1192)

print("connecting to moku... ", end="")
osc = Oscilloscope(moku_address, force_connect = True)
print("done")

# osc.osc_measurement(-1e-6, 3e-6,"Input2",'Rising', 0.04)
osc.set_source(2, source="Input2")
osc.set_acquisition_mode(mode="Precision")

if args.rolling_mode:
    osc.set_timebase(-0.5 , 0.5)
    osc.enable_rollmode(roll=True)
else:        
    osc.set_trigger(
        auto_sensitivity=False,
        hf_reject=False,
        noise_reject=False,
        mode="Normal",
        level=args.trigger_level,
        source="Input2",
    )
    osc.set_timebase(-0.1e-6, 2e-6)
# https://apis.liquidinstruments.com/reference/oscilloscope/

# reset=False will not reset the stages to factory default locations.
print("connecting to xps... ", end="")
xps = XPSController(reset=False)
stage = xps.autocorr_stage
print("done")

# # hardware limits
# min_move = stage.min_limit
# max_move = stage.max_limit

#######
max_pos = round(peak_position + range, 4)
min_pos = round(peak_position - range, 4)

########
pos = np.round(np.arange(min_pos, max_pos + step_size,step_size), 4)

####### create matrices for holding time/voltage data
#stage.absolute_move(peak_position)
print("getting trace dim... ", end="")
scan_pts = len(osc.get_data()["time"])
print("done")

t_matrix = np.zeros((len(pos), num_samples, scan_pts))
v_matrix = np.zeros((len(pos), num_samples, scan_pts))

########
if args.liveview:
    fig, ax = plt.subplots(1, 1)
    ax.autoscale(enable=False, axis="x")
    ax.autoscale(enable=True, axis="y")
    ax.set_xlim([np.min(pos) - 2 * step_size, np.max(pos) + 2 * step_size])


######

v_arr = []

trange = tqdm.tqdm(pos)

for i, loc in enumerate(trange):
    stage.absolute_move(loc)
    stage._wait_end_of_move()
    trange.set_postfix({"Position": f"{loc:.3f}"})

    for n in np.arange(args.num_samples):
        measurement = osc.get_data()
        t = measurement["time"]
        v = measurement["ch2"]
        t_matrix[i, n] = t
        v_matrix[i, n] = v

    v_arr.append(np.nanmean(v_matrix[i]))


    if args.liveview:
        try:
            scatter.remove()
        except NameError:
            pass
        scatter = ax.scatter(pos[: len(v_arr)], v_arr, marker=".", color="k")
        plt.pause(0.001)


stage.absolute_move(args.peak_position) # move stage back to max pos

osc.relinquish_ownership()

#### exporting traces as matrices
with h5py.File(pj(arg_path.parent, arg_path.stem + ".hdf5"), "a") as hf:
    hf.create_dataset("scan_time", data = datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    trace_grp = hf.create_group("trace")
    trace_grp.create_dataset("positions", data=pos)
    trace_grp.create_dataset("time_trace", data=t_matrix)
    trace_grp.create_dataset("voltage_trace", data=v_matrix)

## Generate an FWHM fit with an image saved to the data folder.

if args.rolling_mode:
    signal = v_arr
else:
    signal = np.sum(np.diff(t_matrix, axis=2) * v_matrix[..., :-1], axis=(1, 2))

left_found = False
right_found = False
normed = signal - np.min(signal)
normed /= np.max(normed)
left = 0
right = len(pos) - 1

for i, val in enumerate(normed[1:] - 0.5):
    if val > 0 and not left_found:
        left = i + 1
        left_found = True
    if left_found and val < 0 and not right_found:
        right = i + 1
        right_found = True

# print(
#     "FWHM quick: ".ljust(15) + f"{(pos[right] - pos[left])/1e3/2.998e8/1e-15*2:.2f} fs"
# )

# fitting
def gau(x, x0, s):
    return 1 / np.sqrt(2 / np.pi) * np.exp(-1 / 2 * (x - x0) ** 2 / s**2)


def model(x, A, x0, s, C): 
    return A * gau(x, x0, s) + C

p0 = [2.5, args.peak_position, 0.04, 0.0]

# fit, _ = curve_fit(
#     model,
#     pos,
#     normed,
#     p0=p0,
#     bounds=(
#         [0.001, p0[1] - 0.1, 0.001, p0[-1] - 0.5],
#         [40, p0[1] + 0.1, 0.07, p0[-1] + 1],
#     ),
#     nan_policy="omit"
# )


# A, x0, s, C = fit

# ## conversions
# t_fs = (pos - x0) / 1e3 / 2.998e8 / 1e-15 * 2
t_fs = (pos - args.peak_position) / 1e3 / 2.998e8 / 1e-15 * 2

model = GaussianModel() + ConstantModel()
params = model.make_params(amplitude=240, center=0, sigma=100, c=0)
result = model.fit(normed[1:], params, x=t_fs[1:])  # first datapoint is crooked sometimes
print(result.fit_report())

# fwhm_factor = 2.355
# width = s / 1e3 / 3e8 / 1e-15 * 2

with h5py.File(pj(arg_path.parent, arg_path.stem + ".hdf5"), "a") as hf:
    hf.create_dataset("delay", data=t_fs)
    hf.create_dataset("intensity", data=normed)

## plotting and saving

# print("FWHM fit: ".ljust(15) + f"{fwhm_factor * width:.2f} fs")

if args.pdf:
    f, ax = plt.subplots(1, 1)
    ax.plot(
        t_fs, normed, "k.", lw=0.5, ms=3, alpha=0.7, zorder=-1, markevery=1, label="Data"
    )
    # ax.plot(
    #     t_fs,
    #     model(pos, *fit),
    #     c="b",
    #     lw=1,
    # )
    # ax.plot([], [], lw=1, c="b", label=f"FWHM={width*fwhm_factor:.2f} fs")
    ax.plot(t_fs[1:], result.best_fit, c="b", label=f"FWHM={result.best_values['sigma'] * 2.35482:.2f}")
    ax.axvline(t_fs[left], c="r", ls="--", lw=0.7, alpha=0.7)
    ax.axvline(t_fs[right], c="r", ls="--", lw=0.7, alpha=0.7)
    ax.axhline(
        0.5,
        c="r",
        ls="--",
        lw=0.7,
        alpha=0.7,
        label=f"FWHM={(pos[right] - pos[left])/1e3/2.998e8/1e-15*2:.2f} fs",
    )

    ax.set_xlabel("Delay [fs]")
    ax.set_ylabel("Relative Intensity")
    ax.legend()
    f.tight_layout()

    plt.show()

    f.savefig(pj(arg_path.parent, arg_path.stem +'.pdf'), bbox_inches="tight")
