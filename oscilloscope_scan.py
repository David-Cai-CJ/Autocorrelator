from moku.instruments import Oscilloscope
from uedinst.delay_stage import XPSController
import os
import numpy as np
import socket
import tqdm
import matplotlib
import csv
import matplotlib.pylab as plt
import h5py

from scipy.optimize import curve_fit


matplotlib.use('TKAgg')

#########
folder = input("Logging folder name?:\t")

n_samples = int(input("N samples per position:\t"))

path = os.path.sep.join(['logging', folder])

try:
    os.makedirs(path)
except FileExistsError:
    print("Folder already exists. Choose a different name.")

file_dir = path + os.path.sep + 'data.hdf5'

#########
moku_address = '172.25.12.13' 

## If it says API already connected, close it from fd in the WARNING message. (Probably better ways...)
# socket.socket().close(404)
# socket.socket().close(1192)

osc = Oscilloscope(moku_address, force_connect=True)
# osc.osc_measurement(-1e-6, 3e-6,"Input2",'Rising', 0.04)
osc.set_source(2, source='Input2')
osc.set_acquisition_mode(mode='Precision')
osc.set_trigger(auto_sensitivity=False, hf_reject=False,
                noise_reject=False, mode='Normal', level=0.3, source='Input2')
osc.set_timebase(-0.5e-6, 3e-6)
# https://apis.liquidinstruments.com/reference/oscilloscope/

# reset=False will not reset the stages to factory default locations.
xps = XPSController(reset=False)

print("Hardwares connected")
stage = xps.autocorr_stage

# hardware limits
min_move = stage.min_limit
max_move = stage.max_limit

# signal limits for ~100fs pulse

PEAK_POS_MM = 11.625

# 0.03 mm per 100 fs
RANGE_MM = 0.10
STEP_SIZE_MM = 2e-3

MAX_POS_MM = round(PEAK_POS_MM + RANGE_MM, 4)
MIN_POS_MM = round(PEAK_POS_MM - RANGE_MM, 4)


########
print("Start Scanning\n\n")

pos = np.round(np.arange(MIN_POS_MM, MAX_POS_MM +
               STEP_SIZE_MM, STEP_SIZE_MM), 4)

with h5py.File(file_dir, 'a') as hf:
    hf.create_dataset("positions", data = pos)

########
trange = tqdm.tqdm(pos)


fig, ax = plt.subplots(1, 1)

v_arr = []

for loc in trange:    
    stage.absolute_move(loc)
    trange.set_postfix({'Position': f'{loc}'})

    v_loc = np.zeros(n_samples)

    with h5py.File(file_dir, 'a') as hf:
        grp = hf.create_group(f"{loc}".replace(".", "_"))

        for n in np.arange(n_samples): 
            measurement = osc.get_data()
            t = measurement['time']
            v = measurement['ch2']
            subgrp = grp.create_group(f'{n}') 
            subgrp.create_dataset('time', data = t)
            subgrp.create_dataset('voltage', data =v)
            
            v_loc[n] = np.sum(v)
    v_arr.append(np.mean(v_loc))

    ax.plot(pos[:len(v_arr)], v_arr)
    plt.pause(.1)

osc.relinquish_ownership()

plt.show(block=True)

## Generate an FWHM fit with an image saved to the data folder.

hf = h5py.File(file_dir, 'r')
pos_mm = hf['positions']
v_arr = []

for p in pos:
    subgrp_key = str(p).replace(".", "_")
    subgrp = hf[subgrp_key]
    v_loc = []
    for key in subgrp.keys():
        # can be better integrated with the time stamps
        t=np.array(subgrp[key]['time'])
        v=np.array(subgrp[key]['ch2'])
        cond = np.full(len(v), True)
        v_loc.append(np.sum(np.abs(np.diff(t)[cond[1:]]*v[1:][cond[1:]]))) 
    v_arr.append(np.mean(v_loc))


v_arr = np.array(v_arr) / np.max(v_arr)   # Normalize

def gau(x, x0, s):
    return 1/np.sqrt(2/np.pi)*np.exp(-1/2*(x-x0)**2/s**2)

model = lambda x,A,x0,s,C: A*gau(x,x0,s) + C

p0 = [2.5, 11.136,0.04, 0.]

fit, err = curve_fit( model, pos_mm, v_arr, p0=p0,
                     bounds=([.001, p0[1] - .1,  .001, p0[-1] - 0.5],
                             [40, p0[1] + .1,  .07, p0[-1] + 1]))


## conversions
t_fs = (pos_mm - fit[1])/1e3/2.998e8/1e-15*2

fwhm_factor = 2.355
width = fit[-2]/1e3/3e8 / 1e-15 * 2
e_width = np.sqrt(np.diag(err))[-2]/1e3/3e8 / 1e-15

## plotting and saving

print(f'{width:.2f} +/- {e_width:.2f}')

# Plotting
fig = plt.figure()

ax = fig.add_subplot(1,1)

ax.plot(t_fs, v_arr, 'k.', ms=4, markevery=1, label='Data')
ax.plot(t_fs, model(pos_mm, *fit), c='b', lw=1,  label='Fit')
ax.set_xlabel('Delay (fs)')

fig.tight_layout()

text_out = "$\sigma_{\mathrm{auto.}}=$" + f"${width:.2f}\pm{e_width:.2f}$ fs\n"
# "$\mathrm{FWHM}_{\mathrm{source}}=$" + \
# f"${width/np.sqrt(2) * fwhm_factor:.0f}\pm{e_width/np.sqrt(2) *fwhm_factor:.0f}$ fs\n" +\

ax.text(.05, .95, s=text_out, transform=ax.transAxes, va='top')

print('figsaved')

plt.show()
