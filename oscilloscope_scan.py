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
import argparse
from scipy.optimize import curve_fit


matplotlib.use('TKAgg')

#########
_folder = input("Logging folder name?:\t")

n_samples = int(input("N samples per position:\t"))

path = os.path.sep.join(['logging', _folder])

try:
    os.makedirs(path)
except FileExistsError:
    print("Folder already exists. Choose a different name.")

file_dir = path + os.path.sep + 'data.hdf5'

# #########
moku_address = '172.25.12.13' 

## If it says API already connected, close it from fd in the WARNING message. (Probably better ways...)
# socket.socket().close(404)
# socket.socket().close(1192)

osc = Oscilloscope(moku_address, force_connect=True)
print("Moku connected.")
# osc.osc_measurement(-1e-6, 3e-6,"Input2",'Rising', 0.04)
osc.set_source(2, source='Input2')
osc.set_acquisition_mode(mode='Precision')
osc.set_trigger(auto_sensitivity=False, hf_reject=False,
                noise_reject=False, mode='Normal', level=0.3, source='Input2')
osc.set_timebase(-0.5e-6, 3e-6)
# https://apis.liquidinstruments.com/reference/oscilloscope/

# reset=False will not reset the stages to factory default locations.
xps = XPSController(reset=False)
print("XPS connected. \n")
stage = xps.autocorr_stage

# hardware limits
min_move = stage.min_limit
max_move = stage.max_limit

# signal limits for ~100fs pulse

PEAK_pos = 11.136

# 0.03 mm per 100 fs
RANGE_MM = 0.10
STEP_SIZE_MM = 2e-3

MAX_pos = round(PEAK_pos + RANGE_MM, 4)
MIN_pos = round(PEAK_pos - RANGE_MM, 4)


########

pos = np.round(np.arange(MIN_pos, MAX_pos +
               STEP_SIZE_MM, STEP_SIZE_MM), 4)
    
####### create matrices for holding time/voltage data
scan_pts = len(osc.get_data()['time'])

t_matrix = np.zeros((len(pos), n_samples, scan_pts))
v_matrix = np.zeros((len(pos), n_samples, scan_pts))

########

fig, ax = plt.subplots(1, 1)

ax.autoscale(enable =False, axis = 'x')
ax.autoscale(enable =True, axis = 'y')
ax.set_xlim([np.min(pos) - 2 * STEP_SIZE_MM, np.max(pos) +  2 * STEP_SIZE_MM])


######

v_arr = []

trange = tqdm.tqdm(pos)

for i, loc in enumerate(trange):    
    stage.absolute_move(loc)
    trange.set_postfix({'Position': f'{loc}'})

    for n in np.arange(n_samples): 
        measurement = osc.get_data()
        t = measurement['time']
        v = measurement['ch2']
        t_matrix[i, n] = t
        v_matrix[i, n] = v
        
    v_arr.append(np.mean(v_matrix[i]))

    try:
        scatter.remove()
    except NameError:
        pass

    scatter = ax.scatter(pos[:len(v_arr)], v_arr, marker = '.', color= 'k')
    plt.pause(0.001)


stage.absolute_move(PEAK_pos + 5*RANGE_MM)
osc.relinquish_ownership()


# exporting traces as matrices 
with h5py.File(file_dir, 'a') as hf:
    trace_grp = hf.create_group("trace")
    trace_grp.create_dataset("positions", data = pos)
    trace_grp.create_dataset("time_trace", data = t_matrix)
    trace_grp.create_dataset("voltage_trace", data = v_matrix)

## Generate an FWHM fit with an image saved to the data folder.


signal = np.sum(np.diff(t_matrix, axis= 2) * v_matrix[..., :-1], axis= (1,2))

left_found = False
right_found = False
normed = signal - np.min(signal)
normed /= np.max(normed)

for i, val in enumerate(normed - 0.5):
    if val > 0 and not left_found:
        left = i
        left_found = True
    if left_found and val < 0 and not right_found:
        right = i
        right_found = True

# print(left, right)
print("FWHM quick: ".ljust(15) +  f"{(pos[right] - pos[left])/1e3/2.998e8/1e-15*2:.2f} fs")

# fitting
def gau(x, x0, s):
    return 1/np.sqrt(2/np.pi)*np.exp(-1/2*(x-x0)**2/s**2)

model = lambda x,A,x0,s,C: A*gau(x,x0,s) + C

p0 = [2.5, 11.136,0.04, 0.]

fit, _ = curve_fit( model, pos, normed, p0=p0,
                     bounds=([.001, p0[1] - .1,  .001, p0[-1] - 0.5],
                             [40, p0[1] + .1,  .07, p0[-1] + 1]))


A, x0, s, C = fit


## conversions
t_fs = (pos - x0)/1e3/2.998e8/1e-15*2

fwhm_factor = 2.355
width = s /1e3/3e8 / 1e-15 * 2

## plotting and saving

print('FWHM fit: '.ljust(15) + f'{fwhm_factor * width:.2f} fs')

# Plotting


f, ax =  plt.subplots(1,1)
ax.plot(t_fs, normed, 'k.',lw=.5, ms = 3, alpha = .7, zorder= -1, markevery=1, label='Data')
ax.plot(t_fs, model(pos, *fit), c='b', lw=1, )
ax.plot([],[], lw=1, c="b", label=f"FWHM={width*fwhm_factor:.2f} fs")
ax.axvline(t_fs[left], c='r', ls = '--', lw =.7, alpha = .7)
ax.axvline(t_fs[right], c='r',  ls = '--', lw =.7, alpha = .7)
ax.axhline(0.5, c='r',  ls = '--', lw =.7, alpha = .7, 
           label=f"FWHM={(pos[right] - pos[left])/1e3/2.998e8/1e-15*2:.2f} fs")

np.savetxt(path + os.path.sep + "times.txt" , t_fs)
np.savetxt(path + os.path.sep + "normed_intensity.txt", normed)


with h5py.File(file_dir, 'a') as hf:
    hf.create_dataset('delay', data = t_fs)
    hf.create_dataset('intensity', data = normed)


ax.set_xlabel("Delay [fs]")
ax.set_ylabel("Relative Intensity")
ax.legend()
f.tight_layout()

plt.show()

f.savefig(path + os.path.sep + "//trace.pdf", bbox_inches = 'tight')
