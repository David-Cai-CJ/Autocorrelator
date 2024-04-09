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


matplotlib.use('TKAgg')

#########
folder = input("Logging folder name?\t:")

n_samples = int(input("N samples per position:\t"))

path = os.path.sep.join(['logging', folder])

try:
    os.makedirs(path)
except FileExistsError:
    print("Folder already exists. Chooes a different name.")

    
file = h5py.File(path+os.path.sep + 'dataset.hdf5', 'a')

#########

moku_address = '[fe80:0000:0000:0000:7269:79ff:feb9:1a40%9]'

## If it says API already connected, close it from fd in the WARNING message. (Probably better ways...)
# socket.socket().close(404)
# socket.socket().close(1192)

osc = Oscilloscope(moku_address, force_connect=True)
# osc.osc_measurement(-1e-6, 3e-6,"Input2",'Rising', 0.04)
osc.set_source(2, source='Input2')
osc.set_acquisition_mode(mode='Precision')
osc.set_hysteresis("Absolute", 0.03)
osc.set_trigger(auto_sensitivity=False, hf_reject=False,
                noise_reject=False, mode='Normal', level=0.7, source='Input2')
osc.set_timebase(-3e-6, 5e-6)
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

with h5py.File(path + os.path.sep + "dataset.hdf5", 'a') as hf:
    hf.create_dataset("positions", pos)

########
trange = tqdm.tqdm(pos)

for loc in trange:    
    stage.absolute_move(loc)
    trange.set_postfix({'Position':loc})

    with h5py.File(path + os.path.sep + "dataset.hdf5", 'a') as hf:
        grp = hf.create_group(f"{loc}")

    
    for n in np.arange(n_samples):
        measurement = osc.get_data()
        data = np.array([measurement['time'], measurement['ch2']]).T
        grp["n"] = data

osc.relinquish_ownership()
