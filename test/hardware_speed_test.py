from moku.instruments import Oscilloscope
from uedinst.delay_stage import XPSController
import os
import numpy as np
import socket
import tqdm
import matplotlib
import csv
import matplotlib.pylab as plt
import time

matplotlib.use('TKAgg')

moku_address = '[fe80:0000:0000:0000:7269:79ff:feb9:1a40%9]'


osc = Oscilloscope(moku_address, force_connect=True)
# osc.osc_measurement(-1e-6, 3e-6,"Input2",'Rising', 0.04)
osc.set_source(2, source='Input2')
osc.set_acquisition_mode(mode='Normal')
# osc.set_hysteresis("Absolute", 0)
osc.set_trigger(auto_sensitivity=False, hf_reject=False,
                noise_reject=False, mode='Normal', level=0.0184, source='Input2')
osc.set_timebase(-.6e-6, 1.e-6)
# https://apis.liquidinstruments.com/reference/oscilloscope/
xps = XPSController(reset=False)

stage = xps.autocorr_stage

# hardware limits
min_move = stage.min_limit
max_move = stage.max_limit


PEAK_POS_MM = 11.6550

STEP_SIZE_MM = 5e-4

stage.absolute_move(PEAK_POS_MM - STEP_SIZE_MM)

dt_stage = []
dt_data =[]
for i in np.arange(20):
    print(i)
    for pos in [PEAK_POS_MM, PEAK_POS_MM - STEP_SIZE_MM]:
        # t0 = time.time()
        # stage.absolute_move(pos)
        t1 = time.time()

        # dt_stage.append(t1-t0)

        measurement = osc.get_data()
        # this takes about 30 seconds for 100 data points. 90% of time is spent here
        t2 = time.time()
        dt_data.append(t2-t1)

# print(np.mean(dt_stage) *100)

print(np.mean(dt_data) *100)

osc.relinquish_ownership()
