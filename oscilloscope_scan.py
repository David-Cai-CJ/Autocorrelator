from moku.instruments import Datalogger
from moku.instruments import Oscilloscope
from uedinst.delay_stage import XPSController
from time import sleep
import os
import numpy as np
import socket
import tqdm
import matplotlib
import matplotlib.pylab as plt

matplotlib.use('TKAgg')

moku_address = '[fe80:0000:0000:0000:7269:79ff:feb9:1a40%9]'

# If it says API already connected, close it from fd in the WARNING message. (Probably better ways...)

# socket.socket().close(404)
# socket.socket().close(1192)


osc = Oscilloscope(moku_address, force_connect=True)
# osc.osc_measurement(-1e-6, 3e-6,"Input2",'Rising', 0.04)
osc.set_source(2, source='Input2')
osc.set_acquisition_mode(mode='Precision')
osc.set_hysteresis("Absolute", 0.03)
osc.set_trigger(auto_sensitivity=False, hf_reject=False,
                noise_reject=False, mode='Normal', level=0.09, source='Input2')
osc.set_timebase(-.6e-6, 1.e-6)
# https://apis.liquidinstruments.com/reference/oscilloscope/
xps = XPSController(reset=False)

stage = xps.autocorr_stage

# hardware limits
min_move = stage.min_limit
max_move = stage.max_limit


# signal limits for ~100fs pulse
PEAK_POS_MM = 11.6540

# RANGE_PS = .5
# RANGE_MM = abs(stage.delay_to_distance(RANGE_PS))

# First set -- try to resolve the fringes in the middle
RANGE_MM = 0.025
STEP_SIZE_MM = 1e-4  # 100 nm
# 0.03 mm per 100 fs


# # Second set -- long scan period
# RANGE_MM = 0.1
# STEP_SIZE_MM = 1e-3  # 1um


MAX_POS_MM = round(PEAK_POS_MM + RANGE_MM, 4)
MIN_POS_MM = round(PEAK_POS_MM - RANGE_MM, 4)


###

stage.absolute_move(0)
print(f"Stage moved to {stage.current_position()}")

measurement = osc.get_data()
calibration_data = np.array([measurement['time'], measurement['ch2']]).T
np.savetxt(r'./logging/' + os.sep + 'calibration' +
           '.csv', calibration_data, delimiter=',')

# plt.plot(*calibration_data.T, marker='o', ls=None, ms=.2)

print("Start Scanning\n\n")

pos = np.round(np.arange(MIN_POS_MM, MAX_POS_MM +
               STEP_SIZE_MM, STEP_SIZE_MM), 4)

for loc in tqdm.tqdm(pos):
    stage.absolute_move(loc)
    prefix = f"{stage.current_position():.4f}".replace(".", "_")
    print('\n' + str(loc))

    measurement = osc.get_data()
    data = np.array([measurement['time'], measurement['ch2']]).T
    np.savetxt(r'./logging/' + os.sep + prefix + '.csv', data, delimiter=',')

osc.relinquish_ownership()
