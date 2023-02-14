from moku.instruments import Datalogger
from uedinst.delay_stage import XPSController
from time import sleep
import os
import numpy as np
import matplotlib.pylab as plt
import tqdm
import matplotlib
from scipy.optimize import curve_fit


matplotlib.use('TKAgg')


moku_address = '[fe80:0000:0000:0000:7269:79ff:feb9:1a40%9]'

# If it says API already connected, close it from fd in the WARNING message. (Probably better ways...)


dLogger = Datalogger(moku_address, force_connect=True)
# https://apis.liquidinstruments.com/reference/dloggerilldloggerope/
xps = XPSController(reset=False)

dLogger.start_streaming()
print(dLogger.get_stream_data()['ch2'])

# stage = xps.autocorr_stage

# # hardware limits
# min_move = stage.min_limit
# max_move = stage.max_limit


# # signal limits for ~100fs pulse
# PEAK_POS_MM = 22.659
# RANGE_PS = .22
# RANGE_MM = abs(stage.delay_to_distance(RANGE_PS))
# # STEP_SIZE_MM = 500e-6  # in mm
# N_POINTS = 30

# # Range of motion
# MAX_POS_MM = round(PEAK_POS_MM + RANGE_MM, 4)
# MIN_POS_MM = round(PEAK_POS_MM - RANGE_MM, 4)


# # # Set data logger to dlogger
# dLogger.set_frontend(channel=2, impedance='1MOhm',
#                      coupling="DC", range="10Vpp")
# dLogger.set_acquisition_mode(mode="Precision", strict=True)


# print(f"Moving stage to {MIN_POS_MM}")
# stage.absolute_move(MIN_POS_MM)
# print(f"Stage moved to {stage.current_position()}")

# pos = np.round(np.linspace(MIN_POS_MM, MAX_POS_MM, N_POINTS), 4)

# v_data = []
# e_v_data = []

# fig, (ax1, ax2) = plt.subplots(2, 1)

# for loc in tqdm.tqdm(pos):
#     stage.absolute_move(loc)
#     sleep(1)
#     d = np.array(dLogger.get_stream_data(duration = 4)['ch2'], dtype=float)
#     print(d)
#     print(d.shape)

#     v_data.append(np.mean(d))
#     e_v_data.append(np.std(d))
#     ax1.clear()
#     ax2.clear()
#     ax1.scatter(np.array(pos[:len(v_data)]), np.array(v_data))
#     ax2.plot(d)
#     plt.pause(.1)

# plt.show(block=True)


# def model(x, a, x0, s, C):
#     return a/np.sqrt(2*np.pi)/s*np.exp(-1/2*(x-x0)**2/s**2) +C

# fit, err = curve_fit(model, pos, v_data, sigma=e_v_data, absolute_sigma=True, p0 = [.5, 22.658, .02, .6])

# fig, (ax1, ax2) =  plt.subplots(2,1)
# xx = np.linspace(np.min(pos), np.max(pos), 100)
# ax1.scatter(np.array(pos), np.array(v_data), color = 'k')
# ax1.plot(xx, model(xx, *fit), 'b--', zorder = -1)

# ax2.plot(pos, v_data - model(pos, *fit))

# plt.show(block = True)


dLogger.relinquish_ownership()
# # dlogger.relinquish_ownership()
