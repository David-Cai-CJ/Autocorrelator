from moku.instruments import Oscilloscope
from uedinst.delay_stage import XPSController
import os
import numpy as np
import socket
import tqdm
import matplotlib
import csv
import matplotlib.pylab as plt

matplotlib.use('TKAgg')

#########
folder = r'ch3_problem_solved'
print(folder)

n_samples = 1


try:
    os.makedirs(os.path.sep.join(['logging', folder]))
except FileExistsError:
    pass

#########

moku_address = '172.25.12.13'

# If it says API already connected, close it from fd in the WARNING message. (Probably better ways...)

# socket.socket().close(404)
# socket.socket().close(1192)


osc = Oscilloscope(moku_address, force_connect=True)
# osc.osc_measurement(-1e-6, 3e-6,"Input2",'Rising', 0.04)
osc.set_source(2, source='Input2')
osc.set_acquisition_mode(mode='Precision')
osc.set_trigger(auto_sensitivity=False, hf_reject=False,
                noise_reject=False, mode='Normal', level=0.50, source='Input2')
osc.set_timebase(-0.5e-6, 2e-6)
# https://apis.liquidinstruments.com/reference/oscilloscope/
xps = XPSController(reset=False)

print("hardwares connected")
stage = xps.autocorr_stage

# hardware limits
min_move = stage.min_limit
max_move = stage.max_limit

# signal limits for ~100fs pulse
PEAK_POS_MM = 11.136

RANGE_MM = 0.1 # 0.03 mm per 100 fs
STEP_SIZE_MM = 0.5e-3 # do range divide by 40 or something


MAX_POS_MM = round(PEAK_POS_MM + RANGE_MM, 4)
MIN_POS_MM = round(PEAK_POS_MM - RANGE_MM, 4)

# CALIBRATION

stage.absolute_move(10)

print(f"Stage moved to {stage.current_position()}")

measurement = osc.get_data()
calibration_data = np.array([measurement['time'], measurement['ch2']]).T

np.savetxt(r'./logging' + os.sep + folder + os.path.sep + 'calibration'
           '.csv', calibration_data, delimiter=',')

########
print("Start Scanning\n")

pos = np.round(np.arange(MIN_POS_MM, MAX_POS_MM +
               STEP_SIZE_MM, STEP_SIZE_MM), 4)

fig, (ax, ax2) = plt.subplots(2, 1)

v_data = []
e_v_data = []

for loc in tqdm.tqdm(pos):
    stage.absolute_move(loc)
    prefix = f"{stage.current_position():.4f}".replace(".", "_")

    Vmax = []
    
    step_folder = r'./logging'+os.path.sep + folder + \
        os.path.sep + rf'{loc}'.replace('.', '_')

    try:
        os.makedirs(step_folder)
    except FileExistsError:
        pass

    ax.clear()
    for n in np.arange(n_samples):
        # Current proportional to Voltage. Take max Vout
        measurement = osc.get_data()
        data = np.array([measurement['time'], measurement['ch2']]).T
        np.savetxt(step_folder+os.path.sep +
                   f'{n}' + '.csv', data, delimiter=',')
        Vmax.append(np.sum(measurement['ch2']))
        ax.plot(measurement['time'], measurement['ch2'])

    ax.clear()
    ax2.clear()

    v_data.append(np.mean(Vmax))
    e_v_data.append(np.std(Vmax))

    ax.errorbar(np.array(pos[:len(v_data)]), np.array(v_data), yerr=e_v_data)

    with open(r'./logging' + os.path.sep + folder + os.path.sep +
              'summary.csv', 'a+', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([loc, np.mean(Vmax), np.std(Vmax)])
    plt.pause(.1)

plt.show(block=True)

osc.relinquish_ownership()
