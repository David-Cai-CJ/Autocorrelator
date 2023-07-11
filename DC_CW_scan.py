from moku.instruments import Datalogger
from moku.instruments import Oscilloscope
from uedinst.delay_stage import XPSController
from time import sleep
import os
import numpy as np
import socket
import tqdm

moku_address = '[fe80:0000:0000:0000:7269:79ff:feb9:1a40%9]'

# If it says API already connected, close it from fd in the WARNING message. (Probably better ways...)

# socket.socket().close(404)
# socket.socket().close(1192)

folder = 'interference_or_not'


try:
    os.makedirs('logging' + os.path.sep + folder)
except FileExistsError:
    pass

# osc = Oscilloscope(moku_address, force_connect=True)
dLogger = Datalogger(moku_address, force_connect=True)
# https://apis.liquidinstruments.com/reference/oscilloscope/
xps = XPSController(reset=False)

stage = xps.autocorr_stage

# hardware limits
min_move = stage.min_limit
max_move = stage.max_limit


# signal limits for ~100fs pulse
PEAK_POS_MM = 11.703
# RANGE_PS = .5

# RANGE_MM = abs(stage.delay_to_distance(RANGE_PS))
RANGE_MM = 0.04
# 0.03 mm per 100 fs

STEP_SIZE_MM = 5e-4  # in mm
# STEP_SIZE_MM = RANGE_MM/100  # in mm
# Range of motion
MAX_POS_MM = round(PEAK_POS_MM + RANGE_MM, 4)
MIN_POS_MM = round(PEAK_POS_MM - RANGE_MM, 4)


# Set data logger to osc
dLogger.set_frontend(channel=2, impedance='1MOhm',
                     coupling="DC", range="50Vpp")
dLogger.set_samplerate(1e4)
dLogger.set_acquisition_mode(mode='Precision')

###

filenames = []

print("Start Scanning")
print(f"Moving stage to {MIN_POS_MM}")
stage.absolute_move(MIN_POS_MM)
print(f"Stage moved to {stage.current_position()}")

pos = np.round(np.arange(MIN_POS_MM, MAX_POS_MM +
               STEP_SIZE_MM, STEP_SIZE_MM), 4)

for loc in tqdm.tqdm(pos):
    stage.absolute_move(loc)
    prefix = f"{stage.current_position():.4f}".replace(".", "_")
    print('\n' + str(loc))
    logFile = dLogger.start_logging(duration=3,
                                    file_name_prefix=prefix)

    is_logging = True

    while is_logging:
        sleep(0.5)

        try:
            progress = dLogger.logging_progress()
            remaining_time = progress['time_remaining']
            is_logging = remaining_time >= 0
        except:
            is_logging = False


    dLogger.download("persist",
                     logFile['file_name'],
                     r'./logging'+os.path.sep + folder + \
        os.path.sep + rf'{loc}'.replace('.', '_')+ '.li')
    
    sleep(0.1)


dLogger.relinquish_ownership()
