from moku.instruments import Datalogger
from moku.instruments import Oscilloscope
from uedinst.delay_stage import XPSController
from time import sleep
import os
import numpy as np
import socket


moku_address = '[fe80:0000:0000:0000:7269:79ff:feb9:1a40%9]'

# If it says API already connected, close it from fd in the WARNING message. (Probably better ways...)

# socket.socket().close(404)
# socket.socket().close(1192)


osc = Oscilloscope(moku_address, force_connect=True)
dLogger = Datalogger(moku_address, force_connect=True)
# https://apis.liquidinstruments.com/reference/oscilloscope/
xps = XPSController(reset=False)

stage = xps.autocorr_stage

### hardware limits
min_move = stage.min_limit
max_move = stage.max_limit


### signal limits for ~100fs pulse
PEAK_POS_MM = 22.652
RANGE_PS = .2
RANGE_MM = abs(stage.delay_to_distance(RANGE_PS))
STEP_SIZE_MM = 100e-6  # 100 nm step size

# Range of motion
MAX_POS_MM = round(PEAK_POS_MM + RANGE_MM,4)
MIN_POS_MM = round(PEAK_POS_MM - RANGE_MM,4)


# Set data logger to osc
dLogger.set_frontend(channel=2, impedance='1MOhm',
                     coupling="DC", range="10Vpp")
# Log 100 samples per second
dLogger.set_samplerate(1000)
dLogger.set_acquisition_mode(mode='Precision')


###

print(f"Moving stage to {MIN_POS_MM}")
stage.absolute_move(MIN_POS_MM)
print(f"Stage moved to {stage.current_position()}")

filenames = []


while stage.current_position() <= MAX_POS_MM:
    
    prefix = str(stage.current_position()).replace(".", "_")
    print(prefix.rjust(30), np.mean(osc.get_data()['ch2']))
    logFile = dLogger.start_logging(duration=1,
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

    filenames.append(logFile['file_name'])
    stage.relative_move(STEP_SIZE_MM)


for fname in filenames:
    dLogger.download("persist",
                     fname,
                     './logging' + os.sep + fname)

dLogger.relinquish_ownership()
osc.relinquish_ownership()