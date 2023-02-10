from moku.instruments import Datalogger
from moku.instruments import Oscilloscope

from uedinst.delay_stage import XPSController
from time import sleep
import os


moku_address = '[fe80:0000:0000:0000:7269:79ff:feb9:1a40%9]'


osc = Oscilloscope(moku_address, force_connect=True)
print('done')
dLogger = Datalogger(moku_address, force_connect=True)
# https://apis.liquidinstruments.com/reference/oscilloscope/
xps = XPSController(reset=False)


stage = xps.autocorr_stage
min_move = stage.min_limit
max_move = stage.max_limit


###
PEAK_POS_MM = 22.652
RANGE_PS = .2
RANGE_MM = stage.delay_to_distance(RANGE_PS)
STEP_SIZE_MM = 100e-6  # 100 nm step size

# Range of motion
MAX_POS_MM = PEAK_POS_MM + RANGE_MM
MIN_POS_MM = PEAK_POS_MM - RANGE_MM


# Set data logger to osc
dLogger.set_frontend(channel=2, impedance='1MOhm',
                     coupling="DC", range="10Vpp")
# Log 100 samples per second
dLogger.set_samplerate(1000)
dLogger.set_acquisition_mode(mode='Precision')

print("Logger setup done.")

###

print(f"Moving stage to {PEAK_POS_MM}")
stage.absolute_move(PEAK_POS_MM)
print(f"Stage moved to {PEAK_POS_MM}")

filenames = []


while stage.current_position() < MAX_POS_MM:
    logFile = dLogger.start_logging(duration=10,
                                    file_name_prefix=str(stage.current_position()).replace(".", "_"))

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
                     './log' + os.sep + fname)

dLogger.relinquish_ownership()
