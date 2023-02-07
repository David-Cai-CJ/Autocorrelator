from moku.instruments import Datalogger
import time
import os
from moku.instruments import Oscilloscope
from mfa_delay_stage import MFA_PPD


osc = Oscilloscope('192.168.123.45')
dLogger = Datalogger('192.168.###.###', force_connect=False)
# https://apis.liquidinstruments.com/reference/oscilloscope/
stage = MFA_PPD()


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
                     coupling="DC", range="5Vpp")
# Log 100 samples per second
dLogger.set_samplerate(1000)
dLogger.set_acquisition_mode(mode='Precision')


###
stage.absolute_move(MIN_POS_MM)

while stage.current_position() < MAX_POS_MM:
    logFile = dLogger.start_logging(duration=10,
                                    file_name_prefix=str(stage.current_position()).replace(".", "_"))

    is_logging = True

    while is_logging:
        time.sleep(0.5)

        progress = dLogger.logging_progress()
        remaining_time = int(progress['time_to_end'])
        is_logging = remaining_time > 1
        print(f"Remaining time {remaining_time} seconds")

    dLogger.download("persist", logFile, os.path.join(os.getcwd(), logFile))

    stage.relative_move(STEP_SIZE_MM)

dLogger.relinquish_ownership()
