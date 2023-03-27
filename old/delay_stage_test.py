from uedinst.delay_stage import XPSController
from time import sleep


xps = XPSController()

stage = xps.autocorr_stage
stage.absolute_move(5.5)

min_move = stage.min_limit
max_move = stage.max_limit


while True:
    try:
        stage.relative_move(2)
        sleep(2)
        stage.relative_move(2)
        sleep(2)
    except KeyboardInterrupt:
        break
