from uedinst.delay_stage import XPSController
from time import sleep


xps = XPSController()



xps.delay_stage.absolute_move(0)
xps.compensation_stage.absolute_move(0)

stage = xps.delay_stage
# stage = xps.compensation_stage

min_move = stage.min_limit
max_move = stage.max_limit



while True:
    try:
        stage.absolute_move(min_move)
        sleep(2)
        stage.absolute_move(max_move)
        sleep(2)
    except KeyboardInterrupt:
        break

