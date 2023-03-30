import numpy as np
from lmfit import minimize, Parameters
from lmfit.models import ConstantModel, LorentzianModel, GaussianModel

import matplotlib
import matplotlib.pylab as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks
import os
import sys

matplotlib.use('TKAgg')

folder = 'double_pulse_or_not'
file = os.path.sep.join(['logging', folder, 'summary.csv'])
pos_mm, sig, error = np.loadtxt(file, delimiter=',').T


fig, ax = plt.subplots()
line, = ax.plot(pos_mm, sig, 'k-')
ax.set_xlabel('Stage Position [mm]')

fig.subplots_adjust(left=0.25, bottom=0.25)

model = ConstantModel() + GaussianModel(prefix='g0_')

if len(sys.argv) == 1:
    n_sub_peaks = 0
else:
    n_sub_peaks = int(sys.argv[1])

for n in np.arange(n_sub_peaks):
    for side in ['L', 'R']:
        model += GaussianModel(prefix=f'g{n+1}_{side}_')

params = model.make_params()

params['c'].value = np.min(sig)

for n in np.arange(n_sub_peaks):
    params[f'g{n+1}_L_amplitude'].expr = f'g{n+1}_R_amplitude'
    params[f'g{n+1}_L_sigma'].expr = f'g{n+1}_R_sigma'
    params[f'g{n+1}_L_center'].expr = f'2 * g0_center - g{n+1}_R_center'

print(params['g0_sigma'].max)

plt.show()
