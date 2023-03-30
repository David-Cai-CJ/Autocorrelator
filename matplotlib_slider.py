import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from lmfit import minimize, Parameters
from lmfit.models import ConstantModel, LorentzianModel, GaussianModel

import matplotlib
import matplotlib.pylab as plt
from matplotlib.widgets import Slider, Button
from scipy.signal import find_peaks
import os
import sys


folder = 'double_pulse_or_not'
file = os.path.sep.join(['logging', folder, 'summary.csv'])
pos_mm, sig, error = np.loadtxt(file, delimiter=',').T


class Plotter:
    def __init__(self):
        self.fig, self.ax = plt.subplots()

    def plot(self, obj):
        self.obj = obj
        plt.plot()
        self.l = plt.plot(obj.x_data, obj.series())
        _vars = obj.get_variables()
        plt.subplots_adjust(bottom=0.03*(len(_vars)+2))
        self.sliders = []
        for i, var in enumerate(_vars):
            self.add_slider(i*0.03, var[0], var[1], var[2])
        plt.show()

    def add_slider(self, pos, name, min, max):
        ax = plt.axes([0.1, 0.02+pos, 0.8, 0.02])
        slider = Slider(ax, name, min, max, valinit=getattr(self.obj, name))
        self.sliders.append(slider)

        def update(val):
            setattr(self.obj, name, val)
            self.l[0].set_ydata(self.obj.series())
            self.fig.canvas.draw_idle()
        slider.on_changed(update)


# class SinFunction:
#     def __init__(self):
#         self.freq = 1.0
#         self.amp = 0.5
#         self.t = np.arange(0.0, 1.0, 0.001)

#     def series(self):
#         return self.amp*np.sin(2*np.pi*self.freq*self.t)

#     def get_variables(self):
#         return [
#             ('freq', 0.1, 10),
#             ('amp', 0.1, 1)
#         ]


class Model:
    def __init__(self, x_data, y_data, model, params):
        self.model = model
        self.params = params
        self.x_data = x_data
        self.y_data = y_data

    def series(self):
        return self.model.eval(self.params, x=self.x_data)

    def get_variables(self):
        free_vars = []

        acceptable = ['amp', 'sigma', 'center']
        ranges = [[0, 100], [.0005, .2], [11.4, 11.7]]

        for var in self.params.keys():
            for w_idx, w in acceptable:
                if w in var and self.params[var].expr == None:
                    free_vars.append((var, ranges[w_idx, 0], ranges[w_idx[1]]))

        return free_vars


def gen_model(n_sub_peaks):
    model = ConstantModel() + GaussianModel(prefix='g0_')

    for n in np.arange(n_sub_peaks):
        for side in ['L', 'R']:
            model += GaussianModel(prefix=f'g{n+1}_{side}_')

    params = model.make_params()

    params['c'].value = np.min(sig)

    for n in np.arange(n_sub_peaks):
        params[f'g{n+1}_L_amplitude'].expr = f'g{n+1}_R_amplitude'
        params[f'g{n+1}_L_sigma'].expr = f'g{n+1}_R_sigma'
        params[f'g{n+1}_L_center'].expr = f'2 * g0_center - g{n+1}_R_center'

    return model, params


if len(sys.argv) == 1:
    n_sub_peaks = 0
else:
    n_sub_peaks = int(sys.argv[1])

k = Plotter()
k.plot(Model(pos_mm, sig, *gen_model(n_sub_peaks)))

# k = Plotter()
# k.plot(SinFunction())
