import numpy as np
import matplotlib.pylab as plt
import glob
from scipy import signal
from scipy.stats import moment
import os
from scipy.optimize import curve_fit
import matplotlib.ticker as mtick
import sys
import matplotlib


matplotlib.use('TKAgg')

####
folder = 'menlo_monday_after_ascend'


dir = 'logging' + os.path.sep + folder
li_files =  sorted(glob.glob(dir+os.path.sep+ '[0-9]*.li'))

[os.system(f'liconvert --csv {f}')  for f in li_files if not os.path.exists(f.replace('.li', '.csv'))];
#####

def lor(x, x0, g):
    return 1/np.pi/(1+((x-x0)/g)**2)


def gau(x, x0, s):
    return 1/np.sqrt(2/np.pi)*np.exp(-1/2*(x-x0)**2/s**2)


def model(x, aL, aG, x0, g, s, C):
    return (aG * gau(x, x0, s) + C + aL * lor(x, x0, g))


files = sorted(glob.glob(dir+os.path.sep+ '[0-9]*.csv'))

pos_mm = np.array([float(".".join(os.path.basename(f).split('.')[:-1]).replace('_','.')) for f in files])
# print(".".join(os.path.basename(files[0]).split('.')[:-1]).replace('_','.'))
print(pos_mm)

sig= np.array([np.genfromtxt(f, skip_header= 8, dtype = float, 
                        delimiter = ',').mean(axis = 0)[-1] for f in files])

sig = np.array(sig) / np.max(sig)  
pos_mm, sig = np.array(sorted(zip(pos_mm, sig))).T

# plt.plot(pos_mm, sig)

left_found = False
right_found = False
normed = sig - np.min(sig)
normed /= np.max(normed)

for i, val in enumerate(normed - 0.5):
    if val > 0 and not left_found:
        left = i
        left_found = True
    if left_found and val < 0 and not right_found:
        right = i
        right_found = True

print(left, right)
print((pos_mm[right] - pos_mm[left])/1e3/2.998e8/1e-15*2)


model = lambda x, A,x0,s,C: A * gau(x,x0,s) + C

p0 = [1.,  11.706, .005, 0.]

fit, err = curve_fit(model, pos_mm, normed, p0=p0,
                     bounds=( [0.001, p0[1] - .02,  .0001, -1],
                             [ 1, p0[1] + .02,  .02, .2]))

aG, x0, s, C = fit

t_fs = (pos_mm - fit[1])/1e3/2.998e8/1e-15*2

fwhm_factor = 2.355
width = fit[-2]/1e3/3e8 / 1e-15 * 2
e_width = np.sqrt(np.diag(err))[-2]/1e3/3e8 / 1e-15

width_gamma = fit[3]/1e3/3e8 / 1e-15 * 2
e_width_gamma = np.sqrt(np.diag(err))[3]/1e3/3e8 / 1e-15

print(f'{width:.2f} +/- {e_width:.2f}')


# Plotting
fig = plt.figure()
gs = fig.add_gridspec(2, 1,  height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0)


ax = fig.add_subplot(gs[1, 0])
res_ax = fig.add_subplot(gs[0, 0], sharex=ax)

ax.set_xlabel('Delay (fs)')
# ax.axvline(t_fs[left], c='r')
# ax.axvline(t_fs[right], c='r')
# ax.axhline(0.5, c='r')

ax.plot(t_fs, normed, 'k.', ms=4, markevery=1, label='Data')

# ax.plot(t_fs, model(pos_mm, *p0), c='b', lw=1, dashes =[4,4],  label='Guess')
ax.plot(t_fs, model(pos_mm, *fit), c='b', lw=1,  label='Fit')
ax.set_xlabel('Delay (fs)')
ax.set_ylabel('Signal (arb. unit)')
ax.legend(loc='center left')

res_ax.plot(t_fs, (sig - model(pos_mm, *fit))/sig * 100, 'b-', lw=1)
res_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
res_ax.minorticks_on()
res_ax.set_ylim(-15, 10)
res_ax.set_ylabel('Residuals')
# ax.set_xlim(-250, 250)
plt.setp(res_ax.get_xticklabels(), visible=False)


text_out = "FWHM${\mathrm{auto.}}=$" + f"${fwhm_factor* width:.2f}\pm{fwhm_factor * e_width:.2f}$ fs\n"
# "$\mathrm{FWHM}_{\mathrm{source}}=$" + \
# f"${width/np.sqrt(2) * fwhm_factor:.0f}\pm{e_width/np.sqrt(2) *fwhm_factor:.0f}$ fs\n" +\

ax.text(.05, .95, s=text_out, transform=ax.transAxes, va='top')
fig.tight_layout()
# print('second moment', moment(normed-fit[1], moment=2))
fig.savefig('figures/'+ folder+'.pdf', bbox_inches = 'tight')

plt.show()

