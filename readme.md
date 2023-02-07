The autocorrelator controls a delay stage and uses a Moku Go for readout of Quasi-CW signal using a 10 MHz photodiode.
The readouts are flat DC signals with levels corresponding (supposedly linear) to the SHG intensity.

# Logic:

- Stage Scan func:
  - Move delay & readout in 50 fs steps. Observe any changes in intensity signal and mark the stage position
- Acquisition func:
  - move ~ 100 fs (user set) before the peak, scan across in 100 nm steps. Plotting in plt.
