The autocorrelator controls a delay stage and uses a Moku Go for readout of Quasi-CW signal using a 10 MHz photodiode.
The readouts are flat DC signals with levels corresponding (supposedly linear) to the SHG intensity.
The controller driver and delay stage abstract classes are borrowed from:
https://github.com/Siwick-Research-Group/uedinst

# Logic:

- Stage Scan func:
  - Move delay & readout in 50 fs steps. Observe any changes in intensity signal and mark the stage position
- Acquisition func:
  - move ~ 100 fs (user set) before the peak, scan across in 100 nm steps. Plotting in plt.

NOTE: With `uedinst` under development, I can work on both this project and request pull to `uedinst` by installing `uedinst` with `python setup.py develop`


sys:1: ResourceWarning: unclosed <socket.socket fd=1180, family=AddressFamily.AF_INET6, type=SocketKind.SOCK_STREAM, proto=0, laddr=('fe80::5028:c5cc:a2ae:1ae4%9', 3409, 0, 9), raddr=('fe80::7269:79ff:feb9:1a40%9', 80, 0, 9)>
sys:1: ResourceWarning: unclosed <socket.socket fd=1188, family=AddressFamily.AF_INET6, type=SocketKind.SOCK_STREAM, proto=0, laddr=('fe80::5028:c5cc:a2ae:1ae4%9', 3415, 0, 9), raddr=('fe80::7269:79ff:feb9:1a40%9', 80, 0, 9)>

have to figure out how to close socket connections