# %%
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy.signal import detrend, welch, spectrogram
from scipy.fftpack import fft, ifft
from scipy.signal import firls, firwin, butter, filtfilt
%matplotlib widget

# %%
# Load data
try:
    data = sp.io.loadmat("../filtering/filtering_codeChallenge.mat")
except:
    data = sp.io.loadmat("filtering/filtering_codeChallenge.mat")

x = data["x"][:, 0]
y = data["y"][:, 0]
N = len(x)
fs = data["fs"][0, 0]
fn = fs / 2
hz = np.linspace(0, fn, N // 2)

plt.figure()
plt.plot(x)
plt.plot(y)
plt.show()

# %%
# Plot spectra
xs = np.abs(sp.fftpack.fft(x) / N)**2
ys = np.abs(sp.fftpack.fft(y) / N)**2
plt.figure()
plt.plot(hz, xs[0:len(hz)])
plt.plot(hz, ys[0:len(hz)])
plt.xlim(0, 80)
plt.show()

# %%
# Filter design
band0 = [5, 17, 22, 35]
order = 800
kern0 = firwin(order, band0, fs=fs, pass_zero=False)
kern0_x = np.abs(fft(kern0, N)) ** 2

plt.close("Filter design")
plt.figure("Filter design")
plt.subplot(1, 2, 1)
plt.plot(kern0)
plt.subplot(1, 2, 2)
plt.plot(hz, kern0_x[0:len(hz)])
for i in range(0, len(band0), 2):
    lo = band0[i]
    up = band0[i + 1]
    plt.plot([lo, lo, up, up], [0, 1, 1, 0], 'b')
plt.xlim(0, band0[-1] + 10)
plt.show()

# %%
yf = filtfilt(kern0, 1, y)

plt.figure()
# plt.plot(x)
plt.plot(y)
plt.plot(yf)
plt.show()

yfs = abs(fft(yf) / N)**2

plt.figure()
plt.plot(hz, ys[0:len(hz)], label="Mike")
plt.plot(hz, yfs[0:len(hz)], label="Matt")
plt.xlim(0, 80)
plt.legend()
plt.show()

# %%
