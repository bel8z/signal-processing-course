# %%
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy.signal import detrend, welch, spectrogram
from scipy.fftpack import fft, ifft
%matplotlib widget

# %%
# Load and plot data
try:
    data = sp.io.loadmat("../spectral/spectral_codeChallenge.mat")
except:
    data = sp.io.loadmat("spectral/spectral_codeChallenge.mat")

signal = data["signal"][0]
time = data["time"][0]
srate = data["srate"][0][0]

plt.figure("Signal")
plt.plot(time, signal)
plt.show()

# %%
# create Hann window
winsize = int(2 * srate)  # 2-second window
hannw = .5 - np.cos(2 * np.pi * np.linspace(0, 1, winsize)) / 2

# number of FFT points (frequency resolution)
nfft = winsize

f, welchpow = welch(signal, fs=srate, window=hannw,
                    nperseg=winsize, noverlap=winsize / 4, nfft=nfft)

spectrum = 2 * np.abs(fft(signal, nfft)) / nfft
spec_power = np.abs(fft(signal, nfft))**2

# %%
plt.figure("FFT")
plt.plot(f, spectrum[0:len(f)])
plt.show()

plt.figure("Welch")
plt.semilogy(f, welchpow)
plt.semilogy(f, spec_power[0:len(f)])
# plt.xlim([0, 40])
plt.xlabel('frequency [Hz]')
plt.ylabel('Power')
plt.show()
# %%
