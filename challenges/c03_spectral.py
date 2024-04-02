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

# %%
n = len(signal)

# compute window size and overlap
winlen = 1  # seconds
winsize = int(winlen * srate)
overlap = winsize // 2  # half window

# window onset times
onsets = np.arange(0, n - winsize, winsize - overlap)

# compute frequency (Hz) vector based on window size
hz = np.linspace(0, srate * 0.5, winsize // 2 + 1)

# Hann window
hannw = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, 1, winsize)))
plt.figure("Hann window")
plt.plot(hannw)

# %%
# initialize the power matrix (windows x frequencies)
matrix = np.zeros((len(onsets), len(hz)))

# loop over frequencies
for wi in range(0, len(onsets)):
    chunk = signal[onsets[wi]:][0: winsize]

    # apply Hann taper to data
    chunk = chunk * hannw

    # compute its spectrum
    matrix[wi] = np.abs(fft(chunk)[0:len(hz)]) ** 2


# %%
t = np. linspace(time[0], time[-1], len(onsets))
plt.figure()
plt.pcolormesh(hz, t, matrix)

# %%
