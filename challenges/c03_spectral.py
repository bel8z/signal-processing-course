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

series = np.array([data["time"][0],
                   data["signal"][0]])
srate = data["srate"][0][0]

plt.figure("Signal")
plt.plot(series[0], series[1])

# %%
n = len(series[1])

# compute window size and overlap
winlen = 0.5  # seconds
winsize = int(winlen * srate)
overlap = 0  # winsize // 2  # half window

# window onset times
onsets = np.arange(0, n - winsize, winsize - overlap)

# Hann window
hannw = 0.5 * (1 - np.cos(2 * np.pi * np.linspace(0, 1, winsize)))
plt.figure("Hann window")
plt.plot(hannw)

# %%
nfft = winsize * 2

# compute frequency (Hz) vector based on window size
hz = np.linspace(0, srate * 0.5, nfft // 2 + 1)

# initialize the power matrix (windows x frequencies)
matrix = np.zeros((len(onsets), len(hz)))

# loop over frequencies
for wi in range(0, len(onsets)):
    chunk = series[1][onsets[wi]:][0: winsize]

    # apply Hann taper to data
    chunk = chunk * hannw

    # compute its spectrum
    matrix[wi] = np.abs(fft(chunk, n=nfft)[0:len(hz)]) ** 2


# %%
t = np. linspace(series[0][0], series[0][-1], len(onsets))
plt.figure("Spectrogram")
plt.pcolormesh(t, hz, matrix.T, cmap="hot")
plt.ylim([0, 40])

# %%
plt.show()
