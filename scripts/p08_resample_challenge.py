
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from scipy.signal import detrend, welch, spectrogram
from scipy.fftpack import fft, ifft
import util

data = sp.io.loadmat("resample/resample_codeChallenge.mat")

# data = sp.io.loadmat("spectral/spectral_codeChallenge.mat")

orig = np.array([data["origT"][0], data["origS"][:, 0]])
proc = np.array([data["time"][0], data["signal"][:, 0]])

plt.figure("Signals")
plt.plot(orig[0], orig[1], label="Original")
plt.plot(proc[0], proc[1], label="Processed")
plt.show()

plt.figure("Spectra")
origX = util.power(fft(orig[1]))
procX = util.power(fft(proc[1]))
plt.plot(origX, label="Original")
plt.plot(procX, label="Processed")
plt.show()
