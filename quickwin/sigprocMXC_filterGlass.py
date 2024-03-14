import numpy as np

import scipy.signal as signal
from scipy.io import loadmat

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
matplotlib.style.use('fast')

# load the file

try:
    matfile = loadmat('glassDance.mat')
except FileNotFoundError:
    matfile = loadmat('quickwin/glassDance.mat')

# this is a clip of Philip Glass, Dance VII (https://www.youtube.com/watch?v=LpewOlR-z_4)

glassclip = matfile['glassclip']
srate = matfile['srate'][0][0]

# some variables for convenience

pnts = len(glassclip)
timevec = np.arange(0, pnts)/srate

# setup graph

fig = plt.figure(tight_layout=True)
gs = gridspec.GridSpec(2, 2)

# draw the time-domain signals

p1 = fig.add_subplot(gs[0, 0])
p1.set_xlabel('Time (s)')
p1.plot(timevec, glassclip)
plt.show(block=False)

# inspect the power spectrum

hz = np.linspace(0, srate / 2, int(np.floor(len(glassclip) / 2) + 1))
powr = abs(np.fft.fft(glassclip[:, 0]) / pnts)

p2 = fig.add_subplot(gs[1, 0])
p2.set_xlim([100, 2000])
p2.set_ylim([0, np.max(powr)])
p2.set_xlabel('Frequency (Hz)')
p2.set_ylabel('Amplitude')

p2.plot(hz, powr[:len(hz)])
plt.show(block=False)

# pick frequencies to filter

frange = [300,  460]
frange = [1000, 1100]
frange = [1200, 1450]

# design an FIR1 filter

fkern = signal.firwin(2001, frange, fs=srate, pass_zero=False)

# apply the filter to the signal

filtglass = np.zeros(np.shape(glassclip))
filtglass[:, 0] = signal.filtfilt(fkern, 1, glassclip[:, 0])
filtglass[:, 1] = signal.filtfilt(fkern, 1, glassclip[:, 1])

# plot the raw and filtered signal power spectra

powrF = abs(np.fft.fft(filtglass[:, 0])/pnts)

p2.plot(hz, powrF[:len(hz)], 'r')
plt.show(block=False)

# plot the time-frequency response

frex, time, tf = signal.spectrogram(
    glassclip[:, 0], window=('tukey', .25), fs=srate, noverlap=100)

p3 = fig.add_subplot(gs[:, 1])
p3.set_ylim([0, 2000])
p3.set_xlabel('Time (s)')
p3.set_ylabel('Frequency (Hz)')
p3.plot([timevec[0], timevec[-1]], [frange[0], frange[0]], 'w:')
p3.plot([timevec[0], timevec[-1]], [frange[1], frange[1]], 'w:')
p3.pcolormesh(time, frex, np.log(tf), vmin=-20, vmax=-10)
plt.show(block=True)
