from IPython.display import Audio

import numpy as np

import scipy.signal as signal
from scipy.io import loadmat

import matplotlib
import matplotlib.pyplot as plt

for backend in ("QtAgg", "WebAgg"):
    try:
        matplotlib.use(backend)
        break
    except:
        pass
matplotlib.style.use('fast')

# load the file

try:
    matfile = loadmat('glassDance.mat')
except FileNotFoundError:
    matfile = loadmat('quickwin/glassDance.mat')

# this is a clip of Philip Glass, Dance VII (https://www.youtube.com/watch?v=LpewOlR-z_4)

glassclip = matfile['glassclip']
srate = matfile['srate'][0][0]

# play the music!

Audio(np.array(glassclip[:, 0]), rate=srate)

# some variables for convenience

pnts = len(glassclip)
timevec = np.arange(0, pnts)/srate

# setup graph

fig, (p1, p2, p3) = plt.subplots(3, 1)

# draw the time-domain signals

p1.plot(timevec, glassclip)
# p1.xlabel('Time (s)')
# plt.show()

# inspect the power spectrum

hz = np.linspace(0, srate/2, int(np.floor(len(glassclip)/2)+1))
powr = abs(np.fft.fft(glassclip[:, 0])/pnts)

p2.plot(hz, powr[:len(hz)])
p2.set_xlim([100, 2000])
p2.set_ylim([0, np.max(powr)])
# p2.xlabel('Frequency (Hz)')
# p2.ylabel('Amplitude')
# plt.show()

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

# plt.plot(hz, powr[:len(hz)])
p2.plot(hz, powrF[:len(hz)], 'r')
p2.set_xlim([100, 2000])
p2.set_ylim([0, np.max(powr)])
# p2.xlabel('Frequency (Hz)')
# p2.ylabel('Amplitude')
# plt.show()

# plot the time-frequency response

frex, time, tf = signal.spectrogram(
    glassclip[:, 0], window=('tukey', .25), fs=srate, noverlap=100)

p3.plot([timevec[0], timevec[-1]], [frange[0], frange[0]], 'w:')
p3.plot([timevec[0], timevec[-1]], [frange[1], frange[1]], 'w:')
p3.pcolormesh(time, frex, np.log(tf), vmin=-20, vmax=-10)
p3.set_ylim([0, 2000])
# p3.xlabel('Time (s)')
# p3.ylabel('Frequency (Hz)')

# display all plots

plt.show()


# Play the filtered signal!

Audio(np.array(filtglass[:, 0]), rate=srate)
