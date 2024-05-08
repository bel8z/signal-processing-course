# %%
import numpy as np
import matplotlib.pyplot as plt

import scipy
import scipy.io as sio
from scipy.fftpack import fft, ifft
from scipy.signal import detrend, welch

if __package__:
    from .. import util
    from ..util import wavelet
else:
    import sys
    from os import path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    import util
    from util import wavelet


# %% [markdown]
#
# ---
# # VIDEO: What are wavelets?
# ---
#
plt.figure("Wavelets")

# %%
# general simulation parameters
fs = 1024

# centered time vector (5 seconds)
timevec = np.arange(0, fs * 5) / fs
timevec = timevec - np.mean(timevec)

# for power spectrum
hz = util.freqvec(fs / 2, len(timevec))


# %%
# Morlet wavelet

# parameters
freq = 4  # peak frequency
fwhm = .5  # full-width at half-maximum in seconds

# Morlet wavelet and spectrum
MorletWavelet = wavelet.morlet(timevec, freq, fwhm)
MorletWaveletPow = util.amplitude(fft(MorletWavelet))


# time-domain plotting
plt.subplot(421)
plt.plot(timevec, MorletWavelet, 'k')
plt.xlabel('Time (sec.)')
plt.title('Morlet wavelet in time domain')

# frequency-domain plotting
plt.subplot(422)
plt.plot(hz, MorletWaveletPow[:len(hz)], 'k')
plt.xlim([0, freq * 3])
plt.xlabel('Frequency (Hz)')
plt.title('Morlet wavelet in frequency domain')


# %%
# Haar wavelet

# create Haar wavelet and amplitude spectrum
HaarWavelet = wavelet.haar(timevec, sample_time=1 / fs)
HaarWaveletPow = util.amplitude(fft(HaarWavelet))

# time-domain plotting
plt.subplot(423)
plt.plot(timevec, HaarWavelet, 'k')
plt.xlabel('Time (sec.)')
plt.title('Haar wavelet in time domain')

# frequency-domain plotting
plt.subplot(424)
plt.plot(hz, HaarWaveletPow[:len(hz)], 'k')
plt.xlim([0, freq * 3])
plt.xlabel('Frequency (Hz)')
plt.title('Haar wavelet in frequency domain')

# %%
# Mexican hat wavelet

# create wavelet and amplitude spectrum
MexicanWavelet = wavelet.mexican(timevec, s=0.4)
MexicanPow = util.amplitude(fft(MexicanWavelet))

# time-domain plotting
plt.subplot(425)
plt.plot(timevec, MexicanWavelet, 'k')
plt.xlabel('Time (sec.)')
plt.title('Mexican wavelet in time domain')

# frequency-domain plotting
plt.subplot(426)
plt.plot(hz, MexicanPow[:len(hz)], 'k')
plt.xlim([0, freq * 3])
plt.xlabel('Frequency (Hz)')
plt.title('Mexican wavelet in frequency domain')

# %%
# Difference of Gaussians (DoG)
# (approximation of Laplacian of Gaussian)

# define sigmas
sPos = .1
sNeg = .5

# create wavelet and amplitude spectrum
DoG = wavelet.dog(timevec, sPos, sNeg)
DoGPow = util.amplitude(fft(DoG))

# time-domain plotting
plt.subplot(427)
plt.plot(timevec, DoG, 'k')
plt.xlabel('Time (sec.)')
plt.title('DoG wavelet in time domain')

# frequency-domain plotting
plt.subplot(428)
plt.plot(hz, DoGPow[:len(hz)], 'k')
plt.xlim([0, freq * 3])
plt.xlabel('Frequency (Hz)')
plt.title('DoG wavelet in frequency domain')

plt.show()

# %% [markdown]
#
# ---
# # VIDEO: Convolution with wavelets
# ---
#

# %%
# general simulation parameters
fs = 1024

# centered time vector (5 seconds)
timevec = np.arange(0, 5 * fs) / fs
timevec = timevec - np.mean(timevec)

# for power spectrum
hz = util.freqvec(fs / 2, len(timevec))

# %%
# create wavelets

# parameters
freq = 4  # peak frequency
fwhm = .5  # full-width at half-maximum in seconds

MorletWavelet = wavelet.morlet(timevec, freq, fwhm)
HaarWavelet = wavelet.haar(timevec, 1 / fs)
MexicanWavelet = wavelet.mexican(timevec, s=0.4)
DoG = wavelet.dog(timevec, sPos, sNeg)


# %%
# convolve with random signal

# signal
signal1 = detrend(np.cumsum(np.random.randn(len(timevec))))

# convolve signal with different wavelets
morewav = np.convolve(signal1, MorletWavelet, 'same')
haarwav = np.convolve(signal1, HaarWavelet, 'same')
mexiwav = np.convolve(signal1, MexicanWavelet, 'same')
dogswav = np.convolve(signal1, DoG, 'same')

# amplitude spectra
morewaveAmp = util.amplitude(fft(morewav))
haarwaveAmp = util.amplitude(fft(haarwav))
mexiwaveAmp = util.amplitude(fft(mexiwav))
dogswaveAmp = util.amplitude(fft(dogswav))

# plotting
plt.figure("Convolution with wavelets")

# the signal
plt.plot(timevec, signal1, 'k')
plt.title('Signal')
plt.xlabel('Time (s)')
plt.show()

# the convolved signals
plt.subplot(211)
plt.plot(timevec, morewav, label='Morlet')
plt.plot(timevec, haarwav, label='Haar')
plt.plot(timevec, mexiwav, label='Mexican')
plt.plot(timevec, dogswav, label='DoG')
plt.title('Time domain')
plt.legend()

# spectra of convolved signals
plt.subplot(212)
plt.plot(hz, morewaveAmp[:len(hz)], label='Morlet')
plt.plot(hz, haarwaveAmp[:len(hz)], label='Haar')
plt.plot(hz, mexiwaveAmp[:len(hz)], label='Mexican')
plt.plot(hz, dogswaveAmp[:len(hz)], label='DoG')
plt.yscale('log')
plt.xlim([0, 40])
plt.legend()
plt.xlabel('Frequency (Hz.)')
plt.show()

# %% [markdown]
#
# ---
# # VIDEO: Wavelet convolution for narrowband filtering
# ---
#

# %%
# simulation parameters
srate = 4352  # hz
time = np.arange(0, 8425) / srate
hz = util.freqvec(srate / 2, len(time))

# pure noise signal
signal1 = np.exp(.5 * np.random.randn(len(time)))

# let's see what it looks like
plt.subplot(211)
plt.plot(time, signal1, 'k')
plt.xlabel('Time (s)')

# in the frequency domain
signalX = util.amplitude(fft(signal1))
plt.subplot(212)
plt.plot(hz, signalX[:len(hz)], 'k')
plt.xlim([1, srate / 6])
plt.ylim([0, 300])
plt.xlabel('Frequency (Hz)')
plt.show()


# %%
# create and inspect the Morlet wavelet

# wavelet parameters
ffreq = 34  # filter frequency in Hz
fwhm = .12  # full-width at half-maximum in seconds
# wavelet time vector (same sampling rate as signal!)
wavT = np.arange(-3, 3, 1 / srate)

# create the wavelet
wav = wavelet.morlet(wavT, ffreq, fwhm)


# amplitude spectrum of wavelet
# (note that the wavelet needs its own hz because different length)
wavX = util.amplitude(fft(wav))
wavHz = util.freqvec(srate / 2, len(wav))

# plot it!
plt.subplot(211)
plt.plot(wavT, wav, 'k')
plt.xlim([-.5, .5])
plt.xlabel('Time (sec.)')

plt.subplot(212)
plt.plot(wavHz, wavX[:len(wavHz)], 'k')
plt.xlim([0, ffreq * 2])
plt.xlabel('Frequency (Hz)')
plt.show()

# %%
# now for convolution

convres = scipy.signal.convolve(signal1, wav, 'same')

assert len(signal1) == len(time)
assert len(convres) == len(time)

# show in the time domain
plt.subplot(211)
plt.plot(time, convres, 'r')

# and in the frequency domain
plt.subplot(212)
convresX = util.amplitude(fft(convres))
plt.plot(hz, convresX[:len(hz)], 'r')
plt.show()

# Time-domain wavelet normalization is... annoying and difficult.
# Let's do it in the frequency domain


# %%
# "manual" convolution

nConv = len(time) + len(wavT) - 1
halfw = int(np.floor(len(wavT) / 2))

# spectrum of wavelet
wavX = fft(wav, nConv)

# now normalize in the frequency domain
# note: ensure we're taking the magnitude of the peak;
#  I didn't explain this in the video but it ensures normalization by
#  the magnitude and not the complex value.
wavX = wavX / np.abs(np.max(wavX))
# also equivalent:
wavX = (np.abs(wavX) / max(np.abs(wavX))) * np.exp(1j * np.angle(wavX))

# now for the rest of convolution
convres = ifft(wavX * fft(signal1, nConv))
convres = np.real(convres[halfw:-halfw + 1])

# time domain
plt.plot(time, signal1, 'k', label='original')
plt.plot(time, convres, 'b', label='filtered, norm.')
plt.legend()
plt.xlabel('Time')
plt.show()

# frequency domain
convresX = util.amplitude(fft(convres))
plt.plot(hz, signalX[:len(hz)], 'k', label='original')
plt.plot(hz, convresX[:len(hz)], 'b', label='filtered, norm.')
plt.ylim([0, 300])
plt.xlim([0, 90])
plt.show()

# %%
# to preserve DC offset, compute and add back

convres = convres + np.mean(signal1)

plt.plot(time, signal1, 'k', label='original')
plt.plot(time, convres, 'm', label='filtered, norm.')
plt.legend()
plt.xlabel('Time')
plt.show()

# %% [markdown]
#
# ---
# # Time-frequency analysis with complex wavelets
# ---
#

# %%
# data from http://www.vibrationdata.com/Solomon_Time_History.zip

equake = np.loadtxt('Solomon_Time_History.txt')

# more convenient
times = equake[:, 0]
equake = equake[:, 1]
srate = np.round(1 / np.mean(np.diff(times)))


# plot the signal

# time domain
plt.subplot(211)
plt.plot(times / 60 / 60, equake)
plt.xlim([times[0] / 60 / 60, times[-1] / 60 / 60])
plt.xlabel('Time (hours)')

# frequency domain using pwelch
plt.subplot(212)
winsize = srate * 60 * 10  # window size of 10 minutes
f, welchpow = welch(equake, fs=srate, window=np.hanning(
    winsize), nperseg=winsize, noverlap=winsize / 4)
plt.semilogy(f, welchpow)
plt.xlabel('frequency [Hz]')
plt.ylabel('Power')
plt.ylim([10e-11, 10e-6])
plt.show()


# %%
# setup time-frequency analysis

# parameters (in Hz)
numFrex = 40
minFreq = 2
maxFreq = srate / 2
npntsTF = 1000  # this one's in points

# frequencies in Hz
frex = np.linspace(minFreq, maxFreq, numFrex)

# wavelet widths (FWHM in seconds)
fwhms = np.linspace(5, 15, numFrex)


# time points to save for plotting
tidx = np.arange(1, len(times), npntsTF)


# setup wavelet and convolution parameters
wavet = np.arange(-10, 10, 1 / srate)
halfw = int(np.floor(len(wavet) / 2))
nConv = len(times) + len(wavet) - 1


# create family of Morlet wavelets
cmw = np.zeros((len(wavet), numFrex), dtype=complex)
# loop over frequencies and create wavelets
for fi in range(0, numFrex):
    cmw[:, fi] = np.exp(2 * 1j * np.pi * frex[fi] * wavet) * \
        np.exp(-(4 * np.log(2) * wavet**2) / fwhms[fi]**2)

# plot them
plt.pcolormesh(wavet, frex, np.abs(cmw).T, vmin=0, vmax=1)
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.show()


# %%
# run convolution

# initialize time-frequency matrix
tf = np.zeros((len(frex), len(tidx)))
tfN = np.zeros((len(frex), len(tidx)))

# baseline time window for normalization
basetidx = [0, 0]
basetidx[0] = np.argmin((times - -1000)**2)
basetidx[1] = np.argmin(times**2)
basepow = np.zeros(numFrex)


# spectrum of data
dataX = fft(equake, nConv)

# loop over frequencies for convolution
for fi in range(0, numFrex):

    # create wavelet
    waveX = fft(cmw[:, fi], nConv)

    # note: ensure we're taking the magnitude of the peak;
    #  I didn't explain this in the video but it ensures normalization by
    #  the magnitude and not the complex value.
    waveX = waveX / np.abs(np.max(waveX))

    # convolve
    as1 = ifft(waveX * dataX)
    # trim
    as1 = as1[halfw:-halfw]

    # power time course at this frequency
    powts = np.abs(as1)**2

    # baseline (pre-quake)
    basepow[fi] = np.mean(powts[range(basetidx[0], basetidx[1])])

    tf[fi, :] = 10 * np.log10(powts[tidx])
    tfN[fi, :] = 10 * np.log10(powts[tidx] / basepow[fi])


# %%
# show time-frequency maps

# "raw" power
plt.subplot(211)
plt.pcolormesh(times[tidx], frex, tf, vmin=-150, vmax=-70)
plt.xlabel('Time'), plt.ylabel('Frequency (Hz)')
plt.title('"Raw" time-frequency power')

# pre-quake normalized power
plt.subplot(212)
plt.pcolormesh(times[tidx], frex, tfN, vmin=-15, vmax=15)
plt.xlabel('Time'), plt.ylabel('Frequency (Hz)')
plt.title('"Raw" time-frequency power')
plt.show()


# normalized and non-normalized power
plt.subplot(211)
plt.plot(frex, np.mean(tf, axis=1), 'ks-')
plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (10log_{10})')
plt.title('Raw power')

plt.subplot(212)
plt.plot(frex, np.mean(tfN, axis=1), 'ks-')
plt.xlabel('Frequency (Hz)'), plt.ylabel('Power (norm.)')
plt.title('Pre-quake normalized power')
plt.show()

# %% [markdown]
#
# ---
# # VIDEO: Time-frequency analysis of brain signals
# ---
#

# %%
# load in data
braindat = sio.loadmat('data4TF.mat')
timevec = braindat['timevec'][0]
srate = braindat['srate'][0]
data = braindat['data'][0]

# plot the signal
plt.plot(timevec, data)
plt.xlabel('Time (s)'), plt.ylabel('Voltage (\muV)')
plt.title('Time-domain signal')
plt.show()

# %%
# create complex Morlet wavelets

# wavelet parameters
nfrex = 50  # 50 frequencies
frex = np.linspace(8, 70, nfrex)
fwhm = .2  # full-width at half-maximum in seconds

# time vector for wavelets
wavetime = np.arange(-2, 2, 1 / srate)


# initialize matrices for wavelets
wavelets = np.zeros((nfrex, len(wavetime)), dtype=complex)

# create complex Morlet wavelet family
for wi in range(0, nfrex):
    # Gaussian
    gaussian = np.exp(-(4 * np.log(2) * wavetime**2) / fwhm**2)

    # complex Morlet wavelet
    wavelets[wi, :] = np.exp(1j * 2 * np.pi * frex[wi] * wavetime) * gaussian


# show the wavelets
plt.plot(wavetime, np.real(wavelets[10, :]), label='Real part')
plt.plot(wavetime, np.imag(wavelets[10, :]), label='Imag part')
plt.xlabel('Time')
plt.xlim([-.5, .5])
plt.legend()
plt.show()

plt.pcolormesh(wavetime, frex, np.real(wavelets))
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Real part of wavelets')
plt.xlim([-.5, .5])
plt.show()

# %%
# run convolution using spectral multiplication

# convolution parameters
nconv = len(timevec) + len(wavetime) - 1  # M+N-1
halfk = int(np.floor(len(wavetime) / 2))

# Fourier spectrum of the signal
dataX = fft(data, nconv)

# initialize time-frequency matrix
tf = np.zeros((nfrex, len(timevec)))


# convolution per frequency
for fi in range(0, nfrex):

    # FFT of the wavelet
    waveX = fft(wavelets[fi, :], nconv)
    # amplitude-normalize the wavelet
    waveX = waveX / np.abs(np.max(waveX))

    # convolution
    convres = ifft(waveX * dataX)
    # trim the "wings"
    convres = convres[halfk - 1:-halfk]

    # extract power from complex signal
    tf[fi, :] = np.abs(convres)**2

# %%
# plot the results

plt.pcolormesh(timevec, frex, tf, vmin=0, vmax=1e3)
plt.xlabel('Time (s)'), plt.ylabel('Frequency (Hz)')
plt.title('Time-frequency power')
plt.show()
