
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio
from scipy.fftpack import fft, ifft
from scipy.signal import detrend, welch


def median_filter(s_in, k, threshold):
    s_out = s_in.copy()

    # find data values above the threshold
    over_threshold = np.where(abs(s_in) > threshold)[0]

    for i in over_threshold:
        # lower and upper bounds
        beg = max(0, i - k)
        end = min(i + k + 1, len(s_in))
        # compute median of surrounding points
        s_out[i] = np.median(s_in[beg:end])

    return s_out


def gauss_kernel(k, fwhm, sample_rate=1):
    freq = 1 / sample_rate
    # normalized time vector
    time = np.arange(-k, k + 1) * freq
    # create normalized Gaussian window
    weight = np.exp(- (4 * np.log(2) * time ** 2) / fwhm ** 2)
    return (time, weight / np.sum(weight))


def amplitude(spectrum, spectrum_len=-1):
    spectrum_len = spectrum_len if spectrum_len > 0 else len(spectrum)
    return abs(spectrum) * 2 / spectrum_len


def freqvec(nyquist_freq, spectrum_len):
    return np.linspace(0, nyquist_freq, int(np.floor(spectrum_len / 2) + 1))


class wavelet:
    @staticmethod
    def morlet(timevec, freq, fwhm):
        """
        Morlet wavelet in the time domain
        """
        csw = np.cos(2 * np.pi * freq * timevec)  # cosine wave
        gaussian = np.exp(-(4 * np.log(2) * timevec**2) / fwhm**2)  # Gaussian
        return csw * gaussian

    @staticmethod
    def haar(timevec, sample_time=-1):
        """
        Haar wavelet in the time domain
        """
        sample_time = sample_time if sample_time > 0 else np.diff(timevec)[0]
        wave = np.zeros(len(timevec))
        wave[np.argmin(timevec**2): np.argmin((timevec - .5)**2)] = 1
        wave[np.argmin((timevec - .5)**2)
                       : np.argmin((timevec - 1 - sample_time)**2)] = -1
        return wave

    @staticmethod
    def mexican(timevec, s):
        """
        Mexican hat wavelet in the time domain
        """
        s2 = s**2
        return (2 / (np.sqrt(3 * s) * np.pi**.25)) * (1 - (timevec**2) / s2) * np.exp((-timevec**2) / (2 * s2))

    @staticmethod
    def dog(timevec, sPos, sNeg):
        """
        Difference Of Gaussians (approximation of Laplacian of Gaussian) wavelet 
        in the time domain
        """
        tsq = timevec**2
        # create the two Gaussians
        g1 = np.exp((-tsq) / (2 * sPos**2)) / (sPos * np.sqrt(2 * np.pi))
        g2 = np.exp((-tsq) / (2 * sNeg**2)) / (sNeg * np.sqrt(2 * np.pi))

        # their difference is the DoG
        return g1 - g2
