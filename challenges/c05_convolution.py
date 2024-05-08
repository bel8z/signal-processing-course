#! /usr/bin/env python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft


def main():
    # Load and plot data
    try:
        data = sp.io.loadmat("denoising_codeChallenge.mat")
    except:
        data = sp.io.loadmat("TimeSeriesDenoising/denoising_codeChallenge.mat")

    signal = data['origSignal'][0]
    solution = data['cleanedSignal'][0]
    count = len(signal)

    # Median filtering
    # Threshold and window size are adjusted by looking at the histogram of the
    # dirty and clean signals
    cleaned = median_filter(signal, threshold=5, k=10)
    plt.figure("Histograms")
    plt.hist(signal, 100)
    plt.hist(cleaned, 100)

    # Moving average smoothing as convolution
    h = 25
    w = (2 * h + 1)
    k = np.ones(w) / w

    conv_size = count + w - 1
    cleaned_x = fft(cleaned, conv_size)
    k_x = fft(k, conv_size)
    filter_x = k_x * cleaned_x

    cleaned = np.real(ifft(filter_x))[h:-h]
    freq = np.linspace(0, 1, conv_size // 2 + 1)

    # Final plot
    location = "right"

    plt.figure("Signals")

    plt.subplot(3, 1, 1)
    plt.plot(signal, label="signal")
    plt.plot(solution, label="solution")
    plt.legend(loc=location)

    plt.subplot(3, 1, 2)
    plt.plot(signal, label="signal")
    plt.plot(cleaned, label="cleaned")
    plt.legend(loc=location)

    plt.subplot(3, 1, 3)
    plt.plot(cleaned, label="cleaned")
    plt.plot(solution, label="solution")
    plt.legend(loc=location)

    plt.figure("Spectra")
    plt.subplot(2, 1, 1)
    plt.plot(freq, amplitude(k_x)[:len(freq)])
    plt.subplot(2, 1, 2)
    plt.plot(freq, amplitude(filter_x)[:len(freq)])
    plt.plot(freq, amplitude(cleaned_x)[:len(freq)])
    plt.xlabel("Frequency (normalized)")

    plt.show()


def amplitude(spectrum):
    return abs(spectrum) * 2 / len(spectrum)


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


# Usual business
if __name__ == "__main__":
    main()
