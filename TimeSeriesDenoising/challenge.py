#! /usr/bin/env python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


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
    plt.figure()
    plt.hist(signal, 100)
    plt.hist(cleaned, 100)

    # Smoothing
    k = 200
    w = (2 * k + 1)

    # Gaussian smoothing
    fwhm = w / 3  # ≃ 2.355 / 6
    gt, gw = gauss_kernel(k, fwhm)
    assert len(gw) == w

    plt.figure("Gaussian kernel")
    # plt.plot(gt, gw, 'ko-')
    plt.plot(gt, gw)

    gauss_conv = np.convolve(cleaned, gw, mode='same')
    gauss_loop = np.zeros(count)
    for i in range(k, count - k):
        # each point is the weighted average of k surrounding points
        gauss_loop[i] = np.sum(cleaned[i - k:i + k + 1] * gw)

    # Mean smoothing
    # mean = np.convolve(cleaned, np.ones(w)/w, mode='same')
    mean = cleaned.copy()  # np.zeros(count)
    for i in range(count):
        beg = max(0, i - k)
        end = min(i + k + 1, count)
        mean[i] = np.mean(cleaned[beg:end])

    # Final plot
    location = "right"

    plt.subplot(3, 1, 1)
    plt.plot(signal, label="signal")
    plt.plot(cleaned, label="cleaned")
    plt.plot(solution, label="solution")
    plt.legend(loc=location)

    plt.subplot(3, 1, 2)
    plt.plot(gauss_conv, label="gauss_conv")
    plt.plot(gauss_loop, label="gauss_loop")
    plt.plot(mean, label="mean")
    plt.plot(solution, label="solution")
    plt.legend(loc=location)

    plt.subplot(3, 1, 3)
    plt.plot(gauss_conv - solution, label="gauss_conv")
    plt.plot(gauss_loop - solution, label="gauss_loop")
    plt.plot(mean - solution, label="mean")
    plt.legend(loc=location)

    plt.show()


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
    time = np.arange(-k, k+1) * freq
    # create normalized Gaussian window
    weight = np.exp(- (4 * np.log(2) * time ** 2) / fwhm ** 2)
    return (time, weight / np.sum(weight))


# Usual business
if __name__ == "__main__":
    main()
