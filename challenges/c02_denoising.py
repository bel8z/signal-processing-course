#! /usr/bin/env python3

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import util


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
    cleaned = util.median_filter(signal, threshold=5, k=10)
    plt.figure()
    plt.hist(signal, 100)
    plt.hist(cleaned, 100)

    # Smoothing
    k = 200
    w = (2 * k + 1)

    # Gaussian smoothing
    fwhm = w / 3  # ≃ 2.355 / 6
    gt, gw = util.gauss_kernel(k, fwhm)
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

    plt.figure("Signals")
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


# Usual business
if __name__ == "__main__":
    main()
