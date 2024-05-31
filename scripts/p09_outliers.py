# %%
import numpy as np

import scipy
from scipy import signal
import scipy.io as sio
from scipy.interpolate import griddata

import matplotlib.pyplot as plt

import util

# %% [markdown]
#
# ---
# # VIDEO: Outliers via standard deviation threshold
# ---
#

# %%
# signal is log-normal noise
N = 10000
time = np.arange(0, N) / N
signal1 = np.exp(.5 * np.random.randn(N))

# add some random outiers
nOutliers = 50
randpnts = np.random.randint(0, N, nOutliers)
signal1[randpnts] = np.random.rand(
    nOutliers) * (np.max(signal1) - np.min(signal1)) * 10

# show the signal
plt.plot(time, signal1, 'ks-')

# auto-threshold based on mean and standard deviation
avg, std = util.avg_std(signal1)
threshold = avg + 3 * std
plt.plot([time[0], time[-1]], [threshold, threshold], 'b--')
plt.show()

plt.hist(signal1, 50)
plt.show()


# %%
# interpolate outlier points


# remove supra-threshold points
outliers = signal1 > threshold

# and interpolate missing points
signalR = signal1.copy()
signalR[outliers] = griddata(
    time[~outliers], signal1[~outliers], time[outliers], method='cubic')
signalM = util.median_filter(signal1, 20, threshold)

# and plot the new results
plt.plot(time, signal1, 'k-')
plt.plot(time, signalR, 'r-')
plt.plot(time, signalM, 'g-')

# optional zoom
plt.xlim([.1, .2])

plt.show()


# %% [markdown]
#
# ---
# # VIDEO: Outliers via local threshold exceedance
# ---
#

# %%
# data downloaded from:
# http://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2017

# import data, etc.
matdat = sio.loadmat('outliers/forex.mat')
forex = np.squeeze(matdat['forex'])

N = len(forex)
time = np.arange(0, N) / N


# plot it
plt.figure("Forex")
plt.xlabel('Time (year)')
plt.ylabel('EUR/USD')

# add global thresholds
avg, std = util.avg_std(forex)
threshmax = avg + 3 * std
threshmin = avg - 3 * std
plt.plot([time[0], time[-1]], [threshmax, threshmax], 'r--', label='M+3std')
plt.plot([time[0], time[-1]], [threshmin, threshmin], 'k--', label='M-3std')

# %%
# local threshold

# window size as percent of total signal length
pct_win = 5  # in percent, not proportion!

# convert to indices
k = int(len(forex) * (pct_win / 2 / 100))

# initialize statistics time series to be the global stats
avg_ts, std_ts = util.avg_std(forex, k)
threshmax = avg_ts + 3 * std_ts
threshmin = avg_ts - 3 * std_ts

# compute local outliers
outliers = (forex > threshmax) | (forex < threshmin)

# plotting...
plt.plot(time, forex, 'k')
plt.plot(time, threshmax, 'm--', label='M±3std local')
plt.plot(time, threshmin, 'm--')

# and plot those
plt.plot(time[outliers], forex[outliers], 'ro', label='Outliers')

plt.legend()
plt.xlabel('Time (year)')
plt.ylabel('EUR/USD')
plt.title('Using a %d%% window size' % pct_win)
plt.show()

# %% [markdown]
#
# ---
# # VIDEO: Outlier time windows via sliding RMS
# ---
#

# %%
# generate signal with varying variability
n = 2000
p = 15  # poles for random interpolation

# amplitude modulator
signal1 = np.interp(np.linspace(0, p, n), np.arange(0, p),
                    np.random.rand(p) * 30)
signal1 = signal1 + np.random.randn(n)


# add some high-amplitude noise
signal1[200:221] = signal1[200:221] + np.random.randn(21) * 9
signal1[1500:1601] = signal1[1500:1601] + np.random.randn(101) * 9

# %%
# detect bad segments using sliding std

# window size as percent of total signal length
pct_win = 2  # in percent, not proportion!

# convert to indices
k = int(n * (pct_win / 2 / 100))

# compute sliding average and std deviation
avg_ts, std_ts = util.avg_std(signal1, k)

# pick threshold inspecting the std deviation plot
avg, std = util.avg_std(std_ts)
threshmax = avg + std
threshmin = avg - std

plt.figure("Bad segments detection")

plt.subplot(2, 1, 1)
plt.plot(std_ts, 's-', label='Local STD')
plt.plot([0, n], [threshmax, threshmax], 'r', label='Threshold')
plt.plot([0, n], [threshmin, threshmin], 'r')
plt.legend()

# mark bad regions in original time series
signalR = signal1.copy()
signalR[std_ts > threshmax] = np.nan

plt.subplot(2, 1, 2)
plt.plot(signal1, 'b', label='Original')
plt.plot(signalR, 'r', label='Thresholded')
plt.legend()

plt.show()
