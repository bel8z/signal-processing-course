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
    data = sp.io.loadmat("../filtering/filtering_codeChallenge.mat")
except:
    data = sp.io.loadmat("filtering/filtering_codeChallenge.mat")

# %%
