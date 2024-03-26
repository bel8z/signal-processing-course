# 1. Tools 2h

* **Filters**
* Convolution
* **Wavelets**
* Spectra (FFT)
* Time-frequency (Laplace?)
* Cleaning/denoising
* Interpolation
* **Feature detection**
* SNR/RMS

# 2. Denoising 3h

* Smoothing (removal of high frequency, low amplitude noise) is typically done with convolution methods, moving mean and Gaussian kernel are the most common
* Gaussian smoothing can be used to denoise a time series full of spikes; basically the kernel should be choosen to make spikes akin to a probability distribution
* For Gaussian kernel parameterization, a key concept is the one of Full-Width-Half-Magnitude
* Median filter is useful for removing spike noise; use an histogram to choose the threshold and check the filter quality
* Teager-Kaiser energy-tracking operator (TKEO) useful for isolating a relevant feature of a signal which contains similar noise patterns, by suppressing noise and making the relevant signal larger
* Use **z-score** to further amplify the signal
* Baesyan criterion is useful to evaluate polynomial fitting
* When processing multiple series, a good way to visualize them is plotting a matrix, with time on the horizontal axis, one row for each series and using a color to represent the signal magnitude; this is especially useful to visualize patterns (and pattern removal, via polynomial or least-squares fitting)
* **Brownian noise** with cumsum(randn)
* Average multiple series for denoising

# 3. Spectral analysis 3.3h

* How **zero padding** is useful to FFT?
* Does a very high sampling frequency produce artifacts (harmonic distortion)?
* What is an **Hamm window**?
* How to interpret the FFT output? How does it relate to number of samples, sampling rate and Nyquist frequency?