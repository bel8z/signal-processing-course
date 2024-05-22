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

# 3. Spectral analysis 3:30h

* How **zero padding** is useful to FFT?
* Does a very high sampling frequency produce artifacts (harmonic distortion)?
* What is an **Hamm window**?
* How to interpret the FFT output? How does it relate to number of samples, sampling rate and Nyquist frequency?
* Difference between spectrum, power spectrum, spectral density

# 4. Complex numbers 1h
* Geometrical representation of complex numbers
* Sin/Cos representation
* Why FFT results are complex? How they relate to the frequency components?
* Significance and representation of complex multiplication and division

# 5. Filtering 5h
* Use FIR for more stability, at the expense of processing resources
* IIR filters are smaller and faster, but less stable
* Evaluate ALL filters based on the actual filter outcome
* Evaluate FIR filters inspecting the kernel in both time and frequency domains
* Evaluate IIR filters inspecting the impulse response, in both domains
* FIRLS is more flexible, FIR1 gives good frequency transitions by default
* Zero-phase-shifting is obtained by filtering forward, than backward, then inverting the output again
* Use reflection (of filter kernel size) to reduce edge effects (especially useful for zero phase)
* **When is window-sinc actually useful?**
* **How to choose the order of FIR/IIR filters?**
* **How to choose the frequency transition for filtering? Is it related to roll-off?**

# 6. Convolution 1h
* **does applying a kernel filter using convolution instead of filtfilt produce similar (or same) results?**


# 7. Wavelet 2h

# 8. Resampling 3h
* Upsample requires interpolation (typically a spline)
* When downsampling, low pass filter at the new nyquist (anti-aliasing)
* If the downsampling ratio is not an integer, compute a resampling ratio to first upsample, then downsample
* Linear interpolation is not that bad comparing to spline (especially in the freq domain)
* In case of an irregular signal, resample at higher frequency; the quality of interpolation depends on how the lowest average Nyquist of the irregular sampling relates to the frequency of the original sequences i.e. if it is too low the result is aliased
* Linear extrapolation is more conservative and stable
* Spectral interpolation: fft a window before and after the data hole, average the spectra, ifft and **detrend** because we are adding an artificial trend line to "stitch" to the original signal. This is more a "speculative" reconstruction of the data

# 9. Outlier detection
* A threshold based on mean and standard deviation is good for detecting outlier
* If the signal has a trend, either detrend and apply a global threshold or apply a local threshold by computing mean and std in moving windows
* Local std threshold is good for outlier points, not so for outlier windows
* RMS is good for identifying window, but again a threshold must be selected for the resulting RMS signal; this can be done visually or again applying local thresholds
* **check difference between mean and RMS**

# 10. Feature detection

# 11. Variability

# 12. Bonus

# 13. Resources

* https://en.wikipedia.org/wiki/Fourier_transform
* https://en.wikipedia.org/wiki/Discrete_Fourier_transform
* https://en.wikipedia.org/wiki/Oversampling
* https://en.wikipedia.org/wiki/Upsampling
* https://en.wikipedia.org/wiki/Supersampling
* https://en.wikipedia.org/wiki/Undersampling
* https://en.wikipedia.org/wiki/Spectral_density
* https://www.luisllamas.es/en/fftsharp/
* https://it.mathworks.com/matlabcentral/fileexchange/106725-fourier-analysis
* https://it.mathworks.com/help/signal/ug/power-spectral-density-estimates-using-fft.html
* https://it.mathworks.com/help/signal/filter-design.html
* https://dsp.stackexchange.com/questions/71278/can-fft-convolution-be-faster-than-direct-convolution-for-signals-of-large-sizes
* https://www.biorxiv.org/node/120256.full

## Video
* https://www.youtube.com/watch?v=Ls7AvuZG4kI
* https://www.youtube.com/watch?v=T647CGsuOVU
* https://www.youtube.com/watch?v=7ahrcB5HL0k

## Code 
* https://github.com/swharden/FftSharp/tree/main
* https://github.com/andrewkirillov/AForge.NET/blob/master/Sources/Math/FourierTransform.cs
* https://github.com/Samson-Mano/Fast_Fourier_Transform/tree/main
* https://www.mathdotnet.com
* https://github.com/mathnet/mathnet-numerics
* https://numpy.org/doc/stable/reference/generated/numpy.fft.rfft.html