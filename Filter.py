from scipy.signal import butter, filtfilt, cheby2
from scipy import signal


# chebyshev2 prove to be better PPG - An optimal filter for short photoplethysmogram signals (DOI: 10.1038/sdata.2018.76)
#rp = Passband Ripple, rs = Stopband Attenuation
#Please change the parameter based on your data

#Low and High cutoff - 2 Hz and 5 Hz

def chebyshev2_bandpass_filter(signal, low_cutoff_freq, high_cutoff_freq, sampling_freq, rp, rs, order=4):
    nyquist_freq = 0.5 * sampling_freq
    low_normalized_cutoff = low_cutoff_freq / nyquist_freq
    high_normalized_cutoff = high_cutoff_freq / nyquist_freq
    b, a = cheby2(order, rs, [low_normalized_cutoff, high_normalized_cutoff], btype='band', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
    
    

# GSR - Butterworth
    
def butter_lowpass_filter(signal, cutoff_freq, sampling_freq, order=2):
    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff = cutoff_freq / nyquist_freq
    b, a = butter(order, normalized_cutoff, btype='low', analog=False)
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal
