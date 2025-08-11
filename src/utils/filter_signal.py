import numpy as np
import pywt
from scipy.signal import butter, filtfilt

def apply_filters_with_padding(signal, fs):
    pad_size = min(200, len(signal)//2)

    # Espelha início e fim
    padded = np.concatenate([
        signal[:pad_size][::-1],  # início invertido
        signal,
        signal[-pad_size:][::-1]  # fim invertido
    ])

    # Aplicar os filtros normalmente
    filtered = highpass_filter(padded, fs)
    filtered = notch_filter(filtered, fs)
    filtered = wavelet_denoise(filtered)

    # Remove padding
    return filtered[pad_size:-pad_size]

def wavelet_denoise(signal, wavelet='sym4', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    denoised = pywt.waverec(coeffs_thresh, wavelet)
    return denoised[:len(signal)]

def highpass_filter(signal, fs, cutoff=0.5, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered = filtfilt(b, a, signal)
    return filtered

def notch_filter(signal, fs, freq=50.0, Q=30.0):
    from scipy.signal import iirnotch
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    filtered = filtfilt(b, a, signal)
    return filtered