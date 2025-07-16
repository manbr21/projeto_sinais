import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from hampel import hampel
import wfdb
import ast
import pywt

import wfdb.plot
import wfdb.processing

def apply_filters_with_padding(signal, fs):
    pad_size = 200  # ou fs//2, por exemplo

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

def wavelet_denoise(signal, wavelet='db6', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(signal)))
    coeffs_thresh = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    return pywt.waverec(coeffs_thresh, wavelet)

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

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdrecord(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdrecord(path+f) for f in df.filename_hr]
    #data = np.array([signal for signal, meta in data]) #para rdsamp
    return data

path = ''
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'filtered_database.csv', index_col='ecg_id')

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

#testando para um eletrocardiograma
arr = []
for data in X:
    #parametros
    fs = data.fs
    duration = data.sig_len / fs
    t = np.linspace(0, duration, data.sig_len)

    #sinal
    ecg = data.p_signal[:,1] #canal II do ECG

    ecg_f = apply_filters_with_padding(ecg, fs)

    #picos r
    #distancia minima entre batimentos
    r_peaks, _ = find_peaks(ecg_f, distance=int(fs * 0.6))

    #intervalos RR
    rr_intervals = np.diff(r_peaks) / fs
    mean_rr = np.mean(rr_intervals)
    std_rr = np.std(rr_intervals)
    #heart_rate = 60 / mean_rr #bpm

    #dominio frequencia
    segment = ecg_f[:fs * 5]
    fft_vals = np.abs(np.fft.fft(segment))
    fft_freqs = np.fft.fftfreq(len(segment), 1/fs)
    pos_mask = fft_freqs > 0
    fft_vals = fft_vals[pos_mask]
    fft_freqs = fft_freqs[pos_mask]

    #energia
    band_energy = {
        'low': np.sum(fft_vals[(fft_freqs >= 0.5) & (fft_freqs < 4)]),
        'mid': np.sum(fft_vals[(fft_freqs >= 4) & (fft_freqs < 15)]),
        'high': np.sum(fft_vals[(fft_freqs >= 15) & (fft_freqs < 40)]),
    }

    peak_freq = fft_freqs[np.argmax(fft_vals)]

    #features
    features = {
        'mean_rr_interval_s': mean_rr,
        'std_rr_interval_s': std_rr,
        'band_energy_low': band_energy['low'],
        'band_energy_mid': band_energy['mid'],
        'band_energy_high': band_energy['high'],
        'dominant_frequency_Hz': peak_freq
    }
    arr.append(features)
    
features_df = pd.DataFrame(arr, index=Y.index)
Y_MERGE = pd.merge(Y, features_df, on=Y.index, how='outer')

Y_FINAL = Y_MERGE.drop(columns=['key_0','filename_lr'])

Y_FINAL.to_csv("features.csv")
print("csv criado corretamente")

# plt.figure(figsize=(12, 4))
# plt.plot(t, ecg_f, label='ECG', color='orange')
# plt.plot(t[r_peaks], ecg_f[r_peaks], 'ro', label='Picos R')
# plt.title('Sinal de ECG com Picos R Detectados')
# plt.xlabel('Tempo (s)')
# plt.ylabel('Amplitude (mV)')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# for k, v in features.items():
#     print(f"{k}: {v}")



# Split data into train and test
# test_fold = 10
# # Train
# X_train = X[np.where(Y.strat_fold != test_fold)]
# y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# # Test
# X_test = X[np.where(Y.strat_fold == test_fold)]
# y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
