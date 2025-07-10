import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, medfilt
import wfdb
import ast
import pywt

import wfdb.plot

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

def notch_filter(signal, fs, freq=60.0, Q=30.0):
    from scipy.signal import iirnotch
    w0 = freq / (fs / 2)
    b, a = iirnotch(w0, Q)
    filtered = filtfilt(b, a, signal)
    return filtered

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
        print(type(data))
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

path = ''
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

#testando para um eletrocardiograma
data = wfdb.rdrecord("records100/00000/00001_lr")

# curr_data = data.p_signal[:,1]

# fs = data.fs

# ecg_filtered = highpass_filter(curr_data, fs, cutoff=0.5)
# ecg_filtered = wavelet_denoise(ecg_filtered)
# ecg_filtered = notch_filter(ecg_filtered, fs, freq=50.0)

# plt.figure(figsize=(12, 4))
# plt.plot(curr_data, label='Original', alpha=0.5)
# plt.plot(ecg_filtered, label='Filtrado')
# plt.legend()
# plt.title("ECG filtrado e original")
# plt.show()


rows = ["I/mV", "II/mV", "III/mV", "AVR/mV", "AVL/mV", "AVF/mV", "V1/mV", "V2/mV", "V3/mV", "V4/mV", "V5/mV", "V6/mV"]

fig, axs = plt.subplots(nrows=12, ncols=2, figsize=(12,18))

#normal signal
for i in range(data.n_sig):
    if i == 0:
        axs[i,0].set_title("Space Domain")
    elif i == data.n_sig - 1:
        axs[i,0].set_xlabel("time (s)")

    filtered_data = highpass_filter(data.p_signal[:,i], data.fs, cutoff=0.5)
    filtered_data = wavelet_denoise(filtered_data)
    filtered_data = notch_filter(filtered_data, data.fs, freq=50.0)

    axs[i,0].plot(np.linspace(0,10,data.sig_len), filtered_data)
    axs[i,0].set_ylabel(rows[i])

#transformed signal
for i in range(data.n_sig):

    filtered_data = highpass_filter(data.p_signal[:,i], data.fs, cutoff=0.5)
    filtered_data = wavelet_denoise(filtered_data)
    filtered_data = notch_filter(filtered_data, data.fs, freq=50.0)

    data_fft_signal = np.fft.fft(filtered_data)
    if i == 0:
        axs[i,1].set_title("Frequency Domain")
    elif i == data.n_sig - 1:
        axs[i,1].set_xlabel("freq (Hz)")
    freq = np.fft.fftfreq(len(data_fft_signal), d = 1/data.fs)
    pos = freq > 0
    axs[i,1].plot(freq[pos], np.abs(data_fft_signal[pos]), color='red')
    axs[i,1].set_ylabel(rows[i])

plt.show()
    

#print(data_fft_signal)

# for key, value in vars(data).items():
#     print(f"{key}: {value}")
# print(data.p_signal)
#wfdb.plot_wfdb(record=data, title="teste")

# Load raw signal data
# X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test
# test_fold = 10
# # Train
# X_train = X[np.where(Y.strat_fold != test_fold)]
# y_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
# # Test
# X_test = X[np.where(Y.strat_fold == test_fold)]
# y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
