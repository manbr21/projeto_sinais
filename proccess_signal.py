import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import wfdb
import pywt

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdrecord(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdrecord(path+f) for f in df.filename_hr]
    #data = np.array([signal for signal, meta in data]) #para rdsamp
    return data

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

def wavelet_denoise(signal, wavelet='sym4', level=4):
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

def PSD(ecg_f,fs,low_interval, mid_interval, high_interval):
    #dominio frequencia
    segment = ecg_f[:fs * 5]
    fft_vals = np.abs(np.fft.fft(segment))
    fft_freqs = np.fft.fftfreq(len(segment), 1/fs)
    pos_mask = fft_freqs > 0
    fft_vals = fft_vals[pos_mask]
    fft_freqs = fft_freqs[pos_mask]

    #energia
    band_energy = {
        'low': np.sum(fft_vals[(fft_freqs >= low_interval[0]) & (fft_freqs < low_interval[1])]),
        'mid': np.sum(fft_vals[(fft_freqs >= mid_interval[0]) & (fft_freqs < mid_interval[1])]),
        'high': np.sum(fft_vals[(fft_freqs >= high_interval[0]) & (fft_freqs < high_interval[1])]),
    }

    peak_freq = fft_freqs[np.argmax(fft_vals)]

    return [band_energy, peak_freq]

def extract_r_peaks(ecg_f,distance,height,th_amp, th_samp):
    r_peaks, _ = find_peaks(ecg_f, distance=distance, height=height)
    r_peaks = r_peaks[ecg_f[r_peaks] >= th_amp]
    r_peaks = r_peaks[r_peaks >= 40]
    r_peaks = r_peaks[r_peaks <= th_samp]
    #heart_rate = 60 / mean_rr #bpm
    
    return r_peaks

def extract_peaks(ecg_f,r_peaks):
    q_peaks = []
    for i in range(len(r_peaks)):
        q_peaks.append(np.argmin(ecg_f[max(r_peaks[i]-20,0):r_peaks[i]]) + max(r_peaks[i]-20,0))

    s_peaks = []
    for i in range(len(r_peaks)):
        s_peaks.append(np.argmin(ecg_f[r_peaks[i]:min(r_peaks[i]+12,999)]) + r_peaks[i])

    p_peaks = []
    for i in range(len(q_peaks)):
        p_peaks.append(np.argmax(ecg_f[max(q_peaks[i]-20, 0):q_peaks[i]]) + max(q_peaks[i]-20,0))

    t_peaks = []
    for i in range(len(r_peaks)):
        t_peaks.append(np.argmax(ecg_f[min(s_peaks[i]+3, 999):min(r_peaks[i]+40,999)]) + s_peaks[i]+3)

    return [q_peaks, s_peaks, p_peaks, t_peaks]

def give_name(i):
    if i == 0:
        return "I"
    elif i == 1:
        return "II"
    elif i == 2:
        return "III"
    elif i == 3:
        return "aVL"
    elif i == 4:
        return "aVR"
    elif i == 5:
        return "aVF"
    else:
        return f"V{i-5}"

def plot_with_peaks(ecg_f, duration, sig_len, peaks):
    r_peaks, q_peaks, s_peaks, p_peaks, t_peaks = peaks
    t = np.linspace(0, duration, sig_len)
    plt.figure(figsize=(12, 4))
    plt.plot(t, ecg_f, label='ECG', color='orange')
    plt.plot(t[r_peaks], ecg_f[r_peaks], 'rx', label='Picos R')
    plt.plot(t[q_peaks], ecg_f[q_peaks], 'bx', label='Picos Q')
    plt.plot(t[s_peaks], ecg_f[s_peaks], 'gx', label='Picos S')
    plt.plot(t[p_peaks], ecg_f[p_peaks], 'cx', label='Picos P')
    plt.plot(t[t_peaks], ecg_f[t_peaks], 'kx', label='Picos T')
    plt.title('Sinal de ECG com Picos PQRST')
    plt.xlabel('Tempo (s)')
    plt.ylabel('Amplitude (mV)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_ecg(ecg, fs,duration, sig_len):
    fig, axs = plt.subplots(nrows=12)
    t = np.linspace(0, duration, sig_len)
    for i,ax in enumerate(axs):
        ecg_f = apply_filters_with_padding(ecg[:,i], fs)
        ax.plot(t, ecg_f, label='ECG', color='orange')
        if i == 0:
            ax.set_title('ECG completo')
        if i == 11:
            ax.set_xlabel('t(s)')
        ax.set_ylabel(give_name(i))
        ax.grid(True)

    plt.tight_layout()
    plt.show()

def extract_intervals(x_peak, fs):
    x_intervals = np.diff(x_peak) / fs
    mean_x = np.mean(x_intervals)
    std_x = np.std(x_intervals)

    return [x_intervals, mean_x, std_x]

def extract_features(data, fs, duration, sig_len, ch):
    full_ch_features = []
    for i in ch:
        ecg = data.p_signal[:,i]
        ecg_f = apply_filters_with_padding(ecg, fs)

        #peaks
        r_peaks = extract_r_peaks(ecg_f, int(100*0.6), np.mean(ecg_f), 0.15, 959)

        q_peaks, s_peaks, p_peaks, t_peaks = extract_peaks(ecg_f, r_peaks)
        plot_with_peaks(ecg_f, duration, sig_len, [r_peaks,q_peaks, s_peaks, p_peaks, t_peaks])

        #intervalos
        rr_intervals, mean_rr, std_rr = extract_intervals(r_peaks, fs)
        qq_intervals, mean_qq, std_qq = extract_intervals(q_peaks, fs)
        ss_intervals, mean_ss, std_ss = extract_intervals(s_peaks, fs)
        pp_intervals, mean_pp, std_pp = extract_intervals(p_peaks, fs)
        tt_intervals, mean_tt, std_tt = extract_intervals(t_peaks, fs)

        band_energy, peak_freq = PSD(ecg_f,fs,(0.5,4), (4, 15), (15,40))

        ch_name = give_name(i)

        features = {
            f'mean_rr_interval_s_{ch_name}': mean_rr,
            f'std_rr_interval_s_{ch_name}': std_rr,
            f'mean_qq_interval_s_{ch_name}': mean_qq,
            f'std_qq_interval_s_{ch_name}': std_qq,
            f'mean_ss_interval_s_{ch_name}': mean_ss,
            f'std_ss_interval_s_{ch_name}': std_ss,
            f'mean_pp_interval_s_{ch_name}': mean_pp,
            f'std_pp_interval_s_{ch_name}': std_pp,
            f'mean_tt_interval_s_{ch_name}': mean_tt,
            f'std_tt_interval_s_{ch_name}': std_tt,
            f'band_energy_low_{ch_name}': band_energy['low'],
            f'band_energy_mid_{ch_name}': band_energy['mid'],
            f'band_energy_high_{ch_name}': band_energy['high'],
            f'dominant_frequency_Hz_{ch_name}': peak_freq
        }
        full_ch_features.append(features)
    
    new_feat = {}
    for i in full_ch_features:
        new_feat.update(i)
    
    return new_feat

def test_one_signal(name, ch):
    ecg = wfdb.rdrecord(name)
    feat = extract_features(ecg, ecg.fs, ecg.sig_len / ecg.fs, ecg.sig_len, channels)
    return feat


path = '/home/manbr/Documents/EC/5p/sinais_sistemas/projeto/bd_sinais/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path + 'filtered_database.csv', index_col='ecg_id')

#parameters
channels = [1,8]

#test_feat = test_one_signal(path + "records100/16000/16967_lr", channels)

# # Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

arr = []
for data in X:
    #parametros
    fs = data.fs
    duration = data.sig_len / fs
    t = np.linspace(0, duration, data.sig_len)

    #sinal
    features = extract_features(data, fs, duration, data.sig_len, channels)
    arr.append(features)
    
features_df = pd.DataFrame(arr, index=Y.index)
Y_MERGE = pd.merge(Y, features_df, on=Y.index, how='outer')

Y_FINAL = Y_MERGE.drop(columns=['key_0','filename_lr'])

Y_FINAL.to_csv("features.csv")
print("csv criado corretamente")

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
