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
    #r_peaks = r_peaks[ecg_f[r_peaks] >= th_amp]
    r_peaks = r_peaks[r_peaks >= 40]
    r_peaks = r_peaks[r_peaks <= th_samp]
    #heart_rate = 60 / mean_rr #bpm
    
    return r_peaks

def extract_peaks(ecg_f,r_peaks):
    start_qrs = []
    end_qrs = []
    
    q_peaks = []
    for i in range(len(r_peaks)):
        q_peaks.append(np.argmin(ecg_f[max(r_peaks[i]-20,0):r_peaks[i]]) + max(r_peaks[i]-20,0))
        # extracting qrs starts
        dxdy = np.gradient(ecg_f)
        dxdy = dxdy[max(q_peaks[-1] - 5, 0): q_peaks[-1]-1]
        start_qrs.append(np.argmin(dxdy) + max(q_peaks[-1] - 5, 0))

    s_peaks = []
    for i in range(len(r_peaks)):
        s_peaks.append(np.argmin(ecg_f[r_peaks[i]:min(r_peaks[i]+12,999)]) + r_peaks[i])
        dxdy = np.gradient(ecg_f)
        dxdy = dxdy[s_peaks[-1]+1 : min(s_peaks[-1] + 5, 999)]
        end_qrs.append(np.argmin(dxdy) + s_peaks[-1])
        

    p_peaks = []
    for i in range(len(q_peaks)):
        p_peaks.append(np.argmax(ecg_f[max(q_peaks[i]-20, 0):q_peaks[i]]) + max(q_peaks[i]-20,0))

    t_peaks = []
    for i in range(len(r_peaks)):
        t_peaks.append(np.argmax(ecg_f[min(s_peaks[i]+3, 999):min(r_peaks[i]+40,999)]) + s_peaks[i]+3)

    qrs_peaks = []
    for i in range(len(r_peaks)):
        qrs_peaks.append((start_qrs[i], end_qrs[i]))

    return [q_peaks, s_peaks, p_peaks, t_peaks, qrs_peaks]

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
    r_peaks, q_peaks, s_peaks, p_peaks, t_peaks, qrs_wave = peaks

    qrs_s = [x[0] for x in qrs_wave]
    qrs_e = [x[1] for x in qrs_wave]

    t = np.linspace(0, duration, sig_len)
    plt.figure(figsize=(12, 4))
    plt.plot(t, ecg_f, label='ECG', color='orange')
    plt.plot(t[r_peaks], ecg_f[r_peaks], 'rx', label='Picos R')
    plt.plot(t[q_peaks], ecg_f[q_peaks], 'bx', label='Picos Q')
    plt.plot(t[s_peaks], ecg_f[s_peaks], 'gx', label='Picos S')
    plt.plot(t[p_peaks], ecg_f[p_peaks], 'cx', label='Picos P')
    plt.plot(t[t_peaks], ecg_f[t_peaks], 'kx', label='Picos T')
    plt.plot(t[qrs_s], ecg_f[qrs_s], 'mx', label='Picos QRS_S')
    plt.plot(t[qrs_e], ecg_f[qrs_e], 'yx', label='Picos QRS_E')
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

def get_qrs_intervals(qrs_wave): 
    # (a,b) -> b-a

    qrs_i = [(x[1] - x[0]) for x in qrs_wave]

    mean = np.mean(qrs_i)
    std = np.std(qrs_i)
    return [mean,std]

def extract_features(data, fs, ch, tolerance_num, duration, sig_len):
    full_ch_features = []
    for i in ch:
        ecg = data.p_signal[:,i]
        ecg_f = apply_filters_with_padding(ecg, fs)

        #peaks
        r_peaks = extract_r_peaks(ecg_f, int(100*0.6), np.mean(ecg_f), 0.15, 959)

        if len(r_peaks) >= tolerance_num:
            #print(r_peaks)
            q_peaks, s_peaks, p_peaks, t_peaks, qrs_wave = extract_peaks(ecg_f, r_peaks)
            # plot_with_peaks(ecg_f, duration, sig_len, [r_peaks,q_peaks, s_peaks, p_peaks, t_peaks, qrs_wave])

            #intervalos
            rr_intervals, mean_rr, std_rr = extract_intervals(r_peaks, fs)
            qq_intervals, mean_qq, std_qq = extract_intervals(q_peaks, fs)
            ss_intervals, mean_ss, std_ss = extract_intervals(s_peaks, fs)
            pp_intervals, mean_pp, std_pp = extract_intervals(p_peaks, fs)
            tt_intervals, mean_tt, std_tt = extract_intervals(t_peaks, fs)
            
            mean_qrs, std_qrs = get_qrs_intervals(qrs_wave)

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
                f'mean_qrs_{ch_name}':mean_qrs,
                f'std_qrs_{ch_name}':std_qrs,
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

def test_one_signal(name, ch, tolerance_num):
    ecg = wfdb.rdrecord(name)
    feat = extract_features(ecg, ecg.fs, ch, tolerance_num, ecg.sig_len / ecg.fs, ecg.sig_len)
    return feat

def generate_csv(Y, path, tolerance_num, ch):
    # Load raw signal data
    X = load_raw_data(Y, sampling_rate, path)

    arr = []
    for data in X:
        #parametros
        fs = data.fs
        duration = data.sig_len / fs
        t = np.linspace(0, duration, data.sig_len)

        #sinal
        features = extract_features(data, fs, channels, tolerance_num, data.sig_len / data.fs, data.sig_len)
        arr.append(features)
        
    features_df = pd.DataFrame(arr, index=Y.index)
    Y_MERGE = pd.merge(Y, features_df, on=Y.index, how='outer')

    Y_FINAL = Y_MERGE.drop(columns=['key_0', 'filename_lr'])

    Y_FINAL.to_csv(path + "generated_csv/features.csv")
    print("csv criado corretamente")

if __name__ == "__main__":
    #parameters
    channels = [1,2,5,6,7,8] # channels to extract features
    tolerance_num = 5 # quantos picos devem ser detectados pra considerarmos um sinal válido
    path = '../' # replace to your path
    sampling_rate=100 # 100hz or 500hz

    # load and convert annotation data
    Y = pd.read_csv(path + 'generated_csv/filtered_database.csv', index_col='ecg_id')
    generate_csv(Y, path, tolerance_num, channels)

    # test one signal
    # test_feat = test_one_signal(path + "records100\\00000\\00016_lr", channels, tolerance_num)

    # plot ecg
    # test = wfdb.rdrecord(path + "records100/00000/00175_lr")
    # plot_ecg(test.p_signal, sampling_rate, 10,1000)