import numpy as np
from scipy.signal import find_peaks
from utils.foo import give_name
from utils.filter_signal import apply_filters_with_padding

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