import numpy as np
from scipy.signal import find_peaks
from utils.foo import give_name, return_choose_vec
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

    return band_energy

def extract_r_peaks(ecg_f, th_min,th_max):
    distance = int(0.6 * 100)
    height = np.mean(ecg_f)
    r_peaks, _ = find_peaks(ecg_f, distance=distance, height=height)
    r_peaks = r_peaks[r_peaks >= th_min]
    r_peaks = r_peaks[r_peaks <= th_max]
    
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

def st_features(ecg, fs, r_peaks):
    features = []
    for r in r_peaks:
        # Ponto J ~ 60 ms após R
        j_point = r + int(0.06 * fs)
        
        # Fim do ST ~ 200 ms após R
        st_end = r + int(0.20 * fs)
        
        # Linha de base: média do PR segment (80ms antes do QRS)
        baseline = np.mean(ecg[r - int(0.12 * fs): r - int(0.04 * fs)])
        
        # Elevação do ST (média 60-80ms após R menos baseline)
        st_elev = np.mean(ecg[j_point : j_point + int(0.02 * fs)]) - baseline
        
        # Duração do ST
        st_duration = (st_end - j_point) / fs
        
        # Inclinação do ST
        x = np.arange(j_point, st_end)
        y = ecg[j_point:st_end]
        slope = np.polyfit(x, y, 1)[0]
        
        features.append((st_elev, st_duration, slope))
    
    return np.array(features)  # Cada linha: [elevação, duração, inclinação]



def extract_features(data, fs, ch, tolerance_num, duration, sig_len):
    full_ch_features = []
    for i in ch:
        ecg = data.p_signal[:,i]
        ecg_f = apply_filters_with_padding(ecg, fs)

        #peaks
        r_peaks = extract_r_peaks(ecg_f, 40, 959)

        if len(r_peaks) >= -1: #TODO: remove
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

            st_vals = st_features(ecg_f, fs, r_peaks)

            st_elev_mean = np.mean(st_vals[:, 0])
            st_elev_std  = np.std(st_vals[:, 0])

            st_dur_mean = np.mean(st_vals[:, 1])
            st_dur_std  = np.std(st_vals[:, 1])

            st_slope_mean = np.mean(st_vals[:, 2])
            st_slope_std  = np.std(st_vals[:, 2])

            band_energy = PSD(ecg_f,fs,(0.5,4), (4, 15), (15,40))

            ch_name = give_name(i)

            choose = return_choose_vec(ch_name)

            features = {
                f'{ch_name}_mean_rr_interval': mean_rr,
                f'{ch_name}_std_rr_interval': std_rr,
                f'{ch_name}_mean_qq_interval': mean_qq,
                f'{ch_name}_std_qq_interval': std_qq,
                f'{ch_name}_mean_ss_interval': mean_ss,
                f'{ch_name}_std_ss_interval': std_ss,
                f'{ch_name}_mean_pp_interval': mean_pp,
                f'{ch_name}_std_pp_interval': std_pp,
                f'{ch_name}_mean_tt_interval': mean_tt,
                f'{ch_name}_std_tt_interval': std_tt,
                f'{ch_name}_mean_qrs':mean_qrs,
                f'{ch_name}_std_qrs':std_qrs,
                f'{ch_name}_st_elev_mean':st_elev_mean,
                f'{ch_name}_st_elev_std':st_elev_std,
                f'{ch_name}_st_dur_mean':st_dur_mean,
                f'{ch_name}_st_dur_std':st_dur_std,
                f'{ch_name}_st_slope_mean':st_slope_mean,
                f'{ch_name}_st_slope_std':st_slope_std,
                f'{ch_name}_band_energy_low': band_energy['low'],
                f'{ch_name}_band_energy_mid': band_energy['mid'],
                f'{ch_name}_band_energy_high': band_energy['high'],
            }
            final_dict = {}
            index = 0
            for i in features.keys():
                if choose[index]:
                    final_dict[i] = features[i]
                index+=1


            full_ch_features.append(final_dict)
        
        new_feat = {}
        for i in full_ch_features:
            new_feat.update(i)
    
    return new_feat