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

    return band_energy

def extract_r_peaks(ecg_f, th_min,th_max):
    distance = int(0.6 * 100)
    # height = np.mean(ecg_f)
    #distance = int(100*60/200)
    height = np.percentile(ecg_f, 75)
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

    return [mean_x, std_x]

def get_qrs_intervals(qrs_wave): 
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

def morph_features(ecg, fs, r_peaks, q_peaks, s_peaks, t_peaks, qrs_wave):
    """
    Calcula features morfológicas detalhadas por batimento e agrega por canal.
    """
    t_amps, q_amps, r_amps, s_amps = [], [], [], []
    t_durations, q_durations, r_durations, s_durations = [], [], [], []
    t_areas, q_areas, r_areas, s_areas = [], [], [], []
    t_r_ratios, q_r_ratios, r_s_ratios = [], [], []
    st_areas, st_depressions = [], []
    qts, qtcs, jts = [], [], []
    t_inversions = 0
    q_pathologic_count = 0
    n = len(r_peaks)

    for i in range(n):
        try:
            r = int(r_peaks[i])
            q = int(q_peaks[i])
            s = int(s_peaks[i])
            t = int(t_peaks[i])
            qrs_start, qrs_end = qrs_wave[i]
        except Exception:
            continue

        baseline = np.mean(ecg[max(r - int(0.12 * fs), 0): max(r - int(0.04 * fs), 1)])

        # Amplitudes
        r_amp = ecg[r] - baseline
        t_amp = ecg[t] - baseline
        q_amp = baseline - ecg[q]
        s_amp = baseline - ecg[s]

        r_amps.append(r_amp)
        t_amps.append(t_amp)
        q_amps.append(q_amp)
        s_amps.append(s_amp)

        if t_amp < 0:
            t_inversions += 1

        # Ratios
        if abs(r_amp) > 1e-8:
            t_r_ratios.append(abs(t_amp) / abs(r_amp))
            q_r_ratios.append(abs(q_amp) / abs(r_amp))
            r_s_ratios.append(abs(r_amp) / abs(s_amp) if abs(s_amp) > 1e-8 else 0.0)

        # Q patológica
        if (q_amp > 0.25 * r_amp) and ((r - q) / fs > 0.04):
            q_pathologic_count += 1

        # Durações (aprox.)
        q_durations.append((r - q) / fs)
        r_durations.append((s - r) / fs)
        s_durations.append((qrs_end - s) / fs)

        # T duração
        search_end = min(t + int(0.4 * fs), len(ecg))
        thresh = 0.2 * abs(t_amp)
        seg = ecg[t:search_end] - baseline
        idxs = np.where(np.abs(seg) <= thresh)[0]
        if idxs.size > 0:
            t_end = t + int(idxs[0])
        else:
            t_end = min(t + int(0.12 * fs), search_end)
        t_durations.append((t_end - t) / fs)

        # Áreas
        q_areas.append(np.trapz(ecg[q:r] - baseline, dx=1/fs))
        r_areas.append(np.trapz(ecg[r:s] - baseline, dx=1/fs))
        s_areas.append(np.trapz(ecg[s:qrs_end] - baseline, dx=1/fs))
        t_areas.append(np.trapz(ecg[t:t_end] - baseline, dx=1/fs))

        # ST área / depressão
        j_point = qrs_end
        st_end = min(r + int(0.20 * fs), len(ecg))
        st_seg = ecg[j_point:st_end] - baseline
        st_area = np.trapz(st_seg, dx=1/fs)
        st_areas.append(st_area)
        st_depressions.append(np.mean(st_seg[st_seg < 0]) if np.any(st_seg < 0) else 0.0)

        # QT, QTc, JT
        qt = (t_end - q) / fs
        qts.append(qt)
        qtcs.append(qt / np.sqrt((r_peaks[i+1] - r_peaks[i]) / fs) if i+1 < n else np.nan)
        jts.append(qt - ((qrs_end - q) / fs))

    def safe_mean(x): return float(np.mean(x)) if len(x) > 0 else np.nan

    return {
        't_amplitude': safe_mean(t_amps),
        't_inversion': t_inversions / max(len(t_amps), 1),
        't_r_ratio': safe_mean(t_r_ratios),
        'q_r_ratio': safe_mean(q_r_ratios),
        'r_s_ratio': safe_mean(r_s_ratios),
        'q_amplitude': safe_mean(q_amps),
        'r_amplitude': safe_mean(r_amps),
        's_amplitude': safe_mean(s_amps),
        'q_duration': safe_mean(q_durations),
        'r_duration': safe_mean(r_durations),
        's_duration': safe_mean(s_durations),
        't_duration': safe_mean(t_durations),
        'q_area': safe_mean(q_areas),
        'r_area': safe_mean(r_areas),
        's_area': safe_mean(s_areas),
        't_area': safe_mean(t_areas),
        'st_area': safe_mean(st_areas),
        'st_depression': safe_mean(st_depressions),
        'qt_interval': safe_mean(qts),
        'qt_dispersion': max(qts) - min(qts) if len(qts) > 0 else np.nan,
        'qtc_interval': safe_mean(qtcs),
        'jt_interval': safe_mean(jts),
        'q_pathologic_ratio': q_pathologic_count / max(n, 1)
    }


def extract_features(data, fs, ch,tolerance_num, duration, sig_len):
    full_ch_features = []
    for i in ch:
        ecg = data.p_signal[:,i]
        ecg_f = apply_filters_with_padding(ecg, fs)

        r_peaks = extract_r_peaks(ecg_f, 40, 959)

        if len(r_peaks) >= -1: 
            q_peaks, s_peaks, p_peaks, t_peaks, qrs_wave = extract_peaks(ecg_f, r_peaks)

            mean_rr, std_rr = extract_intervals(r_peaks, fs)
            mean_qq, std_qq = extract_intervals(q_peaks, fs)
            mean_qrs, std_qrs = get_qrs_intervals(qrs_wave)

            #st_related
            st_vals = st_features(ecg_f, fs, r_peaks)
            st_elev_mean = np.mean(st_vals[:, 0])
            st_elev_std  = np.std(st_vals[:, 0])
            st_dur_mean = np.mean(st_vals[:, 1])
            st_dur_std  = np.std(st_vals[:, 1])
            st_slope_mean = np.mean(st_vals[:, 2])
            st_slope_std  = np.std(st_vals[:, 2])

            #morphological
            morph = morph_features(ecg_f, fs, r_peaks, q_peaks, s_peaks, t_peaks, qrs_wave)

            #freq related
            band_energy = PSD(ecg_f, fs, (0.5,4), (4, 15), (15,40))

            ch_name = give_name(i)

            features = {
                f'{ch_name}_mean_rr_interval': mean_rr,
                f'{ch_name}_std_rr_interval': std_rr,
                f'{ch_name}_mean_qq_interval': mean_qq,
                f'{ch_name}_std_qq_interval': std_qq,
                f'{ch_name}_mean_qrs': mean_qrs,
                f'{ch_name}_std_qrs': std_qrs,
                f'{ch_name}_st_elev_mean': st_elev_mean,
                f'{ch_name}_st_elev_std': st_elev_std,
                f'{ch_name}_st_dur_mean': st_dur_mean,
                f'{ch_name}_st_dur_std': st_dur_std,
                f'{ch_name}_st_slope_mean': st_slope_mean,
                f'{ch_name}_st_slope_std': st_slope_std,
                f'{ch_name}_band_energy_low': band_energy['low'],
                f'{ch_name}_band_energy_mid': band_energy['mid'],
                f'{ch_name}_band_energy_high': band_energy['high'],
                f'{ch_name}_t_amplitude': morph['t_amplitude'],
                f'{ch_name}_t_inversion': morph['t_inversion'],
                f'{ch_name}_t_r_ratio': morph['t_r_ratio'],
                f'{ch_name}_q_r_ratio': morph['q_r_ratio'],
                f'{ch_name}_r_s_ratio': morph['r_s_ratio'],
                f'{ch_name}_q_amplitude': morph['q_amplitude'],
                f'{ch_name}_r_amplitude': morph['r_amplitude'],
                f'{ch_name}_s_amplitude': morph['s_amplitude'],
                f'{ch_name}_q_duration': morph['q_duration'],
                f'{ch_name}_r_duration': morph['r_duration'],
                f'{ch_name}_s_duration': morph['s_duration'],
                f'{ch_name}_t_duration': morph['t_duration'],
                f'{ch_name}_q_area': morph['q_area'],
                f'{ch_name}_r_area': morph['r_area'],
                f'{ch_name}_s_area': morph['s_area'],
                f'{ch_name}_t_area': morph['t_area'],
                f'{ch_name}_st_area': morph['st_area'],
                f'{ch_name}_st_depression': morph['st_depression'],
                f'{ch_name}_qt_interval': morph['qt_interval'],
                f'{ch_name}_qt_dispersion': morph['qt_dispersion'],
                f'{ch_name}_qtc_interval': morph['qtc_interval'],
                f'{ch_name}_jt_interval': morph['jt_interval'],
                f'{ch_name}_q_pathologic_ratio': morph['q_pathologic_ratio'],
            }

            full_ch_features.append(features)

        new_feat = {}
        for f in full_ch_features:
            new_feat.update(f)

    return new_feat
