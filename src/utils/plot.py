import matplotlib.pyplot as plt
import numpy as np

from utils.filter_signal import apply_filters_with_padding
from utils.foo import give_name

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
