import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb

from utils.feature_extraction import extract_features
from utils.foo import load_raw_data
from utils.plot import plot_ecg, plot_with_peaks

def test_one_signal(name, ch, min_peaks_detected):
    ecg = wfdb.rdrecord(name)
    feat = extract_features(ecg, ecg.fs, ch, min_peaks_detected, ecg.sig_len / ecg.fs, ecg.sig_len)
    return feat

def generate_csv(Y, path, min_peaks_detected, ch):
    X = load_raw_data(Y, sampling_rate_hz, path)

    arr = []
    for data in X:
        fs = data.fs
        duration = data.sig_len / fs
        t = np.linspace(0, duration, data.sig_len)

        features = extract_features(data, fs, ch, min_peaks_detected, data.sig_len / data.fs, data.sig_len)
        arr.append(features)
        
    features_df = pd.DataFrame(arr, index=Y.index)
    Y_MERGE = pd.merge(Y, features_df, on=Y.index, how='outer')

    Y_FINAL = Y_MERGE.drop(columns=['key_0', 'filename_lr'])

    Y_FINAL.to_csv(path + "generated_csv/features.csv")
    print("Csv created successfully")

if __name__ == "__main__":
    channels = [1,3,5,6,7,10]

    """
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
    """
    min_peaks_detected = 5
    path = '../' 
    sampling_rate_hz=100

    Y = pd.read_csv(path + 'generated_csv/filtered_database.csv', index_col='ecg_id')
    generate_csv(Y, path, min_peaks_detected, channels)

    # test one signal
    # test_feat = test_one_signal(path + "records100\\00000\\00016_lr", channels, min_peaks_detected)

    # # plot ecg
    # test = wfdb.rdrecord(path + "records100/00000/00175_lr")
    # plot_ecg(test.p_signal, sampling_rate_hz, 10,1000)