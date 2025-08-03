#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb

#import our functions
from utils.feature_extraction import extract_features
from utils.foo import load_raw_data
from utils.plot import plot_ecg, plot_with_peaks

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
    #Y = pd.read_csv(path + 'generated_csv/filtered_database.csv', index_col='ecg_id')
    #generate_csv(Y, path, tolerance_num, channels)

    # test one signal
    # test_feat = test_one_signal(path + "records100\\00000\\00016_lr", channels, tolerance_num)

    # # plot ecg
    # test = wfdb.rdrecord(path + "records100/00000/00175_lr")
    # plot_ecg(test.p_signal, sampling_rate, 10,1000)