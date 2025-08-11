import wfdb

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdrecord(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdrecord(path+f) for f in df.filename_hr]
    #data = np.array([signal for signal, meta in data]) #para rdsamp
    return data

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