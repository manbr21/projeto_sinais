import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import pandas as pd

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('../generated_csv/teste_split.csv')

features = ['aVF_q_r_ratio', 'II_t_amplitude', 'II_q_duration', 'V5_st_depression', 'V5_t_amplitude', 'II_r_amplitude', 'V1_st_slope_mean', 'V5_st_elev_mean', 'II_q_r_ratio', 'II_qt_interval', 'V1_s_area', 'V1_r_s_ratio', 'II_r_s_ratio', 'aVL_band_energy_high', 'V1_band_energy_low', 'II_mean_qrs', 'aVL_r_duration', 'II_s_amplitude', 'aVL_r_amplitude', 'age', 'V1_st_area', 'aVF_band_energy_mid', 'V1_s_amplitude', 'V5_t_r_ratio', 'II_band_energy_low', 'II_s_area', 'II_band_energy_high', 'V1_r_duration', 'II_band_energy_mid', 'aVF_band_energy_low', 'V1_q_duration', 'aVL_st_slope_mean', 'aVF_q_area', 'V5_t_area', 'aVL_s_amplitude', 'V1_st_depression', 'aVL_t_r_ratio', 'II_t_duration', 'II_st_elev_std', 'V5_mean_qrs', 'II_st_slope_mean', 'V5_st_slope_mean', 'II_r_area', 'aVL_s_area', 'aVF_s_amplitude', 'V1_t_r_ratio', 'V1_t_amplitude', 'V5_r_s_ratio', 'aVF_r_amplitude', 'aVL_q_duration', 'aVF_st_elev_mean', 'V1_mean_qrs', 'V2_r_amplitude', 'II_st_elev_mean', 'aVF_st_slope_mean', 'V2_band_energy_low', 'aVF_q_amplitude', 'V1_r_area', 'aVL_band_energy_low', 'aVL_q_r_ratio', 'II_q_area', 'V5_t_duration', 'V1_r_amplitude', 'V1_q_r_ratio', 'V1_q_pathologic_ratio', 'V5_st_elev_std', 'aVL_mean_qrs', 'aVF_st_elev_std', 'aVL_r_s_ratio', 'II_st_depression', 'V2_q_r_ratio', 'II_t_area', 'V5_s_amplitude', 'V1_q_amplitude', 'V1_st_elev_mean', 'aVL_st_area', 'aVF_r_area', 'II_qt_dispersion', 'aVF_s_area', 'V1_qt_interval', 'II_r_duration', 'V2_band_energy_mid', 'aVL_qt_interval', 'V2_s_amplitude', 'aVL_q_amplitude', 'aVF_r_s_ratio', 'II_q_pathologic_ratio', 'V5_s_area', 'V5_band_energy_low', 'V5_st_area', 'V5_band_energy_high', 'II_q_amplitude', 'aVL_q_area', 'V1_jt_interval', 'V2_q_area', 'aVF_q_duration', 'V2_r_s_ratio', 'V5_q_r_ratio', 'V5_q_amplitude', 'II_t_r_ratio', 'aVL_r_area', 'aVL_jt_interval', 'aVL_band_energy_mid', 'II_jt_interval', 'aVF_st_slope_std', 'aVF_t_duration', 'V1_q_area', 'aVF_r_duration', 'V1_t_area', 'V5_qt_interval', 'aVF_t_amplitude', 'aVF_band_energy_high', 'aVL_st_slope_std', 'II_s_duration', 'V2_st_elev_mean', 'V2_st_depression', 'aVL_st_depression', 'II_st_area', 'V1_st_elev_std', 'V5_band_energy_mid', 'V5_r_amplitude', 'V2_t_r_ratio', 'aVL_t_amplitude', 'aVL_t_area', 'V1_std_qrs', 'V1_s_duration', 'aVF_mean_qrs', 'V5_jt_interval', 'V1_band_energy_mid', 'aVF_t_area', 'V2_band_energy_high', 'V5_r_duration', 'aVL_std_qrs', 'V2_st_slope_mean', 'II_st_slope_std', 'V2_q_amplitude', 'aVF_st_depression', 'aVL_st_elev_std', 'V5_qt_dispersion', 'V1_std_qq_interval', 'aVL_std_qq_interval', 'V2_st_area', 'V2_jt_interval', 'V1_band_energy_high', 'V2_s_area', 'V5_q_area', 'V5_q_duration', 'V2_r_area', 'V2_st_slope_std', 'V1_st_slope_std', 'V5_r_area', 'aVF_t_r_ratio', 'II_std_qq_interval', 'aVF_qt_dispersion', 'aVF_std_qrs', 'aVF_s_duration', 'II_std_qrs', 'V5_std_qrs', 'V2_std_qrs', 'aVL_t_duration', 'aVF_std_qq_interval', 'V2_st_elev_std', 'V2_qt_interval', 'aVF_qt_interval', 'aVF_st_area', 'V2_std_qq_interval', 'V1_std_rr_interval', 'aVL_std_rr_interval', 'V5_st_slope_std', 'V2_t_area', 'V1_t_duration', 'V2_t_amplitude', 'V2_std_rr_interval', 'aVF_jt_interval', 'II_std_rr_interval', 'aVF_std_rr_interval', 'V5_std_qq_interval', 'V2_mean_qrs', 'aVL_q_pathologic_ratio', 'V1_qt_dispersion', 'V2_t_duration', 'aVL_s_duration', 'V2_qt_dispersion', 'aVL_qt_dispersion', 'V5_std_rr_interval']

X = df[features]
y = df["diagnostic_superclass"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

x_scaled = scaler.transform(X)

y_predict = model.predict(x_scaled)

print(classification_report(y_encoded, y_predict, target_names=le.classes_))