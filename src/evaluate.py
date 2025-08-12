import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

with open('model/model.pkl', 'rb') as f:
    model = pickle.load(f)

df = pd.read_csv('../generated_csv/teste_split.csv')

features = ['aVF_q_r_ratio', 'II_q_duration', 'V5_st_depression', 'V5_t_amplitude', 'II_t_amplitude', 'II_r_amplitude', 'II_q_r_ratio', 'V1_st_slope_mean', 'II_s_amplitude', 'V1_s_area', 'II_r_s_ratio', 'II_qt_interval', 'aVF_band_energy_mid', 'aVF_band_energy_low', 'V1_r_s_ratio', 'V5_st_elev_mean', 'V1_band_energy_low', 'II_band_energy_low', 'V5_t_r_ratio', 'II_mean_qrs', 'aVL_r_duration', 'V1_st_area', 'II_t_duration', 'V1_s_amplitude', 'aVL_r_amplitude', 'aVF_q_area', 'aVL_band_energy_high', 'age', 'aVL_t_r_ratio', 'V2_band_energy_low', 'II_s_area', 'II_band_energy_high', 'II_band_energy_mid', 'V5_mean_qrs', 'aVL_s_amplitude', 'V1_st_depression', 'V1_t_r_ratio', 'II_q_area', 'aVL_st_slope_mean', 'aVL_band_energy_low', 'aVL_s_area', 'II_st_slope_mean', 'V1_q_duration', 'aVF_st_elev_std', 'V5_t_duration', 'aVL_q_duration', 'II_r_area', 'aVF_q_amplitude', 'V5_t_area', 'V2_r_amplitude', 'II_st_elev_mean', 'aVF_st_elev_mean', 'aVL_q_r_ratio', 'aVF_st_slope_mean', 'aVF_s_amplitude', 'V5_r_s_ratio', 'II_t_area', 'aVL_st_area', 'aVF_s_area', 'V5_st_area', 'V5_st_elev_std', 'V5_st_slope_mean', 'V1_t_amplitude', 'V1_r_duration', 'V5_q_duration', 'aVF_r_amplitude', 'II_qt_dispersion', 'V1_r_area', 'aVL_mean_qrs', 'aVL_q_area', 'II_st_elev_std', 'V5_s_amplitude', 'V1_q_amplitude', 'V1_mean_qrs', 'V2_band_energy_mid', 'V1_r_amplitude', 'V5_band_energy_low', 'V5_s_area', 'II_q_pathologic_ratio', 'V1_q_r_ratio', 'aVF_t_amplitude', 'aVF_q_duration', 'aVF_r_s_ratio', 'V2_s_amplitude', 'aVF_st_depression', 'V2_q_r_ratio', 'aVL_q_amplitude', 'II_q_amplitude', 'aVF_r_area', 'aVL_st_depression', 'II_st_depression', 'aVL_r_s_ratio', 'V1_st_elev_mean', 'II_t_r_ratio', 'aVL_band_energy_mid', 'V5_qt_interval', 'V1_t_area', 'V2_r_s_ratio', 'V2_band_energy_high', 'aVF_st_slope_std', 'aVL_st_slope_std', 'II_r_duration', 'V1_qt_interval', 'aVF_t_area', 'V1_s_duration', 'aVL_std_qq_interval', 'II_jt_interval', 'V5_qt_dispersion', 'V2_st_depression', 'aVF_r_duration', 'V5_q_area', 'aVL_t_amplitude', 'V1_q_area', 'aVL_qt_interval', 'V1_band_energy_mid', 'V5_q_r_ratio', 'V5_band_energy_high', 'V1_jt_interval', 'aVL_jt_interval', 'II_s_duration', 'V2_st_elev_mean', 'V5_q_amplitude', 'V5_jt_interval', 'V2_t_r_ratio', 'V2_s_area', 'aVL_std_qrs', 'V5_band_energy_mid', 'aVL_std_rr_interval', 'aVL_r_area', 'V2_q_area', 'II_st_slope_std', 'V5_r_amplitude', 'aVF_band_energy_high', 'aVF_t_duration', 'V1_st_elev_std', 'aVF_qt_interval', 'aVL_st_elev_std', 'II_st_area', 'V1_band_energy_high', 'V2_r_area', 'V1_q_pathologic_ratio', 'aVL_t_duration', 'V2_std_qrs', 'V1_std_qq_interval', 'V5_r_duration', 'aVF_mean_qrs', 'V2_st_area', 'V2_qt_interval', 'V2_mean_qrs', 'aVF_st_area', 'V1_std_qrs', 'aVF_jt_interval', 'V5_st_slope_std', 'V2_q_amplitude', 'V2_t_area', 'V2_std_qq_interval', 'aVF_std_qrs', 'aVL_t_area', 'V1_qt_dispersion', 'II_std_qq_interval', 'V2_st_slope_std', 'aVL_s_duration', 'V1_st_slope_std', 'V1_std_rr_interval', 'V2_t_amplitude', 'aVL_qt_dispersion', 'V5_std_qrs', 'V2_jt_interval', 'V1_t_duration', 'aVF_t_r_ratio', 'V5_std_qq_interval', 'II_std_qrs', 'V2_st_slope_mean', 'V2_t_duration', 'V2_st_elev_std', 'aVF_qt_dispersion', 'V5_r_area', 'V2_std_rr_interval', 'aVF_s_duration', 'II_std_rr_interval', 'aVL_st_elev_mean', 'V5_std_rr_interval', 'aVF_std_rr_interval', 'V2_q_duration', 'aVF_std_qq_interval']

X = df[features]
y = df["diagnostic_superclass"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

x_scaled = scaler.transform(X)

y_predict = model.predict(x_scaled)

print(classification_report(y_encoded, y_predict, target_names=le.classes_))

cm = confusion_matrix(y_predict, y_encoded)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title(f"Matriz de Confusão - SVM")
plt.tight_layout()
plt.show()