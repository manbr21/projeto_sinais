import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# === 1. Carregar CSV ===
df = pd.read_csv("features.csv")

# Remove coluna de índice desnecessária se existir
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# === 2. Limpar coluna de rótulos ===
df["diagnostic_superclass"] = df["diagnostic_superclass"].str.replace(r"[\[\]']", "", regex=True)

# === 3. Separar X (features) e y (classes) ===
X = df.drop(columns=["diagnostic_superclass"])
y = df["diagnostic_superclass"]

# Codificar classes
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 4. Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === 5. Treinar Random Forest para obter ranking das features ===
rf_full = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced')
rf_full.fit(X_train, y_train)

# Obter as 15 melhores features
importances = rf_full.feature_importances_
indices = np.argsort(importances)[::-1]
top_15_features = [X.columns[i] for i in indices[:15]]

print("\n=== Top 15 Features Selecionadas ===")
for rank, feat in enumerate(top_15_features, start=1):
    print(f"{rank:2d}. {feat:30s} -> {importances[indices[rank-1]]:.4f}")

# === 6. Usar só as 15 melhores ===
X_train_top15 = X_train[top_15_features]
X_test_top15 = X_test[top_15_features]

# Escalar os dados (SVM funciona melhor com normalização)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_top15)
X_test_scaled = scaler.transform(X_test_top15)

# === 7. Treinar SVM ===
svm_model = SVC(kernel="rbf", C=10, gamma="scale", random_state=42, class_weight='balanced')
svm_model.fit(X_train_scaled, y_train)

# === 8. Avaliar no teste ===
y_pred_svm = svm_model.predict(X_test_scaled)
acc_svm = accuracy_score(y_test, y_pred_svm)

print(f"\n✅ Acurácia do SVM com as 15 melhores features: {acc_svm*100:.2f}%\n")
print("=== Relatório de Classificação ===")
print(classification_report(y_test, y_pred_svm, target_names=le.classes_))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title("Matriz de Confusão - SVM com 15 Melhores Features")
plt.tight_layout()
plt.show()
