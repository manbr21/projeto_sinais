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
df = pd.read_csv("../generated_csv/features.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# === 2. Limpar coluna de rótulos ===
df["diagnostic_superclass"] = df["diagnostic_superclass"].str.replace(r"[\[\]']", "", regex=True)

# === 3. Separar X e y ===
X = df.drop(columns=["diagnostic_superclass"])
y = df["diagnostic_superclass"]

# Codificar classes
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 4. Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === 5. Treinar Random Forest para ranking das features ===
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf.fit(X_train, y_train)

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names_sorted = [X.columns[i] for i in indices]

print("\n=== Ranking completo das features ===")
for rank, feat in enumerate(feature_names_sorted, start=1):
    print(f"{rank:2d}. {feat:30s} -> {importances[indices[rank-1]]:.4f}")

# === 6. Testar SVM com diferentes números das top features ===
results = []

scaler = StandardScaler()

for n_features in range(5, 219):
    selected_feats = feature_names_sorted[:n_features]

    X_train_sel = X_train[selected_feats]
    X_test_sel = X_test[selected_feats]

    # Escalar
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    # Treinar SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42, class_weight='balanced')
    svm.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    results.append((n_features, acc, selected_feats, y_pred))

    print(f"Com {n_features} features, acurácia SVM: {acc*100:.2f}%")

# === 7. Melhor resultado ===
best = max(results, key=lambda x: x[1])
best_n, best_acc, best_feats, best_pred = best

print(f"\n✅ Melhor acurácia com {best_n} features: {best_acc*100:.2f}%")
print("Features usadas:")
print(best_feats)

print("\n=== Relatório de classificação do melhor modelo ===")
print(classification_report(y_test, best_pred, target_names=le.classes_))

# Matriz de confusão do melhor modelo
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predito")
plt.ylabel("Real")
plt.title(f"Matriz de Confusão - SVM com {best_n} melhores features")
plt.tight_layout()
plt.show()
