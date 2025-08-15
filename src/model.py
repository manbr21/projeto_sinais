import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.evaluate import evaluate_best, evaluate

# === 1. Carregar CSV ===
df = pd.read_csv("../generated_csv/features.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

nan_feat = df.isna().sum()
cols_to_drop = nan_feat[nan_feat == 12084].index
df = df.drop(columns=cols_to_drop)
#df.fillna(0,inplace=True)

# === 2. Limpar coluna de rótulos ===
df["diagnostic_superclass"] = df["diagnostic_superclass"].str.replace(r"[\[\]']", "", regex=True)

# === 3. Dividir em teste e treino/validação ===
remain_df, teste = train_test_split(df,test_size=0.2, stratify=df["diagnostic_superclass"], random_state=20)
teste.to_csv("../generated_csv/teste_split.csv")

# === 4. Separar X e y ===
X = remain_df.drop(columns=["diagnostic_superclass"])
y = remain_df["diagnostic_superclass"]

# Codificar classes
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# === 5. Dividir treino/teste ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# === 6. Treinar Random Forest para ranking das features ===
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

# === 7. Testar SVM com diferentes números das top features ===
results = []
results_test = []
models = []
features = []
scalers = []
#len(df.columns - 1)

for n_features in range(5, len(df.columns)-1):
    scaler = StandardScaler()
    selected_feats = feature_names_sorted[:n_features]

    features.append(selected_feats)

    X_train_sel = X_train[selected_feats]
    X_test_sel = X_test[selected_feats]

    # Escalar
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    scalers.append(scaler)

    # Treinar SVM
    svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42, class_weight='balanced')
    svm.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    acc_test = evaluate(svm, scaler, selected_feats)

    results.append((n_features, acc, selected_feats, y_pred))
    results_test.append((n_features, acc_test, selected_feats, y_pred))
    models.append(svm)

    print(f"Com {n_features} features, acurácia SVM: {acc*100:.2f}%")

# === 8. Melhor resultado ===
best = max(results, key=lambda x: x[1])
best_i = np.argmax([r[1] for r in results])
best_n, best_acc, best_feats, best_pred = best

best_test = max(results_test, key=lambda x: x[1])
best_i_test = np.argmax([r[1] for r in results_test])
best_test_n, best_test_acc, best_test_feats, best_test_pred = best_test

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

df_results = pd.DataFrame(results, columns=["n_features", "accuracy", "features", "y_pred"])
df_results_test = pd.DataFrame(results_test, columns=["n_features", "accuracy", "features", "y_pred"])

# Plotando acurácia vs número de features
plt.figure(figsize=(8, 5))

sns.lineplot(x="n_features", y="accuracy", data=df_results, marker="o", label="Treino")
sns.lineplot(x="n_features", y="accuracy", data=df_results_test, marker="o", label="Teste")

plt.scatter(best_test_n, best_test_acc, color="green", s=100, zorder=5)
plt.text(best_test_n, best_test_acc, f"{best_test_acc:.2f}", color="green", ha="left")

plt.scatter(best_n, best_acc, color="red", s=100, zorder=5)  # ponto ótimo
plt.text(best_n, best_acc, f"{best_acc:.2f}", color="red", ha="left")
plt.xlabel("Quantidade de Features")
plt.ylabel("Acurácia")
plt.title("Acurácia vs Quantidade de Features")
plt.grid(True)
plt.show()

evaluate_best(models[best_i_test], scalers[best_i_test], features[best_i_test], len(features[best_i_test]))