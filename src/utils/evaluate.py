import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model, scaler, features):
    df = pd.read_csv('../generated_csv/teste_split.csv')

    X = df[features]
    y = df["diagnostic_superclass"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    x_scaled = scaler.transform(X)

    y_predict = model.predict(x_scaled)

    result = accuracy_score(y_encoded, y_predict)
    return result

def evaluate_best(model, scaler, features):
    df = pd.read_csv('../generated_csv/teste_split.csv')

    X = df[features]
    y = df["diagnostic_superclass"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

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