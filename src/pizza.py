import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../generated_csv/features.csv")

distribuicao = df['diagnostic_superclass'].value_counts()
classes = df['diagnostic_superclass'].unique()

plt.figure(figsize=(6,6))
plt.pie(distribuicao, labels=classes, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title("Distribuição das Classes")
plt.show()