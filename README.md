# Project Structure

```text
BD_SINAIS/
├── database_csv/              # CSVs originais do banco de dados
│   ├── ptbxl_database.csv
│   └── scp_statements.csv
│
├── generated_csv/             # CSVs gerados durante o processamento
│   ├── features.csv
│   └── filtered_database.csv
│
├── records100/                # ECGs dos diversos pacientes
│
├── src/
│   ├── notebooks/             # Códigos em formato Jupyter Notebook
│   │   ├── proccess_signal_og.ipynb
│   │   ├── proccess_signal.ipynb
│   │   └── rodar_modelo.ipynb
│   │
│   ├── utils/                 # Módulos reutilizáveis com funções auxiliares
│   │   ├── feature_extraction.py
│   │   ├── filter_signal.py
│   │   ├── foo.py
│   │   └── plot.py
│   │
│   ├── create_csv.py          # Script auxiliar para criação de CSVs
│   ├── main.py                # Ponto principal de execução do projeto
│   └── random_forest.py       # Script com o modelo de classificação
│
├── .gitignore
├── LICENSE.txt
├── README.md
└── requirements.txt
