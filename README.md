# Projeto: Predição de Infarto do Miocárdio e Distúrbios de Condução em Sinais de ECG

## Descrição

Este projeto foi desenvolvido na disciplina de Sinais e Sistemas com o objetivo de aplicar técnicas de processamento digital de sinais e aprendizado de máquina em registros de ECG.

A proposta é:

- Realizar pré-processamento dos sinais de ECG utilizando a Transformada de Fourier para filtragem e remoção de ruídos;
- Extrair características relevantes dos sinais (features) para análise;
- Implementar e avaliar modelos de classificação automática capazes de detectar infartos do miocárdio e distúrbios de condução;
- Utilizar Support Vector Machine (SVM) para a classificação dos pacientes.

## Estrutura do Projeto

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
│   │   ├── evaluate.py        # Avalia o modelo para os testes
│   │   ├── feature_extraction.py
│   │   ├── filter_signal.py
│   │   ├── foo.py
│   │   └── plot.py
│   │
│   ├── create_csv.py          # Script auxiliar para criação de CSVs
│   ├── main.py                # Ponto principal de execução do projeto
│   ├── model.py               # Script com o modelo de classificação
│   └── pizza.py               # Script para plotar as distribuições em gráfico pizza
│
├── .gitignore
├── LICENSE.txt
├── Projeto - Sinais.pdf       # Slides apresentados para o professor da disciplina
├── README.md
├── Relatório - Sinais.pdf     # Relatório final com todas a descrição detalhada do projeto
└── requirements.txt

```
## Tecnologias e Conceitos Utilizados

- Python 3
- Bibliotecas: NumPy, Pandas, SciPy, scikit-learn, Matplotlib
- Processamento de Sinais: Transformada de Fourier, filtragem passa-baixa e passa-alta
- Machine Learning: Support Vector Machine (SVM), Random Forest
- Banco de Dados: PTB-XL (ECG de pacientes reais)
