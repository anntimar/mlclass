#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Atividade para trabalhar o pré-processamento dos dados.

Criação de modelo preditivo para diabetes e envio para verificação de peformance
no servidor.

@author: Aydano Machado <aydano.machado@gmail.com>
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import requests

print('\n - Lendo o arquivo com o dataset sobre diabetes')
data = pd.read_csv('diabetes_dataset.csv')

feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

print(' - Realizando pré-processamento dos dados de treino')

# 1) Garante formato numérico
for col in feature_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# 2) Preenche NaN com mediana
medianas = {}
for col in feature_cols:
    medianas[col] = data[col].median()
    data[col] = data[col].fillna(medianas[col])

# 3) Clipping de outliers pelo IQR
limites = {}
for col in feature_cols:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    limite_inf = q1 - 1.5 * iqr
    limite_sup = q3 + 1.5 * iqr
    limites[col] = (limite_inf, limite_sup)
    data[col] = data[col].clip(limite_inf, limite_sup)

print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
X = data[feature_cols].copy()
y = data['Outcome']

# 4) Padronização
medias = {}
desvios = {}
for col in feature_cols:
    medias[col] = X[col].mean()
    desvios[col] = X[col].std()
    if pd.isna(desvios[col]) or desvios[col] == 0:
        desvios[col] = 1
    X[col] = (X[col] - medias[col]) / desvios[col]

print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_csv('diabetes_app.csv')

print(' - Realizando pré-processamento dos dados de aplicação')

for col in feature_cols:
    data_app[col] = pd.to_numeric(data_app[col], errors='coerce')
    data_app[col] = data_app[col].fillna(medianas[col])

for col in feature_cols:
    limite_inf, limite_sup = limites[col]
    data_app[col] = data_app[col].clip(limite_inf, limite_sup)

data_app = data_app[feature_cols].copy()

for col in feature_cols:
    data_app[col] = (data_app[col] - medias[col]) / desvios[col]

y_pred = neigh.predict(data_app)

URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"
DEV_KEY = "arquitetura arm"

data = {
    'dev_key': DEV_KEY,
    'predictions': pd.Series(y_pred).to_json(orient='values')
}

r = requests.post(url=URL, data=data)

print(" - Resposta do servidor:\n", r.text, "\n")