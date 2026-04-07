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
data = pd.read_excel('diabetes_dataset.xlsx')

# Colunas usadas no modelo
feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

# Colunas em que zero costuma indicar valor inválido/ausente
cols_zero_invalido = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# =========================
# PRÉ-PROCESSAMENTO TREINO
# =========================

print(' - Realizando pré-processamento dos dados de treino')

# 1) Troca zeros inválidos por NA
for col in cols_zero_invalido:
    data[col] = data[col].replace(0, pd.NA)

# 2) Preenche faltantes com a mediana da própria coluna
medianas = {}
for col in feature_cols:
    medianas[col] = data[col].median()
    data[col] = data[col].fillna(medianas[col])

# 3) Cria X e y
print(' - Criando X e y para o algoritmo de aprendizagem a partir do arquivo diabetes_dataset')
X = data[feature_cols].copy()
y = data['Outcome']

# 4) Padronização manual
medias = {}
desvios = {}

for col in feature_cols:
    medias[col] = X[col].mean()
    desvios[col] = X[col].std()

    if desvios[col] == 0:
        desvios[col] = 1

    X[col] = (X[col] - medias[col]) / desvios[col]

# =========================
# TREINAMENTO
# =========================

print(' - Criando modelo preditivo')
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)

# =========================
# PRÉ-PROCESSAMENTO APLICAÇÃO
# =========================

print(' - Aplicando modelo e enviando para o servidor')
data_app = pd.read_excel('diabetes_app.xlsx')

print(' - Realizando pré-processamento dos dados de aplicação')

for col in cols_zero_invalido:
    data_app[col] = data_app[col].replace(0, pd.NA)

for col in feature_cols:
    data_app[col] = data_app[col].fillna(medianas[col])

data_app = data_app[feature_cols].copy()

for col in feature_cols:
    data_app[col] = (data_app[col] - medias[col]) / desvios[col]

# =========================
# PREDIÇÃO
# =========================

y_pred = neigh.predict(data_app)

# Enviando previsões realizadas com o modelo para o servidor
URL = "https://aydanomachado.com/mlclass/01_Preprocessing.php"

#TODO Substituir pela sua chave aqui
DEV_KEY = "arquitetura arm"

# json para ser enviado para o servidor
data = {'dev_key': DEV_KEY,
        'predictions': pd.Series(y_pred).to_json(orient='values')}

# Enviando requisição e salvando o objeto resposta
r = requests.post(url=URL, data=data)

# Extraindo e imprimindo o texto da resposta
pastebin_url = r.text
print(" - Resposta do servidor:\n", r.text, "\n")