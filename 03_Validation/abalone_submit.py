#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

print('\n - Lendo a base de treino')
data = pd.read_csv('abalone_dataset.csv')

print(' - Separando atributos e classe')
X = data.drop(columns=['type'])
y = data['type']

categorical_features = ['sex']
numeric_features = [col for col in X.columns if col not in categorical_features]

print(' - Montando pré-processamento')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

print(' - Criando pipeline com o melhor modelo encontrado')
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(C=10, gamma='auto', kernel='rbf', random_state=42))
])

print(' - Treinando modelo final com toda a base')
model.fit(X, y)

print(' - Lendo base de aplicação')
data_app = pd.read_csv('abalone_app.csv')

print(' - Gerando previsões')
y_pred = model.predict(data_app)

URL = "https://aydanomachado.com/mlclass/03_Validation.php"
DEV_KEY = "arquitetura arm"

data = {
    'dev_key': DEV_KEY,
    'predictions': pd.Series(y_pred).to_json(orient='values')
}

print(' - Enviando previsões para o servidor')
r = requests.post(url=URL, data=data)

print(" - Resposta do servidor:\n", r.text, "\n")