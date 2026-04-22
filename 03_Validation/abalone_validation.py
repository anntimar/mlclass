#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

print('\n - Lendo a base de treino')
data = pd.read_csv('abalone_dataset.csv')

print(' - Separando atributos e classe')
X = data.drop(columns=['type'])
y = data['type']

# Identificando colunas
categorical_features = ['sex']
numeric_features = [col for col in X.columns if col not in categorical_features]

print(' - Montando pré-processamento')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

# Modelos a comparar
models = {
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42)
}

print(' - Iniciando validação cruzada\n')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=cv,
        scoring='accuracy'
    )

    mean_score = scores.mean()
    std_score = scores.std()

    results.append((name, mean_score, std_score))

    print(f'Modelo: {name}')
    print(f'Acurácias por fold: {scores}')
    print(f'Média: {mean_score:.4f}')
    print(f'Desvio padrão: {std_score:.4f}\n')

results.sort(key=lambda x: x[1], reverse=True)

print(' - Ranking final dos modelos:')
for name, mean_score, std_score in results:
    print(f'{name}: média={mean_score:.4f} | desvio={std_score:.4f}')

best_model_name = results[0][0]
print(f'\n - Melhor modelo encontrado: {best_model_name}')