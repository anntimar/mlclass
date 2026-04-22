#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
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

categorical_features = ['sex']
numeric_features = [col for col in X.columns if col not in categorical_features]

print(' - Montando pré-processamento')
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Definição dos modelos e grids
search_spaces = {
    'KNN': {
        'pipeline': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier())
        ]),
        'params': {
            'classifier__n_neighbors': [3, 5, 7, 9, 11],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan']
        }
    },

    'DecisionTree': {
        'pipeline': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        'params': {
            'classifier__max_depth': [None, 3, 5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
    },

    'RandomForest': {
        'pipeline': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'params': {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2]
        }
    },

    'SVM': {
        'pipeline': Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', SVC(random_state=42))
        ]),
        'params': {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__gamma': ['scale', 'auto']
        }
    }
}

results = []

print(' - Iniciando busca por melhores cenários\n')

for model_name, config in search_spaces.items():
    print(f'==============================')
    print(f'Modelo: {model_name}')
    print(f'==============================')

    grid = GridSearchCV(
        estimator=config['pipeline'],
        param_grid=config['params'],
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X, y)

    best_score = grid.best_score_
    best_params = grid.best_params_

    results.append({
        'model': model_name,
        'best_score': best_score,
        'best_params': best_params
    })

    print(f'\nMelhor acurácia média para {model_name}: {best_score:.4f}')
    print(f'Melhores parâmetros: {best_params}\n')

print('==============================')
print('RANKING FINAL')
print('==============================')

results = sorted(results, key=lambda x: x['best_score'], reverse=True)

for result in results:
    print(f"{result['model']}: média={result['best_score']:.4f}")
    print(f"Parâmetros: {result['best_params']}\n")

best_overall = results[0]

print('==============================')
print('MELHOR MODELO GERAL')
print('==============================')
print(f"Modelo: {best_overall['model']}")
print(f"Acurácia média: {best_overall['best_score']:.4f}")
print(f"Parâmetros: {best_overall['best_params']}")