#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

from load_data import load_data
from preprocess import (
    remove_extreme_outliers,
    select_features,
    impute_missing_values,
    clip_outliers_iqr,
    scale_features
)
from clustering_models import run_kmeans
from evaluation import compute_silhouette

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "RTVue_20221110_MLClass.xlsx"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "epithelial_clusters.csv"
PROFILE_PATH = BASE_DIR / "outputs" / "tables" / "cluster_profile_means.csv"

FEATURES = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']

def main():
    print("\n- Carregando base...")
    df = load_data(DATA_PATH)

    print("- Removendo outlier extremo identificado...")
    df = remove_extreme_outliers(df)

    print("- Selecionando variáveis do clustering...")
    X = select_features(df)

    print("- Preenchendo valores ausentes com a mediana...")
    X_imputed, medians = impute_missing_values(X)

    print("- Tratando outliers com IQR...")
    X_clipped, limits = clip_outliers_iqr(X_imputed)

    print("- Padronizando variáveis...")
    X_scaled, scaler = scale_features(X_clipped)

    print("- Executando KMeans...")
    model, labels = run_kmeans(X_scaled, n_clusters=2)

    df_result = df.copy()
    df_result["cluster"] = labels

    print("- Avaliando agrupamento...")
    score = compute_silhouette(X_scaled, labels)

    print("\nResumo do clustering")
    print(df_result["cluster"].value_counts().sort_index())
    print(f"\nSilhouette Score: {score:.4f}")

    cluster_profile = df_result.groupby("cluster")[FEATURES].mean()
    print("\nPerfil médio por cluster:")
    print(cluster_profile)

    print("\nComplementos descritivos:")
    if "Age" in df_result.columns:
        print("\nIdade média por cluster:")
        print(df_result.groupby("cluster")["Age"].mean())

    if "Gender" in df_result.columns:
        print("\nFrequência de Gender por cluster:")
        print(pd.crosstab(df_result["cluster"], df_result["Gender"]))

    if "Eye" in df_result.columns:
        print("\nFrequência de Eye por cluster:")
        print(pd.crosstab(df_result["cluster"], df_result["Eye"]))

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_result.to_csv(PROCESSED_PATH, index=False)
    cluster_profile.to_csv(PROFILE_PATH)

    print("\n- Arquivos salvos com sucesso.")

if __name__ == "__main__":
    main()