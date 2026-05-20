#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

from load_data import load_data
from preprocess import select_features, scale_features
from clustering_models import run_kmeans
from evaluation import compute_silhouette

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "barrettII_eyes_clustering.xlsx"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "eyes_clusters.csv"
PROFILE_PATH = BASE_DIR / "outputs" / "tables" / "cluster_profile_means.csv"

def main():
    print("\n- Carregando base...")
    df = load_data(DATA_PATH)

    print("- Selecionando variáveis do clustering...")
    X = select_features(df)

    print("- Padronizando variáveis...")
    X_scaled, scaler = scale_features(X)

    print("- Executando KMeans...")
    model, labels = run_kmeans(X_scaled, n_clusters=2)

    df_result = df.copy()
    df_result["cluster"] = labels

    print("- Avaliando agrupamento...")
    score = compute_silhouette(X_scaled, labels)

    print("\nResumo do clustering")
    print(df_result["cluster"].value_counts().sort_index())

    print(f"\nSilhouette Score: {score:.4f}")

    cluster_profile = df_result.groupby("cluster")[['AL', 'ACD', 'WTW', 'K1', 'K2']].mean()

    print("\nPerfil médio por cluster:")
    print(cluster_profile)

    if "Correto" in df_result.columns:
        print("\nFrequência de 'Correto' por cluster:")
        print(pd.crosstab(df_result["cluster"], df_result["Correto"]))

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)

    df_result.to_csv(PROCESSED_PATH, index=False)
    cluster_profile.to_csv(PROFILE_PATH)

    print("\n- Arquivos salvos com sucesso.")
    print(f"- Base com clusters: {PROCESSED_PATH}")
    print(f"- Perfil médio dos clusters: {PROFILE_PATH}")

if __name__ == "__main__":
    main()