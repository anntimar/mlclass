#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd

from load_data import load_data
from preprocess import select_features, impute_missing_values, scale_features
from clustering_models import run_kmeans

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "RTVue_20221110_MLClass.xlsx"
OUTPUT_PATH = BASE_DIR / "outputs" / "tables" / "possible_outliers.csv"

FEATURES = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']

def main():
    print("\n- Carregando base...")
    df = load_data(DATA_PATH)

    X = select_features(df)
    X_imputed, medians = impute_missing_values(X)
    X_scaled, scaler = scale_features(X_imputed)

    print("- Rodando KMeans com 2 clusters para inspecionar pontos isolados...")
    model, labels = run_kmeans(X_scaled, n_clusters=2)

    df_result = df.copy()
    df_result["cluster"] = labels

    counts = df_result["cluster"].value_counts().sort_index()
    print("\nTamanho dos clusters:")
    print(counts)

    small_clusters = counts[counts <= 5].index.tolist()

    if not small_clusters:
        print("\n- Nenhum cluster minúsculo encontrado.")
        return

    print(f"\n- Clusters pequenos encontrados: {small_clusters}")

    outliers = df_result[df_result["cluster"].isin(small_clusters)].copy()

    cols_to_show = [col for col in ["Index", "pID", "Age", "Gender", "Eye"] if col in outliers.columns] + FEATURES + ["cluster"]
    outliers = outliers[cols_to_show]

    print("\nPossíveis outliers:")
    print(outliers)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    outliers.to_csv(OUTPUT_PATH, index=False)

    print(f"\n- Arquivo salvo em: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()