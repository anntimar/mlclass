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
RESULTS_PATH = BASE_DIR / "outputs" / "tables" / "cluster_test_results.csv"

def main():
    print("\n- Carregando base...")
    df = load_data(DATA_PATH)

    print("- Removendo outlier extremo identificado...")
    df = remove_extreme_outliers(df)

    print("- Selecionando variáveis epiteliais...")
    X = select_features(df)

    print("- Verificando valores ausentes...")
    print(X.isna().sum())

    print("- Preenchendo valores ausentes com a mediana...")
    X_imputed, medians = impute_missing_values(X)

    print("- Tratando outliers com IQR...")
    X_clipped, limits = clip_outliers_iqr(X_imputed)

    print("- Padronizando variáveis...")
    X_scaled, scaler = scale_features(X_clipped)

    results = []

    for n_clusters in [2, 3, 4, 5, 6]:
        model, labels = run_kmeans(X_scaled, n_clusters=n_clusters)
        score = compute_silhouette(X_scaled, labels)
        sizes = pd.Series(labels).value_counts().sort_index().to_dict()

        results.append({
            "n_clusters": n_clusters,
            "silhouette_score": score,
            "cluster_sizes": str(sizes)
        })

        print(f"\nClusters: {n_clusters}")
        print(f"Silhouette: {score:.4f}")
        print(f"Tamanhos: {sizes}")

    results_df = pd.DataFrame(results).sort_values("silhouette_score", ascending=False)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\n- Ranking final:")
    print(results_df)

    print(f"\n- Resultado salvo em: {RESULTS_PATH}")

if __name__ == "__main__":
    main()