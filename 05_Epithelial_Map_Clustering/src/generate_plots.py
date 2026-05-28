#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from load_data import load_data
from preprocess import (
    remove_extreme_outliers,
    select_features,
    impute_missing_values,
    clip_outliers_iqr,
    scale_features
)
from clustering_models import run_kmeans

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "raw" / "RTVue_20221110_MLClass.xlsx"
CHARTS_DIR = BASE_DIR / "outputs" / "charts"
TABLES_DIR = BASE_DIR / "outputs" / "tables"

FEATURES = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']


def save_cluster_size_bar(df_result: pd.DataFrame):
    counts = df_result["cluster"].value_counts().sort_index()

    plt.figure(figsize=(7, 5))
    counts.plot(kind="bar")
    plt.title("Quantidade de mapas epiteliais por cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "cluster_sizes.png", dpi=300)
    plt.close()


def save_boxplots(df_result: pd.DataFrame):
    for col in FEATURES:
        plt.figure(figsize=(7, 5))
        df_result.boxplot(column=col, by="cluster")
        plt.title(f"Distribuição de {col} por cluster")
        plt.suptitle("")
        plt.xlabel("Cluster")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(CHARTS_DIR / f"boxplot_{col}.png", dpi=300)
        plt.close()


def save_pca_scatter(X_scaled, labels):
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title("Visualização dos clusters em 2D com PCA")
    plt.xlabel("Componente principal 1")
    plt.ylabel("Componente principal 2")
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "pca_clusters.png", dpi=300)
    plt.close()

    explained = pd.DataFrame({
        "component": ["PC1", "PC2"],
        "explained_variance_ratio": pca.explained_variance_ratio_
    })
    explained.to_csv(TABLES_DIR / "pca_explained_variance.csv", index=False)


def save_cluster_means_table(df_result: pd.DataFrame):
    cluster_profile = df_result.groupby("cluster")[FEATURES].mean()
    cluster_profile.to_csv(TABLES_DIR / "cluster_profile_means.csv")


def save_age_bar(df_result: pd.DataFrame):
    if "Age" not in df_result.columns:
        return

    age_means = df_result.groupby("cluster")["Age"].mean()

    plt.figure(figsize=(7, 5))
    age_means.plot(kind="bar")
    plt.title("Idade média por cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Idade média")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "age_by_cluster.png", dpi=300)
    plt.close()

    age_means.to_csv(TABLES_DIR / "age_by_cluster.csv")


def save_gender_bar(df_result: pd.DataFrame):
    if "Gender" not in df_result.columns:
        return

    crosstab = pd.crosstab(df_result["cluster"], df_result["Gender"])

    crosstab.plot(kind="bar", figsize=(7, 5))
    plt.title("Frequência de gênero por cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "gender_by_cluster.png", dpi=300)
    plt.close()

    crosstab.to_csv(TABLES_DIR / "gender_by_cluster.csv")


def save_eye_bar(df_result: pd.DataFrame):
    if "Eye" not in df_result.columns:
        return

    crosstab = pd.crosstab(df_result["cluster"], df_result["Eye"])

    crosstab.plot(kind="bar", figsize=(7, 5))
    plt.title("Frequência de olho por cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Quantidade")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CHARTS_DIR / "eye_by_cluster.png", dpi=300)
    plt.close()

    crosstab.to_csv(TABLES_DIR / "eye_by_cluster.csv")


def main():
    print("\n- Carregando base...")
    df = load_data(DATA_PATH)

    print("- Removendo outlier extremo identificado...")
    df = remove_extreme_outliers(df)

    print("- Selecionando variáveis epiteliais...")
    X = select_features(df)

    print("- Preenchendo valores ausentes com a mediana...")
    X_imputed, medians = impute_missing_values(X)

    print("- Tratando outliers com IQR...")
    X_clipped, limits = clip_outliers_iqr(X_imputed)

    print("- Padronizando variáveis...")
    X_scaled, scaler = scale_features(X_clipped)

    print("- Executando KMeans com 2 clusters...")
    model, labels = run_kmeans(X_scaled, n_clusters=2)

    df_result = df.copy()
    df_result["cluster"] = labels

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print("- Gerando gráfico de quantidade por cluster...")
    save_cluster_size_bar(df_result)

    print("- Gerando boxplots...")
    save_boxplots(df_result)

    print("- Gerando PCA 2D...")
    save_pca_scatter(X_scaled, labels)

    print("- Salvando tabela de médias dos clusters...")
    save_cluster_means_table(df_result)

    print("- Gerando gráfico de idade média por cluster...")
    save_age_bar(df_result)

    print("- Gerando gráfico de gênero por cluster...")
    save_gender_bar(df_result)

    print("- Gerando gráfico de olho por cluster...")
    save_eye_bar(df_result)

    print("\n- Gráficos gerados com sucesso.")
    print(f"- Pasta de gráficos: {CHARTS_DIR}")
    print(f"- Pasta de tabelas: {TABLES_DIR}")


if __name__ == "__main__":
    main()