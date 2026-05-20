import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_cluster_counts(df, output_path):
    counts = df['cluster'].value_counts().sort_index()
    counts.to_csv(output_path, header=['count'])


def plot_pca_clusters(X_scaled, labels, output_path):
    pca = PCA(n_components=2, random_state=42)
    comps = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(comps[:, 0], comps[:, 1], c=labels, alpha=0.8)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Visualização dos clusters em 2D (PCA)')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_cluster_means(cluster_profile, output_path):
    ax = cluster_profile.plot(kind='bar', figsize=(10, 6))
    ax.set_title('Perfil médio por cluster')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Valor médio')
    plt.xticks(rotation=0)
    plt.legend(title='Variáveis', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
