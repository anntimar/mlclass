from sklearn.cluster import KMeans


def run_kmeans(X, n_clusters=3, random_state=42):
    """Executa KMeans e devolve modelo e rótulos."""
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels
