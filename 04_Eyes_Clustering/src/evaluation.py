from sklearn.metrics import silhouette_score


def compute_silhouette(X, labels):
    """Calcula silhouette score do agrupamento."""
    return silhouette_score(X, labels)
