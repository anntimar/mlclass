from sklearn.metrics import silhouette_score

def compute_silhouette(X, labels):
    return silhouette_score(X, labels)
