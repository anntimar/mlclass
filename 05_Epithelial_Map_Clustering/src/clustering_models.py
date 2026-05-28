from sklearn.cluster import KMeans, AgglomerativeClustering

def run_kmeans(X, n_clusters=3, random_state=42):
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return model, labels

def run_hierarchical(X, n_clusters=3):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(X)
    return model, labels
