from sklearn.preprocessing import StandardScaler

FEATURES = ['AL', 'ACD', 'WTW', 'K1', 'K2']
OPTIONAL_LABEL = 'Correto'


def select_features(df):
    """Seleciona apenas as variáveis permitidas para o clustering."""
    return df[FEATURES].copy()


def scale_features(X):
    """Padroniza os dados para clustering baseado em distância."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
