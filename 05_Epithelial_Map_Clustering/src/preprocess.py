from sklearn.preprocessing import StandardScaler

FEATURES = ['C', 'S', 'ST', 'T', 'IT', 'I', 'IN', 'N', 'SN']

def remove_extreme_outliers(df):
    df = df.copy()

    # remove o caso extremo já identificado
    if "Index" in df.columns:
        df = df[df["Index"] != 4911]

    return df

def select_features(df):
    return df[FEATURES].copy()

def impute_missing_values(X):
    X = X.copy()
    medians = X.median()
    X = X.fillna(medians)
    return X, medians

def clip_outliers_iqr(X):
    X = X.copy()
    limits = {}

    for col in X.columns:
        q1 = X[col].quantile(0.25)
        q3 = X[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        limits[col] = (lower, upper)
        X[col] = X[col].clip(lower, upper)

    return X, limits

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler