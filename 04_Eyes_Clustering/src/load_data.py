import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """Carrega a base Excel original."""
    return pd.read_excel(path)
