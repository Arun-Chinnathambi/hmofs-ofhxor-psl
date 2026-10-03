"""UCI Heart Disease (Cleveland processed file or the common Kaggle CSV variants)."""
import pandas as pd

from .base import clean_columns, median_impute, numeric_features

UCI_COLUMNS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
               "exang", "oldpeak", "slope", "ca", "thal", "num"]
LABEL_NAMES = ("num", "target", "condition", "HeartDisease", "output")


def load_uci_heart(path):
    """Returns (features_df, y). y = 1 if heart disease present (num > 0)."""
    with open(path) as fh:
        first_token = fh.readline().split(",")[0]
    has_header = any(ch.isalpha() for ch in first_token)
    if has_header:
        df = pd.read_csv(path, na_values=["?"])
    else:                                    # raw processed.cleveland.data
        df = pd.read_csv(path, header=None, names=UCI_COLUMNS, na_values=["?"])
    df = clean_columns(df)
    label = next((c for c in LABEL_NAMES if c in df.columns), None)
    if label is None:
        raise ValueError(f"No label column found among {LABEL_NAMES}")
    y = (pd.to_numeric(df[label], errors="coerce").fillna(0) > 0).astype(int).values
    return numeric_features(df.drop(columns=[label])), y
