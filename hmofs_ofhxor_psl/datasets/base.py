"""Shared helpers for all dataset loaders."""
import numpy as np
import pandas as pd

from ..config import BENIGN_WORDS


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def median_impute(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True)).fillna(0.0)


def attack_labels(series: pd.Series) -> np.ndarray:
    """1 = attack, 0 = benign. Numeric labels: non-zero = attack."""
    if pd.api.types.is_numeric_dtype(series):
        return (series.fillna(0) != 0).astype(int).values
    return (~series.astype(str).str.strip().str.lower().isin(BENIGN_WORDS)).astype(int).values


def numeric_features(df: pd.DataFrame, max_categories: int = 50) -> pd.DataFrame:
    """Factorise low-cardinality text columns, drop free text/IDs, impute medians."""
    df = df.copy()
    for c in list(df.columns):
        if not pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique() > max_categories:
                df = df.drop(columns=c)
            else:
                df[c] = pd.factorize(df[c])[0]
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    return median_impute(df)


def read_labelled_csv(path, label_candidates):
    """Read CSV, strip column names, return (df, [label columns present])."""
    df = clean_columns(pd.read_csv(path))
    found = [c for c in label_candidates if c in df.columns]
    if not found:
        raise ValueError(f"No label column in {path}; expected one of {label_candidates}. "
                         f"Columns: {list(df.columns)[:15]}...")
    return df, found
