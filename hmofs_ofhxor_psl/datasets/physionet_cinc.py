"""PhysioNet / CinC challenge loaders.

Supported layouts
  * CinC 2012 (ICU mortality): a folder of per-patient .txt files (Time,Parameter,Value)
    plus an Outcomes-*.txt file (label: In-hospital_death).
  * CinC 2019 (sepsis): a folder of per-patient .psv files (label: SepsisLabel).
  * Any pre-processed CSV with a label column (Label / label / target).
Each patient's time series is aggregated into one feature row.
"""
import glob
import os

import pandas as pd

from .base import clean_columns, median_impute, numeric_features

STATIC_2012 = ["Age", "Gender", "Height", "ICUType", "Weight"]


def load_cinc2012(set_dir, outcomes_path):
    rows = []
    for f in sorted(glob.glob(os.path.join(set_dir, "*.txt"))):
        d = pd.read_csv(f)
        d["Value"] = pd.to_numeric(d["Value"], errors="coerce")
        rid = d.loc[d["Parameter"] == "RecordID", "Value"]
        if rid.empty:
            continue
        feat = {"RecordID": int(rid.iloc[0])}
        static = d[d["Parameter"].isin(STATIC_2012)].groupby("Parameter")["Value"].first()
        feat.update(static.where(static >= 0).to_dict())          # -1 means missing
        ts = d[~d["Parameter"].isin(STATIC_2012 + ["RecordID"])]
        agg = ts.groupby("Parameter")["Value"].agg(["mean", "min", "max"])
        for p, r in agg.iterrows():
            feat[f"{p}_mean"], feat[f"{p}_min"], feat[f"{p}_max"] = r["mean"], r["min"], r["max"]
        rows.append(feat)
    df = pd.DataFrame(rows).set_index("RecordID")
    out = clean_columns(pd.read_csv(outcomes_path)).set_index("RecordID")
    df = df.join(out["In-hospital_death"], how="inner")
    y = df.pop("In-hospital_death").astype(int).values
    return numeric_features(df), y


def load_cinc2019(psv_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(psv_dir, "*.psv"))):
        d = pd.read_csv(f, sep="|")
        feat = d.drop(columns=["SepsisLabel"]).mean(numeric_only=True)
        feat["ICULOS"] = d["ICULOS"].max()
        feat["label"] = int(d["SepsisLabel"].max())
        rows.append(feat)
    df = pd.DataFrame(rows)
    y = df.pop("label").astype(int).values
    return numeric_features(df), y


def load_cinc(path, outcomes=None):
    """Dispatch on what `path` is. Returns (features_df, y)."""
    if os.path.isdir(path):
        if glob.glob(os.path.join(path, "*.psv")):
            return load_cinc2019(path)
        if glob.glob(os.path.join(path, "*.txt")):
            if not outcomes:
                raise ValueError("CinC 2012 needs the Outcomes file (--cinc-outcomes)")
            return load_cinc2012(path, outcomes)
        raise ValueError(f"No .psv or .txt patient files found in {path}")
    df = clean_columns(pd.read_csv(path))
    label = next((c for c in ("Label", "label", "target", "SepsisLabel",
                              "In-hospital_death") if c in df.columns), None)
    if label is None:
        raise ValueError("Pre-processed CinC CSV needs a label column")
    y = df.pop(label).astype(int).values
    return numeric_features(df), y
