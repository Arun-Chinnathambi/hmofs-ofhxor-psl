"""MIMIC-III Clinical Database (credentialed access via PhysioNet).

`extract_mimic_vitals` streams CHARTEVENTS in chunks, averages six vital signs per
hospital admission, and joins ADMISSIONS.HOSPITAL_EXPIRE_FLAG as the label.
ITEMIDs below cover CareVue + MetaVision; verify against D_ITEMS.csv for your release.

Do NOT commit MIMIC data or derived per-patient files: the PhysioNet data use
agreement forbids redistribution.
"""
import os

import pandas as pd

from .base import clean_columns, median_impute, numeric_features

VITAL_ITEMS = {
    "HR": [211, 220045],
    "SBP": [51, 442, 455, 6701, 220179, 220050],
    "DBP": [8368, 8440, 8441, 8555, 220180, 220051],
    "RR": [615, 618, 220210, 224690],
    "SpO2": [646, 220277],
    "TempC": [223762, 676],
    "TempF": [223761, 678],
}
PLAUSIBLE = {"HR": (20, 300), "SBP": (30, 300), "DBP": (10, 200), "RR": (2, 80),
             "SpO2": (30, 100), "Temp": (25, 45)}


def _find(directory, stem):
    for ext in (".csv", ".csv.gz"):
        p = os.path.join(directory, stem + ext)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"{stem}.csv[.gz] not found in {directory}")


def extract_mimic_vitals(mimic_dir, chunksize=2_000_000, cache_csv=None):
    item2vital = {i: v for v, ids in VITAL_ITEMS.items() for i in ids}
    wanted = {"HADM_ID", "ITEMID", "VALUENUM"}
    acc = None
    reader = pd.read_csv(_find(mimic_dir, "CHARTEVENTS"), chunksize=chunksize,
                         usecols=lambda c: c.upper() in wanted)
    for ch in reader:
        ch.columns = ch.columns.str.upper()
        ch = ch[ch["ITEMID"].isin(item2vital)].dropna(subset=["HADM_ID", "VALUENUM"])
        if ch.empty:
            continue
        ch["HADM_ID"] = ch["HADM_ID"].astype("int64")
        ch["vital"] = ch["ITEMID"].map(item2vital)
        g = ch.groupby(["HADM_ID", "vital"])["VALUENUM"].agg(["sum", "count"])
        acc = g if acc is None else acc.add(g, fill_value=0)
    if acc is None:
        raise ValueError("No vital-sign rows found - check ITEMIDs against D_ITEMS.csv")

    wide = (acc["sum"] / acc["count"]).unstack("vital")
    if "TempF" in wide:
        tf = (wide["TempF"] - 32) * 5 / 9
        wide["Temp"] = wide["TempC"].fillna(tf) if "TempC" in wide else tf
    elif "TempC" in wide:
        wide["Temp"] = wide["TempC"]
    wide = wide.drop(columns=[c for c in ("TempC", "TempF") if c in wide])
    for col, (lo, hi) in PLAUSIBLE.items():
        if col in wide:
            wide[col] = wide[col].where(wide[col].between(lo, hi))

    adm = pd.read_csv(_find(mimic_dir, "ADMISSIONS"),
                      usecols=lambda c: c.upper() in {"HADM_ID", "HOSPITAL_EXPIRE_FLAG"})
    adm.columns = adm.columns.str.upper()
    df = wide.join(adm.set_index("HADM_ID"), how="inner")
    if cache_csv:
        os.makedirs(os.path.dirname(cache_csv) or ".", exist_ok=True)
        df.to_csv(cache_csv)
    y = df.pop("HOSPITAL_EXPIRE_FLAG").astype(int).values
    return median_impute(df), y


def load_mimic3(path, cache_csv="data/mimic3_vitals_cache.csv"):
    """`path` = MIMIC-III folder (extracts vitals) or a pre-extracted CSV."""
    if os.path.isdir(path):
        if cache_csv and os.path.exists(cache_csv):
            path = cache_csv
        else:
            return extract_mimic_vitals(path, cache_csv=cache_csv)
    df = clean_columns(pd.read_csv(path, index_col=0))
    label = next((c for c in ("HOSPITAL_EXPIRE_FLAG", "label", "Label") if c in df.columns), None)
    if label is None:
        raise ValueError("Pre-extracted MIMIC CSV needs HOSPITAL_EXPIRE_FLAG or label column")
    y = df.pop(label).astype(int).values
    return numeric_features(df), y
