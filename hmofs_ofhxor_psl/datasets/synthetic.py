"""Synthetic stand-ins so the project runs (and CI passes) without restricted data."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from ..config import NET, VITALS


def _named(names, n, seed, weights, informative, sep=1.3):
    X, y = make_classification(n, len(names), n_informative=informative,
                               n_redundant=max(1, informative // 3), weights=weights,
                               class_sep=sep, random_state=seed)
    return pd.DataFrame(X, columns=names), y


def synth_wustl_like(n=2500, seed=0):
    """Network features + vital signs with attack-induced drift."""
    rng = np.random.default_rng(seed)
    Xn, y = make_classification(n, len(NET), n_informative=9, n_redundant=5,
                                weights=[0.8, 0.2], class_sep=1.3, random_state=seed)
    pulse = rng.normal(75, 10, n) + y * rng.normal(12, 6, n)
    vit = np.column_stack([
        rng.normal(36.8, 0.4, n) + y * rng.normal(0.7, 0.4, n),
        rng.normal(97, 1.5, n) - y * rng.normal(3, 1.5, n),
        pulse,
        rng.normal(120, 12, n) + y * rng.normal(15, 8, n),
        rng.normal(80, 8, n) + y * rng.normal(8, 5, n),
        pulse + rng.normal(0, 2, n),
        rng.normal(16, 2, n) + y * rng.normal(3, 1.5, n)])
    return pd.DataFrame(np.hstack([Xn, vit]), columns=NET + VITALS), y


def synth_ecu_like(n=2500, seed=1):
    names = ["Protocol", "Length", "Info_len", "ArpOpcode", "PktRate", "IATmean",
             "IATstd", "SynCount", "AckCount", "UniqueSrc", "PayloadEntropy"]
    return _named(names, n, seed, [0.75, 0.25], 6, 1.4)


def synth_clinical(kind, n=1500, seed=2):
    """kind in {'MIMIC-III', 'PhysioNet-CinC', 'UCI-Heart'}; label = clinical outcome."""
    if kind == "MIMIC-III":
        rng = np.random.default_rng(seed)
        y = (rng.random(n) < 0.12).astype(int)
        df = pd.DataFrame({
            "HR": rng.normal(85, 14, n) + y * 8, "SBP": rng.normal(118, 16, n) - y * 10,
            "DBP": rng.normal(62, 10, n) - y * 5, "RR": rng.normal(19, 4, n) + y * 3,
            "SpO2": rng.normal(97, 2, n) - y * 2.5, "Temp": rng.normal(36.9, 0.6, n)})
        return df, y
    if kind == "PhysioNet-CinC":
        names = [f"{p}_{s}" for p in ("HR", "NIDiasABP", "Temp", "GCS", "Lactate", "pH")
                 for s in ("mean", "min", "max")] + ["Age", "Weight"]
        return _named(names, n, seed + 1, [0.85, 0.15], 8)
    if kind == "UCI-Heart":
        names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
                 "exang", "oldpeak", "slope", "ca", "thal"]
        return _named(names, n, seed + 2, [0.54, 0.46], 7)
    raise ValueError(kind)
