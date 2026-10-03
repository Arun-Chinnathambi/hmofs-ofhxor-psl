"""Dataset loaders (WUSTL-EHMS-2020, ECU-IoHT) and synthetic stand-ins for demos."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from .config import BENIGN_WORDS, LABEL_CANDIDATES, NET, VITALS


def load_tabular(path):
    """Generic CSV loader -> (X, y, feature_names, raw_dataframe).

    The label column is auto-detected; any value not in BENIGN_WORDS is an attack.
    Low-cardinality text columns are factorised, free-text/ID columns are dropped.
    """
    df = pd.read_csv(path)
    label_cols = [c for c in LABEL_CANDIDATES if c in df.columns]
    if not label_cols:
        raise ValueError(f"No label column in {path}; expected one of {LABEL_CANDIDATES}")
    lab = df[label_cols[0]]
    y = (~lab.astype(str).str.strip().str.lower().isin(BENIGN_WORDS)).astype(int).values
    feats = df.drop(columns=label_cols)
    for c in feats.columns:
        if feats[c].dtype == object:
            if feats[c].nunique() > 50:
                feats = feats.drop(columns=c)
            else:
                feats[c] = pd.factorize(feats[c])[0]
    feats = feats.apply(pd.to_numeric, errors="coerce").fillna(0)
    return feats.values.astype(float), y, list(feats.columns), df


def synth_wustl_like(n=2500, seed=0):
    """Synthetic network features + vital signs with attack-induced drift."""
    rng = np.random.default_rng(seed)
    Xn, y = make_classification(n, len(NET), n_informative=9, n_redundant=5,
                                weights=[0.8, 0.2], class_sep=1.3, random_state=seed)
    pulse = rng.normal(75, 10, n) + y * rng.normal(12, 6, n)
    vit = np.column_stack([
        rng.normal(36.8, 0.4, n) + y * rng.normal(0.7, 0.4, n),   # Temp
        rng.normal(97, 1.5, n) - y * rng.normal(3, 1.5, n),       # SpO2
        pulse,                                                     # Pulse_Rate
        rng.normal(120, 12, n) + y * rng.normal(15, 8, n),        # SYS
        rng.normal(80, 8, n) + y * rng.normal(8, 5, n),           # DIA
        pulse + rng.normal(0, 2, n),                               # Heart_rate
        rng.normal(16, 2, n) + y * rng.normal(3, 1.5, n)])        # Resp_Rate
    return pd.DataFrame(np.hstack([Xn, vit]), columns=NET + VITALS), y


def synth_ecu_like(n=2500, seed=1):
    """Synthetic packet-level features resembling ARP spoofing / DDoS / MitM traffic."""
    names = ["Length", "Protocol", "SrcPort", "DstPort", "ArpOpcode", "PktRate", "IATmean",
             "IATstd", "SynCount", "AckCount", "UniqueSrc", "PayloadEntropy"]
    X, y = make_classification(n, len(names), n_informative=7, n_redundant=3,
                               weights=[0.75, 0.25], class_sep=1.4, random_state=seed)
    return pd.DataFrame(X, columns=names), y
