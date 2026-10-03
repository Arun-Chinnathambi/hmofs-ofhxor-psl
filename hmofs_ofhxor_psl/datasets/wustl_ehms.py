"""WUSTL-EHMS-2020: network flow metrics + patient biometrics, with spoofing/data-injection attacks."""
from .base import attack_labels, numeric_features, read_labelled_csv

WUSTL_VITALS = ["Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate", "Resp_Rate", "ST"]
# Identifiers leak the label (fixed attacker/victim hosts), so they are dropped by default.
WUSTL_IDENTIFIERS = ["SrcAddr", "DstAddr", "Sport", "Dport", "SrcMac", "DstMac"]


def load_wustl_ehms(path, drop_identifiers=True):
    """Returns (features_df, y). Features include network metrics AND vital signs."""
    df, found = read_labelled_csv(path, ["Label", "Attack Category"])
    y = attack_labels(df[found[0]])
    feats = df.drop(columns=found)
    if drop_identifiers:
        feats = feats.drop(columns=[c for c in WUSTL_IDENTIFIERS if c in feats.columns])
    return numeric_features(feats), y
