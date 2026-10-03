"""ECU-IoHT: packet-level healthcare IoT traffic (ARP spoofing, DoS/DDoS, MitM, scans)."""
from .base import attack_labels, numeric_features, read_labelled_csv

ADDRESS_COLUMNS = ["Source", "Destination"]
INDEX_COLUMNS = ["No.", "Time"]


def load_ecu_ioht(path, keep_addresses=False):
    """Returns (features_df, y). Free-text `Info` becomes `Info_len`.

    Source/Destination addresses are dropped by default (host identity can leak the
    label). Use keep_addresses=True to reproduce papers that include them.
    """
    df, found = read_labelled_csv(path, ["Type", "type", "Label", "label", "Class"])
    y = attack_labels(df[found[0]])
    feats = df.drop(columns=found)
    if "Info" in feats.columns:
        feats["Info_len"] = feats["Info"].astype(str).str.len()
        feats = feats.drop(columns="Info")
    drop = INDEX_COLUMNS + ([] if keep_addresses else ADDRESS_COLUMNS)
    feats = feats.drop(columns=[c for c in drop if c in feats.columns])
    return numeric_features(feats), y
