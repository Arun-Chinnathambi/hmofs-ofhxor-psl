"""Dataset loaders. Every loader returns (features_df, y) with y in {0, 1}."""
from .ecu_ioht import load_ecu_ioht
from .mimic3 import extract_mimic_vitals, load_mimic3
from .physionet_cinc import load_cinc, load_cinc2012, load_cinc2019
from .synthetic import synth_clinical, synth_ecu_like, synth_wustl_like
from .uci_heart import load_uci_heart
from .wustl_ehms import WUSTL_VITALS, load_wustl_ehms

# Encryption-stage sources: clinical telemetry, label = clinical outcome
CLINICAL_SOURCES = ["MIMIC-III", "PhysioNet-CinC", "UCI-Heart"]
# Detection-stage sources: network/biometric traffic, label = attack
DETECTION_SOURCES = ["WUSTL-EHMS-2020", "ECU-IoHT"]

__all__ = ["load_ecu_ioht", "extract_mimic_vitals", "load_mimic3", "load_cinc",
           "load_cinc2012", "load_cinc2019", "load_uci_heart", "load_wustl_ehms",
           "WUSTL_VITALS", "synth_clinical", "synth_ecu_like", "synth_wustl_like",
           "CLINICAL_SOURCES", "DETECTION_SOURCES"]
