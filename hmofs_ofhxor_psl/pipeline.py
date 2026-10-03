"""Integrated HMOFS-OFHXOR-PSL pipeline.

Encryption process : telemetry -> HMOFS -> FHO key K* -> OFHXOR
Detection process  : datasets  -> HMOFS -> PSL (theta = 0.45)
Detection layer    : risk level + feedback loop
Secure storage     : verified, encrypted data
"""
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .config import KEY_LENGTH, THETA
from .detection_layer import DetectionLayer
from .detector import DomainDetector
from .hmofs import HMOFS
from .keygen import optimise_key
from .ofhxor import OFHXOR
from .storage import SecureStorage


class HealthcareSecurityPipeline:
    def __init__(self, key_len=KEY_LENGTH):
        self.key_len = key_len

    # ---- Encryption process ------------------------------------------------
    def build_encryption_stage(self, telemetry: pd.DataFrame, y):
        X = telemetry.values.astype(float)
        hm = HMOFS(filter_k=X.shape[1]).fit(X, y)
        self.tele_mask = hm.mask_
        self.tele_cols = [c for c, m in zip(telemetry.columns, hm.mask_) if m]
        sample = X[:200][:, hm.mask_].astype(np.float32).tobytes()
        self.key, self.key_fit, self.key_hist, self.key_entropy = optimise_key(
            sample, self.key_len)
        self.cipher = OFHXOR(self.key)
        self.storage = SecureStorage(hashlib.sha256(b"mac" + self.key).digest())
        return self

    # ---- Attack detection process --------------------------------------------
    def build_detection_stage(self, domains: dict):
        """domains: name -> (X, y, feature_names)"""
        self.detectors, self.test_sets = {}, {}
        for name, (X, y, names) in domains.items():
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y,
                                                  random_state=7)
            self.detectors[name] = DomainDetector(name, names).fit(Xtr, ytr)
            self.test_sets[name] = (Xte, yte)
        self.layer = DetectionLayer(self.detectors)
        return self

    def evaluate(self):
        """Returns [(name, acc, f1, auc, alpha, n_selected, n_total), ...]."""
        rows = []
        for name, det in self.detectors.items():
            Xte, yte = self.test_sets[name]
            p = det.proba(Xte)
            pred = (p >= THETA).astype(int)
            rows.append((name, accuracy_score(yte, pred), f1_score(yte, pred),
                         roc_auc_score(yte, p), det.psl.alpha_,
                         len(det.selected), Xte.shape[1]))
        return rows

    # ---- Per-record secure monitoring ---------------------------------------
    def process(self, domain, det_x, telemetry_row, session_id, true_label=None):
        p, level, action = self.layer.assess(domain, det_x)
        retrained = False
        if true_label is not None:
            retrained = self.layer.feedback(domain, det_x, p, int(true_label))
        out = dict(session=session_id, p=p, risk=level, action=action, retrained=retrained)
        if action == "BLOCK":
            return out
        payload = np.asarray(telemetry_row, dtype=np.float32)[self.tele_mask].tobytes()
        self.storage.put(session_id, self.cipher.encrypt(payload), level)
        return out

    def read_back(self, session_id):
        """Verify integrity, decrypt, and return (telemetry_values, risk)."""
        blob, risk = self.storage.get(session_id)
        return np.frombuffer(self.cipher.decrypt(blob), dtype=np.float32), risk
