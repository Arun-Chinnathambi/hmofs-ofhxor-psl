"""Integrated HMOFS-OFHXOR-PSL pipeline (multi-source).

Encryption process : clinical telemetry sources -> HMOFS (per source, clinical label)
                     -> one FHO optimal key K* -> OFHXOR
Detection process  : network/biometric datasets -> HMOFS -> PSL (theta = 0.45)
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
from .metrics import encryption_metrics
from .ofhxor import OFHXOR
from .storage import SecureStorage


class HealthcareSecurityPipeline:
    def __init__(self, key_len=KEY_LENGTH):
        self.key_len = key_len
        self.meta = {}                      # session -> telemetry source name

    # ---- Encryption process ------------------------------------------------
    def build_encryption_stage(self, sources: dict):
        """sources: name -> (features DataFrame, clinical label y).

        HMOFS runs per source (each has its own schema); a single optimal key K*
        is searched on a pooled sample of the selected, serialised telemetry.
        """
        self.sources, self.payload_samples, pool = {}, {}, []
        for name, (df, y) in sources.items():
            X = df.values.astype(float)
            hm = HMOFS(filter_k=X.shape[1]).fit(X, y)
            raw = X[:, hm.mask_].astype(np.float32)
            self.sources[name] = dict(cols=[c for c, m in zip(df.columns, hm.mask_) if m],
                                      mask=hm.mask_, n_features=X.shape[1])
            self.payload_samples[name] = [r.tobytes() for r in raw[:60]]
            pool.append(raw[:200].tobytes()[:2048])
        self.key, self.key_fit, self.key_hist, self.key_entropy = optimise_key(
            b"".join(pool), self.key_len)
        self.cipher = OFHXOR(self.key)
        self.storage = SecureStorage(hashlib.sha256(b"mac" + self.key).digest())
        return self

    def evaluate_encryption(self):
        """Returns [(source, n_selected, n_total, entropy, |corr|, avalanche), ...]."""
        rows = []
        for name, info in self.sources.items():
            ent, corr, aval = encryption_metrics(self.cipher, self.payload_samples[name])
            rows.append((name, int(info["mask"].sum()), info["n_features"], ent, corr, aval))
        return rows

    # ---- Attack detection process --------------------------------------------
    def build_detection_stage(self, domains: dict):
        """domains: name -> (X ndarray, y, feature_names)"""
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
    def process(self, domain, det_x, session_id, source, telemetry_row, true_label=None):
        """det_x: traffic features judged by the `domain` detector.
        telemetry_row: full-length row of clinical `source`, encrypted if not blocked."""
        p, level, action = self.layer.assess(domain, det_x)
        retrained = False
        if true_label is not None:
            retrained = self.layer.feedback(domain, det_x, p, int(true_label))
        out = dict(session=session_id, source=source, p=p, risk=level, action=action,
                   retrained=retrained)
        if action == "BLOCK":
            return out
        mask = self.sources[source]["mask"]
        payload = np.asarray(telemetry_row, dtype=np.float32)[mask].tobytes()
        self.storage.put(session_id, self.cipher.encrypt(payload), level)
        self.meta[session_id] = source
        return out

    def read_back(self, session_id):
        """Verify integrity, decrypt. Returns (values, risk, source)."""
        blob, risk = self.storage.get(session_id)
        return (np.frombuffer(self.cipher.decrypt(blob), dtype=np.float32), risk,
                self.meta[session_id])
