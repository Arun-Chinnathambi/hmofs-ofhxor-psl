"""Detection layer: risk-level assignment + feedback loop for AI improvement."""
import numpy as np

from .config import FEEDBACK_RETRAIN_EVERY, FEEDBACK_UNCERTAINTY, RISK_LEVELS, THETA


class DetectionLayer:
    def __init__(self, detectors, retrain_every=FEEDBACK_RETRAIN_EVERY,
                 uncertainty=FEEDBACK_UNCERTAINTY):
        self.detectors = detectors
        self.retrain_every, self.unc = retrain_every, uncertainty
        self.buffer = {k: ([], []) for k in detectors}
        self.retrain_count = 0

    def assess(self, domain, x):
        """Returns (attack_probability, risk_level, action)."""
        p = float(self.detectors[domain].proba(x.reshape(1, -1))[0])
        for upper, level, action in RISK_LEVELS:
            if p < upper:
                return p, level, action

    def feedback(self, domain, x, p, verified_label):
        """Queue misclassified / low-confidence samples; retrain when enough."""
        wrong = int(p >= THETA) != verified_label
        if wrong or abs(p - THETA) < self.unc:
            self.buffer[domain][0].append(x)
            self.buffer[domain][1].append(verified_label)
        xs, ys = self.buffer[domain]
        if len(ys) >= self.retrain_every and len(set(ys)) > 1:
            self.detectors[domain].update(np.array(xs), np.array(ys))
            self.buffer[domain] = ([], [])
            self.retrain_count += 1
            return True
        return False
