"""Per-dataset attack detector: HMOFS feature selection -> PSL."""
import numpy as np

from .config import THETA
from .hmofs import HMOFS
from .psl import ProbabilisticSuperLearner


class DomainDetector:
    def __init__(self, name, feature_names, theta=THETA, seed=0):
        self.name, self.names, self.seed = name, list(feature_names), seed
        self.hmofs = HMOFS(seed=seed)
        self.psl = ProbabilisticSuperLearner(theta, seed)

    def fit(self, X, y):
        self.hmofs.fit(X, y)
        self.Xs_, self.y_ = self.hmofs.transform(X), np.asarray(y)
        self.psl.fit(self.Xs_, self.y_)
        return self

    def update(self, X_new, y_new):
        """Feedback-loop retraining: old data + analyst-verified samples."""
        self.Xs_ = np.vstack([self.Xs_, self.hmofs.transform(X_new)])
        self.y_ = np.concatenate([self.y_, y_new])
        self.psl.fit(self.Xs_, self.y_)

    def proba(self, X):
        return self.psl.predict_proba(self.hmofs.transform(X))

    @property
    def selected(self):
        return [n for n, m in zip(self.names, self.hmofs.mask_) if m]
