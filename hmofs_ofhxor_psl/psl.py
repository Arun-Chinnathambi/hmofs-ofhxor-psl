"""PSL - Probabilistic Super Learner.

Three base learners give posteriors p1, p2, p3. Posterior-probability fusion:
    p = alpha1*p1 + alpha2*p2 + alpha3*p3,   alpha on the simplex,
with alpha learned by minimising validation log-loss.
An attack is flagged iff p >= theta (default 0.45).
"""
import numpy as np
from scipy.optimize import minimize
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import THETA


class ProbabilisticSuperLearner:
    def __init__(self, theta=THETA, seed=0):
        self.theta, self.seed = theta, seed
        self.alpha_ = np.ones(3) / 3

    def _bases(self):
        return [
            RandomForestClassifier(n_estimators=150, random_state=self.seed, n_jobs=-1),
            GradientBoostingClassifier(random_state=self.seed),
            make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        ]

    @staticmethod
    def _softmax(z):
        e = np.exp(z - z.max())
        return e / e.sum()

    @staticmethod
    def _posteriors(models, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in models])

    def fit(self, X, y):
        Xa, Xv, ya, yv = train_test_split(X, y, test_size=0.25, stratify=y,
                                          random_state=self.seed)
        models = [m.fit(Xa, ya) for m in self._bases()]
        P = self._posteriors(models, Xv)
        loss = lambda z: log_loss(yv, np.clip(P @ self._softmax(z), 1e-6, 1 - 1e-6))
        self.alpha_ = self._softmax(minimize(loss, np.zeros(3), method="Nelder-Mead").x)
        self.models_ = [m.fit(X, y) for m in self._bases()]        # refit on all data
        return self

    def predict_proba(self, X):
        return self._posteriors(self.models_, X) @ self.alpha_

    def predict(self, X):
        return (self.predict_proba(X) >= self.theta).astype(int)
