"""HMOFS - Hybrid Multi-Objective Feature Selection.

Stage 1 (filter)  : mutual-information ranking keeps the top-k candidate features.
Stage 2 (wrapper) : FHO searches subsets. Objectives (all minimised):
                    (1) 1 - macro-F1, (2) relative subset size, (3) redundancy.
Result            : Pareto front of every evaluated subset -> knee-point selection.
"""
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from .fho import FireHawkOptimizer


class HMOFS:
    def __init__(self, filter_k=30, pop=10, iters=10, weights=(0.7, 0.2, 0.1), seed=0):
        self.filter_k, self.pop, self.iters = filter_k, pop, iters
        self.w, self.seed = np.array(weights), seed
        self.mask_ = None
        self.pareto_ = []

    def fit(self, X, y):
        n = X.shape[1]
        mi = mutual_info_classif(X, y, random_state=self.seed)
        cand = np.argsort(mi)[::-1][: min(self.filter_k, n)]
        Xc = X[:, cand]
        k = Xc.shape[1]
        corr = np.ones((1, 1)) if k == 1 else np.nan_to_num(np.abs(np.corrcoef(Xc, rowvar=False)))
        clf = DecisionTreeClassifier(max_depth=6, random_state=0)
        cache = {}

        def objectives(mask):
            key = mask.tobytes()
            if key not in cache:
                err = 1 - cross_val_score(clf, Xc[:, mask], y, cv=3, scoring="f1_macro").mean()
                m = int(mask.sum())
                red = 0.0 if m < 2 else (corr[np.ix_(mask, mask)].sum() - m) / (m * (m - 1))
                cache[key] = (np.array([err, m / k, red]), mask.copy())
            return cache[key][0]

        def fitness(v):
            mask = v > 0.5
            return 10.0 if not mask.any() else float(self.w @ objectives(mask))

        FireHawkOptimizer(fitness, k, 0.0, 1.0, self.pop, self.iters, self.seed).run()

        O = np.array([o for o, _ in cache.values()])
        masks = [m for _, m in cache.values()]
        front = [i for i in range(len(O))
                 if not any(np.all(O[j] <= O[i]) and np.any(O[j] < O[i]) for j in range(len(O)))]
        F = O[front]
        norm = (F - F.min(0)) / (np.ptp(F, axis=0) + 1e-12)
        knee = front[int(np.argmin(np.sqrt((self.w * norm ** 2).sum(1))))]

        full = np.zeros(n, dtype=bool)
        full[cand[masks[knee]]] = True
        self.mask_ = full
        self.pareto_ = [(O[i], int(masks[i].sum())) for i in front]
        return self

    def transform(self, X):
        return X[:, self.mask_]
