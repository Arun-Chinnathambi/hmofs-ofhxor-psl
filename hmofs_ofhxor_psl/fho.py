"""Fire Hawk Optimizer (FHO) - continuous, minimisation.

Fire hawks (elite solutions) "spread fire"; prey (remaining solutions) move
relative to their fire hawk's territory and a global safe place.
"""
import numpy as np


class FireHawkOptimizer:
    def __init__(self, fitness, dim, lb, ub, pop=30, iters=50, seed=0):
        self.f, self.dim, self.lb, self.ub = fitness, dim, lb, ub
        self.pop, self.iters = pop, iters
        self.rng = np.random.default_rng(seed)

    def _clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def run(self):
        """Returns (best_vector, best_fitness, convergence_history)."""
        rng, N = self.rng, self.pop
        X = rng.uniform(self.lb, self.ub, (N, self.dim))
        fit = np.array([self.f(x) for x in X])
        best, best_fit = X[fit.argmin()].copy(), fit.min()
        history = [best_fit]

        for _ in range(self.iters):
            order = np.argsort(fit)
            X, fit = X[order], fit[order]
            n_fh = int(rng.integers(1, max(2, N // 5) + 1))
            FH, prey = X[:n_fh].copy(), X[n_fh:].copy()
            owner = np.linalg.norm(prey[:, None] - FH[None], axis=2).argmin(axis=1)
            global_safe = prey.mean(axis=0)
            new_X = X.copy()

            for l in range(n_fh):                                   # fire hawks
                r1, r2 = rng.random(2)
                new_X[l] = self._clip(FH[l] + r1 * best - r2 * FH[rng.integers(n_fh)])

            for q in range(len(prey)):                              # prey
                l = owner[q]
                safe_l = prey[owner == l].mean(axis=0)
                r3, r4, r5, r6 = rng.random(4)
                if rng.random() < 0.5:
                    cand = prey[q] + (r3 * FH[l] - r4 * safe_l)
                else:
                    cand = prey[q] + (r5 * FH[rng.integers(n_fh)] - r6 * global_safe)
                new_X[n_fh + q] = self._clip(cand)

            new_fit = np.array([self.f(x) for x in new_X])
            better = new_fit < fit                                  # greedy selection
            X[better], fit[better] = new_X[better], new_fit[better]
            if fit.min() < best_fit:
                best, best_fit = X[fit.argmin()].copy(), fit.min()
            history.append(best_fit)
        return best, best_fit, history
