import numpy as np

from hmofs_ofhxor_psl import FireHawkOptimizer


def test_fho_minimises_sphere():
    f = lambda x: float(np.sum(x ** 2))
    best, fit, hist = FireHawkOptimizer(f, 5, -10, 10, pop=30, iters=60, seed=0).run()
    assert fit < 1.0
    assert hist[-1] <= hist[0]          # monotone non-increasing best-so-far
    assert np.all(best >= -10) and np.all(best <= 10)
