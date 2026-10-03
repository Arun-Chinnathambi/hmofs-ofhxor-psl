import numpy as np

from hmofs_ofhxor_psl import HMOFS, ProbabilisticSuperLearner
from hmofs_ofhxor_psl.data import synth_ecu_like


def _data():
    df, y = synth_ecu_like(n=500)
    return df.values, y


def test_hmofs_selects_nonempty_subset():
    X, y = _data()
    h = HMOFS(pop=6, iters=4).fit(X, y)
    assert 1 <= h.mask_.sum() <= X.shape[1]
    assert h.transform(X).shape[1] == h.mask_.sum()
    assert len(h.pareto_) >= 1


def test_psl_alpha_simplex_and_probabilities():
    X, y = _data()
    psl = ProbabilisticSuperLearner().fit(X, y)
    assert np.isclose(psl.alpha_.sum(), 1.0) and np.all(psl.alpha_ >= 0)
    p = psl.predict_proba(X)
    assert np.all((p >= 0) & (p <= 1))
    assert psl.theta == 0.45
    assert ((p >= 0.45).astype(int) == psl.predict(X)).all()
