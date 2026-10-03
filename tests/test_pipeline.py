import numpy as np
import pytest

from hmofs_ofhxor_psl import HealthcareSecurityPipeline
from hmofs_ofhxor_psl.config import VITALS
from hmofs_ofhxor_psl.data import synth_ecu_like, synth_wustl_like


@pytest.fixture(scope="module")
def pipe():
    dfw, yw = synth_wustl_like(n=600)
    dfe, ye = synth_ecu_like(n=600)
    p = HealthcareSecurityPipeline(key_len=16)
    p.build_encryption_stage(dfw[VITALS], yw)
    p.build_detection_stage({"W": (dfw.values, yw, list(dfw.columns)),
                             "E": (dfe.values, ye, list(dfe.columns))})
    p.dfw = dfw
    return p


def test_detection_quality(pipe):
    for name, acc, f1, auc, alpha, k, total in pipe.evaluate():
        assert auc > 0.85
        assert np.isclose(alpha.sum(), 1.0)


def test_process_encrypts_and_roundtrips(pipe):
    Xte, _ = pipe.test_sets["W"]
    idx = [list(pipe.dfw.columns).index(c) for c in VITALS]
    stored = None
    for i in range(30):
        r = pipe.process("W", Xte[i], Xte[i][idx], f"t{i}")
        if r["action"] != "BLOCK":
            stored = (f"t{i}", Xte[i][idx])
            break
    assert stored is not None
    vals, _ = pipe.read_back(stored[0])
    expected = stored[1].astype(np.float32)[pipe.tele_mask]
    assert np.allclose(vals, expected)
