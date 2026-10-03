import numpy as np
import pytest

from hmofs_ofhxor_psl import HealthcareSecurityPipeline
from hmofs_ofhxor_psl.config import VITALS
from hmofs_ofhxor_psl.datasets import (CLINICAL_SOURCES, synth_clinical, synth_ecu_like,
                                       synth_wustl_like)


@pytest.fixture(scope="module")
def pipe():
    dfw, yw = synth_wustl_like(n=600)
    dfe, ye = synth_ecu_like(n=600)
    sources = {k: synth_clinical(k, n=400) for k in CLINICAL_SOURCES}
    sources["WUSTL-vitals"] = (dfw[VITALS], yw)
    p = HealthcareSecurityPipeline(key_len=16)
    p.build_encryption_stage(sources)
    p.build_detection_stage({"W": (dfw.values, yw, list(dfw.columns)),
                             "E": (dfe.values, ye, list(dfe.columns))})
    p.clinical = {k: v[0] for k, v in sources.items()}
    return p


def test_detection_quality(pipe):
    for name, acc, f1, auc, alpha, k, total in pipe.evaluate():
        assert auc > 0.85
        assert np.isclose(alpha.sum(), 1.0)


def test_encryption_metrics(pipe):
    rows = pipe.evaluate_encryption()
    assert {r[0] for r in rows} == set(pipe.sources)
    for name, k, total, ent, corr, aval in rows:
        assert 1 <= k <= total
        assert ent > 6.0 and corr < 0.3 and 0.0 <= aval <= 1.0


@pytest.mark.parametrize("source", CLINICAL_SOURCES)
def test_each_source_roundtrips(pipe, source):
    Xte, _ = pipe.test_sets["W"]
    df = pipe.clinical[source]
    row = df.values[0].astype(float)
    sid = None
    for i in range(40):
        r = pipe.process("W", Xte[i], f"{source}-{i}", source, row)
        if r["action"] != "BLOCK":
            sid = r["session"]
            break
    assert sid is not None
    vals, risk, src = pipe.read_back(sid)
    mask = pipe.sources[source]["mask"]
    assert src == source
    assert np.allclose(vals, row.astype(np.float32)[mask])
