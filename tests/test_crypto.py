import numpy as np
import pytest

from hmofs_ofhxor_psl import OFHXOR, SecureStorage, optimise_key
from hmofs_ofhxor_psl.utils import shannon_entropy


def test_key_entropy_and_length():
    sample = np.random.default_rng(0).normal(size=100).astype(np.float32).tobytes()
    key, fit, hist, h = optimise_key(sample, key_len=16, pop=15, iters=15)
    assert len(key) == 16
    assert h > 3.5                       # max is log2(16) = 4
    assert shannon_entropy(key) == pytest.approx(h)


@pytest.mark.parametrize("msg", [b"", b"a", b"hello world", bytes(range(256)) * 3])
def test_ofhxor_roundtrip(msg):
    c = OFHXOR(bytes(range(32)))
    assert c.decrypt(c.encrypt(msg)) == msg


def test_ofhxor_is_randomised():
    c = OFHXOR(bytes(range(32)))
    assert c.encrypt(b"same message") != c.encrypt(b"same message")


def test_storage_detects_tampering():
    s = SecureStorage(b"k" * 32)
    s.put("a", b"ciphertext", "LOW")
    assert s.get("a") == (b"ciphertext", "LOW")
    blob, risk, tag = s.vault["a"]
    s.vault["a"] = (b"Xiphertext", risk, tag)
    with pytest.raises(ValueError):
        s.get("a")
