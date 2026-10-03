"""Encryption quality metrics for OFHXOR."""
import numpy as np

from .utils import shannon_entropy

HEADER = 20  # nonce(16) + len(D1)(4)


def encryption_metrics(cipher, payloads, n_flips=200, seed=0):
    """Returns (ciphertext_entropy_bits, |corr(plain, cipher)|, avalanche_ratio).

    avalanche_ratio: mean fraction of ciphertext bits that change when ONE plaintext
    bit is flipped (fixed nonce). The ideal for a strong cipher is ~0.5; a chained
    XOR stream scheme is far lower, so report this honestly.
    """
    rng = np.random.default_rng(seed)
    bodies = [cipher.encrypt(p)[HEADER:] for p in payloads]
    plain = np.frombuffer(b"".join(payloads), dtype=np.uint8)
    ct = np.frombuffer(b"".join(bodies), dtype=np.uint8)
    entropy = shannon_entropy(ct.tobytes())
    corr = abs(np.corrcoef(plain, ct)[0, 1]) if plain.std() > 0 and ct.std() > 0 else 1.0

    nonce, ratios = bytes(16), []
    for _ in range(n_flips):
        p = bytearray(payloads[rng.integers(len(payloads))])
        c0 = np.unpackbits(np.frombuffer(cipher.encrypt(bytes(p), nonce)[HEADER:], np.uint8))
        bit = int(rng.integers(len(p) * 8))
        p[bit // 8] ^= 1 << (bit % 8)
        c1 = np.unpackbits(np.frombuffer(cipher.encrypt(bytes(p), nonce)[HEADER:], np.uint8))
        ratios.append(float(np.mean(c0 != c1)))
    return entropy, float(corr), float(np.mean(ratios))
