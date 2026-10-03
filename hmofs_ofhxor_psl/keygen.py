"""Optimal key generation with FHO, using H(K) = -sum p(k) log2 p(k)."""
import numpy as np

from .fho import FireHawkOptimizer
from .utils import shannon_entropy, xor_bytes


def optimise_key(sample: bytes, key_len=32, pop=30, iters=50, seed=1):
    """Search key bytes (0-255) giving high key entropy, high ciphertext entropy,
    low plaintext/ciphertext correlation and balanced bits.

    Returns (key, best_fitness, history, key_entropy_bits).
    """
    plain = np.frombuffer(sample, dtype=np.uint8)

    def fitness(vec):
        key = np.round(vec).astype(np.uint8)
        c = np.frombuffer(xor_bytes(sample, key.tobytes()), dtype=np.uint8)
        h_k = shannon_entropy(key.tobytes()) / np.log2(key_len)    # H(K), normalised
        h_c = shannon_entropy(c.tobytes()) / 8.0
        corr = abs(np.corrcoef(plain, c)[0, 1]) if c.std() > 0 and plain.std() > 0 else 1.0
        balance = abs(np.unpackbits(c).mean() - 0.5)
        return (1 - h_k) + (1 - h_c) + corr + balance

    best, fit, hist = FireHawkOptimizer(fitness, key_len, 0, 255, pop, iters, seed).run()
    key = np.round(best).astype(np.uint8).tobytes()
    return key, fit, hist, shannon_entropy(key)
