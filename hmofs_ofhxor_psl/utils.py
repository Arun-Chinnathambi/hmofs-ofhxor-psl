"""Small shared helpers."""
import numpy as np


def shannon_entropy(data) -> float:
    """H(K) = -sum p(k) log2 p(k), in bits per byte (max 8)."""
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    counts = np.bincount(arr, minlength=256).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def xor_bytes(data: bytes, key: bytes) -> bytes:
    """XOR `data` with a repeating `key`."""
    d = np.frombuffer(data, dtype=np.uint8)
    k = np.frombuffer(key, dtype=np.uint8)
    return (d ^ np.resize(k, d.shape)).tobytes()
