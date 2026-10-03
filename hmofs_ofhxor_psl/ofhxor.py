"""OFHXOR - Optimal Fire Hawk XOR encryption (K*, D1, D2).

The plaintext is split into two blocks D1 and D2:
    C1 = D1 XOR K*                          (optimal FHO key)
    C2 = D2 XOR SHAKE256(K* | nonce | C1)   (chained to C1 for diffusion)
Packet layout: nonce(16) | len(D1)(4) | C1 | C2

NOTE: The (K*, D1, D2) block structure is an interpretation of the architecture
diagram. XOR-based schemes are not a substitute for AES-GCM in production.
"""
import hashlib
import os

from .utils import xor_bytes


class OFHXOR:
    def __init__(self, optimal_key: bytes):
        self.key = optimal_key

    def encrypt(self, plaintext: bytes) -> bytes:
        nonce = os.urandom(16)
        half = len(plaintext) // 2
        d1, d2 = plaintext[:half], plaintext[half:]
        c1 = xor_bytes(d1, self.key) if d1 else b""
        ks2 = hashlib.shake_256(self.key + nonce + c1).digest(len(d2))
        c2 = xor_bytes(d2, ks2) if d2 else b""
        return nonce + half.to_bytes(4, "big") + c1 + c2

    def decrypt(self, packet: bytes) -> bytes:
        nonce, half = packet[:16], int.from_bytes(packet[16:20], "big")
        c1, c2 = packet[20:20 + half], packet[20 + half:]
        d1 = xor_bytes(c1, self.key) if c1 else b""
        ks2 = hashlib.shake_256(self.key + nonce + c1).digest(len(c2))
        d2 = xor_bytes(c2, ks2) if c2 else b""
        return d1 + d2
