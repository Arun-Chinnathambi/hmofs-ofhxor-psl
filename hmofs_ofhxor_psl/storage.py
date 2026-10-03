"""Secure storage: encrypted payloads with HMAC-SHA256 integrity verification."""
import hashlib
import hmac


class SecureStorage:
    def __init__(self, mac_key: bytes):
        self.mac_key, self.vault = mac_key, {}

    def _tag(self, sid, blob, risk):
        return hmac.new(self.mac_key, sid.encode() + blob + risk.encode(),
                        hashlib.sha256).digest()

    def put(self, sid, blob, risk):
        self.vault[sid] = (blob, risk, self._tag(sid, blob, risk))

    def get(self, sid):
        """Returns (ciphertext, risk). Raises ValueError if verification fails."""
        blob, risk, tag = self.vault[sid]
        if not hmac.compare_digest(tag, self._tag(sid, blob, risk)):
            raise ValueError(f"Integrity check failed for {sid}")
        return blob, risk
