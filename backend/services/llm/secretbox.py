"""Optional AES-256-GCM sealing for LLM provider credentials at rest."""

import base64
import os
import secrets

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

PREFIX = "enc:v1:"


def _key() -> bytes | None:
    raw = os.environ.get("MILIMO_SECRET_KEY", "").strip()
    if not raw:
        return None
    try:
        value = bytes.fromhex(raw)
        if len(value) == 32:
            return value
    except ValueError:
        pass
    for decoder in (base64.b64decode, base64.urlsafe_b64decode):
        try:
            value = decoder(raw + "=" * (-len(raw) % 4))
            if len(value) == 32:
                return value
        except Exception:
            pass
    raise ValueError("MILIMO_SECRET_KEY must be 32 bytes encoded as hex or base64")


def seal(value: str | None) -> str | None:
    if not value:
        return None
    key = _key()
    if key is None:
        return value
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(key).encrypt(nonce, value.encode(), None)
    return PREFIX + base64.urlsafe_b64encode(nonce + ciphertext).decode().rstrip("=")


def unseal(value: str | None) -> str | None:
    if not value or not value.startswith(PREFIX):
        return value
    key = _key()
    if key is None:
        raise ValueError("encrypted LLM secret found but MILIMO_SECRET_KEY is unset")
    raw = value[len(PREFIX):]
    payload = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4))
    return AESGCM(key).decrypt(payload[:12], payload[12:], None).decode()


def source(value: str | None) -> str:
    if not value:
        return "none"
    return "sealed_db" if value.startswith(PREFIX) else "plaintext_db"
