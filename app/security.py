# app/security.py
"""
Password hashing (stdlib only): PBKDF2-HMAC-SHA256.
"""

import hmac
import base64
import hashlib
import secrets
from typing import Optional, Dict


def _pbkdf2_hash_password(password: str, salt_b64: Optional[str] = None) -> Dict[str, str]:
    """
    Store: algo, iters, salt_b64, hash_b64
    """
    if salt_b64 is None:
        salt = secrets.token_bytes(16)
        salt_b64 = base64.b64encode(salt).decode("utf-8")
    else:
        salt = base64.b64decode(salt_b64.encode("utf-8"))

    iters = 200_000
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=32)
    hash_b64 = base64.b64encode(dk).decode("utf-8")
    return {"algo": "pbkdf2_sha256", "iters": str(iters), "salt_b64": salt_b64, "hash_b64": hash_b64}


def _verify_password(password: str, stored: Dict[str, str]) -> bool:
    if stored.get("algo") != "pbkdf2_sha256":
        return False
    salt_b64 = stored.get("salt_b64")
    iters = int(stored.get("iters", "200000"))
    expected_b64 = stored.get("hash_b64")
    if not salt_b64 or not expected_b64:
        return False
    salt = base64.b64decode(salt_b64.encode("utf-8"))
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iters, dklen=32)
    got_b64 = base64.b64encode(dk).decode("utf-8")
    return hmac.compare_digest(got_b64, expected_b64)
