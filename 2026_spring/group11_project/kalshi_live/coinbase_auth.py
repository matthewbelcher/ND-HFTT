"""Coinbase CDP JWT for Advanced Trade WebSocket (ES256)."""

from __future__ import annotations

import base64
import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec


def load_cdp_signing_key(path: Path):
    data = json.loads(path.read_text())
    raw_key = data["privateKey"]
    if "BEGIN" in raw_key:
        return serialization.load_pem_private_key(
            raw_key.encode("utf-8"), password=None
        ), (data.get("name") or data.get("id") or "")
    raw = base64.b64decode(raw_key)
    if len(raw) < 32:
        raise ValueError("privateKey must be PEM or base64-encoded 32+ byte EC secret")
    priv_int = int.from_bytes(raw[:32], "big")
    return ec.derive_private_key(priv_int, ec.SECP256R1(), default_backend()), (
        data.get("name") or data.get("id") or ""
    )


def build_ws_jwt(key_path: Path, ttl_seconds: int = 120) -> str:
    private_key, api_key_id = load_cdp_signing_key(key_path)
    if not api_key_id:
        raise ValueError("cdp_api_key.json must include 'id' or 'name' (API key id)")
    now = int(time.time())
    payload: Dict[str, Any] = {
        "iss": "cdp",
        "nbf": now,
        "exp": now + ttl_seconds,
        "sub": api_key_id,
    }
    return jwt.encode(
        payload,
        private_key,
        algorithm="ES256",
        headers={"kid": api_key_id, "nonce": secrets.token_hex(16)},
    )
