import hashlib
import hmac
import secrets


PBKDF2_ITERATIONS = 600_000
PASSWORD_SALT_BYTES = 16
SESSION_TOKEN_BYTES = 32


def hash_password(password: str) -> tuple[str, str, int]:
    salt_bytes = secrets.token_bytes(PASSWORD_SALT_BYTES)
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt_bytes,
        PBKDF2_ITERATIONS,
    ).hex()
    return password_hash, salt_bytes.hex(), PBKDF2_ITERATIONS


def verify_password(
    password: str,
    expected_hash: str,
    salt_hex: str,
    iterations: int,
) -> bool:
    candidate_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        bytes.fromhex(salt_hex),
        iterations,
    ).hex()
    return hmac.compare_digest(candidate_hash, expected_hash)


def generate_session_token() -> str:
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def hash_session_token(token: str) -> str:
    return hashlib.sha256(token.encode('utf-8')).hexdigest()
