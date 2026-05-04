import hmac
import json
from abc import ABC, abstractmethod

from config import Settings


class AuthError(Exception):
    pass


class Authenticator(ABC):
    """Protocol for validating bearer tokens presented by connecting clients.

    Concrete implementations are selected by :func:`build_authenticator`
    based on ``settings.auth_mode`` and are consumed by the client-listener
    server (see ``server._ensure_auth``).

    Contract for :meth:`validate`:

    * Returns ``True`` iff ``token`` is a valid, currently-accepted
      credential. Returns ``False`` for any invalid, empty, or malformed
      token -- it must not raise in that case.
    * Implementations must use a constant-time comparison
      (e.g. :func:`hmac.compare_digest`) when checking token equality, to
      avoid leaking information via timing side-channels.
    * May raise :class:`AuthError` only for backend/infrastructure
      failures that prevent the authenticator from making a decision
      (e.g. a secret store is unreachable). Simple "bad token" outcomes
      must be reported as ``False``, not as an exception.
    """

    @abstractmethod
    async def validate(self, token: str) -> bool:
        ...


class StaticTokenAuthenticator(Authenticator):
    def __init__(self, expected_token: str):
        if not expected_token:
            raise AuthError("CLIENT_LISTENER_AUTH_TOKEN is required for static-token auth")
        self._expected_token = expected_token

    async def validate(self, token: str) -> bool:
        return bool(token) and hmac.compare_digest(token, self._expected_token)


class AwsSecretsManagerTokenAuthenticator(Authenticator):
    def __init__(self, expected_token: str):
        if not expected_token:
            raise AuthError("secret value did not contain an auth token")
        self._expected_token = expected_token

    @staticmethod
    def from_secret_string(secret_string: str) -> "AwsSecretsManagerTokenAuthenticator":
        try:
            parsed = json.loads(secret_string)
            token = parsed.get("token") or parsed.get("auth_token")
        except json.JSONDecodeError:
            token = secret_string.strip()
        return AwsSecretsManagerTokenAuthenticator(token)

    async def validate(self, token: str) -> bool:
        return bool(token) and hmac.compare_digest(token, self._expected_token)


def build_authenticator(settings: Settings) -> Authenticator:
    mode = settings.auth_mode.lower()
    if mode == "static-token":
        return StaticTokenAuthenticator(settings.auth_token)

    if mode == "aws-secretsmanager":
        if not settings.auth_secret_arn:
            raise AuthError(
                "CLIENT_LISTENER_AUTH_SECRET_ARN is required for aws-secretsmanager auth"
            )
        try:
            import boto3
        except ImportError as exc:
            raise AuthError("boto3 is required for aws-secretsmanager auth") from exc

        client = boto3.client("secretsmanager", region_name=settings.aws_region)
        response = client.get_secret_value(SecretId=settings.auth_secret_arn)
        secret_string = response.get("SecretString", "")
        return AwsSecretsManagerTokenAuthenticator.from_secret_string(secret_string)

    raise AuthError(f"unsupported auth mode: {settings.auth_mode}")
