"""Session credential helpers for strict per-session isolation."""

from __future__ import annotations

import hmac
import secrets
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionCredentials:
    """Viewer/write credentials issued for one session."""

    viewer_token: str
    write_token: str


def issue_session_credentials() -> SessionCredentials:
    """Generate one viewer token and one writer token."""

    return SessionCredentials(
        viewer_token=secrets.token_urlsafe(24),
        write_token=secrets.token_urlsafe(24),
    )


def token_matches(expected: str, provided: str) -> bool:
    """Constant-time token compare."""

    if not expected or not provided:
        return False
    return hmac.compare_digest(expected, provided)
