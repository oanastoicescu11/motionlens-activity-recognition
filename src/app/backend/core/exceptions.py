"""Application exception hierarchy.

Raise these from business-logic layers; route handlers catch them and
convert to the appropriate HTTP response.

Usage::

    from .core.exceptions import AuthError, RequestError

    raise AuthError("Invalid write token.")
    raise RequestError("sessionId is required.")
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application errors."""

    status_code: int = 500

    def __init__(self, detail: str = "Internal server error") -> None:
        super().__init__(detail)
        self.detail = detail


class AuthError(AppError):
    """Authentication / authorisation failure → HTTP 403."""

    status_code = 403

    def __init__(self, detail: str = "Permission denied") -> None:
        super().__init__(detail)


class RequestError(AppError):
    """Bad request from caller → HTTP 400."""

    status_code = 400

    def __init__(self, detail: str = "Bad request") -> None:
        super().__init__(detail)


class NotFoundError(AppError):
    """Resource not found → HTTP 404."""

    status_code = 404

    def __init__(self, detail: str = "Not found") -> None:
        super().__init__(detail)
