"""Mobile capture page route.

Delegates to ``mobile_capture_page.build_mobile_capture_router`` so the HTML
page lives in one place.
"""

from __future__ import annotations

from ...mobile_capture_page import build_mobile_capture_router

__all__ = ["build_mobile_capture_router"]
