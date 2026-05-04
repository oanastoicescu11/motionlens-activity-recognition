from app.ui.streamlit_app import (
    _backend_api_url,
    _build_mobile_join_url,
    _is_mobile_user_agent,
    _should_auto_open_mobile_capture,
)


def test_is_mobile_user_agent_matches_phone_browsers() -> None:
    assert _is_mobile_user_agent(
        "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148"
    )
    assert _is_mobile_user_agent(
        "Mozilla/5.0 (Linux; Android 15; Pixel 9) AppleWebKit/537.36 Chrome/136.0 Mobile Safari/537.36"
    )


def test_is_mobile_user_agent_rejects_desktop_browsers() -> None:
    assert not _is_mobile_user_agent(
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/136.0 Safari/537.36"
    )
    assert not _is_mobile_user_agent(None)


def test_auto_open_mobile_capture_requires_phone_and_setup_state() -> None:
    assert _should_auto_open_mobile_capture(prefer_direct_mobile_join=True, show_phone_setup=True)
    assert not _should_auto_open_mobile_capture(prefer_direct_mobile_join=False, show_phone_setup=True)
    assert not _should_auto_open_mobile_capture(prefer_direct_mobile_join=True, show_phone_setup=False)


def test_backend_api_url_uses_configured_base_url() -> None:
    assert _backend_api_url("/v1/sessions/start", base_url="https://api.example.com/") == (
        "https://api.example.com/v1/sessions/start"
    )


def test_build_mobile_join_url_uses_public_backend_url() -> None:
    assert _build_mobile_join_url("join-token-123", public_backend_url="https://live.example.com/") == (
        "https://live.example.com/mobile-capture?join_token=join-token-123"
    )