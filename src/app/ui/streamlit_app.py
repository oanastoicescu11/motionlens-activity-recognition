"""Streamlit desktop dashboard for hybrid QR + mobile capture flow."""

from __future__ import annotations

import datetime
import io
import json
from urllib.parse import urlencode

import requests

from app.backend.config import settings

try:
    import streamlit as st
    import streamlit.components.v1 as components
    import qrcode

    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False
    components = None
except Exception:  # pragma: no cover
    st = None
    components = None
    HAS_QRCODE = False

AUTO_REFRESH_SECONDS = 0.35
MIN_AUTO_REFRESH_SECONDS = 0.2
MAX_AUTO_REFRESH_SECONDS = 2.0
SIGNAL_REFRESH_MIN_SECONDS = 0.6
SUMMARY_REFRESH_MIN_SECONDS = 1.0
_MOBILE_USER_AGENT_TOKENS = (
    "android",
    "blackberry",
    "iphone",
    "ipad",
    "ipod",
    "mobile",
    "opera mini",
    "windows phone",
)


def _drain_worker_once() -> None:
    """No-op: inference is now handled by a background thread inside the backend process."""
    pass


def _backend_api_url(path: str, *, base_url: str | None = None) -> str:
    """Build one backend API URL from the configured backend base URL."""

    effective_base_url = (base_url or settings.backend_base_url).rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    return f"{effective_base_url}{normalized_path}"


def _build_mobile_join_url(join_token: str, *, public_backend_url: str | None = None) -> str:
    """Build the public mobile capture URL used in the QR code and phone redirect."""

    effective_public_url = (public_backend_url or settings.public_backend_url).rstrip("/")
    query = urlencode({"join_token": join_token})
    return f"{effective_public_url}/mobile-capture?{query}"


def _generate_qr_code(text: str) -> bytes | None:
    """Generate QR code image as PNG bytes."""
    if not HAS_QRCODE:
        return None
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img_bytes = io.BytesIO()
    img.save(img_bytes, "PNG")
    img_bytes.seek(0)
    return img_bytes.getvalue()


def _is_mobile_user_agent(user_agent: str | None) -> bool:
    """Return True when the request user agent looks like a phone or tablet browser."""
    normalized = (user_agent or "").strip().lower()
    if not normalized:
        return False
    return any(token in normalized for token in _MOBILE_USER_AGENT_TOKENS)


def _client_prefers_direct_mobile_join() -> bool:
    """Prefer a direct join button when the Streamlit page is already open on a phone."""
    if st is None:
        return False

    context = getattr(st, "context", None)
    headers = getattr(context, "headers", None)
    if headers is None:
        return False

    user_agent = headers.get("user-agent") or headers.get("User-Agent")
    return _is_mobile_user_agent(user_agent)


def _should_auto_open_mobile_capture(*, prefer_direct_mobile_join: bool, show_phone_setup: bool) -> bool:
    """Auto-open the mobile capture page when the homepage itself is opened on a phone."""
    return prefer_direct_mobile_join and show_phone_setup


def _render_mobile_capture_redirect(mobile_join_url: str) -> None:
    """Redirect mobile homepage visits straight to the phone capture page."""
    if st is None:
        return

    st.info("Opening phone capture...")
    st.link_button("Open phone capture page", mobile_join_url, use_container_width=True)

    if components is None:
        return

    escaped_url = json.dumps(mobile_join_url)
    components.html(
        f"""
        <script>
        window.top.location.replace({escaped_url});
        </script>
        """,
        height=0,
    )


def _create_session() -> dict | None:
    """Create a desktop bootstrap session via FastAPI backend."""
    try:
        response = requests.post(
            _backend_api_url("/v1/sessions/start"),
            params={
                "owner_id": "streamlit-ui",
                "mode": "desktop",
                "ttl_seconds": 300,
                "join_ttl_seconds": 90,
            },
            timeout=5,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        if st is not None:
            st.error(f"Failed to create session: {e}")
        return None


def _fetch_json(
    url: str,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict:
    response = requests.get(url, params=params, headers=headers, timeout=5)
    response.raise_for_status()
    return response.json()


def _load_session_summary(session_id: str, viewer_token: str) -> dict | None:
    """Fetch the finalized session summary when one exists."""
    summary_url = _backend_api_url(f"/v1/sessions/{session_id}/summary")
    try:
        response = _fetch_json(summary_url, headers={"X-Viewer-Token": viewer_token})
    except Exception:
        return None
    loaded_summary = response.get("summary")
    return loaded_summary if isinstance(loaded_summary, dict) and loaded_summary else None


def _fetch_live_signal(signal_url: str, viewer_token: str) -> tuple[list[list[float]], str | None, bool]:
    """Fetch live signal points and whether they imply capture is active."""
    try:
        signal = _fetch_json(signal_url, headers={"X-Viewer-Token": viewer_token})
    except Exception as e:
        return [], str(e), False
    points = signal.get("points", [])
    return points, None, bool(points)


def _fetch_live_inference(inference_url: str, viewer_token: str) -> tuple[dict | None, str | None, bool]:
    """Fetch the latest inference snapshot and whether capture looks active."""
    _drain_worker_once()
    try:
        inference = _fetch_json(inference_url, headers={"X-Viewer-Token": viewer_token})
    except Exception as e:
        return None, str(e), False
    loaded_snapshot = inference.get("inference")
    if not isinstance(loaded_snapshot, dict):
        return None, None, False
    capture_detected = loaded_snapshot.get("status") in {"success", "queued"}
    return loaded_snapshot, None, bool(capture_detected)


def _mark_capture_started(capture_detected: bool) -> None:
    """Promote the UI into the capture-started state once live data is observed."""
    if capture_detected and not st.session_state.capture_started:
        st.session_state.capture_started = True
        st.rerun()


# ---------------------------------------------------------------------------
# Insight rendering helpers
# ---------------------------------------------------------------------------


_ACTIVITY_EMOJI = {
    "walk": "🚶",
    "run": "🏃",
    "stairs": "𓊍",
    "sit/lay": "🪑",
    "stand": "🧍",
    "transitions": "🔄",
    "locomotion-other": "🚴‍♂️",
}

_CONFIDENCE_TIER_COLOR = {
    "high": "🟢",
    "medium": "🟡",
    "low": "🔴",
}


def _activity_emoji(activity: str) -> str:
    return _ACTIVITY_EMOJI.get(activity, "❓")


def _confidence_badge(tier: str) -> str:
    color = _CONFIDENCE_TIER_COLOR.get(tier, "⚪")
    return f"{color} {tier.upper()}"


def _render_session_summary(summary: dict) -> None:
    """Render the finalized denoised session summary."""
    if st is None:
        return

    st.markdown("## Session Summary")
    summary_text = str(summary.get("summary_text", "")).strip()
    if summary_text:
        st.success(summary_text)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Dominant", str(summary.get("dominant_activity", "unknown")).replace("-", " ").title())
    with metric_cols[1]:
        st.metric("Duration", f"{float(summary.get('total_duration_seconds', 0.0)):.0f}s")
    with metric_cols[2]:
        st.metric("Steps", f"{int(summary.get('estimated_steps', 0))}")
    with metric_cols[3]:
        st.metric("Signal", f"{float(summary.get('signal_quality_mean', 0.0)):.0f}/100")

    detail_cols = st.columns(4)
    with detail_cols[0]:
        st.metric("Walking", f"{float(summary.get('walking_seconds', 0.0)):.0f}s")
    with detail_cols[1]:
        st.metric("Stationary", f"{float(summary.get('stationary_seconds', 0.0)):.0f}s")
    with detail_cols[2]:
        st.metric("Avg cadence", f"{float(summary.get('average_cadence_spm', 0.0)):.0f} SPM")
    with detail_cols[3]:
        st.metric("Longest walk", f"{float(summary.get('longest_walk_bout_seconds', 0.0)):.0f}s")

    time_by_activity = summary.get("time_by_activity") or {}
    if isinstance(time_by_activity, dict) and time_by_activity:
        st.write("**Time by activity**")
        st.bar_chart(time_by_activity)

    notes = summary.get("notes") or []
    if notes:
        st.write("**Notes**")
        for note in notes:
            st.caption(f"- {note}")


def _render_sidebar_panel(
    *,
    mobile_join_url: str,
    show_phone_setup: bool,
    summary_available: bool,
    prefer_direct_mobile_join: bool,
) -> None:
    """Render the session status and phone join affordances in the sidebar."""
    if st is None:
        return

    with st.sidebar:
        st.header("Session")

        if summary_available:
            st.success("Session finalized")
        elif show_phone_setup:
            if prefer_direct_mobile_join:
                st.info("Open capture directly on this phone")
                st.link_button("Open phone capture page", mobile_join_url, use_container_width=True)
                st.caption("For the best activity readout, keep the phone in a trouser pocket while you move.")
            else:
                st.info("Scan with your phone to start live activity capture.")
                qr_bytes = _generate_qr_code(mobile_join_url)
                if qr_bytes:
                    st.image(qr_bytes, caption="Open mobile capture page", width="stretch")
                else:
                    st.warning("QR code generation is unavailable.")
                st.caption("Keep the phone in a trouser pocket for clearer activity recognition.")
        else:
            st.success("Phone connected")
            st.caption("Live capture is active on the phone.")


def _render_home_intro(*, show_phone_setup: bool, summary_available: bool) -> None:
    """Render concise homepage copy without crowding the live panels."""
    if st is None:
        return

    st.caption(
        "MotionLens uses your phone's accelerometer for live activity recognition and a short session recap when you stop."
    )
    if show_phone_setup:
        st.info(
            "Scan the QR code on your computer to start streaming motion data from your phone, then watch the live prediction update here. "
            "For the best result, carry the phone in a trouser pocket while moving."
        )
    elif summary_available:
        st.caption("The live session has ended. The summary below condenses the detected activity into a short recap.")


def _render_insight_card(snapshot: dict) -> None:
    """Render the activity insight card with conditional metrics."""
    if st is None:
        return

    activity = snapshot.get("activity", "?")
    confidence = snapshot.get("confidence", 0.0)
    insights = snapshot.get("insights", {})

    # Headline
    emoji = _activity_emoji(activity)
    st.markdown(f"## {emoji} {activity.upper()}")

    # Confidence tier badge
    tier = insights.get("prediction_confidence_tier", "medium")
    st.markdown(f"**Confidence:** {confidence:.1%}  {_confidence_badge(tier)}")

    # Activity-specific insight text
    insight_text = insights.get("activity_specific_insight", "")
    if insight_text:
        st.info(insight_text)

    # Determine applicable insights
    applicable = set(insights.get("applicable_insights", []))

    # --- Activity-specific metrics ---
    st.markdown("---")
    st.subheader("Activity Metrics")

    metric_cols = st.columns(2)
    col_idx = 0

    # Locomotion metrics
    if "cadence_spm" in applicable:
        with metric_cols[col_idx % 2]:
            cadence = insights.get("cadence_spm", 0.0)
            st.metric("Cadence", f"{cadence:.0f} SPM")
        col_idx += 1

    if "cadence_consistency_pct" in applicable:
        with metric_cols[col_idx % 2]:
            consistency = insights.get("cadence_consistency_pct", 0.0)
            st.metric("Consistency", f"{consistency:.0f}%")
            st.progress(min(int(consistency), 100))
        col_idx += 1

    if "stride_regularity_score" in applicable:
        with metric_cols[col_idx % 2]:
            reg = insights.get("stride_regularity_score", 0.0)
            st.metric("Gait Regularity", f"{reg:.0f}/100")
            st.progress(min(int(reg), 100))
        col_idx += 1

    if "vertical_oscillation_mm" in applicable:
        with metric_cols[col_idx % 2]:
            bounce = insights.get("vertical_oscillation_mm", 0.0)
            st.metric("Vertical Bounce", f"{bounce:.0f} mm")
        col_idx += 1

    if "ground_impact_score" in applicable:
        with metric_cols[col_idx % 2]:
            impact = insights.get("ground_impact_score", 0.0)
            st.metric("Impact", f"{impact:.0f}/100")
            # Color-coded progress
            if impact < 40:
                st.progress(min(int(impact), 100), text="🟢 Soft")
            elif impact < 70:
                st.progress(min(int(impact), 100), text="🟡 Moderate")
            else:
                st.progress(min(int(impact), 100), text="🔴 Hard")
        col_idx += 1

    if "smoothness_score" in applicable:
        with metric_cols[col_idx % 2]:
            smooth = insights.get("smoothness_score", 0.0)
            st.metric("Smoothness", f"{smooth:.0f}/100")
            st.progress(min(int(smooth), 100))
        col_idx += 1

    # Static metrics
    if "posture_stability_score" in applicable:
        with metric_cols[col_idx % 2]:
            stability = insights.get("posture_stability_score", 0.0)
            st.metric("Stillness", f"{stability:.0f}/100")
            st.progress(min(int(stability), 100))
        col_idx += 1

    if "sway_amplitude_deg" in applicable:
        with metric_cols[col_idx % 2]:
            sway = insights.get("sway_amplitude_deg", 0.0)
            st.metric("Sway", f"{sway:.1f}°")
        col_idx += 1

    # --- Universal metrics ---
    st.markdown("---")
    st.subheader("Signal & Quality")

    universal_cols = st.columns(2)
    with universal_cols[0]:
        signal_quality = insights.get("signal_quality_score", 0.0)
        st.metric("Signal Quality", f"{signal_quality:.0f}/100")
        st.progress(min(int(signal_quality), 100))

    with universal_cols[1]:
        intensity_level = insights.get("intensity_level", "unknown")
        intensity_ratio = insights.get("intensity_ratio", 0.0)
        st.metric("Intensity", f"{intensity_level.title()} ({intensity_ratio:.1f}×)")

    # Phone orientation
    orientation = insights.get("phone_orientation", "unknown")
    st.caption(f"📱 Phone orientation: **{orientation}**")

    # Burst / posture change alerts
    burst = insights.get("burst_detected", False)
    posture_change = insights.get("posture_change_detected", False)
    if burst or posture_change:
        alert_cols = st.columns(2)
        if burst:
            with alert_cols[0]:
                st.warning("⚡ Impact burst detected")
        if posture_change:
            with alert_cols[1]:
                st.warning("🔄 Posture change detected")


def _render_signal_panel(
    *,
    signal_url: str,
    viewer_token: str,
    poll_now: bool,
    summary_available: bool,
) -> None:
    """Render only the signal panel so it can refresh independently."""
    if st is None:
        return

    signal_points: list[list[float]] = []
    signal_error: str | None = None

    if poll_now and not summary_available:
        signal_points, signal_error, capture_detected = _fetch_live_signal(signal_url, viewer_token)
        _mark_capture_started(capture_detected)

    if signal_error is not None:
        st.error(f"Failed to fetch signal: {signal_error}")
    elif summary_available:
        st.info("Session finalized — live signal polling stopped.")
    else:
        st.write(f"Raw points: {len(signal_points)}")
        if len(signal_points) >= 2:
            xs = [p[1] for p in signal_points]
            ys = [p[2] for p in signal_points]
            zs = [p[3] for p in signal_points]
            if max(xs) != min(xs) or max(ys) != min(ys) or max(zs) != min(zs):
                st.line_chart({"x": xs, "y": ys, "z": zs})
            else:
                st.info("Signal received but values are constant — sensor may not be moving yet.")
        elif len(signal_points) == 1:
            st.info("1 point received — waiting for more data.")
        elif poll_now:
            st.info("No signal yet — start capture on the phone.")
        else:
            st.info("Enable auto refresh or press Refresh now to load live signal.")


def _render_inference_panel(
    *,
    session_id: str,
    viewer_token: str,
    inference_url: str,
    poll_now: bool,
    summary_available: bool,
) -> None:
    """Render only the inference panel so it can refresh independently."""
    if st is None:
        return

    current_summary = _load_session_summary(session_id, viewer_token) if summary_available else None
    snapshot: dict | None = None
    inference_error: str | None = None

    if poll_now and not summary_available:
        snapshot, inference_error, capture_detected = _fetch_live_inference(inference_url, viewer_token)
        _mark_capture_started(capture_detected)

    if current_summary is not None:
        _render_session_summary(current_summary)
    elif inference_error is not None:
        st.error(f"Failed to fetch inference: {inference_error}")
    elif snapshot is None:
        if poll_now:
            st.info("No inference yet — waiting for enough data (128+ samples).")
        else:
            st.info("Enable auto refresh or press Refresh now to load live inference.")
    else:
        status = snapshot.get("status", "unknown")
        if status == "success":
            _render_insight_card(snapshot)
        elif status == "queued":
            needed = snapshot.get("samples_needed", 128)
            have = snapshot.get("sample_count", 0)
            st.info(f"Collecting data... {have}/{needed} samples")
        elif status == "error":
            st.warning(f"Inference error: {snapshot.get('error', 'unknown')}")
        else:
            st.json(snapshot)


def _watch_summary_state(session_id: str, viewer_token: str) -> None:
    """Update session state when a finalized summary becomes available."""
    if st is None or st.session_state.get("summary_available", False):
        return
    summary = _load_session_summary(session_id, viewer_token)
    if summary is None:
        return
    st.session_state.summary_available = True
    st.session_state.capture_started = True
    st.rerun()


def main() -> None:
    if st is None:
        raise RuntimeError("streamlit is required to run this UI.")

    st.set_page_config(page_title="MotionLens Live", layout="wide")
    st.title("MotionLens Live")

    # Initialize session state
    if "session_data" not in st.session_state:
        st.session_state.session_data = None
        st.session_state.capture_started = False
        st.session_state.summary_available = False
        st.session_state.live_refresh_seconds = AUTO_REFRESH_SECONDS

    # Auto-create session on first load
    if st.session_state.session_data is None:
        st.info("Creating session for you...")
        session_data = _create_session()
        if session_data:
            st.session_state.session_data = session_data
            st.rerun()
        else:
            st.error(f"Failed to create session. Is the backend running at {settings.backend_base_url}?")
            return

    session_data = st.session_state.session_data
    session_id = session_data["session_id"]
    viewer_token = session_data["viewer_token"]
    join_token = session_data["join_token"]
    mobile_join_url = _build_mobile_join_url(join_token)

    signal_url = _backend_api_url(f"/v1/sessions/{session_id}/signal")
    inference_url = _backend_api_url(f"/v1/sessions/{session_id}/inference")

    summary = _load_session_summary(session_id, viewer_token)
    if summary is not None:
        st.session_state.capture_started = True
        st.session_state.summary_available = True

    summary_available = bool(st.session_state.summary_available)
    prefer_direct_mobile_join = _client_prefers_direct_mobile_join()

    show_phone_setup = not st.session_state.capture_started and not summary_available

    if _should_auto_open_mobile_capture(
        prefer_direct_mobile_join=prefer_direct_mobile_join,
        show_phone_setup=show_phone_setup,
    ):
        _render_mobile_capture_redirect(mobile_join_url)
        st.stop()

    _render_sidebar_panel(
        mobile_join_url=mobile_join_url,
        show_phone_setup=show_phone_setup,
        summary_available=summary_available,
        prefer_direct_mobile_join=prefer_direct_mobile_join,
    )

    _render_home_intro(show_phone_setup=show_phone_setup, summary_available=summary_available)

    st.markdown("---")
    st.subheader("🎯 Live View")

    controls_col, refresh_col, button_col = st.columns([1, 1, 1])
    with controls_col:
        auto_refresh = st.checkbox("Auto refresh", value=not summary_available, disabled=summary_available)
    with refresh_col:
        refresh_interval = st.slider(
            "Desktop refresh (s)",
            min_value=MIN_AUTO_REFRESH_SECONDS,
            max_value=MAX_AUTO_REFRESH_SECONDS,
            value=float(st.session_state.live_refresh_seconds),
            step=0.05,
            disabled=summary_available,
        )
        st.session_state.live_refresh_seconds = refresh_interval
    with button_col:
        refresh_now = st.button("Refresh now", disabled=summary_available)

    manual_refresh_requested = refresh_now and not summary_available
    signal_refresh_seconds = max(SIGNAL_REFRESH_MIN_SECONDS, refresh_interval * 2.0)
    summary_refresh_seconds = max(SUMMARY_REFRESH_MIN_SECONDS, refresh_interval * 3.0)

    @st.fragment(run_every=summary_refresh_seconds if auto_refresh and not summary_available else None)
    def _summary_watcher_fragment() -> None:
        _watch_summary_state(session_id, viewer_token)

    _summary_watcher_fragment()

    signal_col, inference_col = st.columns(2)

    with signal_col:
        st.subheader("Signal Plane")

        @st.fragment(run_every=signal_refresh_seconds if auto_refresh and not summary_available else None)
        def _signal_fragment() -> None:
            _render_signal_panel(
                signal_url=signal_url,
                viewer_token=viewer_token,
                poll_now=auto_refresh or manual_refresh_requested,
                summary_available=bool(st.session_state.summary_available),
            )

        _signal_fragment()

    with inference_col:
        st.subheader("Inference Plane")

        @st.fragment(run_every=refresh_interval if auto_refresh and not summary_available else None)
        def _inference_fragment() -> None:
            _render_inference_panel(
                session_id=session_id,
                viewer_token=viewer_token,
                inference_url=inference_url,
                poll_now=auto_refresh or manual_refresh_requested,
                summary_available=bool(st.session_state.summary_available),
            )

        _inference_fragment()

if __name__ == "__main__":
    main()
