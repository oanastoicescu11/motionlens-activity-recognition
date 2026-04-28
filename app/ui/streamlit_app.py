"""Streamlit desktop dashboard for hybrid QR + mobile capture flow."""

from __future__ import annotations

import io
import socket
import requests

try:
    import streamlit as st
    import qrcode
    HAS_QRCODE = True
except ImportError:
    HAS_QRCODE = False
except Exception:  # pragma: no cover
    st = None
    HAS_QRCODE = False


BACKEND_BASE_URL = "http://localhost:8000"

def _drain_worker_once() -> None:
    """No-op: inference is now handled by a background thread inside the backend process."""
    pass


def _get_local_ip() -> str:
    """Get the local IP address visible to other devices on the network."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


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


def _create_session() -> dict | None:
    """Create a desktop bootstrap session via FastAPI backend."""
    try:
        response = requests.post(
            f"{BACKEND_BASE_URL}/v1/sessions/start",
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


def main() -> None:
    if st is None:
        raise RuntimeError("streamlit is required to run this UI.")

    st.set_page_config(page_title="MotionLens Live", layout="wide")
    st.title("MotionLens Live")

    # Initialize session state
    if "session_data" not in st.session_state:
        st.session_state.session_data = None
        st.session_state.local_ip = _get_local_ip()

    # Auto-create session on first load
    if st.session_state.session_data is None:
        st.info("Creating session for you...")
        session_data = _create_session()
        if session_data:
            st.session_state.session_data = session_data
            st.rerun()
        else:
            st.error("Failed to create session. Is backend running on localhost:8000?")
            return

    session_data = st.session_state.session_data
    session_id = session_data["session_id"]
    viewer_token = session_data["viewer_token"]
    join_token = session_data["join_token"]
    local_ip = st.session_state.local_ip

    # Display phone setup section
    st.markdown("---")
    st.subheader("📱 Phone Setup (Scan to Join)")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("**Scan with your phone:**")
        mobile_join_url = (
            f"http://{local_ip}:8000/mobile-capture"
            f"?join_token={join_token}"
        )
        qr_bytes = _generate_qr_code(mobile_join_url)
        if qr_bytes:
            st.image(qr_bytes, caption="Open mobile capture page", width=250)
        else:
            st.warning("QR code library not installed. Install: `pip install qrcode[pil]`")

    with col2:
        st.write("**Or copy manually:**")
        st.code(mobile_join_url, language="url")
        st.write("**Steps:**")
        st.markdown("""
        1. Scan the QR code with your phone.
        2. Tap **Start Analysis** on the phone page.
        3. Grant motion permission when prompted.
        4. Phone starts streaming to this session.
        5. Desktop and phone both show live updates.
        """)

    st.markdown("---")
    st.subheader("🎯 View Live Data")

    signal_url = f"{BACKEND_BASE_URL}/v1/sessions/{session_id}/signal"
    inference_url = f"{BACKEND_BASE_URL}/v1/sessions/{session_id}/inference"

    signal_col, inference_col = st.columns(2)

    auto_refresh = st.checkbox("Auto refresh", value=True)
    if auto_refresh:
        import datetime
        st.caption(f"Auto refresh enabled — last polled at {datetime.datetime.now().strftime('%H:%M:%S')}.")

    with signal_col:
        st.subheader("Signal Plane")
        if st.button("Refresh signal") or auto_refresh:
            try:
                signal = _fetch_json(signal_url, headers={"X-Viewer-Token": viewer_token})
                points = signal.get("points", [])
                import datetime
                st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
                st.write(f"Raw points: {len(points)}")
                if len(points) >= 2:
                    xs = [p[1] for p in points]
                    ys = [p[2] for p in points]
                    zs = [p[3] for p in points]
                    # Only chart if at least one axis has variance
                    if max(xs) != min(xs) or max(ys) != min(ys) or max(zs) != min(zs):
                        st.line_chart({"x": xs, "y": ys, "z": zs})
                    else:
                        st.info("Signal received but values are constant — sensor may not be moving yet.")
                elif len(points) == 1:
                    st.info("1 point received — waiting for more data.")
            except Exception as e:
                st.error(f"Failed to fetch signal: {e}")

    with inference_col:
        st.subheader("Inference Plane")
        if st.button("Refresh inference") or auto_refresh:
            _drain_worker_once()
            try:
                inference = _fetch_json(inference_url, headers={"X-Viewer-Token": viewer_token})
                snapshot = inference.get("inference")
                if snapshot is None:
                    st.info("No inference yet — waiting for enough data (128+ samples).")
                elif isinstance(snapshot, dict):
                    import datetime
                    st.caption(f"Last updated: {datetime.datetime.now().strftime('%H:%M:%S')}")
                    status = snapshot.get("status", "unknown")
                    if status == "success":
                        activity = snapshot.get("activity", "?")
                        confidence = snapshot.get("confidence", 0.0)
                        sample_count = snapshot.get("sample_count", 0)
                        placement_label = snapshot.get("placement_label", "unknown_free_living")
                        window_used = snapshot.get("window_samples_used", 0)
                        st.metric("Activity", activity)
                        st.metric("Confidence", f"{confidence:.1%}")
                        st.metric("Total samples received", sample_count)
                        st.metric("Prediction window", f"{window_used} samples")
                        st.caption("Live prediction uses the latest 128 samples (~2.56s at 50 Hz) after timestamp-based resampling.")
                        st.caption(f"Placement: {placement_label}")
                        top = snapshot.get("top_predictions", [])
                        if top:
                            st.write("**Top predictions:**")
                            for pred in top:
                                st.write(f"- {pred['label']}: {pred['confidence']:.1%}")
                    elif status == "queued":
                        needed = snapshot.get("samples_needed", 128)
                        have = snapshot.get("sample_count", 0)
                        st.info(f"Collecting data... {have}/{needed} samples")
                    elif status == "error":
                        st.warning(f"Inference error: {snapshot.get('error', 'unknown')}")
                    else:
                        st.json(snapshot)
                else:
                    st.info("Waiting for inference...")
            except Exception as e:
                st.error(f"Failed to fetch inference: {e}")

    if auto_refresh:
        import time

        time.sleep(2)
        st.rerun()

    # Debug info (hidden by default)
    with st.expander("Debug Info"):
        st.json({
            "session_id": session_id,
            "viewer_token": "[redacted]",
            "join_token": "[redacted]",
            "local_ip": local_ip,
            "mobile_join_url": mobile_join_url,
        })


if __name__ == "__main__":
    main()
