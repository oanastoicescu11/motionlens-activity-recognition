"""Mobile browser capture page for DeviceMotion streaming."""

from __future__ import annotations

try:
    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse
except Exception:  # pragma: no cover
    APIRouter = None  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]


_MOBILE_PAGE_HTML = """<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>MotionLens Mobile Capture</title>
  <style>
    body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 16px; }
    .card { border: 1px solid #ddd; border-radius: 10px; padding: 12px; margin-bottom: 12px; }
    button { padding: 10px 14px; border-radius: 8px; border: 1px solid #222; background: #111; color: #fff; }
    select { padding: 8px 10px; border-radius: 8px; border: 1px solid #bbb; width: 100%; margin-bottom: 10px; }
    code { background: #f4f4f4; padding: 2px 6px; border-radius: 4px; }
    .muted { color: #666; }
  </style>
</head>
<body>
  <h2>MotionLens Mobile Capture</h2>
  <div class=\"card\">
    <div><strong>Status:</strong> <span id=\"status\">Idle</span></div>
    <div><strong>Session:</strong> <code id=\"session\">-</code></div>
    <div><strong>Accepted points:</strong> <span id=\"accepted\">0</span></div>
    <div><strong>Upload errors:</strong> <span id=\"errors\">0</span></div>
  </div>
  <div class=\"card\">
    <label><strong>Phone placement</strong></label>
    <p class="muted">Detected: Pocket or waist (automatic)</p>
    <button id=\"startBtn\">Start Analysis</button>
    <button id=\"stopBtn\" disabled>Stop</button>
    <p class=\"muted\">Requires HTTPS and user tap for iOS permission.</p>
  </div>
  <div class=\"card\">
    <div><strong>Latest Inference:</strong></div>
    <pre id=\"inference\">(none)</pre>
  </div>

<script>
(() => {
  const params = new URLSearchParams(window.location.search);
  const joinToken = params.get("join_token") || "";
  const initialDeviceId = params.get("device_id") || `phone-${Math.random().toString(36).slice(2,10)}`;

  const statusEl = document.getElementById("status");
  const sessionEl = document.getElementById("session");
  const acceptedEl = document.getElementById("accepted");
  const errorsEl = document.getElementById("errors");
  const inferenceEl = document.getElementById("inference");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");

  const MAX_BUFFER_SAMPLES = 1200;
  const UPLOAD_INTERVAL_MS = 700;
  const MAX_UPLOAD_BATCH = 240;
  const INFERENCE_INTERVAL_MS = 1200;

  let writeToken = null;
  let viewerToken = null;
  let sessionId = null;
  let deviceId = initialDeviceId;
  let messageId = 1;
  let motionHandler = null;
  let uploadTimer = null;
  let inferenceTimer = null;
  let acceptedPoints = 0;
  let uploadErrors = 0;
  let droppedSamples = 0;
  let sampleBuffer = [];
  let uploadInFlight = false;
  let wakeLock = null;
  let noSleepVideo = null;

  const setStatus = (text) => { statusEl.textContent = text; };

  function selectedPlacementLabel() {
    // Pocket_only mode: always use front_pocket (no user selection)
    return "front_pocket";
  }

  function setCounters() {
    acceptedEl.textContent = String(acceptedPoints);
    errorsEl.textContent = String(uploadErrors);
  }

  async function acquireWakeLock() {
    // Primary: Screen Wake Lock API (Chrome, Edge, Safari 16.4+)
    if ("wakeLock" in navigator) {
      try {
        if (wakeLock == null) {
          wakeLock = await navigator.wakeLock.request("screen");
          wakeLock.addEventListener("release", () => {
            wakeLock = null;
          });
        }
        return;
      } catch (_) {
        // Fall through to no-sleep video fallback.
      }
    }
    // Fallback: silent no-sleep video (Firefox, older Safari, iOS WebView)
    startNoSleepVideo();
  }

  function startNoSleepVideo() {
    if (noSleepVideo) return;
    noSleepVideo = document.createElement("video");
    noSleepVideo.setAttribute("playsinline", "");
    noSleepVideo.setAttribute("muted", "");
    noSleepVideo.style.position = "absolute";
    noSleepVideo.style.width = "1px";
    noSleepVideo.style.height = "1px";
    noSleepVideo.style.opacity = "0";
    // Minimal 1-second silent MP4 (base64) — just enough to keep the screen awake.
    noSleepVideo.src = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAA0NtZGF0AAACrgYF//+q3EXpvebZSLeWLNgg2SPu73gyNjmtgGYZjYGYGgY2DmAmJmZmZiZ0ZmZmZkYJzZmZmZmZkZmZmZmaYJzZmZmZmZkZmZmZmYJzZmZmZmZkZmZmZmZ4bWRhdGEAAAQ0ZGF0AAQAAAAAAi0ZGF0AAQAAAAAAjQZGF0AAQAAAAAAkQZGF0AAQAAAAAAlQZGF0AAQAAAAAAmQZGF0AAQAAAAAAnQZGF0AAQAAAAAAoQZGF0AAQAAAAAApQZGF0AAQAAAAAAqQZGF0AAQAAAAAArQZGF0AAQAAAAAAsQZGF0AAQAAAAAAtQZGF0AAQAAAAAAuQZGF0AAQAAAAAAvQZGF0AAQAAAAAAwQZGF0AAQAAAAAAxQZGF0AAQAAAAAAyQZGF0AAQAAAAAAzQZGF0AAQAAAAAA0QZGF0AAQAAAAAA1QZGF0AAQAAAAAA2Q==";
    noSleepVideo.addEventListener("ended", () => {
      if (noSleepVideo) {
        noSleepVideo.currentTime = 0;
        noSleepVideo.play().catch(() => {});
      }
    });
    document.body.appendChild(noSleepVideo);
    noSleepVideo.play().catch(() => {
      setStatus("Streaming... (screen stay-awake not available)");
    });
  }

  function stopNoSleepVideo() {
    if (noSleepVideo) {
      noSleepVideo.pause();
      noSleepVideo.remove();
      noSleepVideo = null;
    }
  }

  async function releaseWakeLock() {
    if (wakeLock) {
      try {
        await wakeLock.release();
      } catch (_) {
        // Ignore release errors; lock may already be gone.
      }
      wakeLock = null;
    }
    stopNoSleepVideo();
  }

  function pushSample(sample) {
    sampleBuffer.push(sample);
    if (sampleBuffer.length > MAX_BUFFER_SAMPLES) {
      const overflow = sampleBuffer.length - MAX_BUFFER_SAMPLES;
      sampleBuffer.splice(0, overflow);
      droppedSamples += overflow;
      setStatus(`Streaming (buffer trimmed, dropped ${droppedSamples})`);
    }
  }

  async function joinSession() {
    if (!joinToken) throw new Error("Missing join_token in URL.");
    const res = await fetch(`/v1/sessions/join`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        join_token: joinToken,
        device_id: deviceId,
      }),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Join failed (${res.status}): ${body}`);
    }
    const data = await res.json();
    writeToken = data.write_token;
    viewerToken = data.viewer_token;
    sessionId = data.session_id;
    sessionEl.textContent = sessionId;
  }

  async function requestMotionPermission() {
    if (typeof DeviceMotionEvent === "undefined") {
      throw new Error("DeviceMotion is not supported on this device/browser.");
    }
    if (typeof DeviceMotionEvent.requestPermission === "function") {
      const state = await DeviceMotionEvent.requestPermission();
      if (state !== "granted") throw new Error("Motion permission denied.");
    }
  }

  async function uploadBatch({ forceAll = false, keepalive = false } = {}) {
    if (uploadInFlight || !writeToken || !sessionId || sampleBuffer.length === 0) {
      return;
    }

    uploadInFlight = true;
    const sendCount = forceAll ? sampleBuffer.length : Math.min(sampleBuffer.length, MAX_UPLOAD_BATCH);
    const samples = sampleBuffer.splice(0, sendCount);
    const payload = {
      source: "web-devicemotion",
      messageId: messageId,
      sessionId: sessionId,
      deviceId: deviceId,
      placementLabel: selectedPlacementLabel(),
      samples: samples,
    };

    try {
      const res = await fetch(`/v1/ingest/${encodeURIComponent(writeToken)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        keepalive: keepalive,
      });
      if (!res.ok) {
        throw new Error(`ingest ${res.status}`);
      }
      const out = await res.json();
      acceptedPoints += Number(out.accepted_points || 0);
      messageId += 1;
      setCounters();
      setStatus(document.hidden ? "Streaming (background)" : "Streaming...");
    } catch (_) {
      uploadErrors += 1;
      setCounters();
      sampleBuffer = samples.concat(sampleBuffer);
      setStatus("Streaming with retries (network issue)");
    } finally {
      uploadInFlight = false;
    }
  }

  function startCapture() {
    motionHandler = (event) => {
      // Model preprocessing expects total acceleration before gravity split.
      // Prefer accelerationIncludingGravity for parity with training datasets.
      const acc = event.accelerationIncludingGravity || event.acceleration;
      if (!acc) {
        return;
      }
      const rr = event.rotationRate || null;
      pushSample({
        timestampMs: Date.now(),
        acc: {
          x: Number(acc.x || 0),
          y: Number(acc.y || 0),
          z: Number(acc.z || 0),
        },
        gyro: rr ? {
          alpha: Number(rr.alpha || 0),
          beta: Number(rr.beta || 0),
          gamma: Number(rr.gamma || 0),
        } : null,
      });
    };
    window.addEventListener("devicemotion", motionHandler, { passive: true });

    uploadTimer = window.setInterval(() => {
      uploadBatch();
    }, UPLOAD_INTERVAL_MS);

    inferenceTimer = window.setInterval(async () => {
      if (!viewerToken || !sessionId) return;
      try {
        const r = await fetch(`/v1/sessions/${encodeURIComponent(sessionId)}/inference`, {
          headers: { "X-Viewer-Token": viewerToken },
        });
        if (!r.ok) return;
        const body = await r.json();
        inferenceEl.textContent = JSON.stringify(body.inference || {}, null, 2);
      } catch (_) {
        // Keep polling loop alive across intermittent network failures.
      }
    }, INFERENCE_INTERVAL_MS);

    acquireWakeLock();
  }

  function stopCapture() {
    if (motionHandler) {
      window.removeEventListener("devicemotion", motionHandler);
      motionHandler = null;
    }
    if (uploadTimer) {
      window.clearInterval(uploadTimer);
      uploadTimer = null;
    }
    if (inferenceTimer) {
      window.clearInterval(inferenceTimer);
      inferenceTimer = null;
    }
    releaseWakeLock();
  }

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      uploadBatch({ forceAll: true });
      if (startBtn.disabled) {
        // Re-acquire wake lock after tab becomes visible again
        // (browsers release it when tab goes background).
        acquireWakeLock();
      }
    }
  });

  window.addEventListener("pagehide", () => {
    uploadBatch({ forceAll: true, keepalive: true });
    releaseWakeLock();
  });

  startBtn.addEventListener("click", async () => {
    try {
      setStatus("Joining session...");
      await joinSession();
      setStatus("Requesting permission...");
      await requestMotionPermission();
      setStatus(`Streaming (${selectedPlacementLabel()})...`);
      startCapture();
      startBtn.disabled = true;
      stopBtn.disabled = false;
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  });

  stopBtn.addEventListener("click", async () => {
    stopCapture();
    await uploadBatch({ forceAll: true, keepalive: true });
    setStatus("Stopped");
    stopBtn.disabled = true;
    startBtn.disabled = false;
  });
})();
</script>
</body>
</html>
"""


def build_mobile_capture_router():
    """Expose a lightweight mobile web page for DeviceMotion capture."""

    if APIRouter is None or HTMLResponse is None:
        return None

    router = APIRouter(tags=["mobile-capture"])

    @router.get("/mobile-capture", response_class=HTMLResponse)
    def mobile_capture_page() -> str:
        return _MOBILE_PAGE_HTML

    return router
