"""Mobile browser capture page for DeviceMotion streaming."""

from __future__ import annotations

try:
    from fastapi import APIRouter
    from fastapi.responses import HTMLResponse
except Exception:  # pragma: no cover
    APIRouter = None  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]


_MOBILE_PAGE_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>MotionLens Mobile Capture</title>
  <style>
    :root {
      --bg: #0d1117;
      --card-bg: #161b22;
      --border: #30363d;
      --text: #c9d1d9;
      --muted: #8b949e;
      --accent: #58a6ff;
      --success: #3fb950;
      --warning: #d29922;
      --danger: #f85149;
    }
    body {
      font-family: -apple-system, Segoe UI, Roboto, sans-serif;
      margin: 0;
      padding: 16px;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 14px;
    }
    .status-bar {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 13px;
      color: var(--muted);
    }
    .status-bar strong { color: var(--text); }
    button {
      padding: 12px 18px;
      border-radius: 10px;
      border: none;
      background: var(--accent);
      color: #fff;
      font-size: 16px;
      font-weight: 600;
      width: 100%;
      margin-bottom: 10px;
      cursor: pointer;
    }
    button:disabled {
      background: var(--border);
      color: var(--muted);
      cursor: not-allowed;
    }
    .muted { color: var(--muted); font-size: 13px; }
    code {
      background: var(--border);
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 12px;
    }

    /* Insight card styles */
    .insight-card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 16px;
      margin-bottom: 14px;
    }
    .activity-headline {
      font-size: 28px;
      font-weight: 800;
      margin-bottom: 4px;
      letter-spacing: -0.5px;
    }
    .confidence-badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 12px;
    }
    .confidence-badge.high { background: rgba(63,185,80,0.15); color: var(--success); }
    .confidence-badge.medium { background: rgba(210,153,34,0.15); color: var(--warning); }
    .confidence-badge.low { background: rgba(248,81,73,0.15); color: var(--danger); }

    .insight-text {
      font-size: 15px;
      color: var(--muted);
      margin-bottom: 14px;
      line-height: 1.4;
    }

    .metric-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 14px;
    }
    .metric {
      background: var(--bg);
      border-radius: 10px;
      padding: 10px 6px;
      text-align: center;
    }
    .metric-value {
      font-size: 22px;
      font-weight: 700;
      color: var(--text);
      margin-bottom: 2px;
    }
    .metric-label {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }

    .quality-bar {
      height: 6px;
      background: var(--bg);
      border-radius: 3px;
      overflow: hidden;
      margin-bottom: 6px;
    }
    .quality-fill {
      height: 100%;
      background: linear-gradient(90deg, var(--danger), var(--warning), var(--success));
      border-radius: 3px;
      transition: width 0.3s ease;
    }
    .quality-label {
      font-size: 12px;
      color: var(--muted);
      text-align: center;
    }

    .alert-row {
      display: flex;
      gap: 8px;
      margin-top: 10px;
    }
    .alert-pill {
      flex: 1;
      padding: 6px 10px;
      border-radius: 8px;
      font-size: 12px;
      font-weight: 600;
      text-align: center;
    }
    .alert-pill.burst { background: rgba(248,81,73,0.12); color: var(--danger); }
    .alert-pill.posture { background: rgba(88,166,255,0.12); color: var(--accent); }

    .orientation-label {
      font-size: 12px;
      color: var(--muted);
      margin-top: 8px;
      text-align: center;
    }
  </style>
</head>
<body>
  <h2 style="margin-top:0; font-size:20px;">MotionLens Mobile</h2>

  <div class="card">
    <div style="font-size:15px; font-weight:600; margin-bottom:6px;">Phone activity recognition</div>
    <div class="muted">
      Stream accelerometer data from this phone for live activity recognition. When you stop streaming,
      a short summary of your activity appears here. For the clearest result, keep the phone in a trouser pocket while moving.
    </div>
    <div style="margin-top:10px; padding:10px 12px; border-radius:12px; background:#fff1d6; color:#7a4b00; font-weight:700;">
      (!) Supported browser: Chrome mobile
    </div>
  </div>

  <div class="card">
    <div class="status-bar">
      <div><strong>Status:</strong> <span id="status">Idle</span></div>
      <div><strong>Points:</strong> <span id="accepted">0</span></div>
    </div>
    <div class="status-bar" style="margin-top:6px;">
      <div><strong>Errors:</strong> <span id="errors">0</span></div>
    </div>
  </div>

  <div class="card">
    <button id="startBtn">Start streaming</button>
    <button id="stopBtn" disabled>Stop streaming</button>
    <p class="muted">Requires HTTPS and a user tap for motion permission on iPhone.</p>
  </div>

  <div id="insightCard" class="insight-card" style="display:none;">
    <div class="activity-headline" id="activityHeadline">-</div>
    <div class="confidence-badge" id="confidenceBadge">-</div>
    <div class="insight-text" id="insightText">-</div>
    <div class="metric-grid" id="metricGrid"></div>
    <div class="quality-bar"><div class="quality-fill" id="qualityFill" style="width:0%"></div></div>
    <div class="quality-label" id="qualityLabel">Signal Quality —</div>
    <div class="orientation-label" id="orientationLabel">📱 —</div>
    <div class="alert-row" id="alertRow"></div>
  </div>

  <div class="card" id="noInferenceCard">
    <div class="muted" style="text-align:center;">Waiting for inference...</div>
  </div>

  <div id="summaryCard" class="card" style="display:none;">
    <div class="activity-headline" id="summaryHeadline">Session Summary</div>
    <div class="insight-text" id="summaryText">-</div>
    <div class="metric-grid" id="summaryMetricGrid"></div>
    <div class="quality-label" id="summaryNotes" style="text-align:left;"></div>
  </div>

<script>
(() => {
  const params = new URLSearchParams(window.location.search);
  const joinToken = params.get("join_token") || "";
  const initialDeviceId = params.get("device_id") || `phone-${Math.random().toString(36).slice(2,10)}`;

  const statusEl = document.getElementById("status");
  const acceptedEl = document.getElementById("accepted");
  const errorsEl = document.getElementById("errors");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");

  const insightCard = document.getElementById("insightCard");
  const noInferenceCard = document.getElementById("noInferenceCard");
  const activityHeadline = document.getElementById("activityHeadline");
  const confidenceBadge = document.getElementById("confidenceBadge");
  const insightText = document.getElementById("insightText");
  const metricGrid = document.getElementById("metricGrid");
  const qualityFill = document.getElementById("qualityFill");
  const qualityLabel = document.getElementById("qualityLabel");
  const orientationLabel = document.getElementById("orientationLabel");
  const alertRow = document.getElementById("alertRow");
  const summaryCard = document.getElementById("summaryCard");
  const summaryHeadline = document.getElementById("summaryHeadline");
  const summaryText = document.getElementById("summaryText");
  const summaryMetricGrid = document.getElementById("summaryMetricGrid");
  const summaryNotes = document.getElementById("summaryNotes");

  const MAX_BUFFER_SAMPLES = 1200;
  const UPLOAD_INTERVAL_MS = 400;
  const MAX_UPLOAD_BATCH = 240;
  const INFERENCE_INTERVAL_MS = 600;

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

  const ACTIVITY_EMOJI = {
    walk: "🚶", run: "🏃", stairs: "🪜",
    "sit/lay": "🪑", stand: "🧍",
    transitions: "🔄", "locomotion-other": "❓"
  };

  const setStatus = (text) => { statusEl.textContent = text; };

  function sleep(ms) {
    return new Promise(resolve => window.setTimeout(resolve, ms));
  }

  function setVisible(element, visible) {
    element.style.display = visible ? "block" : "none";
  }

  function showWaitingState(message = "Waiting for inference...") {
    setVisible(insightCard, false);
    setVisible(summaryCard, false);
    setVisible(noInferenceCard, true);
    noInferenceCard.querySelector(".muted").textContent = message;
  }

  function showInsightState() {
    setVisible(insightCard, true);
    setVisible(summaryCard, false);
    setVisible(noInferenceCard, false);
  }

  function showSummaryState() {
    setVisible(insightCard, false);
    setVisible(summaryCard, true);
    setVisible(noInferenceCard, false);
  }

  function setCaptureButtons({ canStart, canStop }) {
    startBtn.disabled = !canStart;
    stopBtn.disabled = !canStop;
  }

  function setCounters() {
    acceptedEl.textContent = String(acceptedPoints);
    errorsEl.textContent = String(uploadErrors);
  }

  async function acquireWakeLock() {
    if ("wakeLock" in navigator) {
      try {
        if (wakeLock == null) {
          wakeLock = await navigator.wakeLock.request("screen");
          wakeLock.addEventListener("release", () => { wakeLock = null; });
        }
        return;
      } catch (_) {}
    }
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
    noSleepVideo.src = "data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAAA0NtZGF0AAACrgYF//+q3EXpvebZSLeWLNgg2SPu73gyNjmtgGYZjYGYGgY2DmAmJmZmZiZ0ZmZmZkYJzZmZmZmZkZmZmZmaYJzZmZmZmZkZmZmZmYJzZmZmZmZkZmZmZmZ4bWRhdGEAAAQ0ZGF0AAQAAAAAAi0ZGF0AAQAAAAAAjQZGF0AAQAAAAAAkQZGF0AAQAAAAAAlQZGF0AAQAAAAAAmQZGF0AAQAAAAAAnQZGF0AAQAAAAAAoQZGF0AAQAAAAAApQZGF0AAQAAAAAAqQZGF0AAQAAAAAArQZGF0AAQAAAAAAsQZGF0AAQAAAAAAtQZGF0AAQAAAAAAuQZGF0AAQAAAAAAvQZGF0AAQAAAAAAwQZGF0AAQAAAAAAxQZGF0AAQAAAAAAyQZGF0AAQAAAAAAzQZGF0AAQAAAAAA0QZGF0AAQAAAAAA1QZGF0AAQAAAAAA2Q==";
    noSleepVideo.addEventListener("ended", () => {
      if (noSleepVideo) { noSleepVideo.currentTime = 0; noSleepVideo.play().catch(() => {}); }
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
      try { await wakeLock.release(); } catch (_) {}
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
      body: JSON.stringify({ join_token: joinToken, device_id: deviceId }),
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`Join failed (${res.status}): ${body}`);
    }
    const data = await res.json();
    writeToken = data.write_token;
    viewerToken = data.viewer_token;
    sessionId = data.session_id;
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
    if (uploadInFlight || !writeToken || !sessionId || sampleBuffer.length === 0) return;

    uploadInFlight = true;
    const sendCount = forceAll ? sampleBuffer.length : Math.min(sampleBuffer.length, MAX_UPLOAD_BATCH);
    const samples = sampleBuffer.splice(0, sendCount);
    const payload = {
      source: "web-devicemotion",
      messageId: messageId,
      sessionId: sessionId,
      deviceId: deviceId,
      samples: samples,
    };

    try {
      const res = await fetch(`/v1/ingest/${encodeURIComponent(writeToken)}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        keepalive: keepalive,
      });
      if (!res.ok) throw new Error(`ingest ${res.status}`);
      const out = await res.json();
      acceptedPoints += Number(out.accepted_points || 0);
      messageId += 1;
      setCounters();
      setStatus(document.hidden ? "Streaming (background)" : "Streaming...");
    } catch (_) {
      uploadErrors += 1;
      setCounters();
      sampleBuffer = samples.concat(sampleBuffer);
      console.warn("Upload retry scheduled after transient network failure");
      setStatus("Streaming with retries (network issue)");
    } finally {
      uploadInFlight = false;
    }
  }

  async function flushUploadsForFinalize() {
    let attempts = 0;
    while (attempts < 12) {
      attempts += 1;
      if (uploadInFlight) {
        await sleep(120);
        continue;
      }
      if (sampleBuffer.length === 0) {
        return;
      }
      await uploadBatch({ forceAll: true, keepalive: true });
      await sleep(120);
    }
  }

  async function finalizeSession() {
    if (!viewerToken || !sessionId) {
      throw new Error("Session is not ready for finalization.");
    }
    const response = await fetch(`/v1/sessions/${encodeURIComponent(sessionId)}/finalize`, {
      method: "POST",
      headers: { "X-Viewer-Token": viewerToken },
    });
    if (!response.ok) {
      throw new Error(`Finalize failed (${response.status})`);
    }
    const body = await response.json();
    return body.summary || {};
  }

  async function readSessionSummary() {
    if (!viewerToken || !sessionId) {
      return {};
    }
    const response = await fetch(`/v1/sessions/${encodeURIComponent(sessionId)}/summary`, {
      headers: { "X-Viewer-Token": viewerToken },
    });
    if (!response.ok) {
      throw new Error(`Summary read failed (${response.status})`);
    }
    const body = await response.json();
    return body.summary || {};
  }

  function isFinalizedSummary(summary) {
    return Boolean(summary) && summary.status === "finalized";
  }

  async function finalizeAndLoadSummary() {
    const immediate = await finalizeSession();
    if (isFinalizedSummary(immediate)) {
      return immediate;
    }
    for (let attempt = 0; attempt < 10; attempt += 1) {
      await sleep(150);
      const summary = await readSessionSummary();
      if (isFinalizedSummary(summary)) {
        return summary;
      }
    }
    return immediate;
  }

  function renderSessionSummary(summary) {
    const timeByActivity = summary.time_by_activity || {};
    const notes = Array.isArray(summary.notes) ? summary.notes : [];

    showSummaryState();

    summaryHeadline.textContent = `Summary — ${(summary.dominant_activity || "unknown").toUpperCase()}`;
    summaryText.textContent = summary.summary_text || "Session finalized.";

    const metrics = [
      { value: `${Math.round(summary.total_duration_seconds || 0)}s`, label: "Duration" },
      { value: `${summary.estimated_steps || 0}`, label: "Steps" },
      { value: `${Math.round(summary.average_cadence_spm || 0)}`, label: "Avg SPM" },
      { value: `${Math.round(summary.signal_quality_mean || 0)}/100`, label: "Signal" },
      { value: `${Math.round(summary.walking_seconds || 0)}s`, label: "Walking" },
      { value: `${Math.round(summary.stationary_seconds || 0)}s`, label: "Still" },
    ];

    summaryMetricGrid.innerHTML = "";
    metrics.forEach(m => {
      const div = document.createElement("div");
      div.className = "metric";
      div.innerHTML = `<div class="metric-value">${m.value}</div><div class="metric-label">${m.label}</div>`;
      summaryMetricGrid.appendChild(div);
    });

    const activityBits = Object.entries(timeByActivity)
      .map(([activity, seconds]) => `${activity}: ${seconds}s`)
      .join(" | ");
    const noteBits = notes.join(" ");
    summaryNotes.textContent = [activityBits, noteBits].filter(Boolean).join("  ");
  }

  function renderInsightCard(snapshot) {
    const insights = snapshot.insights || {};
    const activity = snapshot.activity || "?";
    const confidence = snapshot.confidence || 0;
    const applicable = new Set(insights.applicable_insights || []);

    showInsightState();

    // Headline
    const emoji = ACTIVITY_EMOJI[activity] || "❓";
    activityHeadline.textContent = `${emoji} ${activity.toUpperCase()}`;

    // Confidence badge
    const tier = insights.prediction_confidence_tier || "medium";
    confidenceBadge.className = `confidence-badge ${tier}`;
    confidenceBadge.textContent = `${(confidence * 100).toFixed(0)}% confident`;

    // Insight text
    insightText.textContent = insights.activity_specific_insight || "";

    // Metric grid — only show applicable metrics
    metricGrid.innerHTML = "";
    const metrics = [];

    if (applicable.has("cadence_spm") && insights.cadence_spm > 0) {
      metrics.push({ value: Math.round(insights.cadence_spm), label: "SPM" });
    }
    if (applicable.has("cadence_consistency_pct")) {
      metrics.push({ value: Math.round(insights.cadence_consistency_pct) + "%", label: "Regular" });
    }
    if (applicable.has("stride_regularity_score")) {
      metrics.push({ value: Math.round(insights.stride_regularity_score), label: "Regularity" });
    }
    if (applicable.has("vertical_oscillation_mm")) {
      metrics.push({ value: Math.round(insights.vertical_oscillation_mm) + "mm", label: "Bounce" });
    }
    if (applicable.has("ground_impact_score")) {
      metrics.push({ value: Math.round(insights.ground_impact_score), label: "Impact" });
    }
    if (applicable.has("smoothness_score")) {
      metrics.push({ value: Math.round(insights.smoothness_score), label: "Smooth" });
    }
    if (applicable.has("posture_stability_score")) {
      metrics.push({ value: Math.round(insights.posture_stability_score), label: "Stillness" });
    }
    if (applicable.has("sway_amplitude_deg")) {
      metrics.push({ value: insights.sway_amplitude_deg.toFixed(1) + "°", label: "Sway" });
    }

    metrics.forEach(m => {
      const div = document.createElement("div");
      div.className = "metric";
      div.innerHTML = `<div class="metric-value">${m.value}</div><div class="metric-label">${m.label}</div>`;
      metricGrid.appendChild(div);
    });

    // Signal quality bar
    const sq = Math.round(insights.signal_quality_score || 0);
    qualityFill.style.width = `${sq}%`;
    qualityLabel.textContent = `Signal Quality ${sq}/100`;

    // Orientation
    const orient = insights.phone_orientation || "unknown";
    orientationLabel.textContent = `📱 ${orient}`;

    // Alerts
    alertRow.innerHTML = "";
    if (insights.burst_detected) {
      const pill = document.createElement("div");
      pill.className = "alert-pill burst";
      pill.textContent = "⚡ Impact";
      alertRow.appendChild(pill);
    }
    if (insights.posture_change_detected) {
      const pill = document.createElement("div");
      pill.className = "alert-pill posture";
      pill.textContent = "🔄 Posture";
      alertRow.appendChild(pill);
    }
  }

  function renderInferenceSnapshot(snapshot) {
    if (snapshot.status === "success") {
      renderInsightCard(snapshot);
      return;
    }
    if (snapshot.status === "queued") {
      showWaitingState(`Collecting data... ${snapshot.sample_count || 0}/${snapshot.samples_needed || 128} samples`);
    }
  }

  function startCapture() {
    motionHandler = (event) => {
      const acc = event.accelerationIncludingGravity || event.acceleration;
      if (!acc) return;
      const rr = event.rotationRate || null;
      pushSample({
        timestampMs: Date.now(),
        acc: { x: Number(acc.x || 0), y: Number(acc.y || 0), z: Number(acc.z || 0) },
        gyro: rr ? { alpha: Number(rr.alpha || 0), beta: Number(rr.beta || 0), gamma: Number(rr.gamma || 0) } : null,
      });
    };
    window.addEventListener("devicemotion", motionHandler, { passive: true });

    uploadTimer = window.setInterval(() => { uploadBatch(); }, UPLOAD_INTERVAL_MS);

    inferenceTimer = window.setInterval(async () => {
      if (!viewerToken || !sessionId) return;
      try {
        const r = await fetch(`/v1/sessions/${encodeURIComponent(sessionId)}/inference`, {
          headers: { "X-Viewer-Token": viewerToken },
        });
        if (!r.ok) return;
        const body = await r.json();
        const snapshot = body.inference || {};
        renderInferenceSnapshot(snapshot);
      } catch (_) {
        console.warn("Inference polling failed; keeping previous mobile UI state");
      }
    }, INFERENCE_INTERVAL_MS);
  }

  function stopCapture() {
    if (motionHandler) {
      window.removeEventListener("devicemotion", motionHandler);
      motionHandler = null;
    }
    if (uploadTimer) { window.clearInterval(uploadTimer); uploadTimer = null; }
    if (inferenceTimer) { window.clearInterval(inferenceTimer); inferenceTimer = null; }
    releaseWakeLock();
  }

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) {
      uploadBatch({ forceAll: true });
      if (startBtn.disabled) acquireWakeLock();
    }
  });

  window.addEventListener("pagehide", () => {
    uploadBatch({ forceAll: true, keepalive: true });
    releaseWakeLock();
  });

  startBtn.addEventListener("click", async () => {
    try {
      // Acquire wake lock immediately while we still have a user gesture.
      // iOS Safari requires the wake-lock request to be in the same event
      // handler as the tap; any preceding await consumes the gesture.
      acquireWakeLock();

      setStatus("Joining session...");
      await joinSession();
      setStatus("Requesting permission...");
      await requestMotionPermission();
      setStatus("Streaming...");
      showWaitingState();
      startCapture();
      setCaptureButtons({ canStart: false, canStop: true });
    } catch (err) {
      setStatus(`Error: ${err.message}`);
    }
  });

  stopBtn.addEventListener("click", async () => {
    try {
      setStatus("Finalizing...");
      stopCapture();
      await flushUploadsForFinalize();
      const summary = await finalizeAndLoadSummary();
      renderSessionSummary(summary);
      setStatus("Finalized");
    } catch (err) {
      setStatus(`Finalize error: ${err.message}`);
    }
    setCaptureButtons({ canStart: false, canStop: false });
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
    def mobile_capture_page() -> HTMLResponse:
      return HTMLResponse(
        content=_MOBILE_PAGE_HTML,
        headers={
          "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
          "Pragma": "no-cache",
          "Expires": "0",
        },
      )

    return router
