"""
Wearable Signal Simulator — Live Streaming Plot
Simulates ACC (3-axis), Respiration, and Skin Temperature for a waist-worn
device across activity states: rest, walking, running, recovery, stress, sleep.

No data is saved to disk. All signals are generated sample-by-sample and
streamed to a live matplotlib figure with a rolling window.
"""

import csv
import os
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════
FS = 50              # Hz — ACC / RSP sampling rate
FS_TEMP = 4          # Hz — skin temperature sampling rate
DT = 1.0 / FS
G = 9.81             # m/s²

# Sensor range (±m/s²) — modelled as ±16g body-worn sensor
ACC_RANGE = 16.0 * G   # ±156.96 m/s²

# How many seconds of history to show in the plot
PLOT_WINDOW = 15.0   # seconds

# How many simulation steps per animation frame (controls speed)
# Higher = faster-than-realtime playback
STEPS_PER_FRAME = 25  # 25 steps @ 50 Hz ≈ 0.5 s of sim time per frame

# Animation interval in ms (matplotlib timer)
FRAME_INTERVAL_MS = 30

# ══════════════════════════════════════════════════════════════════════════════
# Activity sequence — (name, duration_seconds)
# ══════════════════════════════════════════════════════════════════════════════
ACTIVITY_SEQUENCE = [
    ("rest",     30),
    ("walking", 45),
    ("running",  30),
    ("recovery", 40),
    ("rest",     15),
    ("stress",   35),
    ("rest",     15),
    ("sleep",    60),
]

# ══════════════════════════════════════════════════════════════════════════════
# Target parameters per activity
# (f_step, A_z, A_x, A_y, f_resp, B_rsp, IE_ratio, T_core, T_skin, HR)
# ══════════════════════════════════════════════════════════════════════════════
TARGETS = {
    "rest":     (0.0,  0.02, 0.01, 0.01, 0.25, 1.0, 2.0, 37.0, 33.0,  68),
    "walking":  (1.9,  3.7,  1.7,  2.5,  0.33, 1.8, 1.5, 37.3, 31.5, 105),
    "running":  (2.8, 10.0,  5.0,  4.0,  0.60, 3.5, 1.0, 37.8, 29.5, 160),
    "recovery": (0.3,  0.3,  0.15, 0.08, 0.27, 1.5, 1.8, 37.0, 34.0,  68),
    "stress":   (0.0,  0.02, 0.01, 0.01, 0.28, 1.2, 1.8, 37.1, 32.5,  80),
    "sleep":    (0.0,  0.02, 0.01, 0.01, 0.20, 0.8, 2.5, 36.5, 34.5,  55),
}

# ══════════════════════════════════════════════════════════════════════════════
# Time constants (seconds)
# ══════════════════════════════════════════════════════════════════════════════
TAU_ACC = 3.0
TAU_RSP_FREQ = 15.0
TAU_RSP_AMP = 20.0
TAU_CORE = {"rest": 600, "walking": 300, "running": 300,
            "recovery": 600, "stress": 120, "sleep": 1800}
TAU_SKIN = {"rest": 300, "walking": 60, "running": 45,
            "recovery": 120, "stress": 90, "sleep": 900}

# Noise levels per activity
SIGMA_ACC = {"rest": 0.06, "walking": 0.15, "running": 0.3,
             "recovery": 0.05, "stress": 0.06, "sleep": 0.05}
SIGMA_RSP = {"rest": 0.02, "walking": 0.05, "running": 0.08,
             "recovery": 0.04, "stress": 0.03, "sleep": 0.015}
RATE_VAR  = {"rest": 0.015, "walking": 0.025, "running": 0.03,
             "recovery": 0.025, "stress": 0.04, "sleep": 0.01}

# ══════════════════════════════════════════════════════════════════════════════
# Artifact parameters  (calibrated from raw EmoWear sensor data)
# ══════════════════════════════════════════════════════════════════════════════
# Bluetooth / sensor connection gaps — all signals go NaN simultaneously
#   Raw EmoWear back: 22% sample loss, 85k timing gaps in 108 min
#   75 gaps 100-500ms, 1 gap 850ms; most loss is micro-jitter (2-50ms)
#   We model the *noticeable* gaps (>50ms) that affect downstream analysis
GAP_PROB_PER_SAMPLE = 1.0 / (8 * 60 * FS)  # ~1 event per 8 min
GAP_DUR_RANGE = (0.05, 2.0)       # seconds — mostly 50-500ms, occasional >1s

# ACC stuck-value artefact (ADC quantization — consecutive identical readings)
#   Raw EmoWear: 7.8-14% per axis, mostly 1-2 sample runs
ACC_STUCK_PROB = 0.06    # per-sample probability of value getting "stuck"
ACC_STUCK_DUR = (1, 4)   # samples — mostly 1-2, occasionally longer

# Sensor glitch — occasional brief RSP / SKT dropout
#   Real data: BH3 breathing 0% NaN, E4 TEMP 0% NaN across 100 recordings
#   However, real-world use has occasional contact loss, motion artefacts
#   Model as rare, brief events (~1-3% total dropout)
RSP_GLITCH_PROB = 0.00005   # per-sample chance to START a glitch
RSP_GLITCH_DUR = (0.5, 5.0) # seconds — brief contact-loss events
SKT_GLITCH_PROB = 0.00003   # temperature sensor slightly more robust
SKT_GLITCH_DUR = (0.5, 3.0) # seconds

# RSP baseline wander — slow sinusoidal drift
RSP_WANDER_AMP = 0.05   # fraction of current B_rsp amplitude
RSP_WANDER_PERIOD = (40.0, 120.0)  # seconds — very slow

# SKT warmup artefact — sensor contact settling
#   Models the brief thermal settling when sensor first contacts skin.
#   Real E4 data shows slow warmup over 5 min, but that's partly the
#   sensor body mass heating up. We model a 15s exponential settling
#   so it doesn't mask the physiological skin temp changes.
SKT_WARMUP_DUR = 15.0    # seconds — exponential time constant (tau)
SKT_WARMUP_OFFSET = -4.0 # °C below true value at t=0

# SKT spike artefacts (Raw E4: 12 jumps>0.3°C in 107 min = ~6.7/hr)
SKT_SPIKE_PROB = 0.0005  # per TEMP sample (raw E4: ~6.7 jumps>0.3°C/hr → ~0.05% at 4Hz)
SKT_SPIKE_AMP = (0.3, 2.0)  # °C magnitude — raw shows jumps from 0.3 to >1°C

# Activity colours for the status bar
ACTIVITY_COLORS = {
    "rest": "#4CAF50", "walking": "#2196F3", "running": "#F44336",
    "recovery": "#FF9800", "stress": "#9C27B0", "sleep": "#3F51B5",
}


# ══════════════════════════════════════════════════════════════════════════════
# Simulator state (mutable, updated each step)
# ══════════════════════════════════════════════════════════════════════════════
class SimState:
    def __init__(self):
        self.f_step = 0.0
        self.A_z, self.A_x, self.A_y = 0.02, 0.01, 0.01
        self.f_resp = 0.25
        self.B_rsp = 1.0
        self.ie_ratio = 2.0
        self.T_core = 37.0
        self.T_skin = 33.0
        self.HR = 68.0
        self.temp_drift = 0.0

        # Accumulated phases
        self.phase_z = np.random.uniform(0, 2 * np.pi)
        self.phase_x = np.random.uniform(0, 2 * np.pi)
        self.phase_y = np.random.uniform(0, 2 * np.pi)
        self.phase_resp = np.random.uniform(0, 2 * np.pi)

        self.exercise_duration = 0.0
        self.sample_idx = 0  # global sample counter

        # Per-session ACC bias
        self.bias_x = np.random.uniform(-0.15, 0.15)
        self.bias_y = np.random.uniform(-0.15, 0.15)
        self.bias_z = np.random.uniform(-0.15, 0.15)

        # Fidget / shift cooldowns
        self.fidget_cooldown = 0
        self.shift_cooldown = 0

        # Pending fidget/shift overlays (list of remaining samples to add)
        self.fidget_buf_x = deque()
        self.fidget_buf_y = deque()
        self.fidget_buf_z = deque()
        self.shift_buf_x = deque()
        self.shift_buf_y = deque()
        self.shift_buf_z = deque()

        # ── Artifact state ──
        self.gap_remaining = 0          # samples left in current BT gap
        self.rsp_glitch_remaining = 0   # samples left in current RSP glitch
        self.skt_glitch_remaining = 0   # samples left in current SKT glitch
        # RSP baseline wander (very slow oscillation)
        self.rsp_wander_phase = np.random.uniform(0, 2 * np.pi)
        self.rsp_wander_freq = 1.0 / np.random.uniform(*RSP_WANDER_PERIOD)
        # ACC stuck-value state
        self.stuck_remaining = [0, 0, 0]   # samples left stuck per axis
        self.stuck_value = [0.0, 0.0, 0.0] # held value per axis


# ══════════════════════════════════════════════════════════════════════════════
# Helper: asymmetric breathing waveform
# ══════════════════════════════════════════════════════════════════════════════
def breath_waveform(phase, ie_ratio):
    alpha = 1.0 / (1.0 + ie_ratio)
    p = phase % (2 * np.pi)
    insp_end = 2 * np.pi * alpha
    if p < insp_end:
        return np.sin(np.pi * p / insp_end)
    else:
        return np.sin(np.pi + np.pi * (p - insp_end) / (2 * np.pi - insp_end))


# ══════════════════════════════════════════════════════════════════════════════
# Build flat schedule: maps sample index → activity name
# ══════════════════════════════════════════════════════════════════════════════
def build_schedule(sequence, fs):
    schedule = []
    for act, dur in sequence:
        schedule.extend([act] * int(dur * fs))
    return schedule

SCHEDULE = build_schedule(ACTIVITY_SEQUENCE, FS)
TOTAL_SAMPLES = len(SCHEDULE)
TOTAL_TIME = TOTAL_SAMPLES / FS


# ══════════════════════════════════════════════════════════════════════════════
# Step the simulation forward by one sample
# ══════════════════════════════════════════════════════════════════════════════
def step(state: SimState):
    """Advance simulation by one sample. Returns (t, acc_x, acc_y, acc_z, rsp, skt_or_None, activity)."""
    idx = state.sample_idx
    if idx >= TOTAL_SAMPLES:
        return None  # simulation finished

    act = SCHEDULE[idx]
    tgt = TARGETS[act]
    f_step_tgt, Az_tgt, Ax_tgt, Ay_tgt = tgt[0], tgt[1], tgt[2], tgt[3]
    f_resp_tgt, B_tgt, ie_tgt = tgt[4], tgt[5], tgt[6]
    Tc_tgt, Ts_tgt, HR_tgt = tgt[7], tgt[8], tgt[9]

    s = state  # alias

    # ── Exercise duration tracking ──
    if act in ("walking", "running"):
        s.exercise_duration += DT
    else:
        s.exercise_duration = max(0, s.exercise_duration - DT * 0.5)

    # ── Prolonged-exercise skin temp adjustment ──
    Ts_adj = Ts_tgt
    if act in ("walking", "running") and s.exercise_duration > 900:
        Ts_adj = Ts_tgt + min(1.5, 0.1 * (s.exercise_duration - 900) / 60.0)

    # ── Smooth state variables ──
    s.f_step += (DT / TAU_ACC) * (f_step_tgt - s.f_step)
    s.A_z += (DT / TAU_ACC) * (Az_tgt - s.A_z)
    s.A_x += (DT / TAU_ACC) * (Ax_tgt - s.A_x)
    s.A_y += (DT / TAU_ACC) * (Ay_tgt - s.A_y)

    f_resp_coupled = f_resp_tgt + 0.05 * s.f_step
    s.f_resp += (DT / TAU_RSP_FREQ) * (f_resp_coupled - s.f_resp)
    s.B_rsp += (DT / TAU_RSP_AMP) * (B_tgt - s.B_rsp)
    s.ie_ratio += (DT / TAU_RSP_FREQ) * (ie_tgt - s.ie_ratio)

    tau_c = TAU_CORE.get(act, 600)
    tau_s = TAU_SKIN.get(act, 300)
    s.T_core += (DT / tau_c) * (Tc_tgt - s.T_core)
    s.T_skin += (DT / tau_s) * (Ts_adj - s.T_skin)

    tau_hr = 30 if act in ("walking", "running") else 90
    s.HR += (DT / tau_hr) * (HR_tgt - s.HR)

    # ── Accumulate phases ──
    s.phase_z += 2 * np.pi * s.f_step * DT
    s.phase_x += 2 * np.pi * s.f_step * DT
    s.phase_y += 2 * np.pi * s.f_step * DT
    s.phase_resp += 2 * np.pi * s.f_resp * DT

    # ── Step-to-step variability ──
    amp_jitter = 1.0
    if act == "walking":
        amp_jitter = 1.0 + np.random.normal(0, 0.05)
    elif act == "running":
        amp_jitter = 1.0 + np.random.normal(0, 0.08)

    sigma_acc = SIGMA_ACC[act]

    # ── ACC ──
    ax = (s.A_x * amp_jitter * np.sin(s.phase_x + np.pi / 4)
          + s.bias_x + np.random.normal(0, sigma_acc))
    ay = (s.A_y * amp_jitter * np.sin(s.phase_y + np.pi / 2)
          + s.bias_y + np.random.normal(0, sigma_acc))
    az = (s.A_z * amp_jitter * np.sin(s.phase_z)
          + 0.3 * s.A_z * amp_jitter * np.sin(2 * s.phase_z)
          + G + s.bias_z + np.random.normal(0, sigma_acc))

    # ── Fidget / shift overlays ──
    if s.fidget_buf_x:
        ax += s.fidget_buf_x.popleft()
        ay += s.fidget_buf_y.popleft()
        az += s.fidget_buf_z.popleft()
    if s.shift_buf_x:
        ax += s.shift_buf_x.popleft()
        ay += s.shift_buf_y.popleft()
        az += s.shift_buf_z.popleft()

    # ── Trigger new stress fidgets ──
    if s.fidget_cooldown > 0:
        s.fidget_cooldown -= 1
    if (act == "stress" and not s.fidget_buf_x
            and s.fidget_cooldown <= 0
            and np.random.random() < 0.0006):
        flen = int(np.random.uniform(0.5, 1.5) * FS)
        famp = np.random.uniform(0.1, 0.4)
        for j in range(flen):
            sig = famp * np.sin(2 * np.pi * np.random.uniform(3, 6) * j * DT)
            s.fidget_buf_x.append(sig * 0.5)
            s.fidget_buf_y.append(sig * 0.3)
            s.fidget_buf_z.append(sig)
        s.fidget_cooldown = int(np.random.uniform(15, 60) * FS)

    # ── Trigger new sleep position shifts ──
    if s.shift_cooldown > 0:
        s.shift_cooldown -= 1
    if (act == "sleep" and not s.shift_buf_x
            and s.shift_cooldown <= 0
            and np.random.random() < 0.00006):
        slen = int(np.random.uniform(2, 5) * FS)
        speak = np.random.uniform(0.5, 2.0)
        for j in range(slen):
            t_rel = j * DT
            t_dur = slen * DT
            envelope = speak * np.exp(-3 * t_rel / t_dur)
            s.shift_buf_x.append(envelope * np.random.normal(0, 0.3))
            s.shift_buf_y.append(envelope * np.random.normal(0, 0.3))
            s.shift_buf_z.append(envelope * np.random.normal(0, 0.3))
        s.shift_cooldown = int(np.random.uniform(30, 120) * FS)

    # ── ACC clipping ──
    ax = np.clip(ax, -ACC_RANGE, ACC_RANGE)
    ay = np.clip(ay, -ACC_RANGE, ACC_RANGE)
    az = np.clip(az, -ACC_RANGE, ACC_RANGE)

    # ── ACC stuck-value artefact (ADC quantization) ──
    acc_vals = [ax, ay, az]
    for i in range(3):
        if s.stuck_remaining[i] > 0:
            acc_vals[i] = s.stuck_value[i]
            s.stuck_remaining[i] -= 1
        elif np.random.random() < ACC_STUCK_PROB:
            s.stuck_value[i] = acc_vals[i]
            s.stuck_remaining[i] = np.random.randint(*ACC_STUCK_DUR)
    ax, ay, az = acc_vals

    # ── RSP ──
    sigma_rsp = SIGMA_RSP[act]
    rate_var = RATE_VAR[act]
    s.phase_resp += 2 * np.pi * s.f_resp * DT * np.random.normal(0, rate_var) * 0.1
    rsp_val = s.B_rsp * breath_waveform(s.phase_resp, s.ie_ratio) + np.random.normal(0, sigma_rsp)

    # ── RSP baseline wander ──
    s.rsp_wander_phase += 2 * np.pi * s.rsp_wander_freq * DT
    rsp_val += RSP_WANDER_AMP * s.B_rsp * np.sin(s.rsp_wander_phase)

    # ── SKT (lower rate) ──
    skt_val = None
    temp_interval = int(FS / FS_TEMP)
    if idx % temp_interval == 0:
        s.temp_drift += np.random.normal(0, 0.001 * np.sqrt(1.0 / FS_TEMP))
        skt_val = s.T_skin + s.temp_drift + np.random.normal(0, 0.02)

        # ── SKT warmup artefact (exponential approach) ──
        t_now = idx * DT
        if t_now < SKT_WARMUP_DUR * 4:  # tail extends to ~4 tau
            skt_val += SKT_WARMUP_OFFSET * np.exp(-3.0 * t_now / SKT_WARMUP_DUR)

        # ── SKT spike artefact ──
        if np.random.random() < SKT_SPIKE_PROB:
            spike = np.random.uniform(*SKT_SPIKE_AMP)
            skt_val += spike * np.random.choice([-1, 1])

        skt_val = round(skt_val / 0.02) * 0.02

    # ══════════════════════════════════════════════════════════════════
    # Bluetooth connection gap — all signals go NaN
    # ══════════════════════════════════════════════════════════════════
    if s.gap_remaining > 0:
        s.gap_remaining -= 1
        t = idx * DT
        s.sample_idx += 1
        return t, np.nan, np.nan, np.nan, np.nan, np.nan if skt_val is not None else None, act
    elif np.random.random() < GAP_PROB_PER_SAMPLE:
        dur = np.random.uniform(*GAP_DUR_RANGE)
        s.gap_remaining = int(dur * FS)

    # ══════════════════════════════════════════════════════════════════
    # Sensor glitch — occasional brief RSP / SKT dropout
    #   RSP and SKT sensors rarely drop out (0% NaN in raw data)
    #   but real-world use has occasional brief contact-loss events
    # ══════════════════════════════════════════════════════════════════
    if s.rsp_glitch_remaining > 0:
        s.rsp_glitch_remaining -= 1
        rsp_val = np.nan
    elif np.random.random() < RSP_GLITCH_PROB:
        dur = np.random.uniform(*RSP_GLITCH_DUR)
        s.rsp_glitch_remaining = int(dur * FS)

    if skt_val is not None:
        if s.skt_glitch_remaining > 0:
            s.skt_glitch_remaining -= 1
            skt_val = np.nan
        elif np.random.random() < SKT_GLITCH_PROB:
            dur = np.random.uniform(*SKT_GLITCH_DUR)
            s.skt_glitch_remaining = int(dur * FS)

    t = idx * DT
    s.sample_idx += 1
    return t, ax, ay, az, rsp_val, skt_val, act


# ══════════════════════════════════════════════════════════════════════════════
# Live plot setup
# ══════════════════════════════════════════════════════════════════════════════
def main():
    headless = "--headless" in sys.argv

    state = SimState()
    win = int(PLOT_WINDOW * FS)       # number of high-rate samples in window
    win_temp = int(PLOT_WINDOW * FS_TEMP)

    # Rolling buffers (fixed-size deques — no memory growth)
    t_buf = deque(maxlen=win)
    ax_buf = deque(maxlen=win)
    ay_buf = deque(maxlen=win)
    az_buf = deque(maxlen=win)
    rsp_buf = deque(maxlen=win)

    t_temp_buf = deque(maxlen=win_temp)
    skt_buf = deque(maxlen=win_temp)

    # Full signal log for CSV export
    log_rows = []

    if headless:
        # Generate all samples without plotting
        print(f"Generating {TOTAL_SAMPLES} samples ({TOTAL_TIME:.0f}s) in headless mode...")
        for _ in range(TOTAL_SAMPLES):
            result = step(state)
            if result is None:
                break
            t, ax_v, ay_v, az_v, rsp_v, skt_v, act = result
            log_rows.append((t, ax_v, ay_v, az_v, rsp_v, skt_v, act))
        _save_log(log_rows)
        print("Done.")
        return

    # ── Figure layout ──
    fig, axes = plt.subplots(5, 1, figsize=(14, 9), sharex=False)
    fig.patch.set_facecolor("#1e1e2e")
    fig.subplots_adjust(hspace=0.35, left=0.08, right=0.95, top=0.92, bottom=0.06)

    titles = ["ACC X  (anteroposterior)", "ACC Y  (mediolateral)",
              "ACC Z  (vertical)", "Respiration", "Skin Temperature"]
    ylabels = ["m/s²", "m/s²", "m/s²", "a.u.", "°C"]
    colors = ["#42a5f5", "#66bb6a", "#ef5350", "#ffa726", "#ab47bc"]

    lines = []
    for i, ax in enumerate(axes):
        ax.set_facecolor("#2a2a3e")
        ax.tick_params(colors="#ccc", labelsize=8)
        ax.set_ylabel(ylabels[i], color="#ccc", fontsize=9)
        ax.set_title(titles[i], color=colors[i], fontsize=10, fontweight="bold", loc="left")
        for spine in ax.spines.values():
            spine.set_color("#444")
        ln, = ax.plot([], [], color=colors[i], linewidth=0.7)
        lines.append(ln)

    axes[-1].set_xlabel("Time (s)", color="#ccc", fontsize=9)

    # Activity label + progress bar
    activity_text = fig.text(0.5, 0.965, "", ha="center", va="center",
                             fontsize=13, fontweight="bold", color="#eee",
                             fontfamily="monospace")
    progress_text = fig.text(0.5, 0.945, "", ha="center", va="center",
                             fontsize=8, color="#aaa")

    # ── Fixed Y-axis limits (based on worst-case activity ranges) ──
    max_ax = max(t[2] for t in TARGETS.values()) + 0.5   # A_x max + margin
    max_ay = max(t[3] for t in TARGETS.values()) + 0.5   # A_y max + margin
    max_az = max(t[1] for t in TARGETS.values()) + 1.0   # A_z max + margin
    max_rsp = max(t[5] for t in TARGETS.values()) + 0.5  # B_rsp max + margin
    skt_lo = min(t[8] for t in TARGETS.values()) - 2.0   # warmup starts lower
    skt_hi = max(t[8] for t in TARGETS.values()) + 1.0

    FIXED_YLIMS = [
        (-max_ax, max_ax),                   # ACC X
        (-max_ay, max_ay),                   # ACC Y
        (G - max_az - 1.0, G + max_az + 1.0),  # ACC Z (centered on gravity)
        (-max_rsp, max_rsp),                 # Respiration
        (skt_lo, skt_hi),                    # Skin Temperature
    ]

    def init():
        for ln in lines:
            ln.set_data([], [])
        for i, ax in enumerate(axes):
            ax.set_ylim(FIXED_YLIMS[i])
        return lines

    def update(frame):
        # Advance simulation by STEPS_PER_FRAME samples
        for _ in range(STEPS_PER_FRAME):
            result = step(state)
            if result is None:
                # Simulation done — save CSV and stop animation
                _save_log(log_rows)
                log_rows.clear()
                ani.event_source.stop()
                activity_text.set_text("SIMULATION COMPLETE")
                progress_text.set_text(f"Total: {TOTAL_TIME:.0f}s")
                fig.canvas.draw_idle()
                return lines

            t, ax_v, ay_v, az_v, rsp_v, skt_v, act = result
            t_buf.append(t)
            ax_buf.append(ax_v)
            ay_buf.append(ay_v)
            az_buf.append(az_v)
            rsp_buf.append(rsp_v)
            log_rows.append((t, ax_v, ay_v, az_v, rsp_v, skt_v, act))

            if skt_v is not None:
                t_temp_buf.append(t)
                skt_buf.append(skt_v)

        # ── Update line data ──
        t_arr = np.array(t_buf)
        lines[0].set_data(t_arr, np.array(ax_buf))
        lines[1].set_data(t_arr, np.array(ay_buf))
        lines[2].set_data(t_arr, np.array(az_buf))
        lines[3].set_data(t_arr, np.array(rsp_buf))

        if t_temp_buf:
            lines[4].set_data(np.array(t_temp_buf), np.array(skt_buf))

        # ── Adjust x-axis limits (rolling window) ──
        if len(t_buf) > 1:
            t_lo, t_hi = t_buf[0], t_buf[-1]
            for i in range(4):
                axes[i].set_xlim(t_lo, t_hi)
            if t_temp_buf:
                axes[4].set_xlim(t_temp_buf[0], t_temp_buf[-1])

        # ── Activity label ──
        cur_t = t_buf[-1] if t_buf else 0
        act = SCHEDULE[min(state.sample_idx - 1, TOTAL_SAMPLES - 1)]
        color = ACTIVITY_COLORS.get(act, "#fff")
        activity_text.set_text(f"▶  {act.upper()}")
        activity_text.set_color(color)
        pct = 100 * cur_t / TOTAL_TIME
        elapsed_bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        progress_text.set_text(f"{elapsed_bar}  {cur_t:.0f}s / {TOTAL_TIME:.0f}s  ({pct:.0f}%)")

        return lines

    ani = FuncAnimation(fig, update, init_func=init,
                        interval=FRAME_INTERVAL_MS, blit=False, cache_frame_data=False)

    plt.show()

    # Also save if user closes the window before simulation finishes
    if log_rows:
        _save_log(log_rows)
        log_rows.clear()


def _save_log(rows):
    """Write collected signal data to output/sim_output.csv."""
    os.makedirs("output", exist_ok=True)
    path = os.path.join("output", "sim_output.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "acc_x", "acc_y", "acc_z", "rsp", "skt", "activity"])
        for row in rows:
            t, ax, ay, az, rsp, skt, act = row
            w.writerow([f"{t:.4f}", ax, ay, az, rsp, skt if skt is not None else "", act])
    print(f"Saved {len(rows)} samples to {path}")


if __name__ == "__main__":
    main()
