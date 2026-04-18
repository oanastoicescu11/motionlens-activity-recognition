"""
Wearable Signal Simulator — Live Streaming Plot
Simulates ACC (3-axis), Respiration, and Skin Temperature for a waist-worn
device across activity states: rest, walking, running, recovery, stress, sleep.

All parameters calibrated against simulation-spec.md v3 (5 real datasets,
n > 170 subjects: UCI HAR waist, RealWorld2016 waist, EmoWear BH3/E4,
Wearable Dataset E4, InfantSmartWear).

Signals are generated sample-by-sample and streamed to a live matplotlib
figure with a rolling window.  Pass ``--headless`` to skip the plot and
write ``output/sim_output.csv`` directly.
"""

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# ══════════════════════════════════════════════════════════════════════════════
# Configuration  (spec §2 Global Configuration)
# ══════════════════════════════════════════════════════════════════════════════
FS = 50              # Hz — ACC / RSP sampling rate
FS_TEMP = 4          # Hz — skin temperature sampling rate
DT = 1.0 / FS
DT_TEMP = 1.0 / FS_TEMP
G = 9.81             # m/s²
T_AMB = 22.0         # °C — ambient temperature

# Sensor specs
ACC_RANGE = 16.0 * G          # ±156.96 m/s²  (±16 g body-worn sensor)
ACC_RESOLUTION = 0.004         # m/s²  (12-bit ADC at ±16g)
TEMP_RESOLUTION = 0.02         # °C

# Plot / animation
PLOT_WINDOW = 15.0             # seconds of history in the rolling window
STEPS_PER_FRAME = 25           # sim steps per animation frame (~0.5 s)
FRAME_INTERVAL_MS = 30         # matplotlib timer interval

# ══════════════════════════════════════════════════════════════════════════════
# Activity sequence — (name, duration_seconds)
# ══════════════════════════════════════════════════════════════════════════════
ACTIVITY_SEQUENCE = [
    ("rest",     30),
    ("walking",  45),
    ("running",  30),
    ("recovery", 40),
    ("rest",     15),
    ("stress",   35),
    ("rest",     15),
    ("sleep",    60),
]

# ══════════════════════════════════════════════════════════════════════════════
# Target parameters per activity  (spec §3.3, §4.2, §5.3, §6.3)
# (f_step, A_z, A_x, A_y, f_resp, B_rsp, IE_ratio, T_core, T_skin, HR)
# ══════════════════════════════════════════════════════════════════════════════
TARGETS = {
    #              f_step  A_z    A_x    A_y   f_resp  B_rsp  IE     Tc     Ts    HR
    "rest":       (0.0,    0.0,   0.0,   0.0,  0.22,   1.0,  1.5,  37.0,  33.0,  72),
    "walking":    (1.8,    3.5,   4.5,   3.0,  0.35,   1.8,  1.3,  37.3,  31.5, 105),
    "running":    (2.7,    4.0,  10.0,   5.0,  0.70,   3.5,  1.0,  37.8,  30.5, 160),
    "recovery":   (0.3,    0.3,   0.15,  0.08, 0.28,   1.5,  1.3,  37.0,  35.0,  72),
    "stress":     (0.0,    0.0,   0.0,   0.0,  0.28,   1.2,  1.5,  37.1,  32.5,  80),
    "sleep":      (0.0,    0.0,   0.0,   0.0,  0.18,   0.8,  2.0,  36.5,  34.5,  55),
}

# 3rd-harmonic coefficient h3 per gait activity  (spec §3.2)
# Power ratio H3/H1 = h3²; walk target 0.15-0.31, run target ~1.20
H3_COEFF = {"walking": 0.48, "running": 1.10}

# ══════════════════════════════════════════════════════════════════════════════
# Time constants  (spec §3.5, §4.4, §5.3)
# ══════════════════════════════════════════════════════════════════════════════
TAU_ACC = 3.0
TAU_RSP_FREQ = {"rest": 15, "walking": 15, "running": 15,
                "recovery": 4, "stress": 15, "sleep": 20}
TAU_RSP_AMP = 20.0
TAU_CORE = {"rest": 600, "walking": 300, "running": 300,
            "recovery": 600, "stress": 120, "sleep": 1800}
TAU_SKIN = {"rest": 150, "walking": 60, "running": 45,
            "recovery": 30, "stress": 45, "sleep": 200}

# ══════════════════════════════════════════════════════════════════════════════
# Noise / variability per activity  (spec §3.3, §4.2)
# ══════════════════════════════════════════════════════════════════════════════
SIGMA_ACC_SENSOR = 0.02   # m/s² — pure electronic noise, same for all activities

# Micro-movement sway σ during non-gait states  (spec §3.6)
SIGMA_MICRO = {"rest": 0.20, "walking": 0.0, "running": 0.0,
               "recovery": 0.06, "stress": 0.08, "sleep": 0.04}

# Micro-adjustment burst probability per sample at 50 Hz  (spec §3.6)
MICRO_ADJ_PROB = {"rest": 0.002, "walking": 0.0, "running": 0.0,
                  "recovery": 0.0013, "stress": 0.005, "sleep": 0.00033}
MICRO_ADJ_AMP  = {"rest": (0.4, 1.0), "recovery": (0.2, 0.5),
                  "stress": (0.3, 0.8), "sleep": (0.2, 0.5)}

# RSP noise  (spec §4.2)
SIGMA_RSP = {"rest": 0.02, "walking": 0.05, "running": 0.08,
             "recovery": 0.04, "stress": 0.03, "sleep": 0.015}

# Breath-to-breath rate variability  (spec §4.2 — v3 increased values)
RATE_VAR = {"rest": 0.15, "walking": 0.12, "running": 0.10,
            "recovery": 0.12, "stress": 0.18, "sleep": 0.08}

# RSP motion artifact coupling  (spec §4.6)
K_MA = {"walking": 0.05, "running": 0.10}

# LRC coupling strength  (spec §4.5)
K_LRC = 0.02   # Hz

# Movement → breathing coupling  (spec §6.1)
K_MV = 0.02

# ══════════════════════════════════════════════════════════════════════════════
# Inter-subject variability  (spec §2.1)
# Drawn once per SimState instance; scales targets throughout.
# ══════════════════════════════════════════════════════════════════════════════
SUBJ_F_STEP_STD = 0.08   # multiplier std  (CV ≈ 8 %)
SUBJ_AMP_STD    = 0.20   # multiplier std  (CV ≈ 20 %)
SUBJ_SKIN_MEAN  = 33.0   # °C
SUBJ_SKIN_STD   = 1.5    # °C
SUBJ_HR_STD     = 12.0   # bpm (72 ± 12)
SUBJ_F_RESP_STD = 0.015  # Hz  (0.22 ± 0.015)

# ══════════════════════════════════════════════════════════════════════════════
# Sensor placement model  (spec §2.2)
# ══════════════════════════════════════════════════════════════════════════════
THETA_MAX_DEG = 15.0  # ±degrees of random rotation per session

# ══════════════════════════════════════════════════════════════════════════════
# ACC bias  (spec §7.2, §3.7)
# ══════════════════════════════════════════════════════════════════════════════
BIAS_OFFSET_RANGE = 0.15      # ±m/s² per axis (uniform draw)
SIGMA_BIAS_DRIFT  = 0.001     # m/s²/√s  (bounded random walk)
BIAS_DRIFT_CLAMP  = 0.3       # ±m/s²

# ══════════════════════════════════════════════════════════════════════════════
# Artifact parameters  (calibrated from raw EmoWear / Wearable Dataset)
# ══════════════════════════════════════════════════════════════════════════════

# Bluetooth connection gaps — all signals go NaN simultaneously  (spec §7.5)
GAP_PROB_PER_SAMPLE = 1.0 / (8 * 60 * FS)  # ~1 event per 8 min
GAP_DUR_RANGE = (0.05, 2.0)                 # seconds

# ACC stuck-value artefact  (ADC quantization)
ACC_STUCK_PROB = 0.06
ACC_STUCK_DUR = (1, 4)

# Sensor glitches — rare brief RSP / SKT dropout
RSP_GLITCH_PROB = 0.00005
RSP_GLITCH_DUR = (0.5, 5.0)
SKT_GLITCH_PROB = 0.00003
SKT_GLITCH_DUR = (0.5, 3.0)

# RSP baseline wander — slow sinusoidal drift  (spec §7.3)
RSP_WANDER_AMP = 0.025         # fraction of B_rsp (halved to match EmoWear ~12%)
RSP_WANDER_PERIOD = (40.0, 120.0)

# SKT warm-up  (spec §5.6 — τ=180 s, active for 600 s, from T_amb)
SKT_WARMUP_TAU = 180.0   # seconds
SKT_WARMUP_END = 600.0   # seconds — disable after this
SKT_WARMUP_PRE = 600.0   # seconds — pre-warm offset (sensor already worn)
SKT_WARMUP_JUMP_PROB = 0.002  # per temp sample during warm-up
SKT_WARMUP_JUMP_AMP = (0.3, 1.5)  # °C

# SKT spike artefacts  (E4: ~6.7 jumps>0.3 °C/hr)
SKT_SPIKE_PROB = 0.0005
SKT_SPIKE_AMP = (0.3, 2.0)

# Activity colours for plot status bar
ACTIVITY_COLORS = {
    "rest": "#4CAF50", "walking": "#2196F3", "running": "#F44336",
    "recovery": "#FF9800", "stress": "#9C27B0", "sleep": "#3F51B5",
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper: rotation matrix for sensor placement  (spec §2.2)
# ══════════════════════════════════════════════════════════════════════════════
def _rotation_matrix(tx, ty, tz):
    """Build 3×3 rotation from Euler angles (radians)."""
    cx, sx = np.cos(tx), np.sin(tx)
    cy, sy = np.cos(ty), np.sin(ty)
    cz, sz = np.cos(tz), np.sin(tz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


# ══════════════════════════════════════════════════════════════════════════════
# Simulator state  (mutable, updated each step)
# ══════════════════════════════════════════════════════════════════════════════
class SimState:
    def __init__(self):
        # ── Inter-subject variability (spec §2.1) ──
        self.subj_f_step_mult = np.random.normal(1.0, SUBJ_F_STEP_STD)
        self.subj_amp_mult    = np.random.normal(1.0, SUBJ_AMP_STD)
        self.subj_skin_base   = np.random.normal(SUBJ_SKIN_MEAN, SUBJ_SKIN_STD)
        self.subj_hr_offset   = np.random.normal(0, SUBJ_HR_STD)
        self.subj_f_resp_offset = np.random.normal(0, SUBJ_F_RESP_STD)

        # ── Sensor placement (spec §2.2) ──
        theta_max = np.radians(THETA_MAX_DEG)
        tx = np.random.uniform(-theta_max, theta_max)
        ty = np.random.uniform(-theta_max, theta_max)
        tz = np.random.uniform(-theta_max, theta_max)
        R = _rotation_matrix(tx, ty, tz)
        g_vec = R @ np.array([0.0, 0.0, G])
        self.g_x, self.g_y, self.g_z = g_vec

        # ── Smoothed state variables ──
        self.f_step = 0.0
        self.A_z = 0.0
        self.A_x = 0.0
        self.A_y = 0.0
        self.f_resp = 0.22 + self.subj_f_resp_offset
        self.B_rsp = 1.0
        self.ie_ratio = 1.5
        self.T_core = 37.0
        self.T_skin = self.subj_skin_base
        self.HR = 72.0 + self.subj_hr_offset
        self.temp_drift = 0.0

        # ── Accumulated phases ──
        self.phase_z = np.random.uniform(0, 2 * np.pi)
        self.phase_x = np.random.uniform(0, 2 * np.pi)
        self.phase_y = np.random.uniform(0, 2 * np.pi)
        self.phase_resp = np.random.uniform(0, 2 * np.pi)

        self.exercise_duration = 0.0
        self.sample_idx = 0

        # ── ACC bias offset (spec §7.2) ──
        self.bias_x = np.random.uniform(-BIAS_OFFSET_RANGE, BIAS_OFFSET_RANGE)
        self.bias_y = np.random.uniform(-BIAS_OFFSET_RANGE, BIAS_OFFSET_RANGE)
        self.bias_z = np.random.uniform(-BIAS_OFFSET_RANGE, BIAS_OFFSET_RANGE)

        # ── ACC bias instability / drift (spec §3.7) ──
        self.bias_drift_x = 0.0
        self.bias_drift_y = 0.0
        self.bias_drift_z = 0.0

        # ── Micro-movement sway state (LPF at 0.5 Hz, spec §3.6) ──
        self.micro_x = 0.0
        self.micro_y = 0.0
        self.micro_z = 0.0
        self.micro_alpha = DT / (DT + 1.0 / (2 * np.pi * 0.5))

        # ── Micro-adjustment burst state ──
        self.micro_adj_remaining = 0
        self.micro_adj_amp = 0.0
        self.micro_adj_phase = 0.0

        # ── Fidget / shift cooldowns ──
        self.fidget_cooldown = 0
        self.shift_cooldown = 0
        self.fidget_buf_x = deque()
        self.fidget_buf_y = deque()
        self.fidget_buf_z = deque()
        self.shift_buf_x = deque()
        self.shift_buf_y = deque()
        self.shift_buf_z = deque()

        # ── Artifact state ──
        self.gap_remaining = 0
        self.rsp_glitch_remaining = 0
        self.skt_glitch_remaining = 0
        self.rsp_wander_phase = np.random.uniform(0, 2 * np.pi)
        self.rsp_wander_freq = 1.0 / np.random.uniform(*RSP_WANDER_PERIOD)
        self.stuck_remaining = [0, 0, 0]
        self.stuck_value = [0.0, 0.0, 0.0]


# ══════════════════════════════════════════════════════════════════════════════
# Helper: asymmetric breathing waveform  (spec §4.3)
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
    """Advance simulation by one sample.

    Returns (t, acc_x, acc_y, acc_z, rsp, skt_or_None, activity) or None
    when the simulation is finished.
    """
    idx = state.sample_idx
    if idx >= TOTAL_SAMPLES:
        return None

    act = SCHEDULE[idx]
    tgt = TARGETS[act]
    f_step_tgt, Az_tgt, Ax_tgt, Ay_tgt = tgt[0], tgt[1], tgt[2], tgt[3]
    f_resp_tgt, B_tgt, ie_tgt = tgt[4], tgt[5], tgt[6]
    Tc_tgt, Ts_tgt, HR_tgt = tgt[7], tgt[8], tgt[9]

    s = state

    # ── Apply inter-subject scaling (spec §2.1) ──
    f_step_tgt_subj = f_step_tgt * s.subj_f_step_mult
    Az_tgt_subj = Az_tgt * s.subj_amp_mult
    Ax_tgt_subj = Ax_tgt * s.subj_amp_mult
    Ay_tgt_subj = Ay_tgt * s.subj_amp_mult
    HR_tgt_subj = HR_tgt + s.subj_hr_offset
    f_resp_tgt_subj = f_resp_tgt + s.subj_f_resp_offset

    # ── Exercise duration tracking (spec §6.2) ──
    if act in ("walking", "running"):
        s.exercise_duration += DT
    else:
        s.exercise_duration = max(0, s.exercise_duration - DT * 0.5)

    # ── Prolonged-exercise skin temp adjustment (spec §6.2) ──
    Ts_adj = Ts_tgt
    if act in ("walking", "running") and s.exercise_duration > 900:
        Ts_adj = Ts_tgt + min(1.5, 0.1 * (s.exercise_duration - 900) / 60.0)

    # ── Smooth state variables (first-order low-pass, spec §3.5, §4.4) ──
    s.f_step += (DT / TAU_ACC) * (f_step_tgt_subj - s.f_step)
    s.A_z += (DT / TAU_ACC) * (Az_tgt_subj - s.A_z)
    s.A_x += (DT / TAU_ACC) * (Ax_tgt_subj - s.A_x)
    s.A_y += (DT / TAU_ACC) * (Ay_tgt_subj - s.A_y)

    # Breathing: base target + movement coupling (spec §6.1)
    f_resp_coupled = f_resp_tgt_subj + K_MV * s.f_step
    tau_rsp = TAU_RSP_FREQ.get(act, 15)
    s.f_resp += (DT / tau_rsp) * (f_resp_coupled - s.f_resp)
    s.B_rsp += (DT / TAU_RSP_AMP) * (B_tgt - s.B_rsp)
    s.ie_ratio += (DT / tau_rsp) * (ie_tgt - s.ie_ratio)

    tau_c = TAU_CORE.get(act, 600)
    tau_s = TAU_SKIN.get(act, 300)
    s.T_core += (DT / tau_c) * (Tc_tgt - s.T_core)
    s.T_skin += (DT / tau_s) * (Ts_adj - s.T_skin)

    tau_hr = 30 if act in ("walking", "running") else 90
    s.HR += (DT / tau_hr) * (HR_tgt_subj - s.HR)

    # ── Accumulate phases (spec §3.2 — prevents frequency-change discontinuities) ──
    s.phase_z += 2 * np.pi * s.f_step * DT
    s.phase_x += 2 * np.pi * s.f_step * DT
    s.phase_y += 2 * np.pi * s.f_step * DT
    s.phase_resp += 2 * np.pi * s.f_resp * DT

    # ── Locomotor-respiratory coupling (spec §4.5) ──
    if act in ("walking", "running") and s.f_resp > 0:
        n_lrc = max(1, round(s.f_step / s.f_resp))
        lrc_correction = K_LRC * np.sin(n_lrc * s.phase_z - s.phase_resp)
        s.phase_resp += 2 * np.pi * lrc_correction * DT

    # ── Step-to-step variability (spec §3.3) ──
    amp_jitter = 1.0
    if act == "walking":
        amp_jitter = 1.0 + np.random.normal(0, 0.05)
    elif act == "running":
        amp_jitter = 1.0 + np.random.normal(0, 0.08)

    # ── ACC bias instability (spec §3.7 — bounded random walk) ──
    drift_inc = SIGMA_BIAS_DRIFT * np.sqrt(DT)
    s.bias_drift_x = np.clip(s.bias_drift_x + np.random.normal(0, drift_inc),
                             -BIAS_DRIFT_CLAMP, BIAS_DRIFT_CLAMP)
    s.bias_drift_y = np.clip(s.bias_drift_y + np.random.normal(0, drift_inc),
                             -BIAS_DRIFT_CLAMP, BIAS_DRIFT_CLAMP)
    s.bias_drift_z = np.clip(s.bias_drift_z + np.random.normal(0, drift_inc),
                             -BIAS_DRIFT_CLAMP, BIAS_DRIFT_CLAMP)

    # ── Micro-movement noise for non-gait states (spec §3.6) ──
    sigma_micro = SIGMA_MICRO[act]
    # Postural sway (LPF white noise)
    s.micro_x += s.micro_alpha * (np.random.normal(0, sigma_micro) - s.micro_x)
    s.micro_y += s.micro_alpha * (np.random.normal(0, sigma_micro) - s.micro_y)
    s.micro_z += s.micro_alpha * (np.random.normal(0, sigma_micro) - s.micro_z)

    # Sporadic micro-adjustment bursts
    micro_adj_val = 0.0
    if s.micro_adj_remaining > 0:
        s.micro_adj_remaining -= 1
        t_frac = s.micro_adj_remaining / max(1, s.micro_adj_remaining + 1)
        micro_adj_val = s.micro_adj_amp * np.exp(-3 * (1 - t_frac))
    elif act in MICRO_ADJ_PROB and np.random.random() < MICRO_ADJ_PROB[act]:
        dur_samples = int(np.random.uniform(0.5, 2.0) * FS)
        s.micro_adj_remaining = dur_samples
        amp_range = MICRO_ADJ_AMP.get(act, (0.2, 0.5))
        s.micro_adj_amp = np.random.uniform(*amp_range) * np.random.choice([-1, 1])

    # ── Generate ACC signal (spec §3.2) ──
    # 3rd harmonic coefficient
    h3 = H3_COEFF.get(act, 0.0)

    ax = (s.A_x * amp_jitter * np.sin(s.phase_x + np.pi / 4)
          + s.g_x
          + s.bias_x + s.bias_drift_x
          + s.micro_x
          + micro_adj_val * 0.5
          + np.random.normal(0, SIGMA_ACC_SENSOR))

    ay = (s.A_y * amp_jitter * np.sin(s.phase_y + np.pi / 2)
          + s.g_y
          + s.bias_y + s.bias_drift_y
          + s.micro_y
          + micro_adj_val * 0.3
          + np.random.normal(0, SIGMA_ACC_SENSOR))

    az = (s.A_z * amp_jitter * np.sin(s.phase_z)
          + 0.3 * s.A_z * amp_jitter * np.sin(2 * s.phase_z)
          + h3  * s.A_z * amp_jitter * np.sin(3 * s.phase_z)
          + s.g_z
          + s.bias_z + s.bias_drift_z
          + s.micro_z
          + micro_adj_val
          + np.random.normal(0, SIGMA_ACC_SENSOR))

    # ── Fidget / shift overlays (spec §3.4) ──
    if s.fidget_buf_x:
        ax += s.fidget_buf_x.popleft()
        ay += s.fidget_buf_y.popleft()
        az += s.fidget_buf_z.popleft()
    if s.shift_buf_x:
        ax += s.shift_buf_x.popleft()
        ay += s.shift_buf_y.popleft()
        az += s.shift_buf_z.popleft()

    # ── Trigger new stress fidgets (spec §3.4) ──
    if s.fidget_cooldown > 0:
        s.fidget_cooldown -= 1
    if (act == "stress" and not s.fidget_buf_x
            and s.fidget_cooldown <= 0
            and np.random.random() < 0.0006):
        flen = int(np.random.uniform(1.0, 3.0) * FS)
        famp = np.random.uniform(0.5, 2.0)
        for j in range(flen):
            sig = famp * np.sin(2 * np.pi * np.random.uniform(2, 5) * j * DT)
            s.fidget_buf_x.append(sig * 0.5)
            s.fidget_buf_y.append(sig * 0.3)
            s.fidget_buf_z.append(sig)
        s.fidget_cooldown = int(np.random.uniform(30, 120) * FS)

    # ── Trigger new sleep position shifts (spec §3.4) ──
    if s.shift_cooldown > 0:
        s.shift_cooldown -= 1
    if (act == "sleep" and not s.shift_buf_x
            and s.shift_cooldown <= 0
            and np.random.random() < 0.00003):
        slen = int(np.random.uniform(2, 5) * FS)
        speak = np.random.uniform(5, 15)
        for j in range(slen):
            t_rel = j * DT
            t_dur = slen * DT
            envelope = speak * np.exp(-3 * t_rel / t_dur)
            s.shift_buf_x.append(envelope * np.random.normal(0, 0.3))
            s.shift_buf_y.append(envelope * np.random.normal(0, 0.3))
            s.shift_buf_z.append(envelope * np.random.normal(0, 0.3))
        s.shift_cooldown = int(np.random.uniform(5 * 60, 30 * 60) * FS)

    # ── ACC quantization + clipping (spec §7.2) ──
    ax = round(ax / ACC_RESOLUTION) * ACC_RESOLUTION
    ay = round(ay / ACC_RESOLUTION) * ACC_RESOLUTION
    az = round(az / ACC_RESOLUTION) * ACC_RESOLUTION
    ax = np.clip(ax, -ACC_RANGE, ACC_RANGE)
    ay = np.clip(ay, -ACC_RANGE, ACC_RANGE)
    az = np.clip(az, -ACC_RANGE, ACC_RANGE)

    # ── ACC stuck-value artefact ──
    acc_vals = [ax, ay, az]
    for i in range(3):
        if s.stuck_remaining[i] > 0:
            acc_vals[i] = s.stuck_value[i]
            s.stuck_remaining[i] -= 1
        elif np.random.random() < ACC_STUCK_PROB:
            s.stuck_value[i] = acc_vals[i]
            s.stuck_remaining[i] = np.random.randint(*ACC_STUCK_DUR)
    ax, ay, az = acc_vals

    # ── RSP signal (spec §4.1–§4.6) ──
    sigma_rsp = SIGMA_RSP[act]
    rate_var = RATE_VAR[act]
    # Breath-to-breath rate jitter applied as phase perturbation
    s.phase_resp += 2 * np.pi * s.f_resp * DT * np.random.normal(0, rate_var) * 0.1

    rsp_val = s.B_rsp * breath_waveform(s.phase_resp, s.ie_ratio)

    # Motion artifact on RSP during gait (spec §4.6)
    if act in K_MA:
        k_ma = K_MA[act]
        rsp_val += k_ma * abs(az - s.g_z) * np.sign(np.sin(s.phase_z))

    rsp_val += np.random.normal(0, sigma_rsp)

    # RSP baseline wander (spec §7.3)
    s.rsp_wander_phase += 2 * np.pi * s.rsp_wander_freq * DT
    rsp_val += RSP_WANDER_AMP * s.B_rsp * np.sin(s.rsp_wander_phase)

    # ── SKT signal at lower rate (spec §5.2–§5.6) ──
    skt_val = None
    temp_interval = int(FS / FS_TEMP)
    if idx % temp_interval == 0:
        s.temp_drift += np.random.normal(0, 0.001 * np.sqrt(DT_TEMP))
        t_now = idx * DT

        t_warm = t_now + SKT_WARMUP_PRE
        if t_warm < SKT_WARMUP_END:
            # Sensor warm-up: exponential approach from T_amb (spec §5.6)
            warmup_factor = 1.0 - np.exp(-t_warm / SKT_WARMUP_TAU)
            skt_val = T_AMB + (s.T_skin - T_AMB) * warmup_factor
            # Random contact-settling jumps during warm-up
            if np.random.random() < SKT_WARMUP_JUMP_PROB:
                skt_val += np.random.uniform(*SKT_WARMUP_JUMP_AMP) * np.random.choice([-1, 1])
        else:
            skt_val = s.T_skin

        skt_val += s.temp_drift + np.random.normal(0, TEMP_RESOLUTION)

        # SKT spike artefact
        if np.random.random() < SKT_SPIKE_PROB:
            skt_val += np.random.uniform(*SKT_SPIKE_AMP) * np.random.choice([-1, 1])

        # Quantize to sensor resolution (spec §7.4)
        skt_val = round(skt_val / TEMP_RESOLUTION) * TEMP_RESOLUTION

    # ══════════════════════════════════════════════════════════════════
    # Bluetooth connection gap — all signals go NaN (spec §7.5)
    # ══════════════════════════════════════════════════════════════════
    if s.gap_remaining > 0:
        s.gap_remaining -= 1
        t = idx * DT
        s.sample_idx += 1
        return t, np.nan, np.nan, np.nan, np.nan, (np.nan if skt_val is not None else None), act
    elif np.random.random() < GAP_PROB_PER_SAMPLE:
        dur = np.random.uniform(*GAP_DUR_RANGE)
        s.gap_remaining = int(dur * FS)

    # ══════════════════════════════════════════════════════════════════
    # Sensor glitch — occasional brief RSP / SKT dropout
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
    win = int(PLOT_WINDOW * FS)
    win_temp = int(PLOT_WINDOW * FS_TEMP)

    # Rolling buffers
    t_buf = deque(maxlen=win)
    ax_buf = deque(maxlen=win)
    ay_buf = deque(maxlen=win)
    az_buf = deque(maxlen=win)
    rsp_buf = deque(maxlen=win)
    t_temp_buf = deque(maxlen=win_temp)
    skt_buf = deque(maxlen=win_temp)

    log_rows = []

    if headless:
        print(f"Generating {TOTAL_SAMPLES} samples ({TOTAL_TIME:.0f}s) in headless mode...")
        print(f"  Subject profile: f_step_mult={state.subj_f_step_mult:.3f}, "
              f"amp_mult={state.subj_amp_mult:.3f}, "
              f"skin_base={state.subj_skin_base:.2f}°C, "
              f"hr_offset={state.subj_hr_offset:+.1f}bpm, "
              f"f_resp_offset={state.subj_f_resp_offset:+.4f}Hz")
        print(f"  Gravity vector: gx={state.g_x:.3f}, gy={state.g_y:.3f}, gz={state.g_z:.3f}")
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
    for i, ax_plt in enumerate(axes):
        ax_plt.set_facecolor("#2a2a3e")
        ax_plt.tick_params(colors="#ccc", labelsize=8)
        ax_plt.set_ylabel(ylabels[i], color="#ccc", fontsize=9)
        ax_plt.set_title(titles[i], color=colors[i], fontsize=10,
                         fontweight="bold", loc="left")
        for spine in ax_plt.spines.values():
            spine.set_color("#444")
        ln, = ax_plt.plot([], [], color=colors[i], linewidth=0.7)
        lines.append(ln)

    axes[-1].set_xlabel("Time (s)", color="#ccc", fontsize=9)

    activity_text = fig.text(0.5, 0.965, "", ha="center", va="center",
                             fontsize=13, fontweight="bold", color="#eee",
                             fontfamily="monospace")
    progress_text = fig.text(0.5, 0.945, "", ha="center", va="center",
                             fontsize=8, color="#aaa")

    # ── Fixed Y-axis limits ──
    max_ax = max(t[2] for t in TARGETS.values()) + 0.5
    max_ay = max(t[3] for t in TARGETS.values()) + 0.5
    max_az = max(t[1] for t in TARGETS.values()) + 1.0
    max_rsp = max(t[5] for t in TARGETS.values()) + 0.5
    skt_lo = min(T_AMB - 1.0, min(t[8] for t in TARGETS.values()) - 2.0)
    skt_hi = max(t[8] for t in TARGETS.values()) + 1.0

    FIXED_YLIMS = [
        (-max_ax, max_ax),
        (-max_ay, max_ay),
        (G - max_az - 1.0, G + max_az + 1.0),
        (-max_rsp, max_rsp),
        (skt_lo, skt_hi),
    ]

    def init():
        for ln in lines:
            ln.set_data([], [])
        for i, ax_plt in enumerate(axes):
            ax_plt.set_ylim(FIXED_YLIMS[i])
        return lines

    def update(frame):
        for _ in range(STEPS_PER_FRAME):
            result = step(state)
            if result is None:
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

        t_arr = np.array(t_buf)
        lines[0].set_data(t_arr, np.array(ax_buf))
        lines[1].set_data(t_arr, np.array(ay_buf))
        lines[2].set_data(t_arr, np.array(az_buf))
        lines[3].set_data(t_arr, np.array(rsp_buf))

        if t_temp_buf:
            lines[4].set_data(np.array(t_temp_buf), np.array(skt_buf))

        if len(t_buf) > 1:
            t_lo, t_hi = t_buf[0], t_buf[-1]
            for i in range(4):
                axes[i].set_xlim(t_lo, t_hi)
            if t_temp_buf:
                axes[4].set_xlim(t_temp_buf[0], t_temp_buf[-1])

        cur_t = t_buf[-1] if t_buf else 0
        act = SCHEDULE[min(state.sample_idx - 1, TOTAL_SAMPLES - 1)]
        color = ACTIVITY_COLORS.get(act, "#fff")
        activity_text.set_text(f"▶  {act.upper()}")
        activity_text.set_color(color)
        pct = 100 * cur_t / TOTAL_TIME
        elapsed_bar = "█" * int(pct / 2.5) + "░" * (40 - int(pct / 2.5))
        progress_text.set_text(
            f"{elapsed_bar}  {cur_t:.0f}s / {TOTAL_TIME:.0f}s  ({pct:.0f}%)")

        return lines

    ani = FuncAnimation(fig, update, init_func=init,
                        interval=FRAME_INTERVAL_MS, blit=False,
                        cache_frame_data=False)
    plt.show()

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
            w.writerow([f"{t:.4f}", ax, ay, az, rsp,
                        skt if skt is not None else "", act])
    print(f"Saved {len(rows)} samples to {path}")


if __name__ == "__main__":
    main()
