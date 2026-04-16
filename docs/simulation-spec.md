# Wearable Signal Simulation Specification

A code-ready specification for simulating waist-worn sensor data (accelerometer, respiration, skin temperature) across six activity states: **rest, walking, running, recovery, stress, and sleep**.

## Errata vs. Original Report

This specification corrects the following critical errors in `deep-research-report.md`:

1. **Skin temperature direction during exercise** — The original report models skin temp as rising during exercise. Literature confirms the opposite: skin temperature **drops** 1–4 °C during exercise due to sympathetic vasoconstriction and evaporative cooling, even as core temperature rises (Blokker et al., 2022, PMC9666787: trunk skin temp dropped from 33.4 → 29.0 °C during exercise while core rose 37.0 → 37.5 °C).
2. **Skin–core coupling** — The original `T_skin = T_core − 5 + noise` is incorrect. Skin and core temperatures move in opposite directions during exercise, recovery, and sleep. This spec models them independently.
3. **Phase accumulation** — The original pseudocode uses `sin(2π·f·t[i])` with time-varying frequency, producing discontinuities. This spec uses accumulated phase: `phase += 2π·f·dt`.
4. **ACC amplitude mismatch** — The original pseudocode uses amplitudes (0.6 / 1.2 m/s²) far below the stated RMS ranges (1–6 / 6–12.5 m/s²). Corrected here.
5. **Missing activity states** — Stress and sleep were absent. Both are now included.

---

## 1. Activity State Machine

The simulator runs a finite state machine with six states. Each state defines target values for all physiological variables. Transitions use first-order smoothing (no instantaneous jumps).

```
┌──────┐    ┌─────────┐    ┌─────────┐    ┌──────────┐
│ REST │───>│ WALKING │───>│ RUNNING │───>│ RECOVERY │
└──────┘    └─────────┘    └─────────┘    └──────────┘
    │                                          │
    v                                          v
┌────────┐                                ┌────────┐
│ STRESS │                                │ SLEEP  │
└────────┘                                └────────┘
```

Activity sequences are user-defined. Example: `rest(120s) → walking(180s) → running(120s) → recovery(180s) → rest(60s) → stress(120s) → rest(60s) → sleep(600s)`.

---

## 2. Global Configuration

| Parameter | Default | Units | Notes |
|-----------|---------|-------|-------|
| Sampling rate (`fs`) | 50 | Hz | ACC and RSP share this rate |
| Temp sampling rate (`fs_temp`) | 4 | Hz | Lower rate, matching typical sensors |
| Ambient temperature (`T_amb`) | 22.0 | °C | Thermoneutral indoor environment |
| Gravity | 9.81 | m/s² | Added to vertical axis |

---

## 3. Accelerometer (ACC) Model

### 3.1 Coordinate System

Waist-worn device, upright posture:
- **z-axis (vertical)**: Dominant gait component — largest amplitude, contains fundamental + 2nd harmonic
- **x-axis (anteroposterior)**: Medium amplitude — forward/backward trunk oscillation
- **y-axis (mediolateral)**: Smallest amplitude — lateral sway

### 3.2 Signal Equation

For active gait states (walking, running):

$$a_z(t) = A_z \sin(\phi_z(t)) + 0.3 \cdot A_z \sin(2\phi_z(t)) + g + \eta_z(t)$$
$$a_x(t) = A_x \sin(\phi_x(t) + \pi/4) + \eta_x(t)$$
$$a_y(t) = A_y \sin(\phi_y(t) + \pi/2) + \eta_y(t)$$

Where:
- $\phi(t)$ is accumulated phase: $\phi(t) = \phi(t-dt) + 2\pi \cdot f_{step}(t) \cdot dt$
- The 2nd harmonic term ($0.3 \cdot A_z \sin(2\phi)$) models heel-strike impact spikes
- $g = 9.81$ m/s² (gravitational component on z-axis)
- $\eta(t) \sim \mathcal{N}(0, \sigma_{acc}^2)$ is sensor noise

### 3.3 Parameters by Activity

| Parameter | Rest | Walking | Running | Recovery | Stress | Sleep | Units |
|-----------|------|---------|---------|----------|--------|-------|-------|
| $f_{step}$ target | 0 | 1.8 ± 0.2 | 2.8 ± 0.3 | 0.5 → 0 | 0 | 0 | Hz |
| $A_z$ (vertical peak) | 0.02 | 3.0 ± 1.0 | 10.0 ± 3.0 | 0.5 → 0.02 | 0.02 | 0.02 | m/s² |
| $A_x$ (AP peak) | 0.01 | 1.5 ± 0.5 | 5.0 ± 1.5 | 0.25 → 0.01 | 0.01 | 0.01 | m/s² |
| $A_y$ (ML peak) | 0.01 | 0.8 ± 0.3 | 2.5 ± 0.8 | 0.15 → 0.01 | 0.01 | 0.01 | m/s² |
| $\sigma_{acc}$ (noise) | 0.02 | 0.15 | 0.3 | 0.05 | 0.02 | 0.015 | m/s² |
| Step variability | — | ±5% amp, ±3% period | ±8% amp, ±5% period | ±5% | — | — | — |

**Vertical RMS cross-check**: For a sinusoid with peak $A$ and a 2nd harmonic at $0.3A$, RMS ≈ $A \cdot \sqrt{(1 + 0.09)/2}$ ≈ $0.74A$. Walking: $0.74 \times 3.0 = 2.2$ m/s² (within 1–6 range). Running: $0.74 \times 10.0 = 7.4$ m/s² (within 6–12.5 range). ✓

### 3.4 Special Movement Patterns

**Stress fidgets**: Every 30–120 s (random interval), insert a 1–3 s burst of movement:
- Amplitude: 0.5–2.0 m/s² (random)
- Frequency: 2–5 Hz (irregular — use random-phase summed sinusoids)
- Duration: uniformly distributed 1–3 s

**Sleep position shifts**: Every 5–30 min (random interval), insert a large transient:
- Peak amplitude: 5–15 m/s²
- Duration: 2–5 s
- Shape: Gaussian envelope × noise (not periodic)
- Followed by 5–10 s settling (exponential decay to baseline)

### 3.5 Smoothing

Step frequency and amplitude transition between activity states via first-order low-pass:

$$f_{step}(t) = f_{step}(t-dt) + \frac{dt}{\tau_{acc}} \cdot (f_{target} - f_{step}(t-dt))$$

Time constant $\tau_{acc} = 3$ s (gait adjusts within a few steps).

---

## 4. Respiration (RSP) Model

### 4.1 Signal Equation

$$RSP(t) = B(t) \cdot w(\phi_{resp}(t)) + \nu(t)$$

Where:
- $\phi_{resp}(t) = \phi_{resp}(t-dt) + 2\pi \cdot f_{resp}(t) \cdot dt$ (accumulated phase)
- $w(\phi)$ is an asymmetric breathing waveform (see §4.3)
- $B(t)$ is amplitude (proportional to tidal volume)
- $\nu(t) \sim \mathcal{N}(0, \sigma_{rsp}^2)$ is sensor noise

### 4.2 Parameters by Activity

| Parameter | Rest | Walking | Running | Recovery | Stress | Sleep | Units |
|-----------|------|---------|---------|----------|--------|-------|-------|
| $f_{resp}$ target | 0.25 ± 0.03 | 0.40 ± 0.05 | 0.75 ± 0.15 | 0.35 → 0.25 | 0.30 ± 0.05 | 0.20 ± 0.03 | Hz |
| Equivalent bpm | 15 ± 2 | 24 ± 3 | 45 ± 9 | 21 → 15 | 18 ± 3 | 12 ± 2 | bpm |
| $B$ (amplitude) | 1.0 | 1.8 ± 0.3 | 3.5 ± 0.5 | 2.0 → 1.0 | 1.2 ± 0.2 | 0.8 ± 0.1 | normalized |
| I:E ratio | 1:2.0 | 1:1.5 | 1:1.0 | 1:1.5 → 1:2.0 | 1:1.8 | 1:2.5 | — |
| Breath-to-breath variability | ±5% rate | ±8% rate | ±10% rate | ±8% | ±12% rate | ±3% rate | — |
| $\sigma_{rsp}$ (noise) | 0.02 | 0.05 | 0.08 | 0.04 | 0.03 | 0.015 | normalized |

### 4.3 Asymmetric Breathing Waveform

Real breathing has a shorter inspiratory phase and longer expiratory phase. Model using a piecewise sinusoidal waveform parameterized by the I:E ratio $r$:

Let the inspiratory fraction be $\alpha = \frac{1}{1 + r}$ where $r$ is the E:I ratio (e.g., $r=2.0$ gives $\alpha=1/3$).

$$w(\phi) = \begin{cases} \sin\left(\frac{\pi \cdot \phi_{mod}}{2\pi\alpha}\right) & \text{if } \phi_{mod} < 2\pi\alpha \quad \text{(inspiration)} \\ \sin\left(\frac{\pi \cdot (\phi_{mod} - 2\pi\alpha)}{2\pi(1-\alpha)} + \pi\right) \cdot (-1) & \text{if } \phi_{mod} \geq 2\pi\alpha \quad \text{(expiration)} \end{cases}$$

Where $\phi_{mod} = \phi \mod 2\pi$. This produces a waveform that rises quickly (inspiration) and falls slowly (expiration), matching real abdominal sensor traces.

Simplified implementation: remap the phase within each cycle so that the first $\alpha$ fraction maps to 0→π and the remaining $(1-\alpha)$ maps to π→2π, then apply a standard sine.

### 4.4 Smoothing

$$f_{resp}(t) = f_{resp}(t-dt) + \frac{dt}{\tau_{rsp}} \cdot (f_{target,resp} - f_{resp}(t-dt))$$

Time constant $\tau_{rsp} = 15$ s (breathing adjusts over several breath cycles).

Amplitude smoothing uses the same form with $\tau_{rsp,amp} = 20$ s.

---

## 5. Skin Temperature (SKT) Model

### 5.1 Physiological Background

Skin temperature at the waist is governed by:
- **Core temperature** (metabolic heat source)
- **Peripheral blood flow** (controlled by sympathetic vasoconstriction/vasodilation)
- **Evaporative cooling** (sweat)
- **Ambient temperature** (environmental heat exchange)

**Key insight**: During exercise, skin temperature **decreases** despite rising core temperature. This is because:
1. Sympathetic activation constricts peripheral blood vessels (redirecting blood to muscles)
2. Sweat evaporation cools the skin surface
3. Only after prolonged exercise (>15–20 min) does vasodilation partially restore skin blood flow for thermoregulation

Evidence: Blokker et al. (2022) — trunk skin temp dropped 33.4 → 29.0 °C (anterior thorax) and 32.9 → 30.2 °C (back) during exercise, while core rose 37.0 → 37.5 °C.

### 5.2 Two-Variable Model

Model core temperature and skin temperature as **independent first-order systems** with activity-dependent targets:

**Core temperature:**
$$T_{core}(t) = T_{core}(t-dt) + \frac{dt}{\tau_{core}} \cdot (T_{core,target} - T_{core}(t-dt))$$

**Skin temperature:**
$$T_{skin}(t) = T_{skin}(t-dt) + \frac{dt}{\tau_{skin}} \cdot (T_{skin,target} - T_{skin}(t-dt)) + \eta_{temp}(t)$$

Where $\eta_{temp}(t) \sim \mathcal{N}(0, \sigma_{temp}^2)$ is sensor noise/drift.

The target values and time constants depend on activity state (see §5.3). Critically, `T_skin_target` and `T_core_target` move in **opposite directions** during exercise.

### 5.3 Parameters by Activity

| Parameter | Rest | Walking | Running | Recovery | Stress | Sleep | Units |
|-----------|------|---------|---------|----------|--------|-------|-------|
| $T_{core,target}$ | 37.0 | 37.3 | 37.8 | 37.0 | 37.1 | 36.5 | °C |
| $T_{skin,target}$ | 33.0 | 31.5 | 29.5 | 34.0 | 31.5 | 34.5 | °C |
| $\tau_{core}$ | — | 300 | 300 | 600 | 120 | 1800 | s |
| $\tau_{skin}$ | — | 120 | 90 | 180 | 90 | 900 | s |
| $\sigma_{temp}$ (noise) | 0.02 | 0.03 | 0.05 | 0.03 | 0.02 | 0.02 | °C |

### 5.4 Behavior by Activity (Narrative)

**Rest**: Skin ~33 °C, core ~37 °C. Stable with small fluctuations.

**Walking**: Core slowly rises toward 37.3 °C ($\tau$ = 5 min). Skin **drops** toward 31.5 °C ($\tau$ = 2 min) due to vasoconstriction. The skin drop is faster than the core rise because sympathetic vasoconstriction acts within seconds–minutes.

**Running**: Core rises faster toward 37.8 °C. Skin **drops further** toward 29.5 °C ($\tau$ = 90 s) from stronger vasoconstriction and sweat evaporation.

**Recovery**: Core slowly returns toward 37.0 °C ($\tau$ = 10 min). Skin **rises above baseline** toward 34.0 °C ($\tau$ = 3 min) — the body vasodilates peripheral vessels to dump excess heat. This recovery overshoot is a well-documented thermoregulatory response.

**Stress**: Core barely changes (37.1 °C). Skin drops slightly toward 32.5 °C ($\tau$ = 90 s) — real data from a 36-subject stress induction study (Hongn et al., 2024) shows minimal skin temperature change during cognitive stress tasks (32.76 vs 32.52 °C at rest). The original 31.5 °C target was too aggressive.

**Sleep**: Core **drops** toward 36.5 °C (circadian nadir; $\tau$ = 30 min, very slow). Skin **rises** toward 34.5 °C ($\tau$ = 15 min) — distal vasodilation facilitates core heat loss, a key mechanism for sleep onset. This inverse core-skin relationship during sleep is a robust circadian finding.

### 5.5 Sensor Drift

Add a slow random walk to simulate sensor drift:

$$drift(t) = drift(t-dt) + \mathcal{N}(0, \sigma_{drift}^2 \cdot dt)$$

With $\sigma_{drift} = 0.001$ °C/√s. This produces ~0.06 °C drift per hour (realistic for wearable thermopiles).

---

## 6. Cross-Signal Coupling

Activity state is the primary driver. All signals respond to the current activity state through their individual target values and time constants.

```
Activity State ──┬──> f_step_target, A_target    ──> ACC
                 ├──> f_resp_target, B_target     ──> RSP  
                 ├──> T_skin_target, T_core_target ──> SKT
                 └──> HR_target                    ──> HR (auxiliary)
```

### 6.1 Secondary Coupling: Movement → Respiration

Breathing rate has a secondary coupling to actual movement intensity (not just activity label). This captures the effect of walking speed variation:

$$f_{resp,target} = f_{resp,base} + k_{mv} \cdot f_{step}(t)$$

Where $k_{mv} = 0.15$ and $f_{resp,base}$ is the activity-state baseline. This adds ~0.03 Hz per 0.2 Hz step frequency change — a subtle but physiologically correct coupling.

### 6.2 Secondary Coupling: Exercise Duration → Skin Temperature

For prolonged exercise (>15 min), skin temperature begins to recover slightly as thermoregulatory vasodilation overrides vasoconstriction:

$$T_{skin,target,adjusted} = T_{skin,target} + \min(1.5, \; 0.1 \cdot \max(0, \; t_{exercise} - 900))$$

Where $t_{exercise}$ is seconds of continuous exercise. After 15 min (900 s), the skin target rises by 0.1 °C/min up to a 1.5 °C recovery cap. This models the biphasic skin temperature response during prolonged exercise.

### 6.3 Heart Rate (Auxiliary)

HR is modeled as a coupling variable (not a primary output signal) for completeness:

| Activity | HR target (bpm) | $\tau_{HR}$ (s) |
|----------|-----------------|------------------|
| Rest | 68 ± 5 | — |
| Walking | 105 ± 10 | 30 |
| Running | 160 ± 15 | 30 |
| Recovery | 68 | 90 |
| Stress | 85 ± 8 | 20 |
| Sleep | 55 ± 5 | 120 |

---

## 7. Sensor Artifacts

### 7.1 Noise (Applied to All Signals)

Additive white Gaussian noise at levels specified in each signal's parameter table.

### 7.2 ACC-Specific Artifacts

| Artifact | Model | Parameters |
|----------|-------|------------|
| Bias offset | Constant per-axis | ±0.05–0.2 m/s² (random at init) |
| Quantization | Round to resolution | 0.004 m/s² (12-bit, ±16g range) |
| Dropout | Zero random samples | 0.1% probability per sample |
| Sampling jitter | Offset timestamps | ±2 ms uniform |

### 7.3 RSP-Specific Artifacts

| Artifact | Model | Parameters |
|----------|-------|------------|
| Motion artifact | Add ACC-correlated burst | During activity transitions, add 0.1× |a_z| for 2–5 s |
| Baseline drift | Slow sinusoid | 0.05 amplitude, 0.005 Hz |

### 7.4 SKT-Specific Artifacts

| Artifact | Model | Parameters |
|----------|-------|------------|
| Sensor drift | Random walk | σ = 0.001 °C/√s |
| Quantization | Round to resolution | 0.02 °C |
| Response lag | Extra low-pass | τ = 5 s (sensor thermal mass) |

---

## 8. Complete Pseudocode

```python
import numpy as np

# ─── Configuration ───────────────────────────────────────────
fs = 50          # Hz, ACC/RSP sampling rate
fs_temp = 4      # Hz, temperature sampling rate
dt = 1.0 / fs
dt_temp = 1.0 / fs_temp
g = 9.81         # m/s²

# ─── Activity Sequence ───────────────────────────────────────
# User defines: list of (activity_name, duration_seconds)
activity_sequence = [
    ("rest", 120), ("walking", 180), ("running", 120),
    ("recovery", 180), ("rest", 60), ("stress", 120),
    ("rest", 60), ("sleep", 600),
]

total_time = sum(d for _, d in activity_sequence)
n_samples = int(total_time * fs)
n_temp_samples = int(total_time * fs_temp)

# ─── Activity Target Lookup ──────────────────────────────────
TARGETS = {
    #              f_step  A_z    A_x    A_y   f_resp  B_rsp  IE_ratio  T_core  T_skin  HR
    "rest":       (0.0,    0.02,  0.01,  0.01, 0.28,   1.0,   2.0,      37.0,   33.0,   68),
    "walking":    (1.9,    3.0,   1.5,   0.8,  0.40,   1.8,   1.5,      37.3,   31.5,   105),
    "running":    (2.8,    10.0,  5.0,   2.5,  0.75,   3.5,   1.0,      37.8,   29.5,   160),
    "recovery":   (0.3,    0.3,   0.15,  0.08, 0.30,   1.5,   1.8,      37.0,   34.0,   68),
    "stress":     (0.0,    0.02,  0.01,  0.01, 0.30,   1.2,   1.8,      37.1,   32.5,   80),
    "sleep":      (0.0,    0.02,  0.01,  0.01, 0.20,   0.8,   2.5,      36.5,   34.5,   55),
}

# ─── Time Constants (seconds) ────────────────────────────────
TAU_ACC = 3.0        # gait frequency/amplitude smoothing
TAU_RSP_FREQ = 15.0  # breathing rate smoothing
TAU_RSP_AMP = 20.0   # breathing amplitude smoothing
TAU_CORE = {          # core temp time constant per activity
    "rest": 600, "walking": 300, "running": 300,
    "recovery": 600, "stress": 120, "sleep": 1800,
}
TAU_SKIN = {          # skin temp time constant per activity
    "rest": 300, "walking": 120, "running": 90,
    "recovery": 180, "stress": 90, "sleep": 900,
}

# ─── Build Activity Label Array ──────────────────────────────
activity_labels = np.empty(n_samples, dtype=object)
idx = 0
for act, dur in activity_sequence:
    n = int(dur * fs)
    activity_labels[idx:idx+n] = act
    idx += n

# ─── Initialize State Variables ──────────────────────────────
f_step = 0.0
A_z, A_x, A_y = 0.02, 0.01, 0.01
f_resp = 0.25
B_rsp = 1.0
ie_ratio = 2.0
T_core = 37.0
T_skin = 33.0
HR = 68.0
temp_drift = 0.0

# Accumulated phases (critical: prevents frequency-change discontinuities)
phase_z = np.random.uniform(0, 2 * np.pi)
phase_x = np.random.uniform(0, 2 * np.pi)
phase_y = np.random.uniform(0, 2 * np.pi)
phase_resp = np.random.uniform(0, 2 * np.pi)

# Track exercise duration for prolonged-exercise skin temp adjustment
exercise_duration = 0.0

# ACC bias offsets (random per-session)
bias_x = np.random.uniform(-0.15, 0.15)
bias_y = np.random.uniform(-0.15, 0.15)
bias_z = np.random.uniform(-0.15, 0.15)

# ─── Output Arrays ──────────────────────────────────────────
acc_x = np.zeros(n_samples)
acc_y = np.zeros(n_samples)
acc_z = np.zeros(n_samples)
rsp = np.zeros(n_samples)
skt = np.zeros(n_temp_samples)

# ─── Asymmetric Breathing Waveform ──────────────────────────
def breath_waveform(phase_mod, ie_ratio):
    """
    Asymmetric sinusoidal waveform.
    ie_ratio: E:I ratio (e.g. 2.0 means expiration is 2x inspiration).
    Returns value in [-1, 1].
    """
    alpha = 1.0 / (1.0 + ie_ratio)  # inspiratory fraction of cycle
    phase_mod = phase_mod % (2 * np.pi)
    insp_end = 2 * np.pi * alpha
    if phase_mod < insp_end:
        # Inspiration: map [0, insp_end] → [0, π]
        return np.sin(np.pi * phase_mod / insp_end)
    else:
        # Expiration: map [insp_end, 2π] → [π, 2π]
        exp_phase = np.pi + np.pi * (phase_mod - insp_end) / (2 * np.pi - insp_end)
        return np.sin(exp_phase)

# ─── Main Simulation Loop ───────────────────────────────────
temp_idx = 0
temp_interval = int(fs / fs_temp)  # sample SKT every N ACC samples

for i in range(n_samples):
    act = activity_labels[i]
    tgt = TARGETS[act]
    (f_step_tgt, Az_tgt, Ax_tgt, Ay_tgt,
     f_resp_tgt, B_tgt, ie_tgt, Tc_tgt, Ts_tgt, HR_tgt) = tgt

    # ── Track exercise duration for prolonged-exercise adjustment ──
    if act in ("walking", "running"):
        exercise_duration += dt
    else:
        exercise_duration = max(0, exercise_duration - dt * 0.5)  # slow decay

    # ── Adjust skin target for prolonged exercise ──
    Ts_adjusted = Ts_tgt
    if act in ("walking", "running") and exercise_duration > 900:
        Ts_adjusted = Ts_tgt + min(1.5, 0.1 * (exercise_duration - 900) / 60.0)

    # ── Smooth state variables (first-order low-pass) ──
    f_step += (dt / TAU_ACC) * (f_step_tgt - f_step)
    A_z += (dt / TAU_ACC) * (Az_tgt - A_z)
    A_x += (dt / TAU_ACC) * (Ax_tgt - A_x)
    A_y += (dt / TAU_ACC) * (Ay_tgt - A_y)

    # Breathing rate: base target + secondary coupling to movement
    f_resp_coupled = f_resp_tgt + 0.15 * f_step
    f_resp += (dt / TAU_RSP_FREQ) * (f_resp_coupled - f_resp)
    B_rsp += (dt / TAU_RSP_AMP) * (B_tgt - B_rsp)
    ie_ratio += (dt / TAU_RSP_FREQ) * (ie_tgt - ie_ratio)

    # Temperature (at lower rate, but state updates every step)
    tau_c = TAU_CORE.get(act, 600)
    tau_s = TAU_SKIN.get(act, 300)
    T_core += (dt / tau_c) * (Tc_tgt - T_core)
    T_skin += (dt / tau_s) * (Ts_adjusted - T_skin)

    # Heart rate
    tau_hr = 30 if act in ("walking", "running") else 90
    HR += (dt / tau_hr) * (HR_tgt - HR)

    # ── Accumulate phases ──
    phase_z += 2 * np.pi * f_step * dt
    phase_x += 2 * np.pi * f_step * dt
    phase_y += 2 * np.pi * f_step * dt
    phase_resp += 2 * np.pi * f_resp * dt

    # ── Step-to-step variability ──
    amp_jitter = 1.0
    if act == "walking":
        amp_jitter = 1.0 + np.random.normal(0, 0.05)
    elif act == "running":
        amp_jitter = 1.0 + np.random.normal(0, 0.08)

    # ── Generate ACC signal ──
    sigma_acc = {"rest": 0.02, "walking": 0.15, "running": 0.3,
                 "recovery": 0.05, "stress": 0.02, "sleep": 0.015}[act]

    acc_z[i] = (A_z * amp_jitter * np.sin(phase_z)
                + 0.3 * A_z * amp_jitter * np.sin(2 * phase_z)
                + g + bias_z + np.random.normal(0, sigma_acc))

    acc_x[i] = (A_x * amp_jitter * np.sin(phase_x + np.pi / 4)
                + bias_x + np.random.normal(0, sigma_acc))

    acc_y[i] = (A_y * amp_jitter * np.sin(phase_y + np.pi / 2)
                + bias_y + np.random.normal(0, sigma_acc))

    # ── Generate RSP signal ──
    sigma_rsp = {"rest": 0.02, "walking": 0.05, "running": 0.08,
                 "recovery": 0.04, "stress": 0.03, "sleep": 0.015}[act]

    # Breath-to-breath rate variability
    rate_var = {"rest": 0.05, "walking": 0.08, "running": 0.10,
                "recovery": 0.08, "stress": 0.12, "sleep": 0.03}[act]
    # Apply variability to phase increment (subtle rate jitter)
    phase_resp += 2 * np.pi * f_resp * dt * np.random.normal(0, rate_var) * 0.1

    rsp[i] = (B_rsp * breath_waveform(phase_resp, ie_ratio)
              + np.random.normal(0, sigma_rsp))

    # ── Generate SKT signal (at lower sample rate) ──
    if i % temp_interval == 0 and temp_idx < n_temp_samples:
        temp_drift += np.random.normal(0, 0.001 * np.sqrt(dt_temp))
        skt[temp_idx] = T_skin + temp_drift + np.random.normal(0, 0.02)
        # Quantize to 0.02 °C resolution
        skt[temp_idx] = round(skt[temp_idx] / 0.02) * 0.02
        temp_idx += 1

    # ── Stress fidgets ──
    if act == "stress" and np.random.random() < 0.0003:  # ~0.9 events/min at 50Hz
        fidget_len = int(np.random.uniform(1, 3) * fs)
        fidget_end = min(i + fidget_len, n_samples)
        fidget_amp = np.random.uniform(0.5, 2.0)
        for j in range(i, fidget_end):
            fidget_sig = fidget_amp * np.sin(2 * np.pi * np.random.uniform(2, 5) * (j - i) * dt)
            acc_x[j] += fidget_sig * 0.5
            acc_y[j] += fidget_sig * 0.3
            acc_z[j] += fidget_sig

    # ── Sleep position shifts ──
    if act == "sleep" and np.random.random() < 0.00003:  # ~0.09 events/min at 50Hz
        shift_len = int(np.random.uniform(2, 5) * fs)
        shift_end = min(i + shift_len, n_samples)
        shift_peak = np.random.uniform(5, 15)
        for j in range(i, shift_end):
            t_rel = (j - i) * dt
            t_dur = (shift_end - i) * dt
            envelope = shift_peak * np.exp(-3 * t_rel / t_dur)
            acc_x[j] += envelope * np.random.normal(0, 0.3)
            acc_y[j] += envelope * np.random.normal(0, 0.3)
            acc_z[j] += envelope * np.random.normal(0, 0.3)

# ─── Apply Dropout (post-processing) ────────────────────────
dropout_mask = np.random.random(n_samples) < 0.001  # 0.1% dropout
acc_x[dropout_mask] = 0.0
acc_y[dropout_mask] = 0.0
acc_z[dropout_mask] = 0.0

# ─── Output ─────────────────────────────────────────────────
# acc_x, acc_y, acc_z: shape (n_samples,) at fs Hz
# rsp: shape (n_samples,) at fs Hz
# skt: shape (n_temp_samples,) at fs_temp Hz
# activity_labels: shape (n_samples,) activity name per sample
```

---

## 9. Validation Criteria

After implementation, verify:

| Check | Method | Expected |
|-------|--------|----------|
| ACC spectral peaks | PSD of acc_z during walking/running | Peaks at f_step and 2×f_step |
| ACC RMS ranges | RMS of acc_z per activity window | Walk: 1–6 m/s², Run: 6–12.5 m/s² |
| RSP rate histogram | Zero-crossing rate of RSP per activity | Rest: 12–20 bpm, Run: 30–60 bpm, Sleep: 10–16 bpm |
| SKT exercise response | Plot T_skin during rest→exercise | **Decreases** within 2–3 min of exercise onset |
| SKT recovery response | Plot T_skin during exercise→recovery | **Rises above baseline** (overshoot to ~34 °C) |
| SKT sleep response | Plot T_skin during wake→sleep | **Rises** 1–2 °C over 15–30 min |
| Phase continuity | Plot RSP during walking→running transition | Smooth frequency increase, no jumps |
| Coupling consistency | Scatter plot: f_resp vs f_step | Positive correlation during gait |
| Stress fidgets | Visual inspection of ACC during stress | Isolated 1–3 s bursts every 30–120 s |
| Sleep shifts | Visual inspection of ACC during sleep | Rare large transients every 5–30 min |

---

## 10. References

1. Blokker, T. et al. (2022). "Effect of cold ambient temperature on heat flux, skin temperature, and thermal sensation at different body parts in elite biathletes." *Frontiers in Sports and Active Living*. PMC9666787. — Confirms skin temp drops during exercise (trunk: 33.4→29.0 °C), core rises (37.0→37.5 °C).
2. Kräuchi, K. (2007). "The thermophysiological cascade leading to sleep initiation in relation to phase of entrainment." *Sleep Medicine Reviews*. — Distal skin temp rise and core temp drop as sleep-onset mechanism.
3. Schmidt, P. et al. (2018). "Introducing WESAD, a multimodal dataset for wearable stress and affect detection." *ICMI*. — Stress physiology: peripheral skin temp drop, elevated HR and breathing rate.
4. Kavanagh, J.J. & Menz, H.B. (2008). "Accelerometry: A technique for quantifying movement patterns during walking." *Gait & Posture*, 28(1), 1–15. — Trunk accelerometry review: vertical RMS ranges for walking and running.
5. Nicolò, A. et al. (2020). "Respiratory Frequency during Exercise: The Neglected Physiological Measure." *Frontiers in Physiology*, 11, 952. — Breathing rate ranges: rest 12–20 bpm, maximal exercise 40–70 bpm.
6. Pennes, H.H. (1948). "Analysis of tissue and arterial blood temperatures in the resting human forearm." *Journal of Applied Physiology*. — Foundation for bioheat transfer modeling.

---

## 11. Summary of Key Corrections

| Topic | Original Report | Corrected Spec |
|-------|----------------|----------------|
| Skin temp during exercise | Rises +0.5–1 °C | **Drops** 1.5–3.5 °C |
| Skin–core relationship | `T_skin = T_core − 5` | Independent dynamics, opposite directions |
| Phase accumulation | `sin(2π·f·t[i])` | `phase += 2π·f·dt; sin(phase)` |
| ACC vertical amplitude (walk) | 0.6 m/s² peak | 3.0 m/s² peak (RMS ≈ 2.2) |
| ACC vertical amplitude (run) | 1.2 m/s² peak | 10.0 m/s² peak (RMS ≈ 7.4) |
| ACC axis assignment | x = largest | z (vertical) = largest |
| Breathing amplitude | Fixed across activities | Scales 1×–3.5× with exercise |
| I:E ratio | Not modeled | 1:2.0 (rest) → 1:1.0 (run) |
| Stress state | Missing | Included: vasoconstriction, fidgets, elevated HR |
| Sleep state | Missing | Included: vasodilation, position shifts, slow breathing |
| Recovery skin temp | Implicit return to baseline | Overshoot to 34 °C (vasodilation) |
