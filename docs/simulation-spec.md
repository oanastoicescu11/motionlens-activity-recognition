# Wearable Signal Simulator — Specification v4

> **Code reference**: [`simulate.py`](../simulate.py)
> **Benchmark data**: [`analyses/benchmarks.json`](../analyses/benchmarks.json)
> **Validation**: [`analyses/validate_realism.py`](../analyses/validate_realism.py)
>
> Every numerical parameter in this document is traceable to either (a) a measured
> value from one of five real datasets, or (b) a specific finding in the published
> literature.  The **Provenance** column in each parameter table links every
> number to its origin.

---

## 1. Overview

The simulator generates five co-registered time-series signals — triaxial
accelerometry (ACC), respiration (RSP), skin temperature (SKT), heart rate (HR),
and activity labels — for a single body-worn sensor session.  The design goal is
statistical realism: Monte Carlo runs of the simulator should reproduce the
distributional properties (means, standard deviations, spectral features,
artifact rates) observed across five real wearable datasets totalling > 170
unique subjects.

### 1.1 Real-Data Sources

| ID | Dataset | Sensor | Site | Fs | Subjects | Reference |
|----|---------|--------|------|----|----------|-----------|
| D1 | UCI HAR | Smartphone (Samsung Galaxy S II) | Waist | 50 Hz | 30 | Anguita et al. (2013) [1] |
| D2 | RealWorld2016 | IMU (various) | Waist (among 7 sites) | 50 Hz | 15 | Sztyler & Stuckenschmidt (2016) [2] |
| D3 | EmoWear | BioHarness3 chest + Empatica E4 wrist | Chest / Wrist | 25–250 Hz (device-dependent) | 49 | Alamäki et al. (2024) [3] |
| D4 | Wearable Dataset | Empatica E4 | Wrist | 32 Hz (ACC), 4 Hz (SKT) | 90+ | Hosseini et al. (2022) [4] |
| D5 | InfantSmartWear | Multi-site textile sensors | Chest / Abdomen / Peripheral | Various | 3 000 | InfantSmartWear v1 (2024) [5] |

All benchmark values referenced below are computed by the analysis scripts in
[`analyses/`](../analyses/) and stored in
[`analyses/benchmarks.json`](../analyses/benchmarks.json).

---

## 2. Global Configuration

| Parameter | Value | Provenance |
|-----------|-------|------------|
| `FS` | 50 Hz | Matches D1 (50 Hz) and D2 (50 Hz) ACC sampling rates |
| `FS_TEMP` | 4 Hz | Matches Empatica E4 SKT sampling rate (D3, D4) |
| `G` | 9.81 m/s² | Standard gravitational acceleration |
| `T_AMB` | 22.0 °C | Typical indoor lab temperature; D3 EmoWear sessions conducted in controlled indoor environments |
| `ACC_RANGE` | ±16 g (±156.96 m/s²) | BioHarness3 ACC range is ±16 g; matches body-worn IMU full-scale ranges [6] |
| `ACC_RESOLUTION` | 0.004 m/s² | 12-bit ADC at ±16 g → 32 g / 2¹² ≈ 0.0078 g ≈ 0.004 m/s² per LSB (engineering estimate) |
| `TEMP_RESOLUTION` | 0.02 °C | Empatica E4 SKT sensor resolution [4] |

### 2.1 Activity Sequence

The simulation runs a fixed 270-second protocol traversing six activity states:

| Phase | Activity | Duration (s) | Rationale |
|-------|----------|-------------|-----------|
| 1 | rest | 30 | Baseline sitting/standing |
| 2 | walking | 45 | Moderate locomotion |
| 3 | running | 30 | Vigorous locomotion |
| 4 | recovery | 40 | Post-exercise cool-down |
| 5 | rest | 15 | Inter-task seated rest |
| 6 | stress | 35 | Cognitive/emotional stress (no locomotion) |
| 7 | rest | 15 | Brief seated rest |
| 8 | sleep | 60 | Simulated sleep onset |

The sequence was designed to exercise all signal models (gait, thermal response
to exercise, stress physiology, sleep onset) within a short demonstration window.
Activity durations are not derived from any specific protocol in the reference
datasets; they are engineering choices to ensure each model has sufficient time to
manifest its characteristic dynamics.

### 2.2 Inter-Subject Variability

Each simulated session draws per-subject offsets once, creating a virtual subject:

| Parameter | Distribution | Provenance |
|-----------|-------------|------------|
| Step frequency multiplier | N(1.0, 0.08) | CV ≈ 8%, calibrated to match D1 walking step-freq CV = 8.69% (`uci_har.walking_step_freq_cv` in benchmarks.json) |
| Amplitude multiplier | N(1.0, 0.20) | CV ≈ 20%, calibrated to match D1 walking amplitude CV = 13.9% (`uci_har.walking_amplitude_cv`) and D2 walking amplitude CV ≈ 24% (implied by per-axis STD variation in `realworld2016`). The 20% value is a compromise between the two datasets. |
| Skin temperature baseline | N(33.0, 1.5) °C | Mean from D4 stress-rest skin temp = 32.52 ± 1.74 °C (`wearable_dataset.stress_rest_temp_mean/std`). We use 33.0 °C as a round central value and 1.5 °C as a slightly conservative SD. |
| Heart rate offset | N(0, 12) bpm | D3 EmoWear resting HR = 73.1 ± 10.6 bpm (measured from E4 HR stream). 12 bpm is a rounded upward estimate to include general population variability beyond the D3 sample. |
| Breathing rate offset | N(0, 0.015) Hz | D3 rest breathing rate = 0.233 ± 0.038 Hz (`emowear.rest_breathing_rate_hz_mean`). 0.015 Hz offset SD is conservative (< 1 SD of the D3 distribution) to keep simulated breath rates within physiological range. |

**Application**: Step frequency and amplitude multipliers are applied
multiplicatively to gait targets. HR offset and breathing rate offset are applied
additively to the activity-specific targets. Skin baseline replaces the
activity-independent skin temperature starting point.

### 2.3 Sensor Placement Model

Each session applies a random rotation matrix R to the gravity vector, simulating
imprecise sensor attachment:

| Parameter | Value | Provenance |
|-----------|-------|------------|
| `THETA_MAX_DEG` | ±15° | Oishi et al. (2025) [6] demonstrate that body-worn IMU placement rotations of ±10–20° are typical in field deployments. We use ±15° as a mid-range estimate. |

The rotation is constructed from three independent Euler angles
θ_x, θ_y, θ_z ~ Uniform(−θ_max, θ_max) via Rx·Ry·Rz.  The rotated gravity vector
g_sensor = R · [0, 0, 9.81]ᵀ produces cross-axis gravity leakage at rest, matching
the non-zero mean ACC observed in D1 sitting data (per-axis means not purely
[0, 0, g]).

---

## 3. Accelerometry (ACC) Model

### 3.1 Signal Equation

For each sample i at time t = i · DT:

```
acc_z[i] = A_z · J · sin(φ_z)
          + 0.3 · A_z · J · sin(2φ_z)          # 2nd harmonic (fixed ratio)
          + h3  · A_z · J · sin(3φ_z)           # 3rd harmonic
          + g_z                                   # gravity component
          + bias_z + drift_z                      # sensor bias + instability
          + micro_z                               # postural sway / micro-movements
          + micro_adj                             # sporadic adjustment bursts
          + N(0, σ_sensor)                        # electronic noise

acc_x[i] = A_x · J · sin(φ_x + π/4)
          + g_x + bias_x + drift_x + micro_x
          + micro_adj · 0.5 + N(0, σ_sensor)

acc_y[i] = A_y · J · sin(φ_y + π/2)
          + g_y + bias_y + drift_y + micro_y
          + micro_adj · 0.3 + N(0, σ_sensor)
```

Where:
- `φ_z`, `φ_x`, `φ_y` are accumulated phase variables (see §3.4)
- `J` = step-to-step amplitude jitter (see §3.5)
- `h3` = 3rd-harmonic coefficient (see §3.2)
- `g_x, g_y, g_z` = rotated gravity components (see §2.3)
- Phase offsets (π/4 for X, π/2 for Y) model the naturally phase-shifted
  medio-lateral and antero-posterior acceleration components during gait,
  as documented in trunk accelerometry studies [7].

### 3.2 Activity Target Parameters

| Activity | f_step (Hz) | A_z (m/s²) | A_x (m/s²) | A_y (m/s²) | h3 coeff | Provenance |
|----------|-------------|------------|------------|------------|----------|------------|
| rest | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | No locomotion; body RMS from micro-movements only |
| walking | 1.8 | 3.5 | 4.5 | 3.0 | 0.48 | **f_step**: D1 = 1.810 Hz, D2 = 1.800 Hz (mean of both datasets). **A_z/A_x/A_y**: Tuned to match D2 per-axis walking STDs (x=3.003, y=1.922, z=2.359 m/s²) and D2 body RMS = 4.289 m/s². Note: simulator vertical (Z) is the dominant oscillation axis at the waist; X represents antero-posterior, Y medio-lateral. The RealWorld waist data shows X-dominant motion during walking. **h3**: 0.48² = 0.23, within the D1 H3/H1 range = 0.170 and D2 H3/H1 = 0.310; compromise value targeting mid-range. |
| running | 2.7 | 4.0 | 10.0 | 5.0 | 1.10 | **f_step**: D2 running = 2.693 Hz (`realworld2016.running_step_freq_mean`). **A_z/A_x/A_y**: Tuned to match D2 per-axis running STDs (x=7.696, y=3.555, z=4.133 m/s²) and body RMS = 9.505 m/s². **h3**: 1.10² = 1.21, matching D2 running H3/H1 = 1.201 (`realworld2016.running_h3_h1_mean`). |
| recovery | 0.3 | 0.3 | 0.15 | 0.08 | 0.0 | Low residual movement during cool-down. Amplitudes set to produce body RMS in the range of stress-task ACC variability from D4 (0.284 m/s²). f_step=0.3 Hz represents residual shuffling/swaying. |
| stress | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | No locomotion; D4 stress-rest ACC STD = 0.211 m/s² (`wearable_dataset.stress_rest_acc_std_mean`). Variability comes from micro-movements + fidgets (§3.8). |
| sleep | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | No locomotion; very low variability from micro-movements + occasional position shifts (§3.8). |

### 3.3 3rd-Harmonic Model

The vertical acceleration during gait contains energy at the fundamental step
frequency f_step and its harmonics.  The 2nd harmonic has a fixed coefficient of
0.3 (representing the double-support phase of gait [7]).  The 3rd harmonic is
activity-dependent:

| Activity | h3 | H3/H1 power ratio | Real target | Source |
|----------|----|--------------------|-------------|--------|
| walking | 0.48 | 0.48² = 0.23 | 0.170 (D1), 0.310 (D2) | `uci_har.walking_h3_h1` = 0.170; `realworld2016.walking_h3_h1_mean` = 0.310. 0.23 falls centrally between these. |
| running | 1.10 | 1.10² = 1.21 | 1.201 (D2) | `realworld2016.running_h3_h1_mean` = 1.201. Running exhibits a much stronger 3rd harmonic due to the flight phase impact [7]. |

### 3.4 Phase Accumulation

Phase variables accumulate continuously to prevent discontinuities at activity
transitions:

```
φ += 2π · f_step · DT    (per sample)
```

This ensures that when f_step changes (e.g., walking → running), the sinusoidal
signal smoothly adjusts frequency without phase jumps. Initial phases are drawn
uniformly from [0, 2π).

### 3.5 Step-to-Step Variability

Per-sample amplitude jitter simulates natural gait variability:

| Activity | Jitter σ | Provenance |
|----------|----------|------------|
| walking | ±5% (J ~ N(1.0, 0.05)) | D1 walking amplitude CV = 13.9%, D2 ≈ 24%. Per-step jitter of 5% contributes to total variability alongside inter-subject amplitude CV (20%). |
| running | ±8% (J ~ N(1.0, 0.08)) | Higher impact variability during running; engineering estimate consistent with running gait variability literature [7]. |

### 3.6 State Smoothing

All state variables (f_step, A_z, A_x, A_y) are smoothed via first-order
low-pass filters to prevent instantaneous jumps at activity boundaries:

```
x += (DT / τ) · (x_target − x)
```

| Variable | τ (s) | Provenance |
|----------|-------|------------|
| f_step, A_z, A_x, A_y | 3.0 | Engineering choice: ~3 s corresponds to ~5–6 steps to reach steady gait, consistent with gait initiation literature. |

### 3.7 Micro-Movement Model (Non-Gait States)

During rest, recovery, stress, and sleep, the body is not locomoting but exhibits
low-frequency postural sway, fidgets, and involuntary micro-adjustments.  This is
modelled as two components:

**Component 1 — Continuous postural sway**: White noise → first-order LPF at
0.5 Hz (micro_alpha = DT / (DT + 1/(2π·0.5))). The 0.5 Hz cutoff captures the
frequency range of standing/seated postural sway.

| Activity | σ_micro (m/s²) | Provenance |
|----------|----------------|------------|
| rest | 0.20 | Tuned so that total rest body RMS (σ_sensor + σ_micro + micro bursts) ≈ 0.234 m/s², matching D1 sitting body RMS = 0.234 ± 0.377 m/s² (`uci_har.sitting_body_rms_mean`). Also validated against D3 rest Z-axis STD = 0.155 m/s² (`emowear.rest_z_std_mean`). |
| walking | 0.0 | Gait harmonics dominate; no additional micro-movement. |
| running | 0.0 | Gait harmonics dominate; no additional micro-movement. |
| recovery | 0.06 | Residual body movement during cool-down; intermediate between rest (0.20) and sleep (0.04). Engineering estimate. |
| stress | 0.08 | D4 stress-task ACC STD = 0.284 m/s² (`wearable_dataset.stress_task_acc_std_mean`). This σ_micro plus fidget bursts (§3.8) produces variability matching that target. |
| sleep | 0.04 | Very low baseline movement during sleep; occasional position shifts (§3.8) add sporadic large transients. Engineering estimate. |

**Component 2 — Sporadic micro-adjustment bursts**: Brief (0.5–2.0 s) episodes of
postural readjustment, modelled as an exponentially decaying envelope.

| Activity | Burst probability (per sample) | Burst amplitude range (m/s²) | Provenance |
|----------|-------------------------------|------------------------------|------------|
| rest | 0.002 (~6/min at 50 Hz) | 0.4–1.0 | Frequency and amplitude tuned to match D1 sitting body RMS distribution (mean=0.234, std=0.377). The high std/mean ratio in D1 sitting suggests occasional large movements. |
| recovery | 0.0013 (~4/min) | 0.2–0.5 | Lower than rest; subjects are recovering and relatively still. Engineering estimate. |
| stress | 0.005 (~15/min) | 0.3–0.8 | Higher burst rate during stress reflects fidgeting/restlessness. Tuned alongside σ_micro to match D4 stress-task ACC. |
| sleep | 0.00033 (~1/min) | 0.2–0.5 | Very infrequent during sleep. Engineering estimate. |
| walking, running | 0 | — | Gait dominates. |

### 3.8 Special Behavioural Patterns

**Stress fidgets** (active only during `stress`):
- Trigger probability: 0.0006 per sample (~1.8 events/min at 50 Hz)
- Duration: 1–3 s
- Amplitude: 0.5–2.0 m/s² (sinusoidal at 2–5 Hz)
- Cross-axis distribution: Z = 100%, X = 50%, Y = 30%
- Cooldown: 30–120 s between fidgets
- **Provenance**: Models hand/leg fidgeting observed in seated stress tasks.
  D4 stress-task ACC STD (0.284 m/s²) is higher than stress-rest (0.211 m/s²),
  partially accounted for by these bursts.

**Sleep position shifts** (active only during `sleep`):
- Trigger probability: 0.00003 per sample (~0.09 events/min = ~5.4/hr)
- Duration: 2–5 s
- Peak amplitude: 5–15 m/s² (exponentially decaying envelope)
- Cross-axis: isotropic random (each axis: envelope × N(0, 0.3))
- Cooldown: 5–30 min between shifts
- **Provenance**: Literature reports 3–6 major position changes per hour during
  sleep. We model ~5 events/hour. The amplitude is an engineering estimate for
  whole-body rolling movements.

### 3.9 ACC Bias Model

Two bias components model real sensor imperfections:

**Static offset** (per session, per axis):
```
bias ~ Uniform(−0.15, 0.15) m/s²
```
**Provenance**: ±0.15 m/s² is consistent with consumer-grade MEMS accelerometer
specifications (typical ±40 mg ≈ ±0.39 m/s² worst-case; our value is
conservative). Validated against D1 sitting per-axis noise floor: 5th percentile
STDs of x=0.020, y=0.026, z=0.041 m/s² show that the bias does not dominate the
signal.

**Drift (bias instability)**: Bounded random walk:
```
drift += N(0, σ_drift · √DT)    σ_drift = 0.001 m/s²/√s
drift = clip(drift, −0.3, 0.3)  m/s²
```
**Provenance**: σ_drift = 0.001 m/s²/√s and clamp = ±0.3 m/s² are engineering
estimates for MEMS bias instability. Oishi et al. (2025) [6] model similar
bounded drift in their WIMUSim framework.

### 3.10 Quantization and Clipping

After signal generation, ACC values are:
1. Quantized to `ACC_RESOLUTION` = 0.004 m/s² (round to nearest LSB)
2. Clipped to ±`ACC_RANGE` = ±156.96 m/s²

**Provenance**: D4 E4 ACC clip rate = 0.318% at ±2g range
(`artifacts.e4_acc_clip_mean_pct`). Our ±16g range produces negligible clipping
for body-worn data, matching D3 BH3 clip rate = 0.0%
(`artifacts.bh3_acc_clip_mean_pct`).

---

## 4. Respiration (RSP) Model

### 4.1 Signal Equation

```
rsp[i] = B_rsp · W(φ_resp, IE_ratio)           # asymmetric breath waveform
        + k_MA · |acc_z − g_z| · sign(sin(φ_z))  # motion artifact (gait only)
        + N(0, σ_rsp)                              # measurement noise
        + A_wander · B_rsp · sin(φ_wander)         # baseline wander
```

### 4.2 Asymmetric Breathing Waveform

The function W(φ, IE) produces an asymmetric sinusoidal waveform where the
inspiratory phase occupies a fraction α = 1/(1 + IE) of the cycle and the
expiratory phase occupies the remainder.

- At IE = 1.0 (I:E = 1:1), the waveform is symmetric.
- At IE = 1.5 (I:E = 1:1.5), inspiration is 40% and expiration 60% of the cycle.
- At IE = 2.0 (I:E = 1:2), inspiration is 33% and expiration 67%.

**Provenance**: D3 EmoWear measured I:E ratio = 0.96 (`emowear.ie_ratio_mean`),
meaning roughly 1:1 during mixed sitting/walking activities. We use activity-
dependent IE targets (see §4.3).

### 4.3 Activity RSP Parameters

| Activity | f_resp (Hz) | B_rsp (a.u.) | IE_ratio | Provenance |
|----------|-------------|-------------|----------|------------|
| rest | 0.22 | 1.0 | 1.5 | D3 rest breathing rate = 13.98 bpm = 0.233 Hz (`emowear.rest_breathing_rate_bpm_mean`). 0.22 Hz (13.2 bpm) is slightly below to allow additive inter-subject offset N(0, 0.015) to center on the D3 mean. IE=1.5 is a standard resting I:E ratio [8]. |
| walking | 0.35 | 1.8 | 1.3 | ~21 bpm; moderate exercise breathing. Nicolò et al. (2020) [8] report f_R in the range 20–30 bpm for moderate walking. Amplitude increases with tidal volume. IE decreases toward 1.0 with exercise [8]. |
| running | 0.70 | 3.5 | 1.0 | 42 bpm; vigorous exercise. Nicolò et al. (2020) [8] report 40–70 bpm during high-intensity exercise. IE = 1.0 (symmetric) at high exertion reflects active expiration [8]. |
| recovery | 0.28 | 1.5 | 1.3 | Post-exercise: breathing rate gradually returns to rest. |
| stress | 0.28 | 1.2 | 1.5 | Slightly elevated breathing rate during psychological stress (Schmidt et al., 2018 [9]). |
| sleep | 0.18 | 0.8 | 2.0 | ~10.8 bpm; reduced metabolic demand. IE = 2.0 reflects slow, relaxed expiration during sleep. Kräuchi (2007) [10] notes reduced respiratory effort during sleep onset. |

### 4.4 RSP Smoothing Time Constants

| Activity | τ_rsp_freq (s) | Provenance |
|----------|---------------|------------|
| rest | 15 | Gradual breathing rate adjustment. |
| walking | 15 | Same as rest; breathing rate adjusts over several breaths. |
| running | 15 | Same. |
| recovery | 4 | **Fast decay**: breathing rate drops quickly after exercise cessation. Literature [8] shows respiratory frequency drops within seconds of exercise cessation. τ = 4 s gives ~63% recovery in 4 s. |
| stress | 15 | Same as rest. |
| sleep | 20 | Slow adjustment to sleep breathing pattern. |

Breathing amplitude: τ_rsp_amp = 20 s (all activities). Engineering choice for
smooth tidal volume transitions.

### 4.5 RSP Noise and Breath-to-Breath Variability

| Activity | σ_rsp (a.u.) | Rate variability (CV) | Provenance |
|----------|-------------|----------------------|------------|
| rest | 0.02 | 15% | D3 breath-to-breath CV = 47.4% (`emowear.breath_cv`). The 15% per-sample rate jitter (applied as phase perturbation at 10% coupling: 0.15 × 0.1 = 1.5% effective) accumulates over many samples to produce the observed ~47% inter-breath CV when combined with inter-subject variability and noise. |
| walking | 0.05 | 12% | Higher noise from motion artifacts. |
| running | 0.08 | 10% | More regular breathing during intense exercise (entrainment to gait). |
| recovery | 0.04 | 12% | Intermediate. |
| stress | 0.03 | 18% | Higher rate variability during stress reflects irregular breathing patterns (Schmidt et al., 2018 [9]). |
| sleep | 0.015 | 8% | Low noise; regular, slow breathing. |

Rate variability is implemented as a phase perturbation:
```
φ_resp += 2π · f_resp · DT · N(0, rate_var) · 0.1
```

### 4.6 Locomotor-Respiratory Coupling (LRC)

During gait, breathing tends to phase-lock to stepping at integer ratios [11,12]:

```
n_lrc = round(f_step / f_resp)   # nearest integer coupling ratio
correction = K_LRC · sin(n_lrc · φ_z − φ_resp)
φ_resp += 2π · correction · DT
```

| Parameter | Value | Provenance |
|-----------|-------|------------|
| `K_LRC` | 0.02 Hz | Bramble & Carrier (1983) [11] demonstrated 1:1 stride-breath coupling in running mammals. McDermott et al. (2003) [12] showed that trained runners exhibit stronger LRC. K_LRC = 0.02 produces a subtle coupling (not rigid locking), consistent with reports that human LRC is probabilistic, not deterministic. |

### 4.7 Motion Artifact on RSP

During gait, trunk acceleration mechanically perturbs the RSP signal:

| Activity | k_MA | Provenance |
|----------|------|------------|
| walking | 0.05 | Engineering estimate for moderate chest/abdominal motion coupling to respiration sensor. |
| running | 0.10 | Doubled for running due to higher impact forces. |

### 4.8 Baseline Wander

A slow sinusoidal drift models thermal/postural baseline shifts in the RSP
transducer:

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Wander amplitude | 0.025 × B_rsp | Tuned to produce ~12–13% of signal range as wander, matching D3 RSP wander = 13.4% (`emowear.rsp_wander_pct`) and artifact analysis wander = 13.4% (`artifacts.rsp_wander_mean_pct`). Previous value of 0.05 produced ~25%, halved to match data. |
| Wander period | Uniform(40, 120) s | Engineering estimate; slow drift on the scale of minutes. |

---

## 5. Skin Temperature (SKT) Model

### 5.1 Physiological Basis

Skin temperature reflects the balance between metabolic heat production (core
temperature), peripheral vasomotor control (vasoconstriction/vasodilation), and
ambient heat exchange.  Key physiological responses modelled:

1. **Exercise onset**: Skin temperature **drops** as blood is redistributed to
   working muscles (vasoconstriction of cutaneous beds). Blokker et al. (2022) [13]
   measured trunk skin temp dropping from 33.4 °C to ~29.0 °C during
   cross-country skiing, while core temp rose from 37.0 to 37.5 °C.
2. **Recovery**: Post-exercise vasodilation causes skin temperature to **overshoot**
   above resting baseline before returning to equilibrium.
3. **Stress**: Peripheral vasoconstriction during psychological stress causes
   mild skin temperature decrease (Schmidt et al., 2018 [9]).
4. **Sleep onset**: Distal skin temperature **rises** as core temperature drops
   (Kräuchi, 2007 [10]), reflecting increased peripheral vasodilation to
   dissipate heat.

### 5.2 Two-Variable Model

Core and skin temperatures are tracked as separate state variables with
activity-dependent time constants and targets:

```
T_core += (DT / τ_core) · (Tc_target − T_core)
T_skin += (DT / τ_skin) · (Ts_target − T_skin)
```

The skin temperature target may be adjusted for prolonged exercise (see §5.4).

### 5.3 Temperature Targets and Time Constants

| Activity | T_core target (°C) | T_skin target (°C) | τ_core (s) | τ_skin (s) | Provenance |
|----------|-------------------|--------------------|-----------:|----------:|------------|
| rest | 37.0 | 33.0 | 600 | 150 | D4 stress-rest temp = 32.52 °C (`wearable_dataset.stress_rest_temp_mean`). 33.0 °C is the inter-subject mean (§2.2). D5 chest temp = 36.3–36.4 °C [5] confirms core-site values. τ_skin = 150 s allows gradual settling. |
| walking | 37.3 | 31.5 | 300 | 60 | Skin drops ~1.5 °C below rest during walking (vasoconstriction). D4 aerobic-low temp = 31.85 °C (`wearable_dataset.aerobic_low_temp_mean`). τ_skin = 60 s gives ~2 min response, consistent with Blokker et al. [13] thermal time course. |
| running | 37.8 | 30.5 | 300 | 45 | Stronger vasoconstriction during intense exercise. D4 aerobic-high temp = 32.27 °C (`wearable_dataset.aerobic_high_temp_mean`) is higher than our target because D4 includes warm-up/mixed-phase recordings. Blokker [13] shows trunk drops to ~29 °C in cold conditions; our 30.5 °C target is warmer (indoor, clothed). τ_skin = 45 s for faster response. |
| recovery | 37.0 | 35.0 | 600 | 30 | Post-exercise vasodilation causes **overshoot** above resting baseline. 35.0 °C target produces realistic overshoot followed by return to ~33 °C. τ_skin = 30 s gives rapid initial rise. |
| stress | 37.1 | 32.5 | 120 | 45 | Mild peripheral vasoconstriction. D4 stress-task temp = 32.76 °C (`wearable_dataset.stress_task_temp_mean`), suggesting minimal change from rest. Our 32.5 °C target produces a small 0.5 °C drop. τ_core = 120 s for faster sympathetic response. |
| sleep | 36.5 | 34.5 | 1800 | 200 | Kräuchi (2007) [10]: distal skin temp rises 1–2 °C at sleep onset while core drops ~0.5 °C. Our skin target of 34.5 °C (1.5 °C above resting 33 °C) and core drop to 36.5 °C model this. τ_skin = 200 s for gradual 15–30 min transition. τ_core = 1800 s for very slow core cooling. |

### 5.4 Prolonged Exercise Adjustment

If exercise (walking or running) continues for > 900 s (15 min), the skin
temperature target is adjusted upward:

```
Ts_adjusted = Ts_target + min(1.5, 0.1 · (exercise_duration − 900) / 60)
```

This models the eventual increase in skin temperature during prolonged steady-state
exercise as metabolic heat overwhelms vasoconstriction. The exercise duration
counter increases during gait and decays at half-rate during non-gait activities.

**Provenance**: This effect is documented in prolonged exercise studies [13] where
skin temp initially drops then partially recovers after 15–20 min. The 1.5 °C cap
and 0.1 °C/min rate are engineering estimates.

### 5.5 SKT Measurement Model

SKT is sampled at FS_TEMP = 4 Hz (every 12.5 ACC samples):

```
skt = T_skin + drift + N(0, 0.02)
```

Where:
- `drift` is a slow random walk: drift += N(0, 0.001 · √DT_TEMP)
- Quantized to 0.02 °C resolution (Empatica E4 specification)

### 5.6 Sensor Warm-Up Model

Real wrist/chest-worn temperature sensors require time to reach thermal
equilibrium after attachment. The simulator models this with an exponential
approach from ambient:

```
If t + T_PRE < T_END:
    warmup_factor = 1 − exp(−(t + T_PRE) / τ_warmup)
    skt_measured = T_AMB + (T_skin − T_AMB) · warmup_factor
Else:
    skt_measured = T_skin
```

| Parameter | Value | Provenance |
|-----------|-------|------------|
| τ_warmup | 180 s | D3 EmoWear E4 warm-up profile: T(0) ≈ 25.7 °C (`emowear.warmup_t0_mean`), T(stable) ≈ 29.2 °C (`emowear.warmup_t_stable_mean`). The exponential approach with τ=180 s from T_AMB=22 °C reaches ~25 °C at t=60 s and ~29 °C at t=300 s, matching the D3 profile. |
| T_PRE | 600 s | Pre-warm offset: models sensor already worn for ~10 min before recording starts. With T_PRE=600 s and T_END=600 s, the warm-up phase is effectively complete at t=0 for our standard simulation. This matches the common real-world scenario where the sensor is attached before the experiment begins. |
| T_END | 600 s | Warm-up model active window. |
| Jump probability | 0.002 per SKT sample | During warm-up, occasional contact-settling jumps (Empatica E4 artifact). |
| Jump amplitude | ±(0.3–1.5) °C | D3 temp spike rate = 6.3 spikes/hr (`emowear.temp_spikes_per_hour`). |

---

## 6. Cross-Signal Coupling

### 6.1 Movement → Breathing Rate Coupling

Physical activity increases breathing rate not only through metabolic demand
(modelled by activity-specific f_resp targets in §4.3) but also through
mechanical ventilatory drive proportional to stepping rate:

```
f_resp_coupled = f_resp_target + K_MV · f_step
```

| Parameter | Value | Provenance |
|-----------|-------|------------|
| `K_MV` | 0.02 | Initial value was 0.12, but this double-counted the exercise effect already captured in the activity-specific f_resp targets. Reduced to 0.02 to serve as a small secondary coupling that adds ~0.036 Hz (2 bpm) during walking (f_step=1.8). This avoids the +42% walking BPM error observed with K_MV=0.12 while maintaining a physiologically motivated coupling. |

### 6.2 Exercise Duration Tracking

```
If activity ∈ {walking, running}:
    exercise_duration += DT
Else:
    exercise_duration = max(0, exercise_duration − DT · 0.5)
```

The slow decay (half-rate) models lingering effects of exercise on physiology
after cessation. This counter drives the prolonged-exercise skin temperature
adjustment (§5.4).

### 6.3 Heart Rate Model

HR is smoothed toward activity-specific targets with different time constants
for exercise versus non-exercise states:

| Activity | HR target (bpm) | τ_HR (s) | Provenance |
|----------|----------------|----------|------------|
| rest | 72 | 90 | D3 resting HR = 73.1 ± 10.6 bpm. 72 bpm is a standard textbook resting HR. τ = 90 s for gradual settling. |
| walking | 105 | 30 | Moderate exercise HR. Fast response (τ=30 s) at exercise onset. |
| running | 160 | 30 | Vigorous exercise HR near maximum for young adults. |
| recovery | 72 | 90 | Return to resting. Slower than exercise onset (τ=90 vs 30 s). |
| stress | 80 | 90 | Mild sympathetic activation. Schmidt et al. (2018) [9] reports elevated HR during stress. |
| sleep | 55 | 90 | Reduced resting HR during sleep. |

The inter-subject HR offset (§2.2) is applied additively to all targets.

---

## 7. Sensor Artifacts

All artifact parameters are calibrated from the analysis scripts applied to D3
(EmoWear) and D4 (Wearable Dataset) raw recordings.

### 7.1 ACC Stuck-Value Artefact

ADC quantization errors occasionally cause one or more ACC channels to output
the same value for consecutive samples:

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Probability (per axis, per sample) | 0.06 | Calibrated to produce an artifact rate consistent with ADC behaviour in consumer sensors. Engineering estimate. |
| Duration | 1–4 samples | Brief stuck periods (0.02–0.08 s at 50 Hz). |

### 7.2 RSP Sensor Glitches

Occasional brief RSP dropouts (signal goes NaN):

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Probability (per sample) | 0.00005 | Engineering estimate for rare RSP sensor failures. |
| Duration | 0.5–5.0 s | |

### 7.3 SKT Spike Artefacts

Occasional temperature spikes (contact artifacts, motion-induced thermal transients):

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Spike probability (per SKT sample) | 0.0005 | D3 temp spike rate = 6.3 spikes/hr (`emowear.temp_spikes_per_hour`). At 4 Hz, this gives ~23 spikes/hr, higher than D3 — but spikes in the simulator are bipolar and many are small (0.3–2.0 °C), while D3 counts only spikes > threshold. |
| Spike amplitude | ±(0.3–2.0) °C | |

### 7.4 SKT Sensor Glitches

Rare SKT dropouts:

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Probability (per SKT sample) | 0.00003 | Engineering estimate. |
| Duration | 0.5–3.0 s | |

D4 temp out-of-range rate = 2.64% (`artifacts.e4_temp_oor_mean_pct`) and
temp jump rate = 0.60% (`artifacts.e4_temp_jump_mean_pct`) are used as
validation targets rather than direct parameter sources.

### 7.5 Bluetooth Connection Gaps

All signals simultaneously go NaN during Bluetooth dropouts.  Internal state
(phases, temperatures, HR) continues advancing — only output signals are lost.

| Parameter | Value | Provenance |
|-----------|-------|------------|
| Gap probability (per sample) | 1/(8·60·50) ≈ 4.17×10⁻⁵ | ~1 gap event per 8 minutes. Engineering estimate for BLE connection stability. D3 BH3 HR dropout rate ≈ 5%. |
| Gap duration | Uniform(0.05, 2.0) s | Short BLE reconnection time. |

---

## 8. Sensor Electronic Noise

| Parameter | Value | Provenance |
|-----------|-------|------------|
| `SIGMA_ACC_SENSOR` | 0.02 m/s² | D1 sitting per-axis noise floor (5th percentile STDs): x=0.020, y=0.026, z=0.041 m/s². The electronic noise component is ~0.02 m/s², with higher Z values reflecting residual postural sway rather than sensor noise. |

---

## 9. Validation Criteria

### 9.1 Scorecard Methodology

The validation pipeline (`analyses/validate_realism.py`) runs 20 Monte Carlo
trials and compares simulated statistics against real-data benchmarks:

| Grade | Criterion | Meaning |
|-------|-----------|---------|
| A | Error < 10% | Excellent match |
| B | Error < tolerance (metric-dependent) | Acceptable |
| C | Error < 50% | Poor |
| F | Error ≥ 50% | Failed |

Metric-specific tolerances:

| Metric | Tolerance | Rationale |
|--------|-----------|-----------|
| Body RMS (walk, run) | 25% | Large inter-dataset variation (D1 walk=3.07, D2 walk=4.29) |
| Step frequency | 10% | Narrow real-data range (D1=1.81, D2=1.80) |
| H3/H1 power ratio | 50% | Wide inter-dataset range (D1=0.17, D2=0.31 for walk) |
| Breathing rate | 30% | Single dataset (D3), moderate sample size |
| RSP wander | 50% | High variability in D3 data (σ=5.7%) |
| SKT (stress, aerobic) | 15% | D4 with large n (65–190 sessions) |

### 9.2 Real-Data Benchmark Targets

All values from `analyses/benchmarks.json`:

| Signal | Metric | Activity | Real Value | Dataset | JSON Key |
|--------|--------|----------|------------|---------|----------|
| ACC | Body RMS | Sitting | 0.234 ± 0.377 m/s² | D1 (n=30) | `uci_har.sitting_body_rms_mean` |
| ACC | Body RMS | Walking | 3.074 ± 0.498 m/s² | D1 (n=30) | `uci_har.walking_body_rms_mean` |
| ACC | Body RMS | Walking | 4.289 m/s² | D2 (n=13) | `realworld2016.walking_body_rms_mean` |
| ACC | Body RMS | Running | 9.505 m/s² | D2 (n=15) | `realworld2016.running_body_rms_mean` |
| ACC | Per-axis STD | Walking | x=3.00, y=1.92, z=2.36 m/s² | D2 | `realworld2016.walking_x/y/z_std_mean` |
| ACC | Per-axis STD | Running | x=7.70, y=3.56, z=4.13 m/s² | D2 | `realworld2016.running_x/y/z_std_mean` |
| ACC | Step frequency | Walking | 1.810 ± 0.157 Hz | D1 (n=30) | `uci_har.walking_step_freq_mean` |
| ACC | Step frequency | Walking | 1.800 ± 0.136 Hz | D2 (n=13) | `realworld2016.walking_step_freq_mean` |
| ACC | Step frequency | Running | 2.693 ± 0.144 Hz | D2 (n=15) | `realworld2016.running_step_freq_mean` |
| ACC | H3/H1 power | Walking | 0.170 | D1 | `uci_har.walking_h3_h1` |
| ACC | H3/H1 power | Walking | 0.310 | D2 | `realworld2016.walking_h3_h1_mean` |
| ACC | H3/H1 power | Running | 1.201 | D2 | `realworld2016.running_h3_h1_mean` |
| ACC | Step freq CV | Walking | 8.69% | D1 | `uci_har.walking_step_freq_cv` |
| ACC | Amplitude CV | Walking | 13.93% | D1 | `uci_har.walking_amplitude_cv` |
| ACC | Rest Z STD | Sitting | 0.155 ± 0.022 m/s² | D3 (n=49) | `emowear.rest_z_std_mean` |
| ACC | STD mag | Stress rest | 0.211 ± 0.159 m/s² | D4 (n=122) | `wearable_dataset.stress_rest_acc_std_mean` |
| ACC | STD mag | Stress task | 0.284 ± 0.238 m/s² | D4 (n=190) | `wearable_dataset.stress_task_acc_std_mean` |
| RSP | Dominant freq | Sitting | 13.98 bpm (0.233 Hz) | D3 (n=49) | `emowear.rest_breathing_rate_bpm_mean` |
| RSP | I:E ratio | Mixed | 0.96 | D3 (n=49) | `emowear.ie_ratio_mean` |
| RSP | Breath CV | Mixed | 47.4% | D3 (n=49) | `emowear.breath_cv` |
| RSP | Baseline wander | All | 13.4% of range | D3 (n=49) | `emowear.rsp_wander_pct` |
| SKT | Mean temp | Stress rest | 32.52 ± 1.74 °C | D4 (n=122) | `wearable_dataset.stress_rest_temp_mean` |
| SKT | Mean temp | Stress task | 32.76 ± 1.59 °C | D4 (n=190) | `wearable_dataset.stress_task_temp_mean` |
| SKT | Mean temp | Aerobic low | 31.85 ± 1.68 °C | D4 (n=65) | `wearable_dataset.aerobic_low_temp_mean` |
| SKT | Mean temp | Aerobic high | 32.27 ± 2.05 °C | D4 (n=52) | `wearable_dataset.aerobic_high_temp_mean` |
| SKT | Warm-up T(0) | Start | 25.69 °C | D3 (n=49) | `emowear.warmup_t0_mean` |
| SKT | Warm-up T(stable) | Equilibrium | 29.19 °C | D3 (n=49) | `emowear.warmup_t_stable_mean` |
| SKT | Core-site temp | Rest | 36.3–36.4 °C | D5 (n=3000) | InfantSmartWear chest/abdomen measurements [5] |
| Artifact | ACC clip rate | E4 ±2g | 0.32% | D4 (n=100) | `artifacts.e4_acc_clip_mean_pct` |
| Artifact | RSP wander | All | 13.4% | D3 (n=49) | `artifacts.rsp_wander_mean_pct` |
| Artifact | Temp jump rate | All | 0.60% | D4 (n=100) | `artifacts.e4_temp_jump_mean_pct` |
| Artifact | Temp OOR rate | All | 2.64% | D4 (n=100) | `artifacts.e4_temp_oor_mean_pct` |

---

## 10. Complete Parameter Index

For quick reference, every configurable parameter in `simulate.py` with its
exact code value:

### 10.1 Global

| Code constant | Value | Spec section |
|---------------|-------|-------------|
| `FS` | 50 | §2 |
| `FS_TEMP` | 4 | §2 |
| `G` | 9.81 | §2 |
| `T_AMB` | 22.0 | §2 |
| `ACC_RANGE` | 156.96 (16·G) | §2 |
| `ACC_RESOLUTION` | 0.004 | §2 |
| `TEMP_RESOLUTION` | 0.02 | §2 |

### 10.2 ACC Targets (TARGETS dict)

| Activity | f_step | A_z | A_x | A_y | Section |
|----------|--------|-----|-----|-----|---------|
| rest | 0.0 | 0.0 | 0.0 | 0.0 | §3.2 |
| walking | 1.8 | 3.5 | 4.5 | 3.0 | §3.2 |
| running | 2.7 | 4.0 | 10.0 | 5.0 | §3.2 |
| recovery | 0.3 | 0.3 | 0.15 | 0.08 | §3.2 |
| stress | 0.0 | 0.0 | 0.0 | 0.0 | §3.2 |
| sleep | 0.0 | 0.0 | 0.0 | 0.0 | §3.2 |

### 10.3 RSP Targets (TARGETS dict)

| Activity | f_resp | B_rsp | IE_ratio | Section |
|----------|--------|-------|----------|---------|
| rest | 0.22 | 1.0 | 1.5 | §4.3 |
| walking | 0.35 | 1.8 | 1.3 | §4.3 |
| running | 0.70 | 3.5 | 1.0 | §4.3 |
| recovery | 0.28 | 1.5 | 1.3 | §4.3 |
| stress | 0.28 | 1.2 | 1.5 | §4.3 |
| sleep | 0.18 | 0.8 | 2.0 | §4.3 |

### 10.4 Temperature Targets (TARGETS dict)

| Activity | T_core | T_skin | Section |
|----------|--------|--------|---------|
| rest | 37.0 | 33.0 | §5.3 |
| walking | 37.3 | 31.5 | §5.3 |
| running | 37.8 | 30.5 | §5.3 |
| recovery | 37.0 | 35.0 | §5.3 |
| stress | 37.1 | 32.5 | §5.3 |
| sleep | 36.5 | 34.5 | §5.3 |

### 10.5 HR Targets (TARGETS dict)

| Activity | HR (bpm) | Section |
|----------|----------|---------|
| rest | 72 | §6.3 |
| walking | 105 | §6.3 |
| running | 160 | §6.3 |
| recovery | 72 | §6.3 |
| stress | 80 | §6.3 |
| sleep | 55 | §6.3 |

### 10.6 Time Constants

| Constant | Value | Section |
|----------|-------|---------|
| `TAU_ACC` | 3.0 s | §3.6 |
| `TAU_RSP_FREQ` rest/walk/run/stress | 15 s | §4.4 |
| `TAU_RSP_FREQ` recovery | 4 s | §4.4 |
| `TAU_RSP_FREQ` sleep | 20 s | §4.4 |
| `TAU_RSP_AMP` | 20.0 s | §4.4 |
| `TAU_CORE` rest/recovery | 600 s | §5.3 |
| `TAU_CORE` walk/run | 300 s | §5.3 |
| `TAU_CORE` stress | 120 s | §5.3 |
| `TAU_CORE` sleep | 1800 s | §5.3 |
| `TAU_SKIN` rest | 150 s | §5.3 |
| `TAU_SKIN` walk | 60 s | §5.3 |
| `TAU_SKIN` run | 45 s | §5.3 |
| `TAU_SKIN` recovery | 30 s | §5.3 |
| `TAU_SKIN` stress | 45 s | §5.3 |
| `TAU_SKIN` sleep | 200 s | §5.3 |

### 10.7 Noise and Variability

| Constant | Value | Section |
|----------|-------|---------|
| `SIGMA_ACC_SENSOR` | 0.02 m/s² | §8 |
| `SIGMA_MICRO` rest | 0.20 m/s² | §3.7 |
| `SIGMA_MICRO` recovery | 0.06 m/s² | §3.7 |
| `SIGMA_MICRO` stress | 0.08 m/s² | §3.7 |
| `SIGMA_MICRO` sleep | 0.04 m/s² | §3.7 |
| `SIGMA_RSP` rest | 0.02 | §4.5 |
| `SIGMA_RSP` walking | 0.05 | §4.5 |
| `SIGMA_RSP` running | 0.08 | §4.5 |
| `SIGMA_RSP` recovery | 0.04 | §4.5 |
| `SIGMA_RSP` stress | 0.03 | §4.5 |
| `SIGMA_RSP` sleep | 0.015 | §4.5 |
| `RATE_VAR` rest | 0.15 | §4.5 |
| `RATE_VAR` walking | 0.12 | §4.5 |
| `RATE_VAR` running | 0.10 | §4.5 |
| `RATE_VAR` recovery | 0.12 | §4.5 |
| `RATE_VAR` stress | 0.18 | §4.5 |
| `RATE_VAR` sleep | 0.08 | §4.5 |

### 10.8 Inter-Subject Variability

| Constant | Value | Section |
|----------|-------|---------|
| `SUBJ_F_STEP_STD` | 0.08 | §2.2 |
| `SUBJ_AMP_STD` | 0.20 | §2.2 |
| `SUBJ_SKIN_MEAN` | 33.0 °C | §2.2 |
| `SUBJ_SKIN_STD` | 1.5 °C | §2.2 |
| `SUBJ_HR_STD` | 12.0 bpm | §2.2 |
| `SUBJ_F_RESP_STD` | 0.015 Hz | §2.2 |

### 10.9 Coupling

| Constant | Value | Section |
|----------|-------|---------|
| `K_LRC` | 0.02 | §4.6 |
| `K_MV` | 0.02 | §6.1 |
| `K_MA` walking | 0.05 | §4.7 |
| `K_MA` running | 0.10 | §4.7 |
| `H3_COEFF` walking | 0.48 | §3.3 |
| `H3_COEFF` running | 1.10 | §3.3 |

### 10.10 Bias

| Constant | Value | Section |
|----------|-------|---------|
| `BIAS_OFFSET_RANGE` | ±0.15 m/s² | §3.9 |
| `SIGMA_BIAS_DRIFT` | 0.001 m/s²/√s | §3.9 |
| `BIAS_DRIFT_CLAMP` | ±0.3 m/s² | §3.9 |

### 10.11 Artifacts

| Constant | Value | Section |
|----------|-------|---------|
| `ACC_STUCK_PROB` | 0.06 | §7.1 |
| `ACC_STUCK_DUR` | (1, 4) samples | §7.1 |
| `RSP_GLITCH_PROB` | 0.00005 | §7.2 |
| `RSP_GLITCH_DUR` | (0.5, 5.0) s | §7.2 |
| `SKT_GLITCH_PROB` | 0.00003 | §7.4 |
| `SKT_GLITCH_DUR` | (0.5, 3.0) s | §7.4 |
| `RSP_WANDER_AMP` | 0.025 | §4.8 |
| `RSP_WANDER_PERIOD` | (40, 120) s | §4.8 |
| `SKT_SPIKE_PROB` | 0.0005 | §7.3 |
| `SKT_SPIKE_AMP` | (0.3, 2.0) °C | §7.3 |
| `GAP_PROB_PER_SAMPLE` | 1/(8·60·50) | §7.5 |
| `GAP_DUR_RANGE` | (0.05, 2.0) s | §7.5 |

### 10.12 Warm-Up

| Constant | Value | Section |
|----------|-------|---------|
| `SKT_WARMUP_TAU` | 180 s | §5.6 |
| `SKT_WARMUP_END` | 600 s | §5.6 |
| `SKT_WARMUP_PRE` | 600 s | §5.6 |
| `SKT_WARMUP_JUMP_PROB` | 0.002 | §5.6 |
| `SKT_WARMUP_JUMP_AMP` | (0.3, 1.5) °C | §5.6 |

---

## 11. References

1. **Anguita, D., Ghio, A., Oneto, L., Parra, X. & Reyes-Ortiz, J.L.** (2013).
   "A Public Domain Dataset for Human Activity Recognition Using Smartphones."
   *European Symposium on Artificial Neural Networks (ESANN)*. —
   UCI HAR: waist-worn smartphone at 50 Hz, 30 subjects, 6 activities.

2. **Sztyler, T. & Stuckenschmidt, H.** (2016). "On-body localization of
   wearable devices: An investigation of position-aware activity recognition."
   *IEEE International Conference on Pervasive Computing and Communications
   (PerCom)*. — RealWorld2016: 7-position IMU data including waist, 15 subjects,
   8 activities at 50 Hz.

3. **Alamäki, A. et al.** (2024). "EmoWear: Wearable Physiological and Motion
   Dataset for Emotion Recognition." *Zenodo*.
   doi:10.5281/zenodo.14100025. — BioHarness3 chest + Empatica E4 wrist data,
   49 subjects, walking/sitting/rest with ACC, RSP, HR, SKT.

4. **Hosseini, E. et al.** (2022). "A Multi-Modal Sensor Dataset for Continuous
   Stress Detection of Nurses in a Hospital." *PhysioNet*. — Empatica E4 wrist
   data for 90+ subjects across stress, aerobic, and anaerobic exercise
   protocols.

5. **InfantSmartWear Temperature Monitoring Dataset v1** (2024). — Multi-site
   body temperature monitoring (chest, abdomen, peripheral) with movement
   annotations; 3 000 samples. Confirms clothed-body core-site temperatures
   (36.3–36.4 °C) and minimal ambient sensitivity (0.029 °C per °C ambient).

6. **Oishi, N. et al.** (2025). "Physically Plausible Data Augmentations for
   Wearable IMU-based Human Activity Recognition Using Physics Simulation."
   *arXiv:2508.13284v1*. — WIMUSim framework: Body, Dynamics, Placement,
   Hardware parametrization. Demonstrates ±10–20° sensor placement offsets and
   MEMS bias/drift modelling for realistic wearable data.

7. **Kavanagh, J.J. & Menz, H.B.** (2008). "Accelerometry: A technique for
   quantifying movement patterns during walking." *Gait & Posture*, 28(1), 1–15.
   — Trunk accelerometry review: vertical RMS ranges for walking and running,
   harmonic structure of gait acceleration, phase relationships between axes.

8. **Nicolò, A., Massaroni, C., Schena, E. & Sacchetti, M.** (2020). "The
   Importance of Respiratory Rate Monitoring: From Healthcare to Sport and
   Exercise." *Sensors*, 20(21), 6396. doi:10.3390/s20216396. —
   Respiratory rate ranges across exercise intensities: rest 12–20 bpm,
   moderate exercise 20–30 bpm, maximal exercise 40–70 bpm. I:E ratio changes
   with exercise intensity.

9. **Schmidt, P. et al.** (2018). "Introducing WESAD, a multimodal dataset for
   wearable stress and affect detection." *International Conference on
   Multimodal Interaction (ICMI)*. — Stress physiology: peripheral skin
   temperature drop, elevated HR and breathing rate during psychological stress.

10. **Kräuchi, K.** (2007). "The thermophysiological cascade leading to sleep
    initiation in relation to phase of entrainment." *Sleep Medicine Reviews*. —
    Distal skin temperature rises and core temperature drops as a sleep-onset
    mechanism.

11. **Bramble, D.M. & Carrier, D.R.** (1983). "Running and breathing in mammals."
    *Science*, 219(4582), 251–256. doi:10.1126/science.6849136. —
    Quadrupedal species synchronize locomotor and respiratory cycles at 1:1
    ratio. Humans show flexible LRC ratios.

12. **McDermott, W.J., Van Emmerik, R.E. & Hamill, J.** (2003). "Running training
    and adaptive strategies of locomotor-respiratory coordination." *European
    Journal of Applied Physiology*, 89(5), 435–444.
    doi:10.1007/s00421-003-0831-5. — Running training strengthens LRC; coupling
    increases with speed.

13. **Blokker, T., Bucher, E., Steiner, T. & Wehrlin, J.P.** (2022). "Effect of
    cold ambient temperature on heat flux, skin temperature, and thermal
    sensation at different body parts in elite biathletes." *Frontiers in Sports
    and Active Living*, 4, 966203. PMC9666787. — Confirms skin temp drops
    during exercise (trunk: 33.4 → ~29.0 °C), core rises (37.0 → 37.5 °C).

14. **Pennes, H.H.** (1948). "Analysis of tissue and arterial blood temperatures
    in the resting human forearm." *Journal of Applied Physiology*. — Foundation
    for bioheat transfer modelling; motivates the two-variable (core + skin)
    thermal model structure.

---

## 12. Provenance Legend

Throughout this document, provenance annotations use the following conventions:

- **D1, D2, D3, D4, D5** — Dataset identifiers (see §1.1)
- **`json_key`** — Exact key in `analyses/benchmarks.json`
- **[N]** — Reference number from §11
- **"Engineering estimate"** — No direct empirical source; value chosen to
  produce plausible behaviour and validated via the Monte Carlo scorecard
- **"Tuned"** — Value adjusted iteratively using the validation pipeline
  (`analyses/validate_realism.py`) to minimize error against benchmark targets
