# EmoWear Benchmark Report

Comparison of real EmoWear data (participant 27-9VUW, Zephyr BioHarness 3 chest-worn + Empatica E4 wrist) against our `simulate.py` parameters.

## Caveats

| Factor | Detail |
|--------|--------|
| **Sensor placement** | BH3 is **chest-worn** (sternum); our sim models a **waist-worn** device. Chest has lower walking ACC amplitudes and different axis mapping. |
| **Walking type** | EmoWear walks are casual, ~19 s, ~18–19 m around a desk. Not sustained gait. |
| **Activities available** | Only **sitting** and **casual walking**. No exercise, running, stress, or sleep in the sample. |
| **E4 SKT** | Wrist-worn, not waist/trunk. Wrist temps are typically 1–3 °C lower than trunk. |
| **N=1** | Single participant; individual variation is substantial. |

---

## 1. Accelerometer (ACC)

### 1a. Sitting / Rest — ACC magnitude variability

| Metric | Real (BH3 chest) | Sim "rest" | Ratio |
|--------|:-:|:-:|:-:|
| Magnitude STD (m/s²) | **0.100 ± 0.011** | ~0.02 noise | **5×** |
| Quietest segment STD (m/s²) | x=0.154  y=0.085  z=0.164 | 0.02 isotropic | 4–8× |

**Finding:** Real sitting is ~5× noisier than our rest noise floor (σ=0.02 m/s²). Even the quietest video-watching segment shows ~0.085 m/s² on the calmest axis. This is due to breathing motion on the chest strap, micro-movements, and postural sway.

**Recommendation:** Increase `SIGMA_ACC["rest"]` from **0.02 → 0.06–0.10** m/s². The current value produces unnaturally still rest periods. For a waist-worn device (no breathing artifact), 0.05–0.08 would be a reasonable compromise.

### 1b. Walking — ACC

| Metric | Real (BH3 chest) | Sim "walking" | Notes |
|--------|:-:|:-:|:--|
| Magnitude STD (m/s²) | **1.832 ± 0.176** | — | Includes all axes |
| Vertical axis STD (m/s²) | **1.835** | — | BH3 y-axis = gravity |
| Lateral axis STD (m/s²) | **2.010** | — | BH3 x-axis |
| AP axis STD (m/s²) | **1.133** | — | BH3 z-axis |
| Sim A_z (vertical peak) | — | **3.0** | Peak sinusoidal amplitude |
| Sim A_x (lateral peak) | — | **1.5** | |
| Sim A_y (AP peak) | — | **0.8** | |

Sim vertical STD ≈ A_z / √2 ≈ 3.0 / 1.41 ≈ **2.12 m/s²** (close to real 1.835 vertical).  
Sim lateral STD ≈ 1.5 / 1.41 ≈ **1.06** vs real 2.01 (sim is 50% too low).

**Finding:** Our simulated vertical walking amplitude is reasonable (slightly high for chest, appropriate for waist). Lateral axis is underestimated. However, since the BH3 is chest-worn and our sim is waist-worn, direct comparison is approximate. Waist-worn sensors typically show higher vertical and lower lateral amplitudes than chest.

**Recommendation:** These are in the right ballpark for waist-worn. Optionally increase `A_x` walking from **1.5 → 2.0** if targeting a more chest-like location, but for waist this is fine.

### 1c. Step Frequency

| Metric | Real | Sim | |
|--------|:-:|:-:|:-:|
| Step frequency | **1.89 ± 0.04 Hz** | **1.8 Hz** | ✅ Close |
| Range | 1.79–1.98 Hz | fixed 1.8 | |

**Finding:** Our walking step frequency target (1.8 Hz) is an excellent match for casual walking. The real data shows very tight clustering around 1.9 Hz.

**Recommendation:** Optionally adjust `f_step` for walking from **1.8 → 1.9 Hz** to better match the real data. Minor change.

---

## 2. Respiration

| Metric | Real (BH3 BR) | Sim | |
|--------|:-:|:-:|:-:|
| Sitting BR | **17.4 ± 2.5 bpm** | 15 bpm (0.25 Hz) | Sim is ~15% low |
| Walking BR | **17.0 ± 2.0 bpm** | 24 bpm (0.40 Hz) | Sim is 40% high |
| RSP dominant freq (FFT, sitting) | **0.227 Hz = 13.6 bpm** | 0.25 Hz | Close |

**Finding:** Our rest breathing rate (15 bpm) is close to but slightly below the real sitting rate (17.4 bpm). The walking BR (24 bpm sim) is much higher than real (17 bpm), but this makes sense: the EmoWear walks are only 19 seconds of casual desk-walking — too short to elevate breathing rate. The sim's 24 bpm walking target is appropriate for sustained moderate walking (literature supports 20–26 bpm).

**Recommendation:**
- Rest/sitting: Optionally increase from **0.25 Hz (15 bpm) → 0.28 Hz (17 bpm)**. Minor.
- Walking: **Keep at 0.40 Hz (24 bpm)** — the EmoWear walking data is too brief to represent steady-state, and our sim models sustained walking.

---

## 3. Heart Rate

| Metric | Real (BH3 HR) | Sim | |
|--------|:-:|:-:|:-:|
| Sitting HR | **68.0 ± 9.2 bpm** | **68 bpm** | ✅ Exact match |
| Walking HR | **93.1 ± 10.6 bpm** | **105 bpm** | Sim ~13% high |
| HR range (full session) | 0–120 bpm | — | 0 = sensor dropout |

**Finding:** Rest HR matches perfectly. Walking HR is higher in sim, but again the EmoWear walks are only 19 s — not enough to reach steady-state. For sustained walking, 100–110 bpm is expected.

**Recommendation:** **No change needed.** Both rest and walking targets are appropriate. The real walking HR at ~93 bpm after 19 s is consistent with our sim's transient response (τ_HR=30 s), which would still be ramping up at t=19 s.

---

## 4. Skin Temperature (E4 wrist → sim waist)

| Metric | Real (E4 wrist) | Sim "rest" | |
|--------|:-:|:-:|:-:|
| Overall mean | **32.63 ± 1.09 °C** | **33.0 °C** | Close |
| Sitting segments | **32.73 ± 0.50 °C** | **33.0** | Sim slightly high |
| Walking segments | **32.71 ± 0.48 °C** | **31.5** | N/A (walks too short) |
| First 5 min | **29.97 °C** | — | Sensor warming up |
| Last 5 min | **33.64 °C** | — | Equilibrated |

**Finding:** The E4 wrist temperature baseline (~33 °C after equilibration) is remarkably close to our sim's rest target (33.0 °C). Wrist and waist trunk temperatures are often similar at rest (both ~33 °C). The walking segments show no temperature change because they're only 19 seconds long — far too short for thermoregulation to manifest.

**Recommendation:** **No change needed.** The rest baseline is validated. Exercise/sleep temperature dynamics can't be tested with this dataset.

---

## Summary of Recommended Changes

| Parameter | Current | Recommended | Priority | Rationale |
|-----------|---------|-------------|----------|-----------|
| `SIGMA_ACC["rest"]` | 0.02 | **0.06** | **High** | Real sitting is 5× noisier; sim rest looks unnaturally clean |
| `SIGMA_ACC["sleep"]` | 0.015 | **0.03** | Medium | Sleep should also be noisier than current |
| `SIGMA_ACC["stress"]` | 0.02 | **0.06** | Medium | Same reasoning as rest |
| `f_step` walking | 1.8 Hz | **1.9 Hz** | Low | Slightly closer to real casual walking |
| `f_resp` rest | 0.25 Hz (15 bpm) | **0.28 Hz (17 bpm)** | Low | Closer to real resting BR |

### What's already well-calibrated ✓

- **Walking step frequency** (1.8 Hz sim vs 1.89 Hz real)
- **Resting heart rate** (68 bpm sim vs 68.0 bpm real — exact match)
- **Walking ACC amplitude** (right ballpark for waist-worn)
- **Resting skin temperature** (33.0 °C sim vs 32.7 °C real)
- **Overall ACC dynamic range** (sitting 0.1 vs walking 1.8 m/s² — order-of-magnitude ratio is correct)

### What can't be validated with this dataset ✗

- Running/exercise amplitudes and step frequency
- Stress fidgets
- Sleep position shifts
- Skin temperature drop during exercise
- Prolonged-exercise temperature rebound
- HR > 120 bpm responses

---

## Using the Data as a Benchmark

The EmoWear sample is most useful for:

1. **Rest/sitting baseline calibration** — validate noise floor, HR, BR, and SKT at rest
2. **Casual walking signature** — step frequency, walking-to-sitting amplitude ratio, HR transient onset
3. **Transition dynamics** — 38 sit→walk→sit cycles provide excellent transition data for testing smoothing time constants

It is NOT useful for benchmarking: running, sustained exercise, stress, sleep, or thermoregulatory responses.

To make the simulated data more realistic for sitting/walking, applying the **high-priority** noise floor fix (`SIGMA_ACC` rest → 0.06) would be the single most impactful change.
