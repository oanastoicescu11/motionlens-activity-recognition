# Benchmark Report

Comparison of real wearable data against `simulate.py` output, calibrated
across 5 datasets.  All numbers in this document come from the most recent
20-trial validation run (`analyses/validate_realism.py` →
`output/validation_output.txt`).

> **Last validated:** 20-trial statistical validation scorecard:
> **7A / 4B / 0C / 0F**.  All range checks PASS.  3 of 4 physiological
> direction checks PASS (recovery overshoot FAIL — see §4).

## Caveats

| Factor | Detail |
|--------|--------|
| **Sensor placement** | BH3 is **chest-worn** (sternum); simulator models a **waist-worn** device.  Chest has lower walking ACC amplitudes and different axis mapping. |
| **Walking type** | EmoWear walks are casual, ~19 s around a desk.  Not sustained gait. |
| **Activities available** | EmoWear: only **sitting** and **casual walking**.  No exercise, running, stress, or sleep.  Other datasets provide running/stress/aerobic benchmarks. |
| **E4 SKT** | Wrist-worn, not waist/trunk.  Wrist temps are typically 1–3 °C lower than trunk. |
| **Cross-dataset** | Real references span chest (EmoWear), waist (UCI HAR, RealWorld), wrist (Wearable Dataset E4).  Body-position differences cause systematic offsets. |

---

## 1. Accelerometer (ACC)

### 1a. Sitting / Rest — ACC magnitude variability

| Metric | Real | Sim "rest" | Status |
|--------|:-:|:-:|:-:|
| Z STD (m/s², EmoWear BH3 chest n=49) | **0.155 ± 0.022** | 0.201 ± 0.047 | C grade (+30%) |
| X STD (m/s²) | — | 0.108 ± 0.021 | No benchmark |
| Y STD (m/s²) | — | 0.072 ± 0.012 | No benchmark |
| Body RMS (m/s², UCI HAR waist n=30) | **0.234 ± 0.377** | 0.239 ± 0.052 | A grade (+2%) |

Real references:
- Z STD: `benchmarks.json → emowear.rest_z_std_mean` = 0.1546, from BH3 chest
  quiet-window Z-axis STD (windows with body RMS < 0.5 m/s²).
- Body RMS: `benchmarks.json → uci_har.sitting_body_rms_mean` = 0.2338, from
  gravity-subtracted `body_acc` data across ~1776 sitting windows (30 subjects,
  train+test).

Simulator parameters: `SIGMA_MICRO["rest"]` = 0.20 m/s² postural sway with
2.0 Hz LPF cutoff (`simulate.py`).  Rest Z STD is 30% above EmoWear chest —
note the EmoWear measurement is **chest-worn**, not waist.  Body RMS matches
UCI HAR waist within 2%.  Rest ACC metrics are **not scored** in the final
scorecard — the C grade on Z STD is informational only.

### 1b. Walking — ACC

| Metric | Real | Sim "walking" | Status |
|--------|:-:|:-:|:--|
| Body RMS (m/s², RealWorld waist n=13) | **4.289** | 4.367 ± 1.19 | A grade (+2%) |
| Per-axis X STD (RealWorld waist) | **3.003** | 2.915 ± 0.80 | A grade (−3%) |
| Per-axis Y STD (RealWorld waist) | **1.922** | 1.944 ± 0.53 | A grade (+1%) |
| Per-axis Z STD (RealWorld waist) | **2.359** | 2.607 ± 0.71 | B grade (+11%) |
| H3/H1 power ratio | **0.170** (UCI, avg 3 axes) | 0.200 ± 0.004 | B grade (+18%) |

Real references (all from `benchmarks.json → realworld2016`):
- `walking_body_rms_mean` = 4.2885 — per-axis gravity-subtracted RMS,
  waist-mounted, 13 subjects.
- `walking_x/y/z_std_mean` = 3.003 / 1.922 / 2.359 — raw per-axis STD.
- H3/H1: `uci_har.walking_h3_h1` = 0.1700 — average of H3/H1 across x, y, z
  axes on the mean PSD of all walking windows (30 subjects, UCI HAR).

Simulator parameters: `TARGETS["walking"]` = `(f_step=1.8, A_z=3.5, A_x=4.5,
A_y=3.0)`.  `H3_COEFF["walking"]` = 0.48 (power ratio = 0.48² ≈ 0.23).

**Note on H3/H1 methodology:** The real-data value (0.170) uses **single-bin
peak PSD**, while the simulator validation uses **band power** (sum over
±0.3 Hz).  This methodological difference accounts for most of the +18% error.
RealWorld2016 H3/H1 is 0.310 — so the sim value of 0.200 sits between the two
datasets.

### 1c. Step Frequency

| Metric | Real | Sim | Status |
|--------|:-:|:-:|:-:|
| Walk step freq (UCI HAR waist n=30) | **1.810 ± 0.157 Hz** | **1.760 ± 0.17 Hz** | A grade (−3%) |
| Walk step freq (RealWorld waist n=13) | **1.800 ± 0.136 Hz** | **1.760** | Close |
| Run step freq (RealWorld waist n=15) | **2.693 ± 0.144 Hz** | **2.643 ± 0.25 Hz** | A grade (−2%) |
| Inter-subject step freq CV | **7.5–8.7%** | **9.7%** | Slightly above range |
| Inter-subject amplitude CV | **14–24%** | **27.2%** | Above range |

Real references:
- Walking: `uci_har.walking_step_freq_mean` = 1.8097 Hz (n=30 subjects,
  per-window FFT peak in 1.2–3.5 Hz, per-subject means, then grand mean).
  `uci_har.walking_step_freq_cv` = 8.686%.
- Running: `realworld2016.running_step_freq_mean` = 2.6934 Hz (n=15, Welch
  PSD with 5 s segments, peak in 1.5–5.0 Hz).

Simulator parameters: `TARGETS["walking"][0]` = 1.8 Hz,
`TARGETS["running"][0]` = 2.7 Hz.  `SUBJ_F_STEP_STD` = 0.08 (CV ≈ 8%).
`SUBJ_AMP_STD` = 0.20 (CV ≈ 20%).

Step freq CV (9.7%) is slightly above the 7.5–8.7% real range.  Amplitude CV
(27.2%) exceeds the 14–24% real range — the combination of per-run amplitude
randomization (`SUBJ_AMP_STD` = 0.20) and within-run gait jitter inflates
the variability.

---

## 2. Respiration

| Metric | Real | Sim | Status |
|--------|:-:|:-:|:-:|
| Rest BPM (EmoWear sitting n=49) | **14.0 bpm** | **14.6 ± 1.3 bpm** | A grade (+5%) |
| Walking BPM (literature typical 21) | **18–30 bpm** | **20.7 ± 1.5 bpm** | A grade (−2%) |
| Running BPM (literature typical 42) | **30–55 bpm** | **33.7 ± 1.5 bpm** | B grade (−20%) |
| Recovery BPM (literature typical 18) | **13–25 bpm** | **18.5 ± 1.3 bpm** | A grade (+3%) |
| Stress BPM (literature typical 17) | **13–22 bpm** | **15.7 ± 1.4 bpm** | A grade (−8%) |
| Sleep BPM (literature typical 10.8) | **8–14 bpm** | **12.0 ± 1.3 bpm** | B grade (+11%) |
| RSP baseline wander | **13.4%** (EmoWear n=49) | **14.5%** | A grade (+8%) |
| I:E ratio (rest) | **1:1.04** (EmoWear n=49) | **1:1.5** | Sim uses stronger asymmetry |

Real references:
- Rest BPM: `emowear.rest_breathing_rate_bpm_mean` = 13.976 BPM — spectral
  peak in 0.1–0.8 Hz on 60 s windows of BH3 breathing waveform (25 Hz),
  averaged across 49 subjects.
- RSP wander: `emowear.rsp_wander_pct` = 13.388% — 60 s moving-average
  range / signal range, 49 subjects.
- I:E ratio: `emowear.ie_ratio_mean` = 0.960 (above-mean / below-mean sample
  count per 60 s window).
- Walking/running/recovery/stress/sleep BPM: literature-based typical values,
  hard-coded in `validate_realism.py`'s `rsp_ref` dict.  Not from the
  analysis scripts.

Simulator parameters: `TARGETS[act][4]` for `f_resp` (Hz):
rest=0.22, walking=0.35, running=0.70, recovery=0.28, stress=0.28, sleep=0.18.
`SUBJ_F_RESP_STD` = 0.015 Hz.  `K_MV` = 0.02 (movement → breathing coupling).
`RSP_WANDER_AMP` = 0.025.

All 6 activities pass their physiological range checks every trial.  Running
BPM (33.7) is B grade at −20% vs the literature typical of 42 — the sim
target is 0.70 Hz = 42 bpm, but Welch PSD extraction on the 30 s running
segment (1500 samples) resolves a lower spectral peak due to the short window.

---

## 3. Heart Rate

| Metric | Real | Sim | Status |
|--------|:-:|:-:|:-:|
| Sitting HR (EmoWear) | **68.0 ± 9.2 bpm** | **72 bpm** | Close |
| Sitting HR (WD stress-rest n=122) | **79.4 ± 11.9 bpm** | **72 bpm** | Within 1 SD |
| Walking HR | **93.1 ± 10.6 bpm** (EmoWear, 19 s) | **105 bpm** | Expected for sustained walking |

Simulator parameters: `TARGETS["rest"][9]` = 72 bpm,
`TARGETS["walking"][9]` = 105 bpm.  `SUBJ_HR_STD` = 12 bpm.  HR is
**internal only** — it drives breathing rate coupling but is not in the
simulator output signal.

---

## 4. Skin Temperature

| Metric | Real | Sim | Status |
|--------|:-:|:-:|:-:|
| Rest (WD stress-rest n=122) | **32.52 ± 1.74 °C** | **33.15 ± 0.61 °C** | A grade (+2%) |
| Walking (WD aerobic-low n=65) | **31.85 ± 1.68 °C** | **32.39 ± 0.75 °C** | A grade (+2%) |
| Stress task (WD n=190) | **32.76 ± 1.59 °C** | **33.50 ± 0.04 °C** | A grade (+2%) |
| Running SKT | — | **31.66 ± 0.36 °C** | No benchmark |
| Recovery SKT | — | **32.97 ± 0.14 °C** | No benchmark |
| Sleep SKT | expect 33–36 °C | **33.33 ± 0.02 °C** | In range |
| Exercise drops temp | rest > walk > run | 33.15 > 32.39 > 31.66 | PASS |
| Recovery overshoot | Expected > rest | 32.97 < 33.15 | **FAIL** |
| Sleep is warm | Expected > rest | 33.33 > 33.15 | PASS |

Real references (all from `benchmarks.json → wearable_dataset`):
- `stress_rest_temp_mean` = 32.522 °C (n=122 segments, wrist E4).
- `stress_task_temp_mean` = 32.762 °C (n=190 segments).
- `aerobic_low_temp_mean` = 31.853 °C (n=65 segments, 70–75 RPM cycling).

Simulator parameters: `TARGETS[act][8]` for `T_skin` (°C):
rest=33.0, walking=31.5, running=30.5, recovery=35.0, stress=32.5, sleep=34.5.
`SUBJ_SKIN_STD` = 1.5 °C.  `TAU_SKIN` = rest:150, walking:60, running:45,
recovery:30, stress:45, sleep:200 (seconds).

**Recovery overshoot failure:** The recovery target T_skin is 35.0 °C
(above rest's 33.0 °C), but the 40 s recovery segment with
`TAU_SKIN["recovery"]` = 30 s does not always allow full convergence.  When
per-run skin baseline randomization shifts rest temperature upward, recovery
may not exceed rest.  This is a known edge case that would improve with longer
simulation durations.

Warm-up trajectory (from validation output):

| Time | Sim | EmoWear E4 real |
|------|-----|-----------------|
| t = 0 s | 32.72 °C | ~26 °C |
| t = 120 s | 32.77 °C | ~26 °C |
| t = 300 s | NaN | ~29 °C |
| t = 600 s | NaN | ~29 °C |

The simulator starts near body temperature (pre-warmed sensor assumption:
`SKT_WARMUP_PRE` = 600 s), so the warm-up trajectory does not match the
cold-start E4 pattern.  The t = 300 s and t = 600 s values are NaN because
the simulation is only 270 s long.

---

## 5. Running (RealWorld2016 waist, n=15)

| Metric | Real | Sim | Status |
|--------|:-:|:-:|:-:|
| Body RMS (m/s²) | **9.505** | **8.367 ± 2.26** | B grade (−12%) |
| X STD (m/s²) | **7.696** | **6.506 ± 1.75** | B grade (−15%) |
| Y STD (m/s²) | **3.555** | **3.291 ± 0.89** | A grade (−7%) |
| Z STD (m/s²) | **4.133** | **4.104 ± 1.11** | A grade (−1%) |
| Step freq (Hz) | **2.693 ± 0.144** | **2.643 ± 0.25** | A grade (−2%) |
| H3/H1 power ratio | **1.201** | **0.927 ± 0.05** | B grade (−23%) |

Real references (all `benchmarks.json → realworld2016`):
- `running_body_rms_mean` = 9.5048 — per-axis gravity-subtracted RMS.
- `running_x/y/z_std_mean` = 7.696 / 3.555 / 4.133.
- `running_h3_h1_mean` = 1.2010 — Z-axis Welch PSD, single-bin H3 peak / H1
  peak, per subject.

Simulator parameters: `TARGETS["running"]` = `(f_step=2.7, A_z=4.0,
A_x=10.0, A_y=5.0)`.  `H3_COEFF["running"]` = 1.10.

Run body RMS (8.367) is 12% below the real value (B grade with 25% tolerance).
The X STD deficit (−15%) is the main contributor.  H3/H1 (0.927 vs 1.201)
shows −23% error (B grade with 50% tolerance).  The same peak-vs-band
methodology difference applies here.

---

## 6. Artifacts

| Metric | Sim | Real range | Status |
|--------|-----|-----------|--------|
| BT gap rate (% samples) | 0.24 ± 0.30 | ~0.5–2% | PASS |
| RSP dropout rate (%) | 1.13 ± 0.89 | ~0–3% | PASS |
| SKT NaN rate (%) | 0.24 ± 0.28 | ~0–3% | PASS |
| SKT spikes/hr (>0.3 °C) | 9.3 ± 10.4 | ~6–12 /hr | B grade (+39% vs 6.7) |
| ACC stuck-value rate (%) | 14.1 ± 0.4 | 7–14% | B grade (+41% vs 10) |

Real references:
- SKT spikes: `emowear.temp_spikes_per_hour` = 6.338 /hr (EmoWear E4,
  consecutive-sample jumps > 0.5 °C).
- ACC stuck-value: EmoWear BH3 observation range 7.8–14% (ADC quantization
  repeat rate).

Simulator parameters: `SKT_SPIKE_PROB` = 0.0005 per temp sample.
`ACC_STUCK_PROB` = 0.06, `ACC_STUCK_DUR` = (1, 4) samples.

---

## Summary: Multi-Trial Validation Scorecard (20 trials × 270 s)

| Metric | Sim mean | Real | Error | Tol | Grade |
|--------|----------|------|-------|-----|-------|
| Walk body RMS (m/s²) | 4.367 | 4.289 | +2% | 25% | **A** |
| Walk step freq (Hz) | 1.760 | 1.810 | −3% | 10% | **A** |
| Walk H3/H1 | 0.200 | 0.170 | +18% | 50% | **B** |
| Run body RMS (m/s²) | 8.367 | 9.505 | −12% | 25% | **B** |
| Run step freq (Hz) | 2.643 | 2.693 | −2% | 10% | **A** |
| Run H3/H1 | 0.927 | 1.201 | −23% | 50% | **B** |
| Rest BPM | 14.6 | 14.0 | +5% | 30% | **A** |
| Sleep BPM | 12.0 | 10.8 | +11% | 30% | **B** |
| RSP wander (%) | 14.5 | 13.4 | +8% | 50% | **A** |
| Rest SKT (°C) | 33.15 | 32.52 | +2% | 15% | **A** |
| Stress SKT (°C) | 33.50 | 32.76 | +2% | 15% | **A** |

**Overall: 7A / 4B / 0C / 0F** — all range checks PASS, 3 of 4 direction
checks PASS.

### Grading thresholds (from `validate_realism.py`)

```python
def grade(sim, real, tol=25):
    err = abs(sim - real) / abs(real) * 100
    if err < 10:  return "A"
    if err < tol: return "B"   # tol varies per metric
    if err < 50:  return "C"
    else:         return "F"
```

| Metric | B threshold (tol) |
|--------|------------------|
| Walk/Run body RMS | 25% |
| Walk/Run step freq | 10% |
| Walk/Run H3/H1 | 50% |
| Rest/Sleep BPM | 30% |
| RSP wander | 50% |
| Rest/Stress SKT | 15% |

### Issues from validation output

- **Recovery SKT overshoot FAIL:** recovery=32.97 °C < rest=33.15 °C.  The
  40 s recovery segment with TAU_SKIN=30 s does not reliably converge to the
  35.0 °C target when the per-run skin baseline is randomized upward.

### Other observations (not scored)

- **Rest Z STD** = 0.201, +30% above EmoWear chest (0.155).  Graded C in the
  detailed table but **not in the scorecard**.  Note: chest vs. waist placement
  difference.
- **Rest body RMS** = 0.239, +2% above UCI HAR sitting (0.234).  A grade in the
  detailed table.
- **Stress body RMS** = 0.310, +9% above WD stress-task (0.284).  A grade.
- **Step freq CV** = 9.7%, slightly above real range (7.5–8.7%).
- **Amplitude CV** = 27.2%, above real range (14–24%).

### Known Limitations

- **H3/H1 ratios** show +18% to −23% error — driven by methodology difference
  (band power in validation vs single-bin peak PSD in real-data analysis; see
  `benchmark-methodology.md` §11).
- **Run body RMS** is −12% (B grade) — the X-axis STD (6.506 vs 7.696) is
  the main deficit.
- **Sleep BPM** = 12.0 vs literature 10.8 (+11%, B grade) — within the
  [8, 14] range every trial.
- **Running BPM** = 33.7 vs literature 42 (−20%, B grade) — short 30 s segment
  limits spectral resolution for Welch PSD.
- **Recovery overshoot** occasionally fails due to per-run skin baseline
  randomization and short simulation duration (270 s total).
- **Amplitude CV** (27.2%) exceeds real range (14–24%) due to combined per-run
  (`SUBJ_AMP_STD` = 0.20) and within-run gait jitter.

### What's Well-Calibrated

- **Step frequencies** — walking (−3%) and running (−2%) within 3% of measured
  waist data across two independent datasets (UCI HAR n=30, RealWorld n=15).
- **Walk body RMS** — +2% of RealWorld waist per-axis formula (A grade).
- **Per-axis walk STDs** — X within 3%, Y within 1%, Z within 11%.
- **Skin temperature** — rest +2%, stress +2%, walking +2% of Wearable Dataset
  (n=122–190 segments).  Exercise-drops and sleep-warm direction checks PASS.
- **Breathing rates** — rest A (+5%), recovery A (+3%), stress A (−8%).  All 6
  activities within physiological range checks.
- **RSP wander** — 14.5% vs real 13.4% (A grade, +8%).
- **Artifact rates** — BT gaps, RSP dropout, SKT NaN all within expected
  ranges.

---

## Using the Data as Benchmarks

The 5-dataset calibration suite provides:

1. **Rest/sitting baseline** — noise floor, HR, BR, SKT from EmoWear (n=49) +
   UCI HAR (n=30) + Wearable Dataset (n=122+).
2. **Walking gait** — step frequency, per-axis amplitudes, harmonic structure
   from UCI HAR + RealWorld (n=43 waist subjects combined).
3. **Running gait** — step frequency, per-axis amplitudes, H3/H1 from
   RealWorld (n=15 waist subjects).
4. **Stress physiology** — SKT, ACC variability from Wearable Dataset (n=190+
   stress sessions).
5. **Skin temperature** — absolute values, exercise direction, warm-up profile
   from Wearable Dataset + EmoWear E4.
6. **Artifact profiles** — BT gaps, stuck values, RSP dropout, SKT spikes from
   ~100 real recordings.
7. **Respiration** — breathing rate, baseline wander, I:E ratio from EmoWear
   BH3 (n=49).

Remaining gaps: sleep physiology (no real sleep data in any dataset), prolonged
exercise >15 min rebound, HR >120 bpm sustained, running skin temperature.
