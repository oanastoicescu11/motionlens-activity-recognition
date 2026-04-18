# Benchmark Extraction & Validation Methodology

> **Version 2** — every value traced to the originating code and line-level
> reference.  All numbers come from `analyses/benchmarks.json`, which is
> produced by five analysis scripts and consumed by one validation script.
> Nothing in this document is assumed; every claim can be verified against the
> referenced source file.

```
5 analysis scripts  →  benchmarks.json  →  validate_realism.py  →  scorecard
```

---

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Dataset 1: UCI HAR (Accelerometer, Waist)](#2-dataset-1-uci-har)
3. [Dataset 2: RealWorld2016 (Accelerometer, Waist)](#3-dataset-2-realworld2016)
4. [Dataset 3: EmoWear (Chest ACC + Respiration + Wrist Temperature)](#4-dataset-3-emowear)
5. [Dataset 4: Wearable Dataset (Wrist HR + Temperature + ACC)](#5-dataset-4-wearable-dataset)
6. [Dataset 5: Artifact Analysis (EmoWear + Wearable Dataset)](#6-dataset-5-artifact-analysis)
7. [How the Validation Runs Simulations](#7-how-the-validation-runs-simulations)
8. [How Each Metric Is Computed on Simulated Data](#8-how-each-metric-is-computed-on-simulated-data)
9. [Grading System](#9-grading-system)
10. [Activity Mapping: Real Data → Simulator Activities](#10-activity-mapping)
11. [Known Limitations and Cross-Dataset Caveats](#11-known-limitations)

---

## 1. Pipeline Overview

### Data flow

Each analysis script processes one real-world dataset, computes aggregate
statistics, and writes its results into a shared JSON file
(`analyses/benchmarks.json`):

| Script | Writes to key |
|--------|---------------|
| `analyses/analyze_uci_har.py` | `benchmarks.json["uci_har"]` |
| `analyses/analyze_realworld2016.py` | `benchmarks.json["realworld2016"]` |
| `analyses/analyze_emowear_local.py` | `benchmarks.json["emowear"]` |
| `analyses/analyze_wearable_dataset.py` | `benchmarks.json["wearable_dataset"]` |
| `analyses/analyze_artifacts.py` | `benchmarks.json["artifacts"]` |

Each script follows the same save pattern (example from
`analyze_uci_har.py`):

```python
bench_path = os.path.join(os.path.dirname(__file__), "benchmarks.json")
if os.path.exists(bench_path):
    with open(bench_path) as f:
        all_bench = json.load(f)
else:
    all_bench = {}
all_bench["uci_har"] = benchmarks
with open(bench_path, "w") as f:
    json.dump(all_bench, f, indent=2)
```

`analyses/validate_realism.py` then:

1. Loads all benchmark values from `benchmarks.json` into five dicts:
   ```python
   bench = json.load(f)
   uci = bench["uci_har"]
   rw  = bench["realworld2016"]
   emo = bench["emowear"]
   wd  = bench["wearable_dataset"]
   artf = bench["artifacts"]
   ```
2. Runs `N = 20` Monte Carlo trials of the simulator (each 270 s).
3. Extracts the same metrics from simulated data.
4. Computes percentage error: `err = (sim − real) / real × 100`.
5. Assigns a letter grade per metric.
6. Produces a scorecard and issue list, written to
   `output/validation_output.txt`.

---

## 2. Dataset 1: UCI HAR

**Script:** `analyses/analyze_uci_har.py`
**Source:** UCI Human Activity Recognition Using Smartphones
**Sensor:** Samsung Galaxy SII on waist, 50 Hz, ±2 g accelerometer
**Subjects:** 30
**Activities:** Walking, Walking Upstairs, Walking Downstairs, Sitting,
Standing, Laying
**Data format:** Pre-windowed 128-sample segments (2.56 s each at `FS = 50.0`
Hz); both `total_acc` (raw with gravity) and `body_acc` (gravity-subtracted by
the dataset authors via a Butterworth filter)

### Loading

The script loads train+test splits via `load_inertial(split, signal_name,
axis)`, which reads files of the form
`{BASE}/{split}/Inertial Signals/{signal_name}_{axis}_{split}.txt`.

Both `total_acc` and `body_acc` arrays are loaded for x, y, z and stacked
(`np.vstack`) across train and test.  Labels and subject IDs are concatenated
similarly.

All values are originally in units of *g*; the script converts to m/s² with
`G = 9.81`:

```python
total_x_ms2 = total_x * G   # etc. for all axes
body_x_ms2  = body_x  * G
```

### What is extracted

#### Sitting body RMS → `sitting_body_rms_mean`

This is the benchmark for the **simulator rest/sitting** movement level.

For every 128-sample window whose label is `4` (SITTING):

```python
rms_body = np.sqrt(np.mean(body_x_ms2[i]**2 + body_y_ms2[i]**2 + body_z_ms2[i]**2))
```

The body-acceleration arrays (`body_x_ms2`, `body_y_ms2`, `body_z_ms2`) are
the dataset-provided gravity-subtracted signals, already in m/s².

The mean is taken across all sitting windows across all 30 subjects in the
combined train+test split.

**Saved value:** `benchmarks.json → uci_har.sitting_body_rms_mean` =
**0.2338 m/s²** (with `sitting_body_rms_std` = 0.3766).

#### Walking body RMS → `walking_body_rms_mean`

Identical formula to sitting but using windows labeled `1` (WALKING).  This
value is stored in benchmarks.json but is **not used for scorecard grading**
(RealWorld2016 provides the walking body RMS used for grading because it also
covers running).

**Saved value:** `uci_har.walking_body_rms_mean` = **3.074 m/s²**.

#### Walking step frequency → `walking_step_freq_mean`

For each 128-sample walking window, the script:

1. Mean-subtracts each of the three `total_acc` axes.
2. Computes the FFT power spectrum: `psd = np.abs(rfft(sig_c))**2`.
3. Searches each axis for the strongest spectral peak in **1.2–3.5 Hz**:
   ```python
   f_mask = (freqs >= 1.2) & (freqs <= 3.5)
   peak_idx = np.argmax(psd[f_mask])
   ```
4. Picks the axis with the highest peak power and records that frequency.

This is done per window.  Then **per-subject means** are computed (section 4 of
the script), and CV is computed across those per-subject means:

```python
sf_vals = np.array(list(per_subj_step_freq.values()))
```

**Saved value:** `uci_har.walking_step_freq_mean` = **1.810 Hz** (±0.157 Hz).

#### Walking H3/H1 harmonic ratio → `walking_h3_h1`

Section 7 of the script computes an **average PSD across all walking windows**
for each axis:

```python
for i in np.where(walk_mask)[0]:
    sig_c = sig - np.mean(sig)
    psd = np.abs(rfft(sig_c))**2
    avg += psd / n_walk
```

Then per axis:

1. The fundamental `f1` is the strongest peak in **1.2–3.5 Hz**.
2. H3 power is the **single-bin maximum** PSD in
   `[f1 × 2.8, f1 × 3.2]`:
   ```python
   f3_mask = (freqs >= f1*2.8) & (freqs <= f1*3.2)
   p3 = np.max(avg_psd[f3_mask])
   ```
3. `H3/H1 = p3 / p1`.

The final value **averages H3/H1 across the x, y, z axes**:

```python
h3_h1_vals = [v["h3_h1"] for v in walk_harmonics.values()]
walk_h3_h1_mean = float(np.mean(h3_h1_vals))
```

**Saved value:** `uci_har.walking_h3_h1` = **0.170**.

#### Inter-subject variability → `walking_step_freq_cv`, `walking_amplitude_cv`

Section 4 of the script computes per-subject means of step frequency and body
RMS for walking.  The coefficient of variation is:

```python
CV = np.std(vals) / np.mean(vals) * 100
```

**Saved values:**
- `uci_har.walking_step_freq_cv` = **8.686%**
- `uci_har.walking_amplitude_cv` = **13.926%**

Note: `walking_amplitude_cv` uses the per-subject body RMS values from
`per_subj_body_rms`, which are computed using `body_acc` (gravity-subtracted)
windows.

#### Full `uci_har` key in benchmarks.json

```json
"uci_har": {
  "sitting_body_rms_mean": 0.2338,
  "sitting_body_rms_std": 0.3766,
  "walking_body_rms_mean": 3.074,
  "walking_body_rms_std": 0.4984,
  "walking_step_freq_mean": 1.810,
  "walking_step_freq_std": 0.1572,
  "walking_h3_h1": 0.170,
  "walking_step_freq_cv": 8.686,
  "walking_amplitude_cv": 13.926
}
```

---

## 3. Dataset 2: RealWorld2016

**Script:** `analyses/analyze_realworld2016.py`
**Source:** RealWorld (HAR) dataset, University of Mannheim
**Sensor:** Smartphone on waist (and chest), ~50 Hz accelerometer
**Subjects:** 15 (proband1–proband15)
**Activities:** walking, running, sitting, standing, lying, climbingup,
climbingdown, jumping
**Units:** m/s² natively (columns: `id, attr_time (ms), attr_x, attr_y,
attr_z`)
**Key importance:** This is the **only** dataset with waist-mounted running
data.

### Loading

`load_acc_csv(filepath)` reads the CSV and returns `(timestamps_ms, x, y, z)`
arrays.  `find_acc_file(proband_num, activity, position)` locates the file
(proband1 has a flat layout; proband2+ have a nested `acc_{activity}_csv/`
directory).

Sampling rate is estimated from timestamps:

```python
fs = 1000.0 / np.median(np.diff(timestamps[:1000]))
```

The first and last 2 seconds are trimmed to remove transition artifacts:

```python
trim = int(2 * fs)
x = x[trim:-trim]  # etc.
```

### What is extracted

#### Per-activity waist ACC statistics

For each proband × activity × position combination, the script computes:

1. **Per-axis STD:** `std_x = np.std(x)`, etc.

2. **Magnitude-based RMS** (stored as `rms`, not used for scorecard grading):
   ```python
   mag = np.sqrt(x**2 + y**2 + z**2)
   rms = np.sqrt(np.mean((mag - np.mean(mag))**2))
   ```
   This detrends the scalar magnitude, which underestimates variability because
   cross-terms cancel.

3. **Per-axis body RMS** (stored as `body_rms`, **used for scorecard grading**):
   ```python
   body_rms = np.sqrt(np.mean((x - np.mean(x))**2 + (y - np.mean(y))**2 + (z - np.mean(z))**2))
   ```
   This subtracts the per-axis mean (gravity) independently, preserving the
   full vector variance.  The same formula is used on the simulator side in
   `validate_realism.py`.

All values for a given activity are collected across probands and averaged.

#### Walking benchmarks

| Key in `benchmarks.json` | Value | How computed |
|--------------------------|-------|--------------|
| `walking_body_rms_mean` | 4.289 m/s² | `np.mean([d["body_rms"] for d in results["walking_waist"]])` |
| `walking_x_std_mean` | 3.003 m/s² | `np.mean([d["std_x"] ...])`  |
| `walking_y_std_mean` | 1.922 m/s² | `np.mean([d["std_y"] ...])`  |
| `walking_z_std_mean` | 2.359 m/s² | `np.mean([d["std_z"] ...])`  |
| `walking_step_freq_mean` | 1.800 Hz | Welch PSD, see below |
| `walking_h3_h1_mean` | 0.310 | Per-subject Z-axis H3/H1, see below |

#### Running benchmarks

| Key in `benchmarks.json` | Value | How computed |
|--------------------------|-------|--------------|
| `running_body_rms_mean` | 9.505 m/s² | Same as walking body RMS |
| `running_x_std_mean` | 7.696 m/s² | Same method |
| `running_y_std_mean` | 3.555 m/s² | Same method |
| `running_z_std_mean` | 4.133 m/s² | Same method |
| `running_step_freq_mean` | 2.693 Hz | Welch PSD |
| `running_h3_h1_mean` | 1.201 | Z-axis H3/H1 |

#### Step frequency extraction (Welch method)

Unlike UCI HAR (pre-windowed), RealWorld has continuous recordings.  The script
uses a Welch-like approach in the main data-collection loop:

```python
seg_len = int(5 * fs)               # 5-second segments
n_segs = max(1, len(sig_c) // seg_len)
avg_psd = np.zeros(...)
for s in range(n_segs):
    seg = sig_c[s*seg_len:(s+1)*seg_len]
    psd = np.abs(rfft(seg))**2
    avg_psd += psd / n_segs
```

The peak is found in each axis within a frequency band that depends on the
activity:
- Walking: **[1.0, 3.5] Hz**
- Running: **[1.5, 5.0] Hz**

The axis with the strongest peak is selected.  One value per subject is stored.

#### Harmonic analysis (H3/H1)

Applied to the **Z-axis only** using the same 5-second-segment Welch PSD:

```python
avg_psd_z = np.zeros(len(seg_freqs))
for s in range(n_segs):
    seg = z[s*seg_len:(s+1)*seg_len] - np.mean(z)
    avg_psd_z += np.abs(rfft(seg))**2 / n_segs
```

Then:
1. `f1` = strongest peak in [1.0, 3.5] Hz (walking) or [1.5, 3.5] Hz
   (running).
2. H2 power = **single-bin max** in `[f1 × 1.8, f1 × 2.2]`.
3. H3 power = **single-bin max** in `[f1 × 2.8, f1 × 3.2]`.
4. `H3/H1 = p3 / p1`, recorded per subject.

Note the contrast with validate_realism.py on the simulator side, which uses
**band power** (sum over a frequency band).  This methodological difference
(single-bin peak vs. band-sum) contributes to the H3/H1 discrepancy between
real and sim.

#### Full `realworld2016` key in benchmarks.json

```json
"realworld2016": {
  "walking_x_std_mean": 3.003,
  "walking_y_std_mean": 1.922,
  "walking_z_std_mean": 2.359,
  "walking_rms_mean": 3.163,
  "walking_body_rms_mean": 4.289,
  "walking_h3_h1_mean": 0.310,
  "walking_step_freq_mean": 1.800,
  "walking_step_freq_std": 0.136,
  "running_x_std_mean": 7.696,
  "running_y_std_mean": 3.555,
  "running_z_std_mean": 4.133,
  "running_rms_mean": 6.212,
  "running_body_rms_mean": 9.505,
  "running_h3_h1_mean": 1.201,
  "running_step_freq_mean": 2.693,
  "running_step_freq_std": 0.144
}
```

---

## 4. Dataset 3: EmoWear

**Script:** `analyses/analyze_emowear_local.py`
**Source:** EmoWear study raw data
**Sensors:**
- BH3 (BioHarness 3): chest-worn, 100 Hz accelerometer (raw 12-bit ADC),
  25 Hz breathing waveform
- E4 (Empatica E4): wrist-worn, 4 Hz skin temperature

**Subjects:** All subdirectories in `data/Emowear-raw/` (49 subjects with
usable data, as recorded in `n_subjects_acc`, `n_subjects_rsp`,
`n_subjects_temp`)
**Protocol:** Seated emotional-stimuli viewing sessions (~40–90 min each)

The script defines three per-subject analysis functions —
`analyze_bh3_acc(bh3_dir)`, `analyze_bh3_rsp(bh3_dir)`,
`analyze_e4_temp(e4_dir)` — and loops over all subject directories, calling
each.

### BH3 Accelerometer → Rest Z STD

**Function:** `analyze_bh3_acc(bh3_dir)`

**ADC to m/s² conversion:**

```python
ADC_OFFSET = 2048
ADC_PER_G  = 100
x_ms2 = (bh3_x - ADC_OFFSET) / ADC_PER_G * 9.81
```

**Processing:**

1. The full recording is split into 30-second non-overlapping windows
   (`win_sec = 30`, `win_samples = int(win_sec * fs)` where `fs = 100.0`).
2. Per window, the script computes:
   - `win_stds_z.append(np.std(wz))` — standard deviation of the Z-axis.
   - Body RMS:
     ```python
     mag = np.sqrt((wx - np.mean(wx))**2 + (wy - np.mean(wy))**2 + (wz - np.mean(wz))**2)
     win_rms.append(np.sqrt(np.mean(mag**2)))
     ```
3. Windows are classified:
   - **Quiet:** `win_rms < 0.5` m/s² (sitting still)
   - **Active:** `win_rms > 1.0` m/s² (moving)
4. The function returns `quiet_z_std_mean = np.mean(win_stds_z[quiet_mask])` for
   each subject.

**Aggregation (benchmark-save section):**

```python
quiet_z_stds = collect(all_acc_results, "quiet_z_std_mean")
benchmarks["rest_z_std_mean"] = float(np.mean(quiet_z_stds))
```

**Saved value:** `emowear.rest_z_std_mean` = **0.1546 m/s²** (±0.0221,
across 49 subjects).

### BH3 Breathing Waveform → Breathing Rate, Baseline Wander, Breath CV

**Function:** `analyze_bh3_rsp(bh3_dir)`

The raw BH3 breathing signal is at `fs_rsp = 25.0` Hz.

#### Breathing rate extraction

1. The signal is split into 60-second non-overlapping windows
   (`win_sec = 60`, `win_samples = int(win_sec * fs_rsp)`).
2. Per window: mean-subtract, compute `psd = np.abs(rfft(seg_c))**2`, find the
   strongest peak in **0.1–0.8 Hz**:
   ```python
   f_mask = (freqs >= 0.1) & (freqs <= 0.8)
   peak_f = freqs[f_mask][np.argmax(psd[f_mask])]
   breath_rates.append(peak_f)
   ```
3. Per-subject mean is computed in Hz, then converted to BPM (×60).

**Saved value:** `emowear.rest_breathing_rate_bpm_mean` = **13.976 BPM**
(±2.295, across 49 subjects).  Also saved:
`rest_breathing_rate_hz_mean` = 0.233 Hz.

#### I:E ratio

Per 60-second window, the script counts samples above and below the mean:

```python
above = np.sum(seg_c > 0)
below = np.sum(seg_c <= 0)
ie_ratios.append(above / below)
```

**Saved value:** `emowear.ie_ratio_mean` = **0.960**.

#### Baseline wander

Computed only when the recording is longer than 60 s:

```python
win_ma = int(60 * fs_rsp)          # 1500 samples
cs = np.cumsum(rsp)
cs = np.insert(cs, 0, 0)
baseline = (cs[win_ma:] - cs[:-win_ma]) / win_ma   # 60s moving average
wander_range = np.max(baseline) - np.min(baseline)
rsp_range = np.max(rsp) - np.min(rsp)
result["wander_pct"] = float(100 * wander_range / rsp_range)
```

This is a cumulative-sum moving average with a **60-second kernel**.

**Saved value:** `emowear.rsp_wander_pct` = **13.388%** (±5.709, across 49
subjects).

#### Breath-to-breath variability (CV)

The script detrends the RSP signal with a 10-second moving average, then finds
positive zero-crossings, computes inter-breath intervals, filters to 1–10 s,
and computes CV:

```python
rsp_detrend = rsp - np.convolve(rsp, np.ones(int(10*fs_rsp))/int(10*fs_rsp), mode='same')
crossings = np.where(np.diff(np.sign(rsp_detrend)) > 0)[0]
breath_intervals = np.diff(crossings) / fs_rsp
valid = (breath_intervals > 1.0) & (breath_intervals < 10.0)
result["brv_cv"] = float(np.std(bi) / np.mean(bi) * 100)
```

**Saved value:** `emowear.breath_cv` = **47.355%**.

### E4 Skin Temperature → Warm-up Trajectory + Spike Rate

**Function:** `analyze_e4_temp(e4_dir)`

Reads `TEMP.csv` (E4 format: row 1 = start timestamp, row 2 = sample rate,
rows 3+ = data).  The sample rate comes from `float(f.readline().strip())`.

#### Spike rate

```python
diffs = np.abs(np.diff(temp_data))
result["spikes_gt_05"] = int(np.sum(diffs > 0.5))
result["spikes_per_hour"] = float(np.sum(diffs > 0.5) / dur_hrs)
```

A "spike" is any consecutive-sample jump where `|T[i] − T[i−1]| > 0.5 °C`.

**Saved value:** `emowear.temp_spikes_per_hour` = **6.338 /hr**.

#### Warm-up trajectory

The script samples temperature at three time windows:

```python
t1 = temp_data[:int(30 * fs_temp)]                            # first 30 s
t2 = temp_data[int(60 * fs_temp):int(120 * fs_temp)]          # 60–120 s
t_stable = temp_data[int(300 * fs_temp):int(600 * fs_temp)]   # 300–600 s (if long enough)
```

**Saved values:**

| Key | Value | Time window |
|-----|-------|-------------|
| `warmup_t0_mean` | 25.694 °C | 0–30 s |
| `warmup_t60_120_mean` | 26.115 °C | 60–120 s |
| `warmup_t_stable_mean` | 29.189 °C | 300–600 s |

**Caveat:** This is **wrist** temperature (E4), not waist or chest.

#### Full `emowear` key in benchmarks.json

```json
"emowear": {
  "rest_z_std_mean": 0.1546,
  "rest_z_std_std": 0.0221,
  "rest_breathing_rate_hz_mean": 0.2329,
  "rest_breathing_rate_bpm_mean": 13.976,
  "rest_breathing_rate_bpm_std": 2.295,
  "ie_ratio_mean": 0.960,
  "rsp_wander_pct": 13.388,
  "breath_cv": 47.355,
  "temp_noise_mean_abs_diff": 0.00466,
  "temp_spikes_per_session": 10,
  "temp_spikes_per_hour": 6.338,
  "warmup_t0_mean": 25.694,
  "warmup_t60_120_mean": 26.115,
  "warmup_t_stable_mean": 29.189,
  "n_subjects_acc": 49,
  "n_subjects_rsp": 49,
  "n_subjects_temp": 49
}
```

---

## 5. Dataset 4: Wearable Dataset

**Script:** `analyses/analyze_wearable_dataset.py`
**Source:** Published E4 wearable dataset with STRESS, AEROBIC, and ANAEROBIC
protocols
**Sensor:** Empatica E4 (wrist): ACC 32 Hz (1/64 g units), TEMP 4 Hz (°C),
HR 1 Hz (BPM), EDA 4 Hz (µS)
**Subjects:** ~90+ across 3 protocols (~40 STRESS, ~25 AEROBIC, ~25 ANAEROBIC)
**Data format:** E4 CSV format (row 1 = start timestamp as
`YYYY-MM-DD HH:MM:SS`, row 2 = sample rate, rows 3+ = data)
**Skipped subjects** (from `SKIP` dict in the script): STRESS/S02,
AEROBIC/{S03\_, S07\_, S11\_a, S11\_b, S06, S07\_},
ANAEROBIC/{S01, S06\_, S16\_a, S16\_b}.

### Protocol segmentation

Each subject directory contains a `tags.csv` with timestamps marking protocol
transitions.  The script reads them via `read_tags(filepath)` which parses
`YYYY-MM-DD HH:MM:SS` strings into `datetime` objects.

Segment extraction uses `time_to_sample(tag, start_time_str, fs)` to convert
a tag timestamp into a sample index:

```python
start = datetime.strptime(start_time_str.strip(), "%Y-%m-%d %H:%M:%S")
delta = (tag - start).total_seconds()
return int(delta * fs)
```

#### STRESS protocol

Handled by `process_stress(subj_dir, tags, is_v2)`.  Subjects with
prefix `f` are V2 (9 tags), prefix `S` are V1 (13 tags).

- **V2 (9 tags):** tag[0]→tag[1] = baseline rest (`stress_rest`);
  tag[1]→tag[2] = first task (`stress_task`); tag[3]→tag[4] and tag[5]→tag[6]
  = additional tasks; tag[2]→tag[3] = rest between tasks; tag[-2]→tag[-1] =
  final rest.
- **V1 (7+ tags):** tag[0]→tag[1] = baseline rest; tag[1]→tag[2] = Stroop
  (`stress_task`); tag[2]→tag[3] = rest; tag[3]→tag[4] = math
  (`stress_task`); tag[4]→tag[5] = rest; remaining tags are classified by
  duration: `< 120 s` → `stress_task`, else → `stress_rest`.

#### AEROBIC protocol

Handled by `process_aerobic(subj_dir, tags, is_v2)`.  Progressive cycling
at increasing RPM.

- **V2 (8 tags):** tag[0]→tag[1] = baseline (`aerobic_rest`);
  tag[1]→tag[2] and tag[2]→tag[3] = `aerobic_low` (70–75 RPM);
  tag[3]→tag[4] and tag[4]→tag[5] = `aerobic_med` (80–85 RPM);
  tag[5]→tag[6] = `aerobic_high` (90–95+ RPM);
  tag[6]→tag[7] = `aerobic_cool`.
- **V1 (12 tags):** tag[0]→tag[1] = baseline; tags 1–4 = low; tags 4–7 =
  med; tags 7–10 = high; tags 10–11 = cooldown.

#### ANAEROBIC protocol

Handled by `process_anaerobic(subj_dir, tags, is_v2)`.

- tag[0]→tag[1] = baseline rest (`anaerobic_rest`).
- Subsequent tag pairs classified by gap duration:
  - `< 60 s` → `anaerobic_sprint` (~30–45 s all-out sprint)
  - `60–300 s` → `anaerobic_cool` (~4-min recovery)
  - `> 300 s` → `anaerobic_rest`

### Signal extraction per segment

`extract_segment(subj_dir, tag_start, tag_end, label)` reads each E4 signal
between the two bounding tags and computes:

- **HR:** `np.mean(seg[seg > 0])` — filters out zero readings.
- **TEMP:** `np.mean(seg)` in °C.
- **EDA:** `np.mean(seg)` in µS.
- **ACC STD:**
  ```python
  mag = np.sqrt(seg[:,0]**2 + seg[:,1]**2 + seg[:,2]**2) * 9.81 / 64.0
  results[label]["acc_mag"].append(np.std(mag))
  ```
  This converts from 1/64 g to m/s² and takes the STD of the scalar magnitude.
  Note: this is **not** per-axis body RMS but rather `std(||a||)`.

Each subject contributes one mean value per segment per signal.

### Benchmarks saved

The save section iterates over four labels and stores temperature and ACC
statistics:

```python
for label in ["stress_rest", "stress_task", "aerobic_low", "aerobic_high"]:
    benchmarks[f"{label}_temp_mean"] = m_val     # np.mean of all subjects' mean temps
    benchmarks[f"{label}_temp_std"]  = s_val
    benchmarks[f"{label}_temp_n"]   = n          # count of subjects
    benchmarks[f"{label}_acc_std_mean"] = m_val   # np.mean of all subjects' ACC STDs
    benchmarks[f"{label}_acc_std_std"]  = s_val
```

| Key | Value | Source segment | N |
|-----|-------|----------------|---|
| `stress_rest_temp_mean` | 32.522 °C | Baseline rest periods | 122 |
| `stress_rest_temp_std` | 1.741 °C | | |
| `stress_rest_acc_std_mean` | 0.211 m/s² | Baseline rest ACC | |
| `stress_task_temp_mean` | 32.762 °C | Stroop/math/emotional tasks | 190 |
| `stress_task_acc_std_mean` | 0.284 m/s² | | |
| `aerobic_low_temp_mean` | 31.853 °C | 70–75 RPM cycling | 65 |
| `aerobic_high_temp_mean` | 32.273 °C | 90–95+ RPM cycling | 52 |

**Critical caveat:** These temperatures are **wrist-worn E4** readings.  Wrist
skin temperature at rest is typically 32–34 °C.  The simulator models a
hypothetical waist/chest sensor where resting temperature is ~33 °C.  The
validation compares absolute values, so there is an inherent body-position
offset of 0–2 °C.

---

## 6. Dataset 5: Artifact Analysis

**Script:** `analyses/analyze_artifacts.py`
**Sources:** EmoWear BH3 (all subjects in `data/Emowear-raw/`) + Wearable
Dataset E4 (all subjects across STRESS/AEROBIC/ANAEROBIC)
**Purpose:** Characterize real-world signal artifacts to calibrate simulator
artifact injection.

### BH3 ACC clipping

For each EmoWear subject, the script reads the raw ADC accelerometer file
(same file as `analyze_emowear_local.py`), centers at ADC offset 2048, and
checks:

```python
arr = np.column_stack([xs, ys, zs]).astype(float) - 2048
n_clip = int(np.sum(np.abs(arr) > 1580))
n_total = arr.size
bh3_clip_pcts.append(100 * n_clip / n_total)
```

The threshold of **1580 ADC units from center** corresponds to approximately
±16 g at the BH3's ADC scale.

**Saved value:** `artifacts.bh3_acc_clip_mean_pct` = **0.000%** (expected since
EmoWear is a seated protocol with minimal movement).

### BH3 RSP baseline wander

The identical algorithm to `analyze_emowear_local.py` (60-second cumulative-sum
moving average):

```python
win = 60 * fs_rsp  # = 1500 samples
cs = np.cumsum(rsp)
cs = np.insert(cs, 0, 0)
baseline = (cs[win:] - cs[:-win]) / win
wander_range = np.max(baseline) - np.min(baseline)
rsp_range = np.max(rsp) - np.min(rsp)
bh3_wander_pcts.append(100 * wander_range / rsp_range)
```

**Saved value:** `artifacts.rsp_wander_mean_pct` = **13.388%** (±5.709, same
data as `emowear.rsp_wander_pct`).

### E4 TEMP anomalies

Across all Wearable Dataset subjects in all three protocols:

**Jump rate:**

```python
diffs = np.abs(np.diff(temp))
n_jumps = np.sum(diffs > 0.5)
temp_jump_rates.append(n_jumps / len(temp))
```

Fraction of consecutive-sample pairs where `|T[i] − T[i−1]| > 0.5 °C`.

**Saved value:** `artifacts.e4_temp_jump_mean_pct` = **0.604%** (max 7.463%).

**Out-of-range rate:**

```python
n_bad = np.sum((temp < 20) | (temp > 42))
temp_nan_rates.append(n_bad / len(temp))
```

Fraction of samples where T < 20 °C or T > 42 °C.

**Saved value:** `artifacts.e4_temp_oor_mean_pct` = **2.643%** (max 100% for
subjects with entirely cold/disconnected sensors).

### E4 ACC clipping

E4 has a ±2 g accelerometer reporting in 1/64 g units.  Clipping occurs when
any axis value reaches ±127 (the 8-bit saturation level):

```python
clip_count = np.sum(np.abs(acc[:, :3]) >= 127)
acc_clip_rates.append(clip_count / acc[:, :3].size)
```

**Saved value:** `artifacts.e4_acc_clip_mean_pct` = **0.318%**.

#### Full `artifacts` key in benchmarks.json

```json
"artifacts": {
  "bh3_acc_clip_mean_pct": 0.000,
  "bh3_acc_clip_max_pct": 0.000,
  "bh3_n_acc_subjects": 49,
  "rsp_wander_mean_pct": 13.388,
  "rsp_wander_std_pct": 5.709,
  "bh3_n_rsp_subjects": 49,
  "e4_acc_clip_mean_pct": 0.318,
  "e4_temp_jump_mean_pct": 0.604,
  "e4_temp_jump_max_pct": 7.463,
  "e4_temp_oor_mean_pct": 2.643,
  "e4_temp_oor_max_pct": 100.0
}
```

---

## 7. How the Validation Runs Simulations

**Script:** `analyses/validate_realism.py`

The script runs `N = 20` independent simulation trials.  Each trial creates a
fresh `SimState` (which randomises subject parameters — height, weight, skin
baseline, etc.) and steps through the full activity sequence:

```python
for trial in range(N):
    s = simulate.SimState()
    ...
    for _ in range(simulate.TOTAL_SAMPLES):   # 270 s × 50 Hz = 13 500 samples
        r = simulate.step(s)
```

`simulate.step(s)` returns a 7-element tuple per sample:

```
(t, ax, ay, az, rsp, skt, activity)
```

- `t` — timestamp in seconds.
- `ax, ay, az` — 3-axis accelerometer in m/s².  May be `NaN` during BT gap
  artifacts.
- `rsp` — respiration waveform (arbitrary units).  May be `NaN` during
  dropouts.
- `skt` — skin temperature in °C.  Is `None` on non-SKT sample slots (since
  SKT is sampled at 4 Hz, only every 12th or 13th ACC sample carries a SKT
  value).  May be `NaN` during dropouts.
- `activity` — string label (`"rest"`, `"walking"`, `"running"`, `"recovery"`,
  `"stress"`, `"sleep"`).

The fixed activity sequence is: rest (30 s) → walking (45 s) → running (30 s)
→ recovery (40 s) → rest (15 s) → stress (35 s) → rest (15 s) → sleep (60 s)
= **270 s total** (`simulate.TOTAL_TIME`).

### Data collection during each trial

Samples are partitioned into per-activity buckets using NaN checks:

```python
if ax == ax:                         # NaN check (NaN != NaN)
    act_data[act]["ax"].append(ax)   # etc. for ay, az
if rsp == rsp:
    act_data[act]["rsp"].append(rsp)
if skt is not None:
    skt_time.append((t, skt))
    if skt == skt:
        act_data[act]["skt"].append(skt)
```

---

## 8. How Each Metric Is Computed on Simulated Data

All metric computations below are in `validate_realism.py`, inside the
per-trial loop.

### Body RMS (accelerometer)

For each activity segment:

```python
means = np.array([np.mean(ax_a), np.mean(ay_a), np.mean(az_a)])
ax_ng = ax_a - means[0]
ay_ng = ay_a - means[1]
az_ng = az_a - means[2]
body_rms = np.sqrt(np.mean(ax_ng**2 + ay_ng**2 + az_ng**2))
```

This subtracts the per-axis mean (removing gravity) and computes vector-RMS.
This is identical to the formula used in `analyze_realworld2016.py` for
`body_rms`, ensuring consistency.

### Per-axis STD

```python
per_act[act]["x_std"].append(np.std(ax_a))
per_act[act]["y_std"].append(np.std(ay_a))
per_act[act]["z_std"].append(np.std(az_a))
```

### Step frequency

For walking and running segments with > 200 samples:

```python
az_detrend = az_a - np.mean(az_a)
freqs = rfftfreq(len(az_detrend), 1.0 / simulate.FS)
psd = np.abs(rfft(az_detrend))**2
mf = (freqs >= 1.0) & (freqs <= 4.0)
per_act[act]["step_freq"].append(freqs[mf][np.argmax(psd[mf])])
```

The search band is **1.0–4.0 Hz**, applied to the Z-axis (vertical) FFT.

### H3/H1 harmonic ratio

On the same Z-axis FFT:

```python
mf1 = (freqs >= 1.0) & (freqs <= 3.5)
f1 = freqs[mf1][np.argmax(psd[mf1])]
# H1 power: SUM in [f1 − 0.3, f1 + 0.3]
h1_mask = (freqs >= f1 - 0.3) & (freqs <= f1 + 0.3)
h1_power = np.sum(psd[h1_mask])
# H3 power: SUM in [3*f1 − 0.4, 3*f1 + 0.4]
h3_mask = (freqs >= 3*f1 - 0.4) & (freqs <= 3*f1 + 0.4)
h3_power = np.sum(psd[h3_mask])
h3_h1 = h3_power / h1_power
```

**Critical difference from real-data analysis:** The real data scripts
(`analyze_uci_har.py`, `analyze_realworld2016.py`) use **single-bin peak
PSD** (`np.max(avg_psd[mask])`), while the validation uses **band power**
(`np.sum(psd[mask])`).  Band power is more robust to spectral leakage but
yields systematically different values.  This is the primary reason the H3/H1
tolerance is set to 50%.

### Respiration BPM

For each activity with > 256 RSP samples:

```python
arr = np.array(rsp_data)
freqs_r, psd_r = sig.welch(arr - np.mean(arr), fs=simulate.FS,
                            nperseg=min(1024, len(arr)))
rmask = (freqs_r > 0.08) & (freqs_r < 1.0)
per_act[act_name]["rsp_bpm"].append(freqs_r[rmask][np.argmax(psd_r[rmask])] * 60)
```

Uses `scipy.signal.welch` with `nperseg = min(1024, N)`.  Peak frequency in
**0.08–1.0 Hz** is converted to BPM (×60).

### RSP baseline wander

Computed on the **first 30-second rest block only** (`first_rest_n = int(30 *
simulate.FS)` = 1500 samples):

```python
kernel = np.ones(200) / 200          # 200-sample (4-second) moving average
baseline = np.convolve(arr, kernel, mode="valid")
wander = (np.max(baseline) - np.min(baseline)) / max(0.001, np.ptp(arr))
per_act["rest"]["rsp_wander"].append(wander * 100)
```

The kernel is **200 samples = 4 s** at 50 Hz, much shorter than the
60-second kernel used in the real-data analysis.  This is because the
simulator rest block is only 30 seconds, so a 60-second kernel would not fit.

### SKT mean per activity

```python
if len(d["skt"]) > 5:
    per_act[act]["skt_mean"].append(np.mean(d["skt"]))
```

Average of all valid (non-NaN) skin temperature samples in each activity
segment.

### SKT spikes

Isolated spike detection using a both-neighbour check on the time-ordered
valid SKT samples across the full trial:

```python
arr_skt = np.array(valid_skt)
spike_count = 0
for i in range(1, len(arr_skt) - 1):
    d_prev = arr_skt[i] - arr_skt[i - 1]
    d_next = arr_skt[i] - arr_skt[i + 1]
    if abs(d_prev) > 0.3 and abs(d_next) > 0.3 and d_prev * d_next > 0:
        spike_count += 1
```

Requirements for a spike: the sample deviates > 0.3 °C from **both** its
neighbours, and in the same direction (`d_prev * d_next > 0`).  This
distinguishes true spikes from gradual activity-transition ramps.

The spike count is normalized to per-hour in the report section:

```python
spikes_hr = spikes / (simulate.TOTAL_TIME / 3600)
```

### Artifact rates

Computed across the full trial (all activities combined):

- **BT gap rate:** `100 * nan_acc / total` — percentage of samples where `ax`
  is NaN (checked via `r[1] != r[1]`).
- **RSP dropout rate:** `100 * nan_rsp / total` — percentage where `rsp` is
  NaN (checked via `r[4] != r[4]`).
- **SKT NaN rate:** `100 * nan_skt / skt_total` — percentage of
  SKT-emitting samples where `skt` is NaN.
- **ACC stuck-value rate:** percentage of consecutive valid ACC pairs where
  `valid_ax[i] == valid_ax[i-1]` (exact equality — quantization artifacts):
  ```python
  stuck_count = sum(1 for i in range(1, len(valid_ax)) if valid_ax[i] == valid_ax[i-1])
  art["stuck"].append(100 * stuck_count / len(valid_ax))
  ```

### Warm-up trajectory

The script checks SKT values near specific timestamps across the full trial:

```python
for t_check, key in [(0.5, "skt_t0"), (120, "skt_t120"),
                     (300, "skt_t300"), (600, "skt_t600")]:
    vals = [v for t_s, v in skt_time if abs(t_s - t_check) < 2.0 and v == v]
    if vals:
        art[key].append(np.mean(vals))
```

These are compared against EmoWear E4 warm-up values but are **not graded** in
the scorecard (the simulator starts at body temperature, not ambient).

---

## 9. Grading System

### Grade function

```python
def grade(sim, real, tol=25):
    if real == 0 or np.isnan(sim):
        return "—"
    err = abs(sim - real) / abs(real) * 100
    if err < 10:
        return "A"
    elif err < tol:
        return "B"
    elif err < 50:
        return "C"
    else:
        return "F"
```

The error is always **unsigned** (absolute value) for grading.  The displayed
error in the report is **signed** (`pct_err` function: `(sim − real) / real ×
100%`) so you can see over- vs under-estimation.

### Per-metric tolerance overrides

Not all metrics use the default `tol = 25`.  The tolerances are set in the
`scorecard` list in `validate_realism.py`:

```python
scorecard = [
    ("Walk body RMS",      sim, rw["walking_body_rms_mean"],          25),
    ("Walk step freq",     sim, uci["walking_step_freq_mean"],        10),
    ("Walk H3/H1",         sim, uci["walking_h3_h1"],                 50),
    ("Run body RMS",       sim, rw["running_body_rms_mean"],          25),
    ("Run step freq",      sim, rw["running_step_freq_mean"],         10),
    ("Run H3/H1",          sim, rw["running_h3_h1_mean"],             50),
    ("Rest breath rate",   sim, emo["rest_breathing_rate_bpm_mean"],  30),
    ("Sleep breath rate",  sim, 10.8,                                 30),
    ("RSP wander",         sim, emo["rsp_wander_pct"],                50),
    ("Rest SKT",           sim, wd["stress_rest_temp_mean"],          15),
    ("Stress SKT",         sim, wd["stress_task_temp_mean"],          15),
]
```

| Metric | Tolerance | Real-data source | Rationale |
|--------|-----------|-----------------|-----------|
| Walk body RMS | 25% | `rw["walking_body_rms_mean"]` = 4.289 | Default |
| Walk step freq | 10% | `uci["walking_step_freq_mean"]` = 1.810 | Tight: well-characterised |
| Walk H3/H1 | 50% | `uci["walking_h3_h1"]` = 0.170 | Analysis method differs (peak vs band) |
| Run body RMS | 25% | `rw["running_body_rms_mean"]` = 9.505 | Default |
| Run step freq | 10% | `rw["running_step_freq_mean"]` = 2.693 | Tight |
| Run H3/H1 | 50% | `rw["running_h3_h1_mean"]` = 1.201 | Same as walk H3/H1 |
| Rest breath rate | 30% | `emo["rest_breathing_rate_bpm_mean"]` = 13.976 | Biological variability |
| Sleep breath rate | 30% | Hard-coded `10.8` (published literature) | No dataset reference |
| RSP wander | 50% | `emo["rsp_wander_pct"]` = 13.388 | High inter-subject STD (±5.7) |
| Rest SKT | 15% | `wd["stress_rest_temp_mean"]` = 32.522 | Temperature is precise |
| Stress SKT | 15% | `wd["stress_task_temp_mean"]` = 32.762 | Temperature is precise |

### Additional detailed metrics (not in scorecard)

The validation also prints (but does not grade in the scorecard):

| Metric row | Real reference | Source in code |
|------------|---------------|----------------|
| Rest X/Y/Z STD | None | Displayed only |
| Rest body RMS | `uci["sitting_body_rms_mean"]` = 0.2338 | Graded at `tol=25` in the detailed table |
| Walk X/Y/Z STD | `rw["walking_x/y/z_std_mean"]` | Per-axis from RealWorld |
| Run X/Y/Z STD | `rw["running_x/y/z_std_mean"]` | Per-axis from RealWorld |
| Stress body RMS | `wd["stress_task_acc_std_mean"]` = 0.284 | WD stress-task ACC |
| Sleep Z STD | None | Displayed only |
| All respiration BPM rows | See §10 for activity-specific references | |
| All artifact rates | Range checks, not graded | |

### Range checks

The script also performs pass/fail range checks (not letter-graded):

```python
in_range(walk_rms, 1, 6)            # Walk body RMS in [1, 6] m/s²
in_range(run_rms, 6, 12.5)          # Run body RMS in [6, 12.5] m/s²
in_range(rest_z, 0.08, 0.25)        # Rest Z STD in [0.08, 0.25] m/s²
```

### Respiration range checks

Each activity's BPM is checked against a literature range:

```python
rsp_ref = {
    "rest":     (12, 20, emo["rest_breathing_rate_bpm_mean"], "lit:12-20, EmoWear"),
    "walking":  (18, 30, 21, "lit:18-30"),
    "running":  (30, 55, 42, "lit:30-55"),
    "recovery": (13, 25, 18, "lit:13-25"),
    "stress":   (13, 22, 17, "lit:13-22"),
    "sleep":    (8, 14, 10.8, "lit:8-14"),
}
```

The 3rd element is the "typical" value used for percentage-error grading (with
`tol = 30`).  Only `rest` uses a real-data reference
(`emo["rest_breathing_rate_bpm_mean"]`).  All others use literature-based
typical values hard-coded in the script.

### Physiological direction checks

The script checks four expected physiological relationships:

```python
("Exercise drops skin temp",   skt_walk_m < skt_rest_m)
("Running drops further",      skt_run_m < skt_walk_m)
("Recovery overshoots rest",   skt_recov_m > skt_rest_m)
("Sleep is warm",              skt_sleep_m > skt_rest_m)
```

These are pass/fail checks, not letter-graded.

### Issues

The script collects violation flags at the end:

- Rest Z STD outside [0.08, 0.25]
- Walk body RMS outside [1, 6]
- Run body RMS outside [6, 12.5]
- Rest BPM outside [10, 20]
- Sleep BPM outside [8, 16]
- SKT does not drop during exercise
- SKT recovery does not overshoot rest
- Walk step freq outside [1.5, 2.2]
- Run step freq outside [2.3, 3.2]

---

## 10. Activity Mapping: Real Data → Simulator Activities

The simulator has 6 activities: rest, walking, running, recovery, stress,
sleep.  Not all datasets contain all activities.

### Rest (simulator: stationary, low noise)

| Source | Metric used | Benchmark key |
|--------|-------------|---------------|
| UCI HAR "SITTING" (activity ID 4), waist, n=30, body_acc RMS | Rest body RMS | `uci_har.sitting_body_rms_mean` = 0.2338 |
| EmoWear BH3 quiet windows (body RMS < 0.5), chest, n=49, Z-axis STD | Rest Z STD | `emowear.rest_z_std_mean` = 0.1546 |
| EmoWear BH3, chest, n=49, spectral breathing rate | Rest breath rate | `emowear.rest_breathing_rate_bpm_mean` = 13.976 |
| EmoWear BH3, chest, n=49, 60 s MA wander | RSP wander | `emowear.rsp_wander_pct` = 13.388 |
| Wearable Dataset stress baseline, wrist E4, n=122 | Rest SKT | `wearable_dataset.stress_rest_temp_mean` = 32.522 |

### Walking (simulator: ~1.8 Hz step frequency, moderate amplitude)

| Source | Metric used | Benchmark key |
|--------|-------------|---------------|
| UCI HAR "WALKING" (ID 1), waist, n=30 | Step frequency, H3/H1 | `uci_har.walking_step_freq_mean` = 1.810, `uci_har.walking_h3_h1` = 0.170 |
| RealWorld2016 "walking", waist, n=13 | Body RMS, per-axis STD | `realworld2016.walking_body_rms_mean` = 4.289 |
| Wearable Dataset aerobic_low (70–75 RPM cycling), wrist E4 | Walking SKT | `wearable_dataset.aerobic_low_temp_mean` = 31.853 |

**Caveat on walking SKT:** Light stationary cycling is NOT walking.  It was
chosen as the closest available low-intensity exercise with skin temperature
data.  Cycling involves less body displacement and different convective cooling
than overground walking.

### Running (simulator: ~2.7 Hz, high amplitude)

| Source | Metric used | Benchmark key |
|--------|-------------|---------------|
| RealWorld2016 "running", waist, n=15 | Body RMS, per-axis STD, step freq, H3/H1 | `realworld2016.running_*` |

No running skin temperature data is available from any of the five datasets.

### Recovery (simulator: post-exercise, decreasing HR/breathing)

No direct benchmark data.  Validated only via:
- **Direction check:** `skt_recovery > skt_rest` (skin temperature overshoots
  after exercise).
- **BPM range check:** `[13, 25]` BPM (literature reference, typical = 18).

### Stress (simulator: stationary, elevated breathing/SKT)

| Source | Metric used | Benchmark key |
|--------|-------------|---------------|
| Wearable Dataset stress-task (Stroop/math), wrist E4 | Stress SKT | `wearable_dataset.stress_task_temp_mean` = 32.762 |
| Wearable Dataset stress-task ACC STD | Stress body RMS | `wearable_dataset.stress_task_acc_std_mean` = 0.284 |

**Caveat on stress body RMS:** The ACC STD from the Wearable Dataset is
wrist-worn at 32 Hz ±2 g in 1/64 g units, while the simulator models
waist-worn ±16 g at 50 Hz.  The absolute values are not directly comparable.
The validation compares them anyway — the relative scale (stress ≈ rest) is
the key check.

### Sleep (simulator: stationary, low breathing rate, warm skin)

**No real sleep data exists in any of the 5 datasets.**

- **Sleep breathing rate:** The benchmark is `10.8` BPM, hard-coded in
  `validate_realism.py`'s `scorecard` list and `rsp_ref` dict.  This comes
  from published literature on normal adult sleep respiration rates.
- **Sleep SKT:** Validated only via direction check (`skt_sleep > skt_rest`).
- **Sleep movement level:** Displayed but not graded (no benchmark).

---

## 11. Known Limitations and Cross-Dataset Caveats

### Body position mismatches

| Dataset | Sensor location | Simulator models |
|---------|----------------|-----------------|
| UCI HAR | Waist (smartphone) | Waist ✓ |
| RealWorld2016 | Waist (smartphone) | Waist ✓ |
| EmoWear BH3 | Chest (strap) | Waist ✗ |
| EmoWear E4 | Wrist (band) | Waist ✗ |
| Wearable Dataset E4 | Wrist (band) | Waist ✗ |

Chest accelerometer readings show less movement than waist during walking
(the waist swings more).  Wrist skin temperature differs from waist/chest by
1–3 °C and responds differently to exercise.  These mismatches are noted in
the validation output but not corrected.

### Sensor range and resolution differences

| Sensor | Range | Rate | Resolution |
|--------|-------|------|------------|
| UCI HAR (Galaxy SII) | ±2 g | 50 Hz | ~0.001 g |
| RealWorld2016 (phone) | varies (~±8–16 g) | ~50 Hz | varies |
| EmoWear BH3 | ±16 g | 100 Hz | 12-bit ADC (~0.1 m/s²) |
| E4 ACC | ±2 g | 32 Hz | 8-bit (1/64 g ≈ 0.15 m/s²) |
| Simulator | ±16 g | 50 Hz | float (then optionally quantised) |

### Activity protocol differences

The Wearable Dataset "aerobic" protocol uses **stationary cycling**, not
walking or running.  The skin temperature response to cycling differs from
overground locomotion because there is no convective cooling from forward
movement and a different thermoregulatory load.

### H3/H1 methodology difference

The real-data analyses (`analyze_uci_har.py` section 7,
`analyze_realworld2016.py` harmonics section) use **peak PSD** (single-bin
maximum) for harmonic power.  The simulator validation (`validate_realism.py`)
uses **band power** (sum over a ±0.3–0.4 Hz band).  Band power is more robust
but yields systematically higher values.  This is the primary reason the H3/H1
tolerance is set to 50%.

### RSP wander kernel difference

The real data uses a **60-second (1500-sample) kernel** at 25 Hz.  The
simulator validation uses a **200-sample (4-second) kernel** at 50 Hz.  The
simulator's rest block is only 30 s, so a 60-second kernel would not fit.
This kernel-size difference means the wander metric is not strictly
comparable; the 50% tolerance accounts for this.

### Metrics with no real-data reference

The following simulator outputs have no graded benchmark:

- Rest X STD, Rest Y STD (only Z STD from EmoWear)
- Running SKT, Recovery SKT, Sleep SKT (no direct data)
- Sleep movement levels
- Recovery breathing rate (literature range-check only)

### Statistical considerations

- **N = 20 trials** provides moderate confidence in simulator means
  (SE = SD / √20 ≈ 0.22 · SD) but does not capture the full population range.
- **Benchmark sample sizes** vary: 13 (RealWorld walking), 15 (RealWorld
  running), 30 (UCI HAR), 49 (EmoWear), 65–190 (Wearable Dataset segments).
- **Grading is against the point estimate** (benchmark mean), not against a
  confidence interval.  A metric that grades B (+20%) might actually be within
  the 95% CI of the real data.
