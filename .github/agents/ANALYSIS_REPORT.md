# ANALYSIS REPORT
> Written by agent during Phase 1. Must be approved by user before Phase 2.
> Every finding cites its source (dataset + file, or DOI + section).
> Historical note: many findings were first written under earlier simulator-first or coaching-demo framing. The findings remain valid as raw evidence, but the current product scope is now **MotionLens**, an interpretable ACC signal-intelligence app, with any simulator treated as supporting analysis infrastructure only.

## Status: COMPLETE — PHASE 2 APPROVED

Analysis date: 2026-04-21  
Script: `analyses/phase1_data_analysis.py`

---

## 1. Literature review

### 1.1 Placement effects on ACC signals

**Sotirakis et al. 2025** [DOI:10.1016/j.jbiomech.2025.112975] is the most directly relevant paper. It compared an abdomen-worn IMU against an L5 lumbar reference in Parkinson's disease (PD) patients and healthy controls during walking and quiet stance.

Key findings from this paper:
- Coherence > 0.9 between abdomen and L5 for most axes during walking in both groups — meaning abdomen signal tracks lumbar signal closely for gross locomotion.
- Coherence drops < 0.9 for **mediolateral acceleration** and **pitch rotation** during walking — these are noisier at the abdomen due to abdominal tissue compliance.
- Coherence is substantially lower for **anteroposterior (AP) acceleration** and **pitch rotation** during quiet stance — the abdomen decouples from the lumbar spine when there is no strong driving motion.
- Conclusion: "Abdomen device placement provides clinically valid movement data for most outcomes. Caution advised for specific parameters, as abdominal movement may introduce noise."

This has direct design implications: the project must treat the abdomen as a **softly coupled** body site, not a rigid extension of the trunk. Axis-dependent noise is real and measurable.

**Mannini & Sabatini 2010** and the wider literature (surveyed in Sotirakis 2025) establish that trunk-worn sensors (waist, sternum, lower back) produce the most reliable gait features, with thorax > abdomen in signal fidelity for fine gait metrics.

### 1.2 Anterior waist/abdomen as a sensor region

The **EmoWear** dataset [DOI:10.1038/s41597-024-03429-3] places its STb sensor at the **xiphoid process level** (sternum bottom / epigastric region), which is the superior boundary of the front waist/abdomen region. The STb IMU captures **seismocardiography (SCG)** — cardio-respiratory vibrations of the thorax/abdomen wall — in addition to gross body motion. The paper notes:
- The dorsoventral (z) and mediolateral (x) axes carry the SCG signal.
- During walking, trunk motion dominates and SCG becomes harder to isolate.
- Front and back sensors at the same sternal level show high inter-device correlation during locomotion and somewhat lower during rest (because each side reflects its own wall dynamics).

The **WoW** dataset [Mendeley: DOI:10.17632/9h2y3z8x5r/1] is the only dataset with patches directly on the **front torso** (thoracic and abdominal wall), using strain gauges + IMU. Three patch variants:
- **P1**: one patch on the rib cage (thoracic) + one perpendicular patch on the stomach (umbilical region)
- **PS1**: single perpendicular patch at the stomach only
- **PD1**: dual-sensor patch (two perpendicular gauges, same patch, on stomach)

Activities covered are static/postural only (sitting, standing, Fowler position). No walking or dynamic activities are present.

### 1.3 Activity biomechanics at the waist/abdomen region

From RealWorld2016 data (waist and chest, 15 probands):

| Activity | Dynamic? | ACC magnitude std (waist, m/s²) | Notes |
|----------|----------|----------------------------------|-------|
| Lying | No | 0.123 | Near-zero dynamics; gravity is dominant |
| Standing | No | 0.191 | Postural sway only |
| Sitting | No | 0.259 | Slight breathing artefact |
| Climbing up | Moderate | 2.357 | Rhythmic, asymmetric |
| Climbing down | Moderate | 3.636 | Higher impact than up |
| Walking | High | 2.191 | Rhythmic, ~1–2 Hz steps |
| Running | Very high | 6.063 | High-impact, broad spectrum |

At the front abdomen specifically (from WoW, g units, static positions only):
- Sitting: std_mag ≈ 0.010–0.013 g — essentially static
- Standing: std_mag ≈ 0.037–0.062 g — postural sway + breathing
- Fowler (semi-reclined): std_mag ≈ 0.010 g — similar to sitting

The abdomen wall is significantly more compliant than the lumbar spine or sternum. During static poses, the dominant signals are: breathing artefact (~0.2–0.5 Hz), postural tremor (~3–10 Hz), and gravitational projection based on body orientation.

---

## 2. Dataset analysis

### 2.1 Relevance assessment

| Dataset | Relevant? | Reasoning | Usable for |
|---------|-----------|-----------|------------|
| **WoW** (`DataSet_RR/`) | **HIGH** | Only dataset with direct front torso (thoracic + abdominal) patches. Has IMU alongside strain gauges. | Signal magnitude reference for static/postural positions on the front waist/abdomen |
| **WARD** (`WARD1.0/`) | **HIGH** | One node is fixed at the front center of the waist. Covers rest, walking variants, turns, stairs, jog, jump, push at 20 Hz. | Direct front-center waist activity ordering and low-rate locomotion context |
| **iPhone placement sweep** (`Acceleration/`) | **HIGH** | Direct local recordings across the actual coarse zones in scope: front-center, front-side, and lateral placements. | Coarse-zone sanity checks, left/right symmetry checks, and local transfer calibration |
| **EmoWear** (`Emowear-raw/`) | **MEDIUM** | STb front sensor at xiphoid level (upper-front torso boundary). High sample rate but above the locomotion-oriented scope. | Upper-front placement reference; SCG characterization; inter-subject variability |
| **RealWorld2016** (`realworld2016_dataset/`) | **HIGH** | Includes an explicit waist placement plus chest. 15 probands × 7 activities × 7 body positions. 50 Hz. | Activity-specific ACC statistics; waist–chest comparison; locomotion dynamics |
| **UCI HAR** (`human+activity+recognition+using+smartphones/`) | **MEDIUM** | Waist / beltline placement with a lateral side-mounted example. 30 subjects. | Walking/stairs dynamics for side or front-side waist placements |
| **WISDM** (`WISDM_ar_v1.1/`) | **MEDIUM** | Controlled Android-phone data from the front pants-leg pocket at 20 Hz. This is a `front_pocket` carry source, not a body-fixed waist/abdomen sensor. | Smartphone-style front-pocket locomotion and posture contrast |
| **HARTH** (`harth/`) | **LOW** | High-quality free-living dataset, but sensors are on lower back and front thigh only. | Indirect trunk/lumbar and posture-transition context |
| **Wearable/PhysioNet** (`Wearable_Dataset/`) | **NONE** | Wrist only (E4). Irrelevant to anterior abdomen simulation. | — |

### 2.2 Signal characteristics by activity

#### RealWorld2016 — waist vs chest (m/s², 50 Hz, 15 probands averaged)

| Activity | Waist mean_mag | Waist std_mag | Waist std_xyz (x,y,z) | Chest mean_mag | Chest std_mag | Chest std_xyz (x,y,z) |
|----------|----------------|---------------|-----------------------|----------------|---------------|-----------------------|
| Walking | 10.196 | 2.191 | (2.150, 1.415, 2.141) | 9.858 | 2.014 | (1.597, 2.025, 1.274) |
| Running | 10.226 | 6.063 | (6.417, 2.210, 3.171) | 9.647 | 6.293 | (4.068, 6.437, 3.025) |
| Sitting | 9.851 | 0.259 | (0.831, 0.411, 1.576) | 9.682 | 0.273 | (0.943, 0.448, 0.994) |
| Standing | 9.925 | 0.191 | (0.454, 0.279, 3.772) | 9.768 | 0.275 | (2.178, 1.496, 2.109) |
| Lying | 9.725 | 0.123 | (0.859, 0.283, 1.279) | 9.476 | 0.133 | (0.948, 0.678, 0.891) |
| Climbing up | 10.091 | 2.357 | (2.285, 1.271, 1.941) | 9.923 | 2.482 | (1.735, 2.548, 1.509) |
| Climbing down | 10.118 | 3.636 | (3.640, 1.756, 2.125) | 10.025 | 3.582 | (3.547, 2.226, 1.520) |

Notes: Units m/s². Mean magnitude ≈ 9.81 m/s² (gravity) plus motion. Sample rate 50 Hz.

#### WoW — anterior torso patches (g, ~11–14 Hz, 4–6 subjects averaged)

| Patch | Position | mean_mag (g) | std_mag (g) | std_xyz (x,y,z) (g) | ~Hz |
|-------|----------|-------------|------------|----------------------|-----|
| P1 | Sitting | 1.0196 | 0.0123 | (0.043, 0.018, 0.048) | 14.1 |
| P1 | Standing | 1.0211 | 0.0365 | (0.100, 0.048, 0.092) | 14.1 |
| P1 | Fowler | 1.0175 | 0.0104 | (0.043, 0.031, 0.037) | 14.1 |
| PS1 | Sitting | 1.0242 | 0.0133 | (0.127, 0.040, 0.102) | 13.6 |
| PS1 | Standing | 1.0265 | 0.0619 | (0.103, 0.070, 0.104) | 13.6 |
| PD1 | Sitting | 1.0047 | 0.0101 | (0.045, 0.034, 0.041) | 11.2 |
| PD1 | Standing | 1.0015 | 0.0550 | (0.079, 0.070, 0.090) | 11.2 |
| PD1 | Moving | 1.0013 | 0.0653 | (0.085, 0.117, 0.243) | 11.2 |

Note: "fuller" variant in P1 data (test3_fuller) is a fourth position (extended Fowler): std_mag=0.010 g.

#### EmoWear STb — front vs back (g, 1600 Hz nominal, 10 subjects, whole session)

| Position | mean_mag (g) | std_mag (g) | std_x (g) | std_y (g) | std_z (g) |
|----------|-------------|-------------|-----------|-----------|-----------|
| Front (anterior) | 1.026 | 0.077 | 0.064 | 0.114 | 0.149 |
| Back (posterior) | 0.992 | 0.069 | 0.066 | 0.105 | 0.148 |

Note: whole-session stats (rest + walk + talk + drink mixed). The y-axis is vertical (gravity-aligned) and shows the largest std; z-axis (dorsoventral) reflects cardiorespiratory and SCG motion. Units converted from mg to g.

#### WISDM — front pants-leg pocket smartphone (`front_pocket` carry source; dataset docs: `10 ~= 1 g`, 20 Hz, 36 users in local raw file)

| Activity | mean_mag | std_mag | std_xyz |
|----------|----------|---------|---------|
| Walking | 11.449 | 4.715 | (5.826, 5.022, 4.019) |
| Jogging | 13.431 | 7.240 | (9.168, 9.217, 5.847) |
| Upstairs | 10.697 | 4.298 | (5.495, 4.891, 3.568) |
| Downstairs | 10.832 | 4.437 | (4.956, 4.905, 3.707) |
| Sitting | 9.847 | 0.392 | (4.759, 3.258, 3.736) |
| Standing | 9.815 | 0.360 | (3.235, 1.265, 1.377) |

Notes: The original controlled collection paper states the phone was carried in the **front pants leg pocket** and sampled every 50 ms (20 Hz). This is useful as a front-pocket smartphone reference, but pocket dynamics include fabric coupling and thigh contact rather than direct attachment to the waist/abdomen wall. In the current MotionLens design contract this maps to the explicit `front_pocket` carry label rather than any waist label.

#### WARD — front-center waist node (raw counts, 20 Hz, 21 subjects in local copy)

| Activity | mean_mag | std_mag | mean_samples |
|----------|----------|---------|--------------|
| standing_rest | 1052.960 | 7.978 | 514.3 |
| sitting_rest | 1035.691 | 5.993 | 503.2 |
| lying_rest | 1044.571 | 8.679 | 504.2 |
| walk_forward | 1093.172 | 299.464 | 376.1 |
| upstairs | 1112.359 | 336.942 | 440.0 |
| downstairs | 1070.825 | 447.845 | 399.1 |
| jog | 1210.480 | 681.973 | 337.1 |
| jump | 1049.578 | 808.690 | 415.9 |
| push | 1108.168 | 176.780 | 414.0 |

Notes: The WARD protocol places sensor 3 at the **front center of the waist**. Local MAT inspection confirms five nodes per trial and usable waist-node accelerometer channels. Values are raw sensor counts rather than calibrated g, so WARD is strongest for comparative ordering across activities, not for absolute amplitude calibration.

#### HARTH — lower-back and front-thigh free-living accelerometers (22 subjects, 50/100 Hz local files)

- Local files confirm fixed dual-sensor placement: lower back and right front thigh.
- Lower-back magnitude stays near 1 g for the quieter labels in the released CSVs and increases in variability for more dynamic labels, while thigh variability is consistently larger than back variability.
- Because the released CSV labels are integer-coded and the placements do not include any front/side waist location, HARTH is best treated as an indirect trunk-motion benchmark rather than a direct placement-transfer source.

#### Local iPhone placement sweep — coarse front/side waist-abdomen zones (~100 Hz, one subject)

- 42 clips across 7 placements and 6 activities were analyzed after trimming 10 s from clip edges when possible (note: the canonical trim rule per protocol_readme.txt is 5 s from start and 15 s from end; Phase 1 used symmetric 10 s as a conservative approximation; the analysis script has been corrected to the canonical rule).
- Sample rate is highly stable at ~99.64 Hz across recordings.
- Walking cadence clusters tightly around 1.77-1.91 Hz and running around 2.76-2.79 Hz across placements.
- Front-side left/right walking clips differ by only `std_mag=0.044` and cadence `0.001 Hz` after trimming, which supports using left/right front-side zones as near-symmetric coarse categories.
- Front-center walking shows an ordered lower-to-upper change in `std_mag` (`1.446 -> 1.567 -> 1.643`), while front-center running shows a different ordered trend (`4.897 -> 4.531 -> 3.840`), suggesting that vertical effects exist but are activity-dependent and should be used as local calibration cues rather than universal scaling laws.

### 2.3 Placement comparison

**RealWorld2016 waist vs chest:** The signal magnitude std at the chest is 5–8% lower than waist during walking and running, reflecting less hip-swing influence. The axis distribution differs: chest has more y-axis (vertical) and less x-axis (mediolateral) variability during walking. For static positions (sitting, lying), chest and waist are nearly identical (< 0.05 m/s² difference in std_mag). This confirms that for placements spanning the front and sides of the waist/abdomen region, signal characteristics interpolate between waist-like and chest-like behavior in a position-dependent manner.

**EmoWear front vs back at sternum level:** Front and back show nearly identical std patterns (within 0.01 g per axis), indicating that at the sternal/xiphoid level the anterior and posterior walls are strongly coupled mechanically, especially during locomotion. The slight difference (front std_mag 0.077 vs back 0.069) may reflect anterior abdominal wall compliance adding a small noise floor.

**WoW standing vs sitting std_mag ratio:** Standing shows ~3× higher std_mag than sitting across all three patch types, driven by postural sway. This standing-to-sitting ratio is consistent across patch variants (P1: 0.0365/0.0123=2.97; PS1: 0.0619/0.0133=4.65; PD1: 0.0550/0.0101=5.44), suggesting patch compliance (stomach vs rib) amplifies postural dynamics.

### 2.4 Inter-subject variability

- **RealWorld2016:** 15 probands all included in averages. Within-activity variability is implicitly captured in the std values reported (which are temporal std within each proband, then averaged). Cross-proband variability not directly computed but the dataset has consistent activity labels.
- **EmoWear:** 48 subjects (only 10 sampled here). Whole-session stats are stable across subjects given large sample sizes (~3–5M rows/subject). Body composition (BMI, abdominal adiposity) is not reported in the dataset but would be expected to modulate abdominal wall compliance.
- **WoW:** Only 4–6 subjects per patch variant. Very small sample; variability in patch placement on the abdomen likely dominates over inter-subject physiology.

### 2.5 Data quality notes

- **WoW sample rate (~11–14 Hz):** This is below the Nyquist frequency for meaningful locomotion dynamics (step frequency ~2 Hz, harmonics up to ~6 Hz). The WoW IMU data can only characterize static/postural positions and very low-frequency signals (breathing, slow sway). No walking dynamics are available.
- **EmoWear timing:** The STb sensor runs at 1600/1100/208 Hz (hardware-dependent). The CSV timestamps are at millisecond resolution. High sample rate is appropriate for SCG but is much higher than needed for gross locomotion simulation.
- **RealWorld2016 header:** The CSV files have a UUID in the header row (must be skipped). The actual columns are `id, attr_time, attr_x, attr_y, attr_z`. The `attr_time` is Unix epoch in milliseconds; median spacing = 20 ms = 50 Hz confirmed.
- **WoW units:** IMU values are in **g** (not m/s²). Gravity component ≈ 1.0 g, confirming upright sensor orientation during sitting/standing tests.
- **EmoWear units:** Raw CSV values are in **mg** (milligravity). Division by 1000 gives g. Values like accY ≈ –1000 mg = –1.0 g confirm gravity along –y when worn on sternum.

---

## 3. Findings

**F1 — Abdomen sensor coherence with lumbar reference is axis-dependent**  
Source: Sotirakis et al. 2025 [DOI:10.1016/j.jbiomech.2025.112975]  
Coherence between abdomen and L5 lumbar IMU exceeds 0.9 for vertical and most axes during walking, but drops below 0.9 for mediolateral acceleration and pitch rotation during walking, and substantially below 0.9 for anteroposterior acceleration and pitch rotation during quiet stance. The abdomen is not a rigid extension of the lumbar spine; abdominal tissue compliance introduces axis-specific, posture-dependent noise.

**F2 — The only dataset with direct front abdominal IMU data (WoW) is limited to static positions and low sample rate (~11–14 Hz)**  
Source: WoW dataset (`data/DataSet_RR/`), `phase1_data_analysis.py` output  
WoW covers only sitting, standing, and Fowler positions with 4–6 subjects per patch variant. There are no walking or dynamic activity recordings. The IMU sample rate (~11–14 Hz) is insufficient to capture locomotion dynamics. This is the most directly relevant dataset for true front abdominal placements but covers only static use cases.

**F3 — Static front abdominal ACC magnitude is ~1.0 g with very low std (0.010–0.065 g depending on patch and posture)**  
Source: WoW dataset, `phase1_data_analysis.py`  
Sitting: std_mag ≈ 0.010–0.013 g. Standing: std_mag ≈ 0.037–0.062 g. Standing generates ~3–5× more variability than sitting due to postural sway. Patch compliance (stomach vs rib) amplifies these dynamics — the abdominal patch (stomach) shows higher sway response than the thoracic patch (rib).

**F4 — Waist and chest ACC provide complementary proxies for front/side waist-abdomen macro-dynamics; differences are axis-specific**  
Source: RealWorld2016 (`data/realworld2016_dataset/`), 15 probands, `phase1_data_analysis.py`  
During walking, chest std_mag = 2.014 m/s² vs waist std_mag = 2.191 m/s² (8% lower). During sitting/lying, difference is < 5%. The chest shows relatively more y-axis (vertical) variability and less x-axis (mediolateral) variability than waist during locomotion, reflecting reduced hip sway influence at the thorax. For a simulator covering the front and sides of the waist/abdomen, chest is a vertical/upward proxy while waist is the closest side/front-side beltline proxy.
During walking, chest std_mag = 2.014 m/s² vs waist std_mag = 2.191 m/s² (8% lower). During sitting/lying, difference is < 5%. The chest shows relatively more y-axis (vertical) variability and less x-axis (mediolateral) variability than waist during locomotion, reflecting reduced hip sway influence at the thorax. For MotionLens, chest is an upward trunk proxy while waist is the closest side/front-side beltline proxy.

**F5 — Axis distribution of ACC variability is activity-specific and must be preserved per-axis, not just by magnitude**  
Source: RealWorld2016 (`data/realworld2016_dataset/`), `phase1_data_analysis.py`  
For walking at waist: std ratios are x:y:z = 2.15:1.42:2.14 m/s². For running: 6.42:2.21:3.17. For sitting: 0.83:0.41:1.58. For standing: 0.45:0.28:3.77. Each activity has a distinct per-axis signature that a classifier or reporting pipeline should preserve; collapsing to scalar magnitude loses this structure.

**F6 — EmoWear front (xiphoid level, anterior) and back have nearly identical std patterns; front shows ~10% higher std_mag**  
Source: EmoWear (`data/Emowear-raw/`), 10 subjects, `phase1_data_analysis.py`  
Front std_mag = 0.077 g; back = 0.069 g. Per-axis: front std_z = 0.149 g, back = 0.148 g (dorsoventral identical). Front std_x = 0.064 g, back = 0.066 g (mediolateral essentially identical). Front std_y = 0.114 g, back = 0.105 g (vertical slightly higher on front). The 10% front vs back magnitude difference suggests the anterior abdominal wall adds a small noise contribution but the main motion signal is the same.

**F7 — The vertical (y) axis is gravity-dominated; lateral and dorsoventral axes carry SCG and postural sway signals**  
Source: EmoWear (`data/Emowear-raw/`), WoW (`data/DataSet_RR/`), `phase1_data_analysis.py`  
In EmoWear front data: mean accY ≈ –1000 mg = –1.0 g (gravity). In WoW: mean_mag ≈ 1.0 g (upright gravity). The dorsoventral and mediolateral axes have near-zero mean and carry dynamic content. This axis decomposition is consistent with the gravity separation approach used in UCI HAR (0.3 Hz Butterworth low-pass to extract gravity component [DOI:10.24432/C54S4K]).

**F8 — The repo covers front abdominal statics directly and waist/side locomotion directly, but not all placements across the full front-plus-side region**  
Sources: WoW (static only, ~14 Hz), EmoWear (sternal/xiphoid, mixed activities), RealWorld2016 (chest/waist proxy, 50 Hz), **UCI HAR dataset_uci** README + placement image (`data/human+activity+recognition+using+smartphones/UCI HAR Dataset/dataset_uci/`)  
Walking and climbing are directly supported for waist / beltline placements through RealWorld2016 waist and UCI side-waist data, while front abdominal locomotion still requires proxy treatment with acknowledged uncertainty. Running is represented at the waist in RealWorld2016 but not directly at front abdominal placements. The remaining gap is therefore not “waist/abdomen locomotion absent,” but “continuous placement coverage across front-center, front-side, and lateral waist/abdomen is incomplete.”

**F9 — Sample rate of 50 Hz is adequate for locomotion and low-frequency respiration-related ACC, but not for SCG-faithful simulation**  
Source: UCI HAR [DOI:10.24432/C54S4K] (50 Hz, waist), RealWorld2016 (50 Hz, confirmed), EmoWear (1600 Hz for SCG)  
Step frequency ≈ 1–2 Hz; harmonics relevant for gait are within the passband of 50 Hz sampling. This is adequate for gross locomotion and low-frequency posture or breathing-related acceleration structure. SCG-focused analysis still requires materially higher sampling rates (EmoWear uses 1600 Hz), so 50 Hz should be treated as sufficient for MotionLens locomotion, cadence, and quiet-window respiratory-proxy features, but not for SCG-level interpretation.

**F10 — Subject count in directly relevant data (WoW) is very small (4–6 subjects, static only); RealWorld2016 (15 subjects, 7 activities) at proxy locations is the statistical backbone**  
Sources: WoW, RealWorld2016, `phase1_data_analysis.py`  
Generalization from WoW alone would be unreliable. MotionLens must therefore use WoW only as a static abdominal reference and rely on RealWorld2016 chest/waist data (F4), plus the other public locomotion datasets, for its activity and cadence backbone.

**F12 — UCI HAR `dataset_uci` is a relevant waist / lateral-beltline locomotion reference, not front-abdomen ground truth**  
Source: `data/human+activity+recognition+using+smartphones/UCI HAR Dataset/dataset_uci/README.txt`, image `waist_mounted_phone.PNG`, data inspection 2026-04-21  
The `dataset_uci` add-on documents a smartphone "worn around the waist" at 50 Hz across six activities in 30 participants aged 22–79. The included image is consistent with a right lateral beltline / side-waist placement. This makes it directly relevant for side or front-side waist placements and as a preprocessing reference for gravity separation, but not as evidence for a centered front-abdomen placement specifically.

The dataset also requires explicit reconstruction before sample-level activity statistics can be treated as design evidence: the accelerometer files contain 384611 sample rows in total, whereas the label files contain 5744 rows. Any per-sample activity statistics, covariance matrices, or phase offsets derived from this dataset must therefore document the label-expansion or alignment method in a checked-in script before they are promoted to Phase 2 evidence.

**F13 — UCI/RealWorld differences are useful for side-waist versus higher-trunk sensitivity analysis, but not for front-abdomen-specific calibration**  
Source: F12 data-cardinality review vs ANALYSIS_REPORT §2.2 (RealWorld2016 chest)  
Comparisons between `dataset_uci` and RealWorld2016 may still be useful for exploratory sensitivity analysis across side-waist versus higher-trunk placements. However, because the UCI add-on is only documented as waist-mounted and its sample-level activity reconstruction is not yet checked into the repo, these differences should not be attributed directly to centered front-abdomen compliance or used to calibrate front-abdomen-specific noise parameters.

**F14 — UCI `dataset_uci` covariance and phase-fit claims are currently non-reproducible in this repo and should not be used as approved design evidence**  
Source: citation audit of `analyses/_covariance_fit.py` in this repo; `dataset_uci` file cardinalities reviewed 2026-04-21

This report previously cited `analyses/_covariance_fit.py` for per-activity covariance matrices and walking phase offsets, but that script is not present in the repository. Combined with the unresolved label-expansion step described in F12, the reported covariance matrices and phase offsets cannot currently be independently reproduced from the repo contents.

Multivariate coupling across axes remains plausible for trunk-mounted motion data, but Phase 2 should treat that as an open analysis task until a checked-in script reproduces the result from raw files.

**F15 — WARD adds direct evidence for a fixed front-center waist placement and confirms the expected activity-intensity ordering at that location**  
Source: WARD protocol text supplied by user from the original paper; local MAT inspection of `data/WARD1.0/`; `phase1_data_analysis.py`

WARD is the strongest newly added direct dataset because one node is explicitly located at the **front center of the waist**. Even in raw counts, the waist-node summary is highly structured: rest postures are quiet, walking and stairs are distinctly more variable, and jog/jump are the highest-dynamic activities. WARD therefore strengthens confidence that the simulator should preserve a strong frontal waist locomotion signature, but it does not solve continuous placement transfer because only one frontal waist location is measured.

**F16 — WISDM is a controlled front-pocket smartphone proxy, not a body-fixed waist/abdomen reference**  
Source: Kwapisz et al. 2010 [DOI:10.1145/2003653.2003656], paper PDF; local raw file `data/WISDM_ar_v1.1/WISDM_ar_v1.1_raw.txt`; `phase1_data_analysis.py`

The WISDM paper explicitly states that subjects carried the Android phone in their **front pants leg pocket** and recorded at 20 Hz. Its walking, jogging, stairs, sitting, and standing classes are clearly separated in the local summary, so it is useful as a controlled smartphone-style front-pocket reference. However, pocket coupling to clothing and the thigh makes it a proxy for a front-pocket carry mode rather than direct evidence for a sensor fixed to the waist/abdomen wall. In the current MotionLens placement contract this supports the explicit `front_pocket` label, not any waist label.

**F17 — HARTH improves confidence in free-living trunk-motion structure, but remains indirect for this project’s spatial scope**  
Source: Logacjov et al. 2021 [DOI:10.3390/s21237853]; local files `data/harth/S*.csv`; `phase1_data_analysis.py`

HARTH provides professionally annotated, fixed-placement free-living accelerometer data with strong posture and locomotion coverage, but its sensors are on the lower back and front thigh only. This makes it useful as an indirect trunk/lumbar reference and as supporting evidence that lower-back and thigh carry separable motion content in free living. It does not provide direct evidence for front-center, front-side, or lateral waist/abdomen placements.

**F18 — The new datasets strengthen specific anchor points but do not remove the need for coarse zone-based placement transfer**  
Source: F12–F17

WARD adds one direct front-center waist anchor, WISDM adds a controlled front-pocket proxy, and HARTH adds indirect trunk context. None of these datasets sample a dense grid across front-center, front-side, and lateral waist/abdomen placements under the same protocol. The central design constraint therefore remains unchanged: placement transfer should stay coarse and evidence-backed rather than pretending to be a dense anatomical interpolation.

**F19 — The local iPhone placement sweep is the only dataset in the repo that directly samples the project’s coarse zone geometry, but it should be treated as calibration-only evidence**  
Source: `data/Acceleration/`; `phase1_data_analysis.py`

The iPhone session is the strongest direct evidence for the actual zone layout used in the design because it includes front-center vertical levels plus front-side and lateral placements under the same device and same-day protocol. Its trimmed summaries support coarse left/right symmetry and confirm plausible cadence consistency across placements. However, because it is a single-subject, single-session dataset with remounting and no population diversity, it should constrain coarse transfer sanity checks rather than serve as the primary statistical backbone for amplitude calibration.

**F11 — WoW dataset contains BioZ respiration waveforms usable as a candidate breathing-shape reference**
Source: WoW dataset (`data/DataSet_RR/`), `BioZ.csv` and `rr_out.csv` files; WoW README PDF
`BioZ.csv` files are present for patch P1 alongside IMU recordings in sitting, standing, and Fowler tests, and `rr_out.csv` files provide processed respiration traces. This supports using a real torso-mounted waveform family instead of a purely synthetic sinusoid for the breathing component. However, any claimed breathing-rate distribution or fitted waveform parameterization should be backed by a checked-in extraction script before being treated as calibrated Phase 2 evidence.

**F20 — The installed WHAR Datasets library adds a reproducible standardization layer across 37 public HAR datasets**
Source: installed `whar_datasets` package inventory generated by `analyses/whar_dataset_integration.py`; `output/whar/whar_dataset_inventory.csv`

The library exposes 37 dataset configs with common metadata fields (sampling rate, subjects, activities, channels, and source URLs) and a shared preprocessing pipeline that emits `activity_df`, `session_df`, and `window_df`. This makes it practical to expand MotionLens beyond the repo-local datasets without building one-off parsers for each new source.

**F21 — MotionSense is the closest public proxy for live iPhone sensor streaming in MotionLens**
Source: MotionSense dataset README / IoTDI 2019 citation [DOI:10.1145/3302505.3310068]; `output/whar/whar_preprocessing_summary.csv` generated by `analyses/whar_dataset_integration.py` on 2026-04-22

MotionSense was collected with an iPhone 6s in the front pocket using iOS Core Motion via SensingKit at 50 Hz and includes attitude, gravity, rotation rate, user acceleration, plus accelerometer and gyroscope views. In the WHAR export path it standardizes to 360 sessions and 21,528 windows across 24 subjects and six activities. This is the strongest public evidence that MotionLens should support an ACC-first live phone path with optional gravity and rotation features when the device/browser exposes them. In the current placement contract, MotionSense is a `front_pocket` carry source rather than a waist placement source.

**F22 — HHAR shows that device heterogeneity is a first-class accuracy risk for smartphone activity recognition**
Source: HHAR dataset page [DOI:10.24432/C5689X]; `output/whar/whar_preprocessing_summary.csv` generated by `analyses/whar_dataset_integration.py` on 2026-04-22

HHAR was gathered across eight smartphones, four smartwatches, and realistic walking, stairs, and biking routes to reflect sensing heterogeneity expected in real deployments. The WHAR export produced 10,813 sessions and 24,544 windows across nine users and six activities. MotionLens therefore should not promise a single deployment-wide accuracy number without conditioning on device family, sampling behavior, and available sensor channels.

**F23 — Real-Life HAR is the best public proxy for unconstrained smartphone carrying, but it forces ACC-first fallbacks**
Source: Real-Life HAR site abstract and dataset description; `output/whar/whar_preprocessing_summary.csv` generated by `analyses/whar_dataset_integration.py` on 2026-04-22

Real-Life HAR was collected from subjects using their own smartphones in real-life conditions with no fixed placement or orientation. The source explicitly notes irregular sampling frequency on Android devices and missing gyroscope or magnetometer channels in some sessions. The WHAR export produced 468 sessions and 443,642 windows across 17 subjects and four coarse activities. This supports an ACC-first inference path with optional richer sensors, lower confidence under uncontrolled phone usage, and a separate free-living evaluation tier.

**F24 — WISDM remains a controlled laboratory baseline rather than a deployment-faithful smartphone benchmark**
Source: WISDM dataset page; Kwapisz et al. 2010 [DOI:10.1145/2003653.2003656]; `output/whar/whar_preprocessing_summary.csv` generated by `analyses/whar_dataset_integration.py` on 2026-04-22

WISDM is explicitly collected under controlled laboratory conditions and the dataset page directs users elsewhere for real-world data. The WHAR export produced 179 sessions and 20,352 windows across 36 users and six activities. MotionLens can use it as a clean locomotion and posture ACC baseline, but not as evidence that browser-streamed phone inference will transfer unchanged to unconstrained everyday use.

**F25 — Unified session/window manifests are now reproducible repo artifacts, not just a future idea**
Source: `analyses/whar_dataset_integration.py`; `output/whar/whar_unified_session_manifest.csv`; `output/whar/whar_unified_window_manifest.csv`

The repo now contains a checked-in script that inventories the WHAR catalog and exports standardized per-dataset metadata plus unified manifests that map dataset, subject, activity, session, and window IDs to the library's cached parquet artifacts. This is sufficient to build cross-dataset evaluation splits and a larger training pool without flattening all raw sensor files into one bespoke format first.

---

## 4. Implications for MotionLens design

### 4.1 Evidence-backed roles in the MotionLens pipeline

| Dataset | Role in MotionLens |
|---------|--------------------|
| **RealWorld2016 waist/chest** | Primary backbone for activity segmentation, cadence estimation, and motion-intensity summaries; chest remains the main upward trunk proxy [F4, F5] |
| **MotionSense** | Best public proxy for iPhone `DeviceMotion` streaming, including gravity and rotation-derived channels [F21] |
| **HHAR** | Heterogeneity stress test across phone/watch models and realistic routes; strongest evidence that deployment accuracy must be tiered by device context [F22] |
| **Real-Life HAR** | Free-living smartphone robustness source for ACC-first fallbacks and unconstrained carry evaluation [F23] |
| **UCI HAR / `dataset_uci`** | Supplemental walk/stairs/posture training data and the clearest in-repo reference for gravity/body separation at 0.3 Hz [F7, F12] |
| **HARTH** | Free-living trunk-motion robustness for activity segmentation, not direct placement evidence [F17] |
| **WARD waist node** | Direct front-center waist anchor for frontal activity ordering and heuristic quality checks [F15] |
| **WISDM** | Smartphone carry-style robustness and supplemental cadence contrast for phone-like recordings [F16] |
| **Local iPhone placement sweep** | Weak but direct same-device evidence for front-center/front-side/lateral placement-quality heuristics [F19] |
| **WoW** | Static front-abdominal sway and respiration reference for quiet-window breathing-proxy work [F2, F3, F11] |
| **EmoWear front** | Front-torso respiratory context and motion-contamination caution for low-motion respiratory plausibility checks [F6, F7] |
| **WHAR unified manifests** | Reproducible session/window export layer for cross-dataset training, evaluation, and ablation work [F20, F25] |
| Wearable/PhysioNet | Not part of the MotionLens MVP evidence chain because placement is wrist-only |

### 4.2 Feasible MotionLens outputs

| Feature | Evidence quality | Evidence base | Constraints |
|---------|------------------|---------------|-------------|
| Gravity/dynamic separation | STRONG | UCI HAR preprocessing + RealWorld2016/UCI signal behavior [F7, F9, DOI:10.24432/C54S4K] | Phase 2 still needs explicit unit normalization and filter implementation details |
| Activity segmentation | STRONG | RealWorld2016 + UCI HAR + HARTH + MotionSense + HHAR + Real-Life HAR [F4, F5, F12, F17, F21, F22, F23] | Label harmonization and evaluation protocol remain open |
| Cadence estimation | STRONG | RealWorld2016 + UCI HAR + MotionSense + WISDM + local iPhone sweep [F9, F12, F16, F19, F21] | Best for rhythmic locomotion windows; confidence should drop for nonperiodic motion |
| Motion-intensity timeline | STRONG | RealWorld2016 + WARD + MotionSense + WISDM + Real-Life HAR + local iPhone sweep [F3, F15, F16, F19, F21, F23] | Use relative intensity language, not energy-expenditure claims |
| Quiet-window breathing-rate proxy | HONEST DEMO | WoW + EmoWear + Nicolò et al. 2020 [F2, F3, F6, F7, F11, DOI:10.3390/s20216396] | Only for static/quiet windows; motion artifacts and sensor location materially affect signal quality |
| Placement-quality indicator | WEAK | Local iPhone sweep + WARD + cross-dataset heuristics [F15, F18, F19] | Red/yellow/green only; not continuous anatomical inference or a population-validated placement classifier |

The portfolio strength of MotionLens is that it exposes the full processing chain rather than asking users to trust a black-box placement classifier. The weakest prior framing was dense placement inference across the front-plus-side region; MotionLens replaces that with transparent DSP plus explicit quality gating.

The new WHAR-backed exports also imply that MotionLens should present at least two evaluation tiers in Phase 2: controlled-phone/body-worn performance and free-living heterogeneous-phone performance. HHAR and Real-Life HAR make it clear that collapsing those into one headline number would overstate real deployment accuracy [F22, F23, F25].

### 4.3 Signal properties MotionLens must preserve or estimate

1. **Gravity/body decomposition**: ~1.0 g (9.81 m/s²) along the orientation-dependent gravity axis, with a stable low-pass/body split suitable for before/after plots [F7, F9].
2. **Per-axis activity structure**: activity evidence is not scalar magnitude only; the x/y/z variance pattern is informative and should remain visible in features and plots [F5].
3. **Step periodicity in the locomotion band**: walking/running-like windows carry clear ~1–3 Hz periodic structure that should drive cadence estimates and their confidence [F9, F19].
4. **Quiet-window noise floor**: low-motion sitting/standing baselines help distinguish actual rest, poor mounting, and windows suitable for breathing-proxy attempts [F3].
5. **Respiratory-band quality gating**: quiet-window respiratory estimates should only be attempted after screening for low motion, stable gravity angle, and plausible front-torso dynamics because motion artifacts and sensor location strongly affect respiratory signal quality [F1, F3, F7, DOI:10.3390/s20216396].
6. **Heuristic placement quality, not dense placement regression**: the evidence supports stability/range checks and coarse same-device sanity rules, not fine-grained placement inference across anatomy [F18, F19].
7. **Sensor-availability fallback path**: live phone inference must work with accelerometer only, while optionally using gravity and rotation-derived channels when the runtime exposes them; public evidence does not support assuming a fixed full sensor bundle in real-life phone use [F21, F23].

### 4.4 Interpretation scope and limits

- MotionLens can accept generic raw ACC input, but the strongest evidence-backed interpretations come from waist/trunk recordings.
- Live phone capture should be ACC-first with optional `DeviceMotion` enrichment, not hard-required gravity, gyro, or magnetometer channels [F21, F23].
- Front-plus-side waist/abdomen remains the only placement-quality calibration domain in this repo; the back is excluded.
- Fifty-hertz data is adequate for locomotion, cadence, motion intensity, and a low-frequency respiratory demo, but not for SCG-faithful or clinical-grade cardiorespiratory analysis [F9].
- Respiratory output should be framed as a quality-gated estimate from quiet windows, not as continuous breathing monitoring or a clinical respiratory monitor [DOI:10.3390/s20216396].
- Placement output should be framed as signal-quality guidance rather than placement classification.
- Controlled-lab and free-living accuracy claims should be reported separately because phone heterogeneity, unconstrained orientation, and missing channels materially change deployment difficulty [F22, F23].

### 4.5 Unresolved questions

**UQ1 — Cross-dataset label harmonization:** RealWorld2016, UCI HAR, HARTH, WISDM, and WARD do not expose the same label sets or sampling assumptions. Phase 2 must define a harmonized MVP taxonomy before training or evaluation.

**UQ2 — CSV normalization and unit handling:** MotionLens will ingest raw CSVs and browser `DeviceMotion`, but the evidence base mixes g, m/s², raw counts, and phone-specific units. Phase 2 must define a reproducible normalization strategy.

**UQ3 — UCI sample-level reconstruction reproducibility:** UCI remains useful for preprocessing and window-level activity evidence, but the missing checked-in reconstruction step still blocks some sample-level claims [F12, F14].

**UQ4 — Placement-quality thresholds beyond one-subject calibration:** the iPhone sweep is directly relevant but still single-subject. Any thresholds used in a red/yellow/green quality indicator should be presented as heuristic unless more subjects are collected [F19].

**UQ5 — Quiet-window breathing validation:** WoW and EmoWear support a cautious respiratory demo, but Phase 2 still needs explicit gating and confidence logic that shows when motion contamination makes the estimate untrustworthy [F2, F6, F11, DOI:10.3390/s20216396].

**UQ6 — Live browser sensor contract:** MotionSense supports the value of gravity and rotation-derived channels for a phone-streamed workflow, but Real-Life HAR shows that real devices and sessions do not guarantee uniform sensor availability or fixed sampling. Phase 2 must decide the required minimum input contract for live capture versus optional enrichments [F21, F23].

**UQ7 — Controlled vs free-living evaluation tiers:** the new WHAR exports make it feasible to report separate controlled, heterogeneous-device, and free-living evaluation tiers, but Phase 2 still needs a concrete benchmarking protocol and a rule for what accuracy claims are allowed in the UI and README [F22, F23, F25].

