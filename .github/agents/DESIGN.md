# DESIGN
> Agent writes approved design decisions here. Nothing is implemented until it appears here.
> Each entry requires a citation.
> Reset note: previous app-first decisions are superseded by the 2026-04-22 MotionLens project reset and must not be treated as approved for implementation.

## Status
Phase 2 is in progress under the 2026-04-22 MotionLens reframing.
Next step: resolve Batch 2 derived-metric decisions, starting with the rhythm-consistency metric definition.

## Entry format
```
### [topic]
Decision: …
Evidence: [ANALYSIS_REPORT §X] or [DOI, claim]
Approved: yes
```

---

### Input contract
Decision: MotionLens uses an ACC-first canonical input contract. Required fields are timestamp plus 3-axis accelerometer samples. Optional enrichment fields may include gravity, user acceleration, gyroscope, magnetometer, GPS, or rotation-derived channels when they are available. Live browser capture and uploaded CSV workflows must remain functional with accelerometer-only input.
Evidence:
- MotionSense [DOI:10.1145/3302505.3310068] — iPhone 6s front-pocket, Core Motion via SensingKit at 50 Hz; provides acc, gravity, rotation-rate, and user-acceleration channels — shows that optional enrichment is available on iOS but cannot be assumed universally; ACC is the guaranteed base channel [F21]
- Real-Life HAR [ANALYSIS_REPORT §F23] — irregular Android sampling and missing gyroscope/magnetometer channels in real-life sessions; confirms ACC must be the sole required channel or inference breaks in deployment [F23, UQ6]
- HHAR [DOI:10.24432/C5689X] — eight smartphone and four smartwatch models across heterogeneous routes; requiring a fixed full sensor bundle would exclude the majority of real device configurations [F22]
- RealWorld2016, UCI HAR, WARD, WISDM — all include only 3-axis accelerometer plus timestamp as the minimal common denominator across the repo's direct evidence base [F4, F12, F15, F16]
Approved: yes

### MVP gait and intensity metric scope
Decision: MotionLens narrows the initial gait and intensity analytics MVP to two reportable walking-window metrics: impact intensity and rhythm consistency. Step symmetry and stability remain deferred beyond the MVP until the project has stronger validation for fixed placement, window definitions, and controlled versus free-living behavior.
Evidence:
- RealWorld2016 (Sztyler & Stuckenschmidt, Uni Mannheim), waist placement, 15 probands, 50 Hz: walking std_mag = 2.191 m/s² vs running = 6.063 m/s² and stair descent = 3.636 m/s² — large inter-activity magnitude differences confirm impact intensity and rhythm as clearly separable reportable targets [F4, F5]
- Local iPhone placement sweep [ANALYSIS_REPORT §F19], 7 placements: trimmed walking cadence 1.77–1.91 Hz and running cadence 2.76–2.79 Hz are stable across front-center, front-side, and lateral zones — supports cadence-based rhythm consistency as a valid per-window output [F19]
- UCI HAR [DOI:10.24432/C54S4K], 30 subjects, waist-mounted Samsung Galaxy S II, 50 Hz: step symmetry and stability features in the published 561-feature set require a fixed lateral-beltline placement and controlled lab conditions that the free-living and multi-device populations in HHAR and Real-Life HAR cannot match — deferring symmetry and stability avoids overclaiming on unconstrained phone data [F12, F22, F23]
- Sotirakis et al. 2025 [DOI:10.1016/j.jbiomech.2025.112975]: abdomen-L5 coherence drops below 0.9 for mediolateral acceleration during walking, confirming that fine asymmetry metrics at the abdomen carry axis-specific noise that would require dedicated validation before release [F1]
Approved: yes

### Preprocessing contract
Decision: MotionLens canonicalizes both uploaded CSV data and live phone capture into one internal accelerometer preprocessing contract before ML inference or reporting. The required internal fields are monotonic elapsed time plus acc_x, acc_y, and acc_z in m/s^2. Accepted input timestamps may be ISO datetimes, Unix seconds, Unix milliseconds, or elapsed milliseconds, but they are normalized into one monotonic timeline while preserving the raw source format in metadata. Accepted acceleration units may be g or m/s^2, but they are normalized to m/s^2 before feature extraction. Preprocessing includes sorting by time, duplicate removal, invalid-row rejection, short-gap interpolation, long-gap session splitting, resampling to 50 Hz, and basic glitch or timestamp hygiene. Main denoising, smoothing, and gravity or body separation filter choices remain a separate design decision.
Evidence:
- UCI HAR [DOI:10.24432/C54S4K]: units in g (÷9.81 for m/s²), sampled at 50 Hz, noise-filtered with a median filter followed by a 3rd-order Butterworth LPF at 20 Hz, windowed into 128-sample (2.56 s) segments with 50% overlap — establishes 50 Hz as the canonical target rate and g→m/s² as the required unit conversion [F9, F12]
- RealWorld2016 (Sztyler & Stuckenschmidt), 15 probands, 7 body positions: units m/s², `attr_time` is Unix epoch milliseconds, confirmed median inter-sample spacing = 20 ms = 50 Hz; UUID in header row must be skipped — demonstrates that raw CSV files from real datasets require header and timestamp hygiene before processing [F4, F5]
- WISDM [DOI:10.1145/2003653.2003656], 20 Hz, Android front pants-leg pocket: sensor units where `10 ≈ 1 g` (not standard g or m/s²) — illustrates the heterogeneous unit convention problem that mandates a normalization step before any cross-dataset feature extraction [F16, UQ2]
- EmoWear [DOI:10.1038/s41597-024-03429-3], STb front IMU at ~50 Hz: raw values in mg (milligravity); mean accY ≈ −1000 mg = −1.0 g confirms gravity along −y — confirms unit detection and conversion are non-trivial and must be handled explicitly [F7, §2.5]
- Local iPhone placement sweep [ANALYSIS_REPORT §F19], iPhone 16, ~100 Hz: measured sample rate = 99.64 Hz — confirms that real phone ACC data requires resampling to the canonical 50 Hz target before features or comparisons are computed [F19]
- Real-Life HAR [ANALYSIS_REPORT §F23]: irregular Android sampling intervals and missing gyroscope/magnetometer channels across sessions — motivates the gap-detection, session-splitting, and short-gap interpolation steps so that uneven real-life recordings do not silently corrupt window timing [F23, UQ6]
Approved: yes

### Gravity and dynamic separation pipeline
Decision: MotionLens uses a UCI-aligned per-axis gravity split on the canonical 50 Hz accelerometer stream. Gravity is estimated with a 0.3 Hz Butterworth low-pass filter on each accelerometer axis, and body acceleration is defined as total acceleration minus gravity acceleration per axis. MotionLens retains three internal streams: total acceleration for raw visualizations and quality checks, gravity acceleration for orientation and quiet-window quality logic, and body acceleration as the default signal for ML activity timeline inference, cadence, rhythm consistency, and impact intensity. No additional aggressive smoothing or denoising is applied by default beyond the approved preprocessing hygiene.
Evidence:
- UCI HAR [DOI:10.24432/C54S4K], Anguita et al. 2013: explicitly applies a 3rd-order Butterworth LPF at 0.3 Hz to the raw 50 Hz total accelerometer to extract the gravity component; body acceleration is defined as total − gravity per axis — this is the exact filter topology MotionLens adopts; also separates gravity into a dedicated channel used for orientation estimation [F7, F9, F12]
- RealWorld2016 (Sztyler & Stuckenschmidt), waist, 15 probands: mean_mag ≈ 9.81 m/s² (≈ g) across all seven activities, confirming that gravity is always the dominant DC component in trunk-mounted ACC and must be removed before locomotion features can be computed [F4, F5]
- EmoWear [DOI:10.1038/s41597-024-03429-3], STb front sensor at xiphoid level: mean accY ≈ −1000 mg = −1.0 g, confirming that the gravity vector sits on whichever axis aligns with vertical and that per-axis gravity separation is needed — gravity is not a fixed global z offset across all devices and orientations [F7]
- Sotirakis et al. 2025 [DOI:10.1016/j.jbiomech.2025.112975]: abdomen–L5 coherence > 0.9 for the vertical axis during walking — confirms that the body (dynamic) component recovered after the 0.3 Hz split retains a valid locomotion signal at the waist/abdomen for the walking case [F1]
- WARD, front-center waist node, 21 subjects, 20 Hz: standing_rest std_mag = 7.978 raw counts vs walk_forward std_mag = 299.464 counts — gravity is the near-constant DC floor in rest; removing it leaves the locomotion signal as the dominant component, confirming the separation strategy at the exact front-waist placement in scope [F15]
Approved: yes

### Activity label set for the MVP report
Decision: MotionLens uses a 7-class activity label set for the MVP report: walk, run, stairs (up and down merged into one class), sit, stand, lay. Running is treated as a distinct class because it is clearly separable from walking in the primary locomotion backbone data. Stairs-up and stairs-down are merged to maintain adequate training-window counts. The label set is applied in a tier-aware way: the controlled tier (UCI HAR + WISDM) uses a 6-class compatible subset (no run class); the heterogeneous-device tier (HHAR + MotionSense) includes the run class where it is present; the free-living tier (Real-Life HAR) maps to 4 coarse classes (locomotion, static-upright, sedentary, lying) via a label compatibility layer. The run class is omitted from the free-living tier accuracy claim.
Evidence:
- RealWorld2016 (Sztyler & Stuckenschmidt), waist, 15 probands, 50 Hz: walking std_mag = 2.191 m/s² vs running std_mag = 6.063 m/s² — the 2.8× difference in impact magnitude makes run vs walk the most clearly separable pair in the locomotion backbone and justifies adding running as a distinct class [F4, F5]
- UCI HAR [DOI:10.24432/C54S4K], Anguita et al. 2013, 30 subjects: ships a fixed 6-class set (walk, stairs-up, stairs-down, sit, stand, lay) with no running activity — the 6-class set is the most cross-dataset compatible training target and remains the controlled-tier subset [F12]
- MotionSense [DOI:10.1145/3302505.3310068], iPhone 6s front-pocket, 50 Hz: includes jogging as a distinct label alongside walking, sitting, standing, and stairs (up/down) — confirms the run class is present and labelled in one of the four WHAR training-pool datasets [F21]
- Real-Life HAR [ANALYSIS_REPORT §F23]: only 4 coarse activities; the label compatibility layer maps walk+run+stairs to locomotion; sit+stand to static-upright; lying to lying; cycling to locomotion-other [F23]
- WHAR unified manifest [ANALYSIS_REPORT §F25]: UCI 9,499 + MotionSense 21,528 + HHAR 24,544 + WISDM 20,352 windows; walk/sit/stand/lay are the universal common labels; run/jog is present in MotionSense and RealWorld2016 but not in UCI or WISDM; stairs present in UCI and HHAR but counts are low if split by direction [F25]
Approved: yes

### Windowing strategy and feature set
Decision: 128-sample windows (2.56 s) with 50% overlap at 50 Hz, matching the UCI HAR contract. Feature set per window:
- Time domain: mean per axis (DC / gravity proxy), std, RMS, MAD, min, max, zero-crossing rate, inter-axis correlation coefficients (xy, xz, yz)
- Frequency domain: dominant frequency (FFT peak in 0–25 Hz), spectral entropy (STFT-based), signal energy in 0.5–4 Hz locomotion band, signal energy in 0.1–0.5 Hz breathing band
- Gravity channel: mean gravity magnitude, gravity-vector angle stability (std of gravity angle over window)
All features computed on body-acceleration (gravity-removed) channels except gravity-channel features which use the gravity stream.
Evidence:
- UCI HAR [DOI:10.24432/C54S4K], Anguita et al. 2013: 128 samples at 50 Hz = 2.56 s windows with 50% overlap; this is the canonical reference configuration shared across all four WHAR training datasets and must not be changed without invalidating the pre-trained artifact [F9, F12]
- Mannini & Sabatini 2010 [DOI:10.3390/s100201154]: applied Pudil SFFS feature selection to an 85-feature space over 7 activities; the 17-feature optimal subset retained 4 DC components (per-axis mean = gravity proxy) and 13 inter-axis correlation coefficients, establishing these as the most discriminative single-frame features for locomotion vs. posture classification from trunk ACC
- Mannini & Sabatini 2010: spectral entropy (STFT-derived) discriminates activities that differ in signal complexity — walking generates high-frequency foot-impact noise (high entropy) compared to cycling or standing; this entropy principle applies to the locomotion cadence band at the waist/abdomen
- Bulling, Blanke & Schiele 2014 [DOI:10.1145/2499621]: Activity Recognition Chain framework confirms 50% overlapping sliding windows as the standard for stationary-feature HAR pipelines
Approved: yes

### Classifier choice and training protocol
Decision: XGBoost (xgboost XGBClassifier, joblib-serialized artifact shipped with the app). Pre-trained fixed artifact; no user-facing retrain path in MVP. Training input: [signal_features + placement_onehot]. Full training pool: HAPT, WISDM, MotionSense, HHAR, RealWorld2016, iPhone Placement Sweep, PAMAP2, MHEALTH, SAD, UMA_FALL — except held-out test subjects per dataset (defined below in the Training/test split decision). Gradient-boosted trees consistently outperform Random Forest by 1–3% on tabular feature vectors with identical `feature_importances_` API and the same deployment footprint (xgboost pip package, no PyTorch/TensorFlow).
Considered and rejected alternatives:
- Random Forest (sklearn RandomForestClassifier): same feature_importances_ API; outperformed by XGBoost by 1–3% on tabular HAR feature vectors due to gradient boosting's sequential error correction; no deployment advantage over XGBoost
- CNN-BiLSTM-GRU (Lalwani & Ganeshan 2024 [DOI:10.1007/s44196-024-00689-0]): hybrid deep learning model achieving 99.7% accuracy on WISDM. Rejected because: (1) WISDM is a controlled-lab single-dataset benchmark that is not deployment-faithful per F24 — this accuracy figure overstates cross-dataset generalization; (2) CNN produces no per-window feature importances, which violates the MotionLens interpretability goal; (3) requires PyTorch/TensorFlow dependency conflicting with lightweight Streamlit deployment; (4) encoding placement as a categorical input feature is trivial in tree methods but requires dual-stream architectural changes in CNN
- SVM (RBF kernel): competitive single-frame accuracy per Mannini & Sabatini 2010 Table 3, but provides no feature importances and does not accept categorical placement features without manual encoding overhead
- Sequential HMM: 99.1% vs 98.5% best single-frame per Mannini & Sabatini 2010, but requires subject-specific transition matrices and provides no per-window feature importances
Evidence:
- Mannini & Sabatini 2010 [DOI:10.3390/s100201154]: single-frame classifiers on the approved feature set achieve 98.5% (NM) for 7-class locomotion/posture classification from trunk ACC, confirming that tree-based single-frame methods are sufficient without sequential modeling
- Lalwani & Ganeshan 2024 [DOI:10.1007/s44196-024-00689-0]: CNN-BiLSTM-GRU achieves 99.7% on WISDM; abstract confirms WISDM-only evaluation; WISDM is controlled-lab baseline per F24
- F22: HHAR device heterogeneity is the primary accuracy risk in real deployment; XGBoost with placement encoding addresses device-conditional signal patterns more directly than fixed-architecture CNN
Approved: yes

### Training/test split and placement-aware evaluation
Decision: Subject-level holdout protocol (~20% per dataset) with explicit placement encoding as a one-hot model input feature (9 dimensions: `chest`, `front_center_mid`, `front_center_lower`, `front_side_left_mid`, `front_side_right_mid`, `lateral_left_lower`, `lateral_right_lower`, `front_pocket`, `unknown_free_living`). All windows are windowed at 128 samples / 50 Hz / 50% overlap.

**Held-out test subjects:**
- HAPT (30 subjects): {4, 9, 14, 19, 24, 29} (6 = 20%)
- WISDM (36 subjects): {4, 9, 14, 19, 24, 29, 34} (7 ≈ 19%)
- MotionSense (24 subjects): {4, 9, 14, 19} (4 ≈ 17%)
- HHAR (9 subjects, users a–i): {b, g} (2 = 22%)
- RealWorld2016 (15 subjects): {3, 8, 13} (3 = 20%); `chest` and `front_center_mid` streams only
- PAMAP2 (9 subjects): {2, 7} (2 ≈ 22%); `chest` channel only
- MHEALTH (10 subjects): {2, 7, 10} (3 = 30%); `chest` channel only
- SAD (10 subjects): {2, 7, 10} (3 = 30%); belt channel only
- UMA_FALL (19 subjects): every 5th ID (≈20%); chest + waist + pocket channels, each windowed separately
- iPhone Placement Sweep (1 subject): no test split — training only, excluded from accuracy reporting
- Real-Life HAR (17 subjects): entire dataset held out as free-living evaluation only; never used in training

**Placement one-hot assignment per dataset:**
- HAPT: `lateral_right_lower` (Samsung Galaxy S II right lateral beltline)
- WISDM: `front_pocket` (paper states front pants-leg pocket; side-specific laterality not reported, so the model bucket remains side-agnostic)
- MotionSense: `front_pocket` (README states trousers' front pocket; side-specific laterality not reported, so the model bucket remains side-agnostic)
- HHAR: `unknown_free_living` (placement positions unverifiable as torso)
- RealWorld2016: `chest` (chest stream; the source names the body position simply `chest`, not a stricter centerline upper-front mount); `front_center_mid` (waist stream)
- PAMAP2: `chest` (Colibri IMU on the chest; exact sublocation not specified beyond chest)
- MHEALTH: `chest` (sensor on the subject's chest; exact sublocation not specified beyond chest)
- SAD: `lateral_right_lower` (belt at right lateral waist)
- UMA_FALL: `chest` (chest stream); `front_center_lower` (waist stream — front midline lower abdomen); `front_pocket` (raw `right_pocket` stream folded into the side-agnostic `front_pocket` model bucket; pocket carry kept distinct from waist)
- iPhone Placement Sweep: 7 raw folder labels; raw `front_center_upper` maps to model label `chest`, while `front_center_mid`, `front_center_lower`, `front_side_left_mid`, `front_side_right_mid`, `lateral_left_lower`, and `lateral_right_lower` map directly
- Real-Life HAR: `unknown_free_living`

**Accuracy reporting:**
- Per-dataset LOSO accuracy table (one row per dataset using its held-out subjects)
- HHAR placement-group breakdown (phone_position_a vs phone_position_b) to quantify carrying-position effect
- Free-living row: Real-Life HAR 4-class accuracy (locomotion / static-upright / sedentary / lying)
- No combined headline accuracy number across all datasets
- `front_pocket` is a side-agnostic classifier carry-mode label, not part of the torso-only placement-quality heuristic calibrated on the iPhone sweep
- Default at inference when carry mode is undeclared: `unknown_free_living` (do not invent a specific waist or pocket site from missing metadata)

Evidence:
- HHAR [DOI:10.24432/C5689X], Stisen et al. 2015: device heterogeneity — including placement position — is the primary source of accuracy variation; encoding placement as model input separates device-conditional signal patterns from activity patterns [F22]
- F19: local iPhone sweep shows per-axis std varies by placement zone; placement-conditional signal statistics justify placement as an explicit model input
- RealWorld2016 dataset page lists the upper-trunk body position simply as `chest`; PAMAP2 [DOI:10.24432/C5NW2H] states "1 IMU on the chest"; MHEALTH dataset page states sensors were placed on the subject's chest, right wrist, and left ankle. These sources support a broader `chest` bucket rather than a stricter `front_center_upper` centerline label.
- F16: WISDM front pants-leg pocket carry is thigh-coupled and not a direct waist/abdomen attachment, so it should not be collapsed into a waistband label
- F21: MotionSense was collected with an iPhone kept in the participant's trousers' front pocket; it is a front-pocket carry source, not a chest or waistband source
- UMA_FALL local CSV header: `RIGHTPOCKET`, `WAIST`, and `CHEST` are enumerated as separate sensor positions, so pocket carry is a distinct placement source from waist-mounted SensorTags
- F23: Real-Life HAR has no fixed placement and irregular sampling; complete holdout as free-living evaluation set is the only defensible use
- F24: WISDM is a controlled-lab baseline; WISDM holdout subjects give controlled-lab accuracy reference, not deployment accuracy
- Mannini & Sabatini 2010 [DOI:10.3390/s100201154]: DC component is placement-sensitive; encoding placement explicitly allows the model to learn placement-conditional DC thresholds without inflating intra-class variance
Approved: yes

### Cadence estimation method and confidence scoring
Decision: FFT peak on body-acceleration magnitude in the 0.8–3.5 Hz search band as the primary cadence estimator. Augmented with a harmonic cross-check: if the detected peak frequency $f$ satisfies $f/2 > 0.8$ Hz, also evaluate whether $f/2$ has a peak-to-mean spectral ratio > 1.5; if yes, flag the cadence as "possibly a harmonic" and report the lower value ($f/2$) as the secondary estimate. Confidence thresholds:
- **High**: body-acc std > 0.8 m/s² AND FFT peak in band AND peak-to-mean spectral ratio > 3
- **Low**: peak in band but ratio 1.5–3
- **Suppressed** (no cadence displayed): body-acc std ≤ 0.8 m/s² OR no clear peak in band
Evidence:
- Zijlstra & Hof 2003 [DOI:10.1016/S0966-6362(02)00190-X]: "duration of subsequent stride cycles and left/right steps... can be obtained from lower trunk accelerations" via autocorrelation; the autocorrelation function shows peaks at both the stride period and step period (stride/2), confirming that the FFT spectrum can land on the harmonic of the fundamental step frequency — directly motivates the harmonic cross-check
- F19: local iPhone sweep, 7 placements: trimmed walking cadence 1.77–1.91 Hz and running cadence 2.76–2.79 Hz are stable across front-center, front-side, and lateral zones — validates the 0.8–3.5 Hz search band
- F3: WoW standing std_mag ≈ 0.037–0.062 g ≈ 0.36–0.61 m/s²; the 0.8 m/s² high-confidence threshold lies above the maximum observed static-standing noise floor, ensuring cadence is not reported for non-locomotion windows
Approved: yes

### Training pool dataset details and placement vocabulary
Decision: All datasets are windowed at 128 samples / 50 Hz / 50% overlap. Placement vocabulary uses 7 torso zones plus one explicit `front_pocket` carry label and one `unknown_free_living` fallback label. The upper-front torso bucket is named `chest` rather than `front_center_upper` because the public chest-mounted datasets do not support a strict centerline claim. Non-target body positions (forearm, head, shin, thigh, upper-arm, wrist, and ankle) are discarded. WARD (uncalibrated raw counts), HARTH (lower_back only — label retired), HAR70 (lower_back only — label retired), EmoWear, WoW, and Wearable Dataset (PhysioNet) are excluded.

**Per-dataset channel selection and activity mapping:**

- **HAPT** (Reyes-Ortiz et al. 2016, UCI [DOI:10.24432/C54G7P]): 30 subjects, 50 Hz, Samsung Galaxy S II right lateral beltline. Channels: `acc_x, acc_y, acc_z`. Activity mapping: Walking→walk, Walking Upstairs/Downstairs→stairs, Sitting→sit, Standing→stand, Laying→lay, all postural transitions→transitions. Supersedes UCI HAR (strict superset with 6 transition labels added).

- **WISDM** (Weiss et al. 2019, UCI [DOI:10.24432/C5X61X]): 36 subjects, 20 Hz, front pants-leg pocket (side not reported). Channels: `accel_x, accel_y, accel_z`. Activity mapping: Walking→walk, Jogging→run, Upstairs/Downstairs→stairs, Sitting→sit, Standing→stand.

- **MotionSense** (Malekzadeh et al. 2019, GitHub [DOI:10.1145/3302505.3310068]): 24 subjects, 50 Hz, trousers front pocket (side not reported). Channels: `userAcceleration.x/y/z`. Activity mapping: Walking→walk, Jogging→run, Upstairs/Downstairs→stairs, Sitting→sit, Standing→stand.

- **HHAR** (Stisen et al. 2015, UCI [DOI:10.24432/C5689X]): 9 subjects, 20 Hz, free-living phone carry. Channels: `accel_x, accel_y, accel_z`. Activity mapping: Walk→walk, Stairsup/Stairsdown→stairs, Sit→sit, Stand→stand. Bike→locomotion-other.

- **RealWorld2016** (Sztyler & Stuckenschmidt, Uni Mannheim): 15 subjects, 50 Hz. Channels: `acc_chest_x/y/z` (placement: `chest`) and `acc_waist_x/y/z` (placement: `front_center_mid`). All other body positions discarded. Activity mapping: Walking→walk, Running→run, Climbingup/Climbingdown→stairs, Sitting→sit, Standing→stand, Lying→lay. Jumping→excluded.

- **iPhone Placement Sweep** (`data/Acceleration/`, protocol_readme.txt): 1 subject, 100 Hz → 50 Hz. 7 raw folder labels × 6 activities = 42 clips. For the model contract, raw `front_center_upper` maps to `chest`; the remaining six labels map directly. Trim: 5 s from start, 15 s from end. Activity mapping: standingstill→stand, sitting→sit, laying→lay, walking→walk, running→run, sittingstanding→transitions. No test split (single subject).

- **PAMAP2** (Reiss & Stricker 2012, UCI [DOI:10.24432/C5NW2H]): 9 subjects, 100 Hz → 50 Hz. Channels: `chest_acc_x, chest_acc_y, chest_acc_z` only (hand + ankle discarded). Activity mapping: Lying→lay, Sitting→sit, Standing→stand, Walking→walk, Running→run, Ascending/Descending Stairs→stairs, Cycling→locomotion-other. Excluded: Ironing, Vacuum Cleaning, Nordic Walking, Rope Jumping, Other.

- **MHEALTH** (Banos et al. 2014, UCI [DOI:10.24432/C5TS36]): 10 subjects, 50 Hz. Channels: `chest_acc_x, chest_acc_y, chest_acc_z` only (left ankle + right arm discarded). Activity mapping: Standing Still→stand, Sitting And Relaxing→sit, Lying Down→lay, Walking→walk, Jogging/Running→run, Climbing Stairs→stairs. Excluded: Unknown, Waist Bends Forward, Frontal Elevation Of Arms, Knees Bending, Cycling, Jump Front And Back.

- **SAD** (Shoaib et al. 2014, University of Twente [DOI:10.3390/s141020164]): 10 subjects, 50 Hz. Channels: `Belt_Ax, Belt_Ay, Belt_Az` only (left/right pocket, wrist, upper arm discarded). Activity mapping: Walking→walk, Jogging→run, Sitting→sit, Standing→stand, Upstairs/Downstairs→stairs, Biking→locomotion-other.

- **UMA_FALL** (Casilari et al. 2017, figshare [DOI:10.6084/m9.figshare.4214283]): 19 subjects, 200 Hz → 50 Hz. Channels used: `chest_acc_x/y/z` (placement: `chest`), `waist_acc_x/y/z` (placement: `front_center_lower` — front midline lower abdomen), and `right_pocket_phone_acc_x/y/z` (raw source stream mapped to side-agnostic placement label `front_pocket`). Each sensor position generates its own windows with its respective placement label. Wrist and ankle discarded. Activity mapping: Walking→walk, Jogging→run, Go Upstairs/Go Downstairs→stairs, Lying Down On A Bed→lay, Sitting Getting Up On A Chair→transitions. Excluded: Bending, Hands Up, Hopping, Making A Call, Opening Door, Aplausing, Forward Fall, Backward Fall, Lateral Fall.

**Placement vocabulary (9 dimensions: 7 torso zones + `front_pocket` + `unknown_free_living`, one-hot):**

| Label | Source datasets |
|---|---|
| `chest` | iPhone Sweep raw `front_center_upper`; RealWorld2016 chest; PAMAP2 chest; MHEALTH chest; UMA_FALL chest |
| `front_center_mid` | iPhone Sweep; RealWorld2016 waist |
| `front_center_lower` | iPhone Sweep; UMA_FALL waist |
| `front_side_left_mid` | iPhone Sweep |
| `front_side_right_mid` | iPhone Sweep |
| `lateral_left_lower` | iPhone Sweep |
| `lateral_right_lower` | iPhone Sweep; HAPT; SAD |
| `front_pocket` | WISDM front pocket; MotionSense front pocket; UMA_FALL raw `RIGHTPOCKET` mapped into the same side-agnostic front-pocket bucket |
| `unknown_free_living` | HHAR; Real-Life HAR |

**Pocket remapping policy:** Explicit front-trouser-pocket carries map to the side-agnostic `front_pocket` model label whether the raw source says left pocket, right pocket, or simply front pocket. Current evidence gives a raw `RIGHTPOCKET` stream for UMA_FALL, while WISDM and MotionSense only report a generic front pocket with no side. Lateral beltline or belt-mounted devices remain `lateral_left_lower` / `lateral_right_lower` only when the source explicitly places them at the waist or beltline. HHAR phone_position_a/b → `unknown_free_living`.

**Scope note:** `front_pocket` is a classifier-conditioning carry label. It sits outside the torso-only placement-quality heuristic domain, which remains calibrated only on the front-plus-side waist/abdomen geometry sampled in the iPhone placement sweep.

**Training pool summary:**

| Dataset | Subjects | Placement(s) | Hz |
|---|---|---|---|
| HAPT | 30 | `lateral_right_lower` | 50 |
| WISDM | 36 | `front_pocket` | 20 |
| MotionSense | 24 | `front_pocket` | 50 |
| HHAR | 9 | `unknown_free_living` | 20 |
| RealWorld2016 | 15 | `chest`, `front_center_mid` | 50 |
| iPhone Sweep | 1 | 7 front/lateral labels | 100→50 |
| PAMAP2 | 9 | `chest` | 100→50 |
| MHEALTH | 10 | `chest` | 50 |
| SAD | 10 | `lateral_right_lower` | 50 |
| UMA_FALL | 19 | `chest`, `front_center_lower`, `front_pocket` | 200→50 |

Evidence:
- HAPT [DOI:10.24432/C54G7P], Reyes-Ortiz et al. 2016, Neurocomputing 171:754–767: same 30 subjects and Samsung Galaxy S II waist placement as UCI HAR; adds 6 postural-transition labels; strictly supersedes UCI HAR
- WISDM [DOI:10.1145/2003653.2003656], Kwapisz et al. 2010: Android phone carried in the front pants-leg pocket at 20 Hz; thigh/fabric coupling makes it a front-pocket carry source rather than a waistband label [F16]
- MotionSense [DOI:10.1145/3302505.3310068], Malekzadeh et al. 2019: iPhone 6s stored in the participant's trousers' front pocket at 50 Hz; front-pocket carry is the intended collection mode [F21]
- RealWorld2016 dataset page: body positions are published as `chest`, `forearm`, `head`, `shin`, `thigh`, `upperarm`, `waist`; the source does not define a stricter centerline chest subsite
- PAMAP2 [DOI:10.24432/C5NW2H], Reiss & Stricker 2012: 1 Colibri IMU is placed "on the chest" at 100 Hz; chest_acc is the only torso ACC channel
- MHEALTH dataset page: sensors are placed on the subject's chest, right wrist, and left ankle at 50 Hz; the source supports a chest bucket but not a stricter centerline label
- SAD [DOI:10.3390/s141020164], Shoaib et al. 2014: belt sensor at right lateral waist, 50 Hz; only dataset with a purpose-worn belt sensor rather than a phone clipped to a belt
- UMA_FALL [DOI:10.6084/m9.figshare.4214283], Casilari et al. 2017 + local CSV header inspection: simultaneous chest, waist, and right-pocket sensors at 200 Hz; fall windows excluded; `CHEST`, `WAIST`, and `RIGHTPOCKET` are distinct source positions
- RealWorld2016 (Sztyler & Stuckenschmidt): 7 simultaneous body positions, 15 subjects, 50 Hz; chest and waist streams directly train on matched torso signal shapes [F4, F5]
- DATA_CATALOG.md, WARD entry: “values are raw counts rather than calibrated g” — direct incompatibility with the m/s² feature scale
Approved: yes

## Pending decisions

> Ordered by dependency. Batch 1 (Decisions 1–5, signal-processing/ML layer) is fully resolved. Batch 2 decisions define derived metrics and quality gates. Batch 3 defines the Streamlit app structure, live-capture path, and deployment.

### Batch 2 — Derived metrics and quality gates (depends on Batch 1)

6. **Rhythm-consistency metric definition**  
   How to quantify regularity from cadence estimates within a locomotion window (e.g., coefficient of variation of inter-peak intervals, autocorrelation peak height, or spectral entropy), and what constitutes a "consistent" vs "irregular" walking signal for the MVP report. [F5, F9, F19]

7. **Motion-intensity metric and report color scale**  
   Whether to use RMS of body acceleration, signal magnitude area (SMA), or a percentile of the body-acc magnitude distribution; the thresholds separating low / moderate / high / vigorous; and whether to anchor these thresholds to the RealWorld2016 waist statistics (lying std_mag = 0.123 m/s² through running std_mag = 6.063 m/s²). [F4, F5, F15]

8. **Quiet-window breathing-rate proxy gating, bandpass, and confidence logic**  
   The criteria that qualify a window as "quiet enough" for a respiratory estimate (gravity-vector stability, body-acc std below a threshold derived from WoW/EmoWear statics), the bandpass used for the respiratory band (~0.1–0.5 Hz, i.e., ~6–30 breaths/min), and exactly what confidence label and caveat text are shown vs when the feature is hidden entirely. [F2, F3, F6, F7, F11, F19, UQ5, DOI:10.3390/s20216396]

9. **Placement-quality heuristic definition and UX wording**  
   The rule set mapping body-acc std, gravity-vector consistency, and cadence plausibility to red / yellow / green (e.g., green = gravity stable + cadence in physiological range + std_mag in expected locomotion band; red = near-zero dynamics with implausible orientation for declared activity), and the exact disclaimer text. [F1, F15, F18, F19, UQ4]

### Batch 3 — Streamlit app structure and deployment (depends on Batches 1–2)

10. **Streamlit page structure and user flow**  
    Whether to use a single-page layout with `st.tabs` (Upload / Live Capture / About) or `st.sidebar` navigation with multipage routing, how intermediate results (preprocessing summary, gravity split plot, activity timeline) are staged across the page, and what the report download artifact is (HTML, PDF via `weasyprint`, or Streamlit's own `st.download_button` with a JSON/CSV bundle).

11. **Live phone DeviceMotion capture path in Streamlit**  
    Streamlit runs Python server-side and cannot natively call `window.DeviceMotion`. Options: (A) a minimal custom Streamlit component (`streamlit-javascript` or a bespoke HTML component) that streams sensor data back to Python via WebSocket or polling; (B) a separate static capture page (plain HTML served via `st.components.v1.html`) that writes a CSV the user then uploads; (C) defer live capture to a later phase and ship upload-only first. This is the highest-risk open item for the live-capture requirement. [F21, F23]

12. **Python package and module structure**  
    Whether to organize as a flat `app/` folder or a proper installable package (`motionlens/` with `preprocessing`, `features`, `classifier`, `metrics`, `report` submodules), how to manage the pre-trained classifier artifact, and what goes in `requirements.txt` vs `pyproject.toml`.

13. **Deployment target**  
    Streamlit Community Cloud (zero-ops, free tier, GitHub-connected), Docker + self-hosted, or local-only for the demo phase. Affects whether large WHAR cache files can be committed to the repo or must stay out-of-tree.
