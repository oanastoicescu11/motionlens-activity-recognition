# BRIEF

## Goal
Build **MotionLens**: an interpretable wearable ACC signal-intelligence tool. The primary product is a web app where a user uploads raw 3-axis accelerometer CSV data, or records live via phone-browser `DeviceMotion`, and receives an annotated signal-processing report with interactive visualizations. Core outputs are gravity/dynamic separation, activity segmentation, cadence estimation, motion-intensity timelines, a quiet-window breathing-rate proxy with a confidence flag, and a simple red/yellow/green placement-quality indicator. The data backbone now includes both repo-local specialty datasets and WHAR-standardized public HAR datasets exported into unified session/window manifests. Evidence-backed interpretation is strongest for waist/trunk recordings; the placement-quality heuristic is calibrated only for the front-plus-side waist/abdomen domain, with the back excluded.

## Hard requirements
- Product = end-to-end ACC reporting app, not a simulator-first artifact
- Inputs = uploaded raw 3-axis CSV plus planned live browser `DeviceMotion` capture
- Core outputs = gravity/dynamic split, activity segmentation, cadence, motion intensity, quiet-window breathing proxy, and placement-quality / signal-quality indicator
- Report must surface interpretable DSP visuals, not only black-box labels
- Evidence-backed domain = waist/trunk ACC overall; front-plus-side waist/abdomen only for placement-quality heuristics
- Live phone path must work in ACC-only mode and degrade gracefully when gravity, rotation, or other phone sensors are absent
- Training/evaluation data layer = reproducible WHAR-backed session/window manifests plus repo-local specialty datasets
- Accuracy reporting must distinguish controlled, heterogeneous-device, and free-living evaluation tiers
- No fake precision on placement or breathing estimates
- Language = Python
- Every design decision must cite evidence (data finding or paper)
- No medical or clinical diagnosis claims

## Workflow
```
Phase 1: analyze data + read papers for the MotionLens scope → ANALYSIS_REPORT.md (user approves before Phase 2)
Phase 2: propose app/system decisions one at a time, each with citation → write only approved decisions into DESIGN.md
Phase 3: implement only what is in DESIGN.md
If project framing changes materially, reset DESIGN.md and restart from Phase 1
```

## Datasets (see more details in DATA_CATALOG.md)

| Name | Path | Placement | Signal | Hz | Main MotionLens role |
|------|------|-----------|--------|----|----------------------|
| HARTH | `data/harth/` | lower back + front thigh | ACC | 50/100 | Free-living trunk-motion robustness for activity segmentation |
| iPhone Placement Sweep | `data/Acceleration/` | front-center / front-side / lateral waist-abdomen zones | ACC | ~100 | Weak but direct evidence for placement-quality heuristics and same-device calibration |
| MotionSense | `output/whar_datasets_cache/motion_sense/` | iPhone front pocket (`front_pocket` carry label) | DeviceMotion + ACC + gyro | 50 | Closest public proxy for live iPhone streaming and optional gravity/orientation features |
| HHAR | `output/whar_datasets_cache/hhar/` | mixed smartphone/watch carry | ACC + gyro | ~20 | Device heterogeneity robustness across phone/watch models and routes |
| Real-Life HAR | `output/whar_datasets_cache/real_life_har/` | unconstrained smartphone placement/orientation | ACC + gyro + mag + GPS | irregular / adapted ~20 | Free-living robustness, missing-sensor fallback, and deployment realism |
| UCI HAR | `data/human+activity+recognition+using+smartphones/UCI HAR Dataset/` | waist / lateral beltline | ACC (g) | 50 | Classifier/cadence backbone and documented gravity/body separation reference |
| RealWorld2016 | `data/realworld2016_dataset/proband{1..15}/data/` | chest/forearm/head/shin/thigh/upperarm/**waist** | ACC (m/s²) | ~50 | Main locomotion backbone for classification, cadence, and motion intensity |
| WARD | `data/WARD1.0/` | **front-center waist** + 4 other nodes | ACC + gyro | 20 | Direct frontal-waist anchor for activity ordering and heuristic quality checks |
| WISDM | `data/WISDM_ar_v1.1/` | front pants-leg pocket (`front_pocket` carry label) | ACC | 20 | Smartphone carry-style robustness for cadence and activity contrast |
| EmoWear | `data/Emowear-raw/` | chest (100Hz) / wrist (32Hz) / front+back IMU (~50Hz) | ACC + resp + SKT | varies | Low-motion respiratory context and motion-artifact caution for breathing proxy work |
| Wearable (PhysioNet) | `data/Wearable_Dataset/` | wrist | ACC (32Hz) + SKT + EDA + BVP | 32 | Out-of-scope comparison dataset; not part of the MotionLens MVP evidence chain |
| WoW | `data/DataSet_RR/` | **anterior torso patches** | ACC + gyro + resp | varies | Front-abdominal quiet-window sway and breathing reference |

## File formats (quick ref)
- **WHAR preprocessing summary:** `output/whar/whar_preprocessing_summary.csv` — per-dataset exported counts (`activities`, `sessions`, `windows`, `subjects`) for the latest standardized preprocessing run
- **WHAR unified manifests:** `output/whar/whar_unified_{session,window}_manifest.csv` — cross-dataset metadata linking dataset, subject, activity, session, and window IDs to cached parquet artifacts
- **UCI HAR:** `Inertial Signals/total_acc_{x,y,z}_{split}.txt` — rows = 128-sample windows; `y_{split}.txt` = labels (1=walk 2=stairs+ 3=stairs- 4=sit 5=stand 6=lay); units = g
- **RealWorld2016:** `acc_{activity}_{position}.csv` — cols: `id, attr_time, attr_x, attr_y, attr_z`; units = m/s²
- **WoW:** per-subject folders in `DataSet_P1/`, `DataSet_PS1/`, `DataSet_PD1/`; IMU files = `timestamp, ax, ay, az, gx, gy, gz`
