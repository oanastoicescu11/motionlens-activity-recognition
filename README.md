# MotionLens

MotionLens is an interpretable accelerometer signal-intelligence pipeline and live web app. It analyzes smartphone accelerometer data from uploaded CSVs or live browser `DeviceMotion` capture and produces activity, segmentation, cadence, intensity, gravity/body separation, and signal-quality outputs.

The current model and evidence scope are **smartphone pocket carry only**. Training uses front-pocket phone datasets plus lateral side-hip iPhone recordings. MotionLens is ACC-first: timestamp plus 3-axis acceleration is the only required input. Optional browser/device signals such as gravity, user acceleration, rotation rate, gyroscope, orientation, magnetometer, or GPS may be used when available, but the app must continue working without them.

MotionLens is not a medical device. It does not make clinical or diagnostic claims. Breathing output is only a low-confidence quiet-window proxy, and placement quality is a coarse signal-quality guide, not anatomical placement classification.

## What MotionLens does

Core outputs:

- Activity inference for pocket-phone motion windows
- Activity segmentation over time
- Gravity/body acceleration separation
- Cadence and periodicity estimates for rhythmic locomotion
- Impact/motion intensity timeline
- Rhythm consistency
- Low-confidence breathing proxy for quiet/static windows only
- Red/yellow/green signal or placement-quality guidance
- Interpretable DSP and feature-based reporting rather than black-box-only labels

Current activity labels:

- `walk`
- `run`
- `stairs`
- `sit/lay`
- `stand`
- `transitions`
- `locomotion-other`

The main user-facing labels are `walk`, `run`, `stairs`, `sit/lay`, and `stand`. `transitions` and `locomotion-other` are useful for robustness and should be presented carefully.

## Current scope

### In scope

- Smartphone pocket carry
- Front pants/trouser/jeans pocket recordings
- Lateral side-hip iPhone recordings
- Uploaded raw 3-axis accelerometer CSVs
- Live browser `DeviceMotion` capture
- ACC-only fallback behavior
- Interpretable signal-processing outputs
- Per-dataset accuracy reporting
- Local development with FastAPI, Streamlit, and optional Redis-backed worker mode

### Out of scope

- Medical or clinical diagnosis
- Clinical respiratory monitoring
- Dense anatomical placement inference
- Wrist, chest, waist, back, ankle, or arbitrary free-living carry as training targets
- One global deployment accuracy number
- Energy expenditure or calorie claims
- User-facing model retraining in the MVP

## Evidence-backed data scope

The current pocket-only model keeps only these placement labels:

| Label | Meaning |
|---|---|
| `front_pocket` | Phone in front pants/trouser/jeans pocket |
| `lateral_left_lower` | Left lateral side-hip iPhone sweep |
| `lateral_right_lower` | Right lateral side-hip iPhone sweep |

These placement labels are metadata for dataset filtering and interpretation. The current trained model does **not** use placement one-hot features.

Training datasets:

| Dataset | Placement used | Hz | Role | Citation |
|---|---:|---:|---|---|
| WISDM | front pants-leg pocket | 20 → 50 | Controlled pocket HAR baseline | [D1] |
| MotionSense | trousers front pocket | 50 | Closest public iPhone/Core Motion proxy | [D2] |
| WISDM v2 | front pants-leg pocket | 20 → 50 | Larger Android pocket dataset | [D3] |
| RealWorld2016 | thigh stream only | ~50 | Front-pocket proxy from multi-phone study | [D4] |
| UniMiB-SHAR | front pants pocket | 50 | ADL and transition examples | [D5] |
| UMA Fall | right pants pocket only | 200 → 50 | Pocket locomotion and transitions | [D6] |
| Shoaib 2013 | right jeans pocket | 50 | Small controlled pocket set | [D7] |
| Shoaib Sensors | left + right jeans pockets | 50 | Bilateral pocket data | [D8] |
| UT Complex | smartphone at pocket | 50 | Complex activity subset | [D8], [D9] |
| iPhone Placement Sweep | lateral left/right lower only | ~100 → 50 | Local lateral side-hip calibration | repo-local capture |

Excluded from training:

- UCI/HAPT
- HHAR
- Real-Life HAR
- HARTH
- WARD
- WoW
- EmoWear
- PhysioNet/Wearable
- chest, waist, wrist, ankle, back, and other non-pocket streams

Some excluded datasets remain useful as analysis/reference evidence, but they are not part of the current pocket-only training pool.

## Signal-processing contract

All data is normalized into a canonical accelerometer contract before inference.

```text
raw samples
→ timestamp/unit normalization
→ monotonic elapsed time
→ acc_x / acc_y / acc_z in m/s²
→ sort, dedupe, invalid-row rejection
→ gap handling
→ resample to 50 Hz
→ gravity/body split
→ 128-sample windows
→ feature extraction
→ XGBoost inference
→ guard/refinement layers
→ causal HMM temporal decoding
→ cadence, periodicity, intensity, and quality outputs
```

Core parameters:

| Parameter | Value |
|---|---:|
| Target sample rate | 50 Hz |
| Window size | 128 samples |
| Window duration | 2.56 seconds |
| Window step | 64 samples |
| Window overlap | 50% |
| Gravity split | per-axis 0.3 Hz Butterworth low-pass |
| Body acceleration | total acceleration minus gravity acceleration |

Internal acceleration streams:

- `total_acc`: raw total acceleration for visualizations and quality checks
- `gravity_acc`: low-pass gravity estimate for orientation and quiet-window gating
- `body_acc`: gravity-removed motion signal for activity inference, cadence, rhythm, and intensity

Important signal facts:

- Gravity dominates raw ACC near 9.81 m/s².
- Gravity direction depends on device orientation.
- Per-axis structure matters; activity evidence is not just magnitude.
- Walking/running cadence is usually in the low-Hz locomotion band.
- Cadence is suppressed when motion is too weak or no clear periodic peak exists.
- Quiet/static windows are required before attempting any breathing proxy.

## Feature extraction

The current model uses engineered features from each 128-sample window.

Feature families include:

- Time-domain statistics
- Frequency-domain features
- Time-frequency features
- Autoregressive features
- Cross-axis coordination features
- Gravity/body decomposition features
- Jerk and impact features
- Cadence and periodicity features

The latest evaluated pocket-only run in the repo uses **151 features**.

Key shared modules:

```text
src/core/features/window_features.py
src/core/inference/model.py
src/app/worker/_steps.py
processing/train_pocket_only_v1.py
```

## Model

The current training pipeline is implemented in:

```text
processing/train_pocket_only_v1.py
```

Model design:

- XGBoost multiclass classifier
- 128-sample windows at 50 Hz
- No placement one-hot features: `PLACEMENT_LABELS = []`
- Subject-level holdout per dataset where possible
- Temporal decoding options: `none` and `causal-hmm`
- Static specialist/refinement for difficult static posture cases
- Final runtime bundle serialized with `joblib`

Deployment runtime bundle directory:

```text
src/artifacts/model/
```

Committed live runtime file:

```text
src/artifacts/model/inference_bundle.joblib
```

The live app does not call a standalone `predict_with_artifacts(...)` helper. The
worker loads `src/artifacts/model/inference_bundle.joblib` in
`src/app/worker/_steps.py` and runs live inference through `src/app/worker/pipeline.py`.

Training and evaluation artifacts belong under `output/...`, not in `src/artifacts/model/`.
That includes metrics, confusion matrices, feature schemas, transition stats, and
component-model dumps. The runtime bundle is the only artifact expected under
`src/artifacts/model/` for deployment.

Live inference loads:

```text
src/artifacts/model/inference_bundle.joblib
```

Live inference does **not** re-estimate transition probabilities. It uses the saved `log_init_probs` and `log_trans_probs` from the trained artifact bundle.

For deployment, the app expects the bundled runtime artifact to be present. It does not
fall back to a secondary raw-model file.

## Current model metrics

The repo currently contains two model states:

- Deployed runtime bundle: `src/artifacts/model/inference_bundle.joblib`
- Latest evaluated pocket-only training run: `output/model_pocket_ultra_tuned/`

Latest evaluated run metrics (`output/model_pocket_ultra_tuned/metrics.json`):

| Metric | Value |
|---|---:|
| Decoded accuracy | 0.904712 |
| Raw accuracy | 0.848810 |
| Train windows | 153,870 |
| Test windows | 38,263 |
| Feature count | 151 |

Per-dataset decoded accuracy (`output/model_pocket_ultra_tuned/per_dataset_metrics.csv`):

| Dataset | Accuracy |
|---|---:|
| MotionSense | 0.8960 |
| RealWorld2016 | 0.8838 |
| Shoaib Sensors | 0.9980 |
| UMA Fall | 0.6622 |
| UniMiB-SHAR | 0.9360 |
| WISDM | 0.9240 |
| WISDM v2 | 0.8591 |

Do not report a single deployment-wide accuracy claim without context. Controlled datasets, heterogeneous devices, and live browser capture differ substantially.

## Repository structure

High-level structure:

```text
.
├── src/app/                     # Live backend, worker, and UI
├── src/core/                    # Shared preprocessing, features, inference, guards
├── processing/                  # Offline dataset contract + model training
├── src/artifacts/model/         # Current trained runtime model bundle
├── output/motionlens_contract/  # Canonical training contract outputs
├── tests/                       # Unit and integration tests
└── run_worker.py                # Worker entrypoint
```

## App architecture

The `src/app/` folder contains the live inference backend and UI. After the app refactor, the worker pipeline is split into smaller step, guard, decoding, and serialization modules instead of a single large monolithic pipeline function.

```text
src/app/
├── __init__.py
├── insights.py
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── ingest_routes.py
│   ├── session_routes.py
│   ├── mobile_capture_page.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── dependencies.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── ingest.py
│   │       ├── mobile.py
│   │       └── sessions.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── exceptions.py
│   │   ├── ingest_parser.py
│   │   └── security.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── ingest.py
│   │   ├── protocols.py
│   │   └── session.py
│   └── storage/
│       ├── __init__.py
│       ├── memory.py
│       └── redis_impl.py
├── ui/
│   ├── __init__.py
│   └── streamlit_app.py
└── worker/
    ├── __init__.py
    ├── consumer.py
    ├── main.py
    ├── pipeline.py
    ├── _steps.py
    ├── _guards.py
    ├── _decoding.py
    └── _serialization.py
```

Main components:

- FastAPI backend: session lifecycle, auth, ingest, mobile capture page, signal/inference endpoints
- Streamlit UI: desktop dashboard and QR-code mobile join flow
- Mobile capture page: browser DeviceMotion capture and upload
- Worker pipeline: resampling, feature extraction, XGBoost inference, guards, artifact scoring, cadence extraction, HMM decoding, and inference snapshot serialization
- Store abstraction: in-memory store for local development, Redis implementation for multi-process/runtime use

## Live capture flow

```text
Desktop Streamlit UI
→ starts desktop session
→ displays QR code

Phone browser
→ opens /mobile-capture?join_token=...
→ joins session
→ requests DeviceMotion permission on user tap
→ captures accelerationIncludingGravity when available
→ batches samples
→ POSTs to /v1/ingest/{write_token}

Backend
→ validates token/session/device binding
→ deduplicates by session/device/message_id
→ stores raw points
→ enqueues inference task

Worker
→ reads recent raw points
→ resamples to 128 samples at 50 Hz
→ extracts features
→ runs XGBoost
→ applies guards and calibration
→ runs temporal decoder
→ stores latest inference snapshot

UI and mobile page
→ poll signal/inference endpoints
→ show raw signal and live predictions
```

The live browser path prefers total acceleration / `accelerationIncludingGravity` when available so that gravity/body separation remains possible.

## API summary

| Endpoint | Method | Auth | Purpose |
|---|---|---|---|
| `/health` | GET | none | Health check |
| `/v1/sessions/start` | POST | none | Start mobile or desktop session |
| `/v1/sessions/join` | POST | join token | Attach phone to desktop session |
| `/v1/sessions` | POST | none | Create simple session |
| `/v1/sessions/{id}/signal` | GET | viewer token | Read raw accelerometer points |
| `/v1/sessions/{id}/inference` | GET | viewer token | Read latest inference snapshot |
| `/v1/ingest/{write_token}` | POST | write token | Ingest sensor batch |
| `/mobile-capture` | GET | join token | Mobile browser capture page |

Security model:

- Write tokens are bound to a session and device.
- Viewer tokens are read-only.
- Ingest rejects mismatched session/device/write-token combinations.
- Messages are deduplicated by `(session_id, device_id, message_id)`.
- Sessions expire by TTL.
- Redis is available for multi-process storage.

## Running locally

Use Windows PowerShell from the repository root.

Activate the virtual environment first:

```powershell
.venv\Scripts\activate
```

Install the project once in editable mode so `app` and `core` resolve from `src/`:

```powershell
.venv\Scripts\python.exe -m pip install -e .
```

For plain local `http://localhost` development, either copy `.env.example` to `.env`
or set `MLIVE_ALLOW_INSECURE_LOCAL=1` before starting the backend. Otherwise the
backend rejects requests with `HTTPS is required.`

### Option 1: backend only

Use this for ingest/API testing without the dashboard.

```powershell
$env:MLIVE_ALLOW_INSECURE_LOCAL="1"
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Create a session:

```powershell
.venv\Scripts\python.exe -c "import requests; r = requests.post('http://localhost:8000/v1/sessions/start', params={'owner_id':'local-dev','mode':'desktop','ttl_seconds':300,'join_ttl_seconds':90}, timeout=10); r.raise_for_status(); print(r.json())"
```

Read signal data:

```powershell
Invoke-RestMethod -Headers @{ "X-Viewer-Token" = "<VIEWER_TOKEN>" } -Uri "http://localhost:8000/v1/sessions/<SESSION_ID>/signal"
```

### Option 2: backend + Streamlit UI

Recommended for local live testing.

Terminal 1:

```powershell
$env:MLIVE_ALLOW_INSECURE_LOCAL="1"
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```powershell
$env:MLIVE_BACKEND_BASE_URL="http://localhost:8000"
$env:MLIVE_PUBLIC_BACKEND_URL="http://<LAN-IP>:8000"
.venv\Scripts\python.exe -m streamlit run src/app/ui/streamlit_app.py
```

Then open:

```text
http://localhost:8501
```

Scan the QR code with your phone, tap Start Analysis, and keep the phone and computer on the same network.

### Option 3: full stack with Redis

Use this when testing a production-like backend + worker + UI setup.

Terminal 1:

```powershell
$env:MLIVE_ALLOW_INSECURE_LOCAL="1"
$env:MLIVE_STORE_BACKEND="redis"
$env:MLIVE_REDIS_URL="redis://localhost:6379/0"
$env:MLIVE_REDIS_PREFIX="mlive"
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```powershell
$env:MLIVE_STORE_BACKEND="redis"
$env:MLIVE_REDIS_URL="redis://localhost:6379/0"
$env:MLIVE_REDIS_PREFIX="mlive"
.venv\Scripts\python.exe run_worker.py --store-backend redis
```

Terminal 3:

```powershell
$env:MLIVE_BACKEND_BASE_URL="http://localhost:8000"
$env:MLIVE_PUBLIC_BACKEND_URL="http://<LAN-IP>:8000"
.venv\Scripts\python.exe -m streamlit run src/app/ui/streamlit_app.py
```

If using Redis, make sure a Redis server is running and the Python `redis` package is installed.

## Phone setup notes

- Phone and computer must be on the same local network.
- Use `ipconfig` to find the computer LAN IP.
- iOS/Safari-style motion access may require an explicit user tap.
- Some browsers require secure-context behavior for motion/orientation permission APIs.
- Local development may allow insecure local exceptions depending on browser and backend settings.
- If capture does not start, check browser permissions, network reachability, and backend logs.

Useful helper scripts:

```text
run_worker.py       # starts worker process
```

## Testing

Run the full test suite:

```powershell
.venv\Scripts\python.exe -m unittest discover tests -v
```

Useful focused suites:

```powershell
.venv\Scripts\python.exe -m unittest tests.test_ingest_schema tests.test_ingest_dedupe -v
.venv\Scripts\python.exe -m unittest tests.test_session_isolation -v
.venv\Scripts\python.exe -m unittest tests.test_worker_pipeline tests.test_integration_e2e -v
.venv\Scripts\python.exe -m unittest tests.test_live_insights -v
.venv\Scripts\python.exe -m unittest tests.test_preprocessing_accelerometer -v
.venv\Scripts\python.exe -m unittest tests.test_motionlens_contract_processing -v
```

Tests cover:

- DeviceMotion ingest schema parsing
- Message deduplication
- Session isolation and token binding
- Raw signal vs inference lane separation
- Worker queue processing
- End-to-end ingest → inference → read flow
- Preprocessing, resampling, and gravity/body split
- Live insights
- Training contract processing

## Offline processing pipeline

The `processing/` folder builds the canonical dataset contract and trains the model.

Important files:

```text
processing/motionlens_contract_processing.py
processing/prepare_pocket_only_training_data.py
processing/training_transition_stats.py
processing/train_pocket_only_v1.py
```

Canonical dataset outputs:

```text
output/motionlens_contract/
├── session_manifest.parquet
├── window_manifest.parquet
├── dataset_summary.csv
└── parquet partitions
```

Training output:

```text
output/model_pocket_ultra_tuned/
```

Current runtime artifact path:

```text
src/artifacts/model/inference_bundle.joblib
```

Promoting a model to live inference is a separate deployment step. Training runs write
their evaluation artifacts under `output/...`; the app only loads the committed runtime
bundle from `src/artifacts/model/inference_bundle.joblib`.

## Development rules

- Keep the app ACC-first.
- Do not require optional sensors.
- Do not add medical claims.
- Do not report fake precision for breathing or placement quality.
- Do not revive older waist/abdomen or dense placement-classification framing without a new approved design reset.
- Do not train on non-pocket streams in the current pocket-only model.
- Keep public functions typed and documented.
- Use `ValueError` with clear messages for bad inputs.
- Use `logging`, not `print`, in application code.
- Add or update tests when behavior changes.
- Run relevant tests before marking work done.
- New design decisions must be evidence-backed.

## References and dataset sources

Active pocket-scope citations:

- [D1] Kwapisz, Jennifer R., Gary M. Weiss, and Samuel A. Moore. "Activity recognition using cell phone accelerometers." ACM SIGKDD Explorations Newsletter 12, no. 2 (2011): 74-82. DOI: `10.1145/1964897.1964918`.
- [D2] Malekzadeh, Mohammad, Richard G. Clegg, Andrea Cavallaro, and Hamed Haddadi. "Mobile sensor data anonymization." Proceedings of the International Conference on Internet of Things Design and Implementation (IoTDI '19) (2019): 49-58. DOI: `10.1145/3302505.3310068`.
- [D3] Weiss, Gary. "WISDM Smartphone and Smartwatch Activity and Biometrics Dataset." UCI Machine Learning Repository (2019). DOI: `10.24432/C5HK59`.
- [D4] Sztyler, Timo, and Heiner Stuckenschmidt. "On-body localization of wearable devices: An investigation of position-aware activity recognition." 2016 IEEE International Conference on Pervasive Computing and Communications (PerCom). DOI: `10.1109/PERCOM.2016.7456521`.
- [D5] Micucci, Daniela, Marco Mobilio, and Paolo Napoletano. "UniMiB SHAR: A Dataset for Human Activity Recognition Using Acceleration Data from Smartphones." Applied Sciences 7, no. 10 (2017): 1101. DOI: `10.3390/app7101101`.
- [D6] Casilari, Eduardo, Jose A. Santoyo-Ramon, and Jose M. Cano-Garcia. "UMAFall: A Multisensor Dataset for the Research on Automatic Fall Detection." Procedia Computer Science 110 (2017): 32-39. DOI: `10.1016/j.procs.2017.06.110`. Dataset distribution: "UMAFall: Fall Detection Dataset (Universidad de Malaga)." DOI: `10.6084/m9.figshare.4214283`.
- [D7] Shoaib, Muhammad. "Human Activity Recognition Using Hetrogenious Sensors." Adjunct Proceedings of UbiComp 2013. Citation requested in `data/activity-recognition-dataset-shoaib/Activity_Recognition_DataSet/Readme.txt`.
- [D8] Shoaib, Muhammad, Stephan Bosch, Ozlem Durmaz Incel, Hans Scholten, and Paul J. M. Havinga. "Complex Human Activity Recognition Using Smartphone and Wrist-Worn Motion Sensors." Sensors 16, no. 4 (2016): 426. DOI: `10.3390/s16040426`.
- [D9] Shoaib, M., S. Bosch, H. Scholten, P. J. Havinga, and O. D. Incel. "Towards detection of bad habits by fusing smartphone and smartwatch sensors." 2015 IEEE International Conference on Pervasive Computing and Communication Workshops (PerCom Workshops): 591-596. DOI: `10.1109/PERCOMW.2015.7134104`.
- [D10] Anguita, Davide, Alessandro Ghio, Luca Oneto, Xavier Parra, and Jorge L. Reyes-Ortiz. "A Public Domain Dataset for Human Activity Recognition Using Smartphones." European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN), 2013. UCI dataset page DOI: `10.24432/C54S4K`.
- [D11] Zijlstra, Wiebren, and At L. Hof. "Assessment of spatio-temporal gait parameters from trunk accelerations during human walking." Gait & Posture 18, no. 2 (2003): 1-10. DOI: `10.1016/S0966-6362(02)00190-X`.

The repo-local iPhone Placement Sweep is internal capture data and has no external citation. MotionLens also keeps analysis/reference notes for excluded datasets such as HHAR, Real-Life HAR, HARTH, WARD, WoW, and EmoWear, but these are not training streams for the current pocket-only model.