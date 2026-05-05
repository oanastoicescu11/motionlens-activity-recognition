# MotionLens

MotionLens is a pocket-phone accelerometer activity-analysis app. It accepts live browser `DeviceMotion` streams, normalizes them to a shared 50 Hz contract, separates gravity from body motion, extracts engineered window features, runs XGBoost with guard logic and causal HMM decoding, and serves the results through a FastAPI backend and Streamlit dashboard.

MotionLens is ACC-first: timestamp plus `acc_x`, `acc_y`, and `acc_z` are enough to run the pipeline. The current training and interpretation scope is smartphone pocket carry.

## What MotionLens does

Core outputs:

- Activity inference for pocket-phone motion windows
- Activity segmentation over time
- Gravity/body acceleration separation
- Cadence and periodicity estimates for rhythmic locomotion
- Impact/motion intensity timeline
- Rhythm consistency
- Signal-quality scoring and phone-orientation context
- Interpretable DSP and feature-based reporting rather than black-box-only labels

Current activity labels:

- `walk`
- `run`
- `stairs`
- `sit/lay`
- `stand`
- `transitions`
- `locomotion-other`

Runtime predictions are emitted over the seven labels above.

## Data scope

MotionLens is trained on pocket-phone recordings from these sources. The current checked-in evaluation run reports held-out metrics only for datasets with test subjects.

| Dataset | Carry position | Hz | Split in current run | Role | Citation |
|---|---:|---:|---|---|---|
| WISDM | front pants-leg pocket | 20 → 50 | held-out eval | Controlled pocket HAR baseline | [D1] |
| MotionSense | trousers front pocket | 50 | held-out eval | Closest public iPhone/Core Motion proxy | [D2] |
| WISDM v2 | front pants-leg pocket | 20 → 50 | held-out eval | Larger Android pocket dataset | [D3] |
| RealWorld2016 | thigh stream only | ~50 | held-out eval | Front-pocket proxy from multi-phone study | [D4] |
| UniMiB-SHAR | front pants pocket | 50 | held-out eval | ADL and transition examples | [D5] |
| UMA Fall | right pants pocket only | 200 → 50 | held-out eval | Pocket locomotion and transitions | [D6] |
| Shoaib 2013 | right jeans pocket | 50 | train_only | Small controlled pocket set | [D7] |
| Shoaib Sensors | left + right jeans pockets | 50 | held-out eval | Bilateral pocket data | [D8] |
| UT Complex | smartphone at pocket | 50 | train_only | Complex activity subset | [D8], [D9] |
| iPhone Placement Sweep | lateral left/right lower only | ~100 → 50 | train_only | Local lateral side-hip calibration | repo-local capture |

All sources are normalized into the same accelerometer contract before feature extraction, training, and live inference. The published per-dataset metrics in the current pocket-only run cover WISDM, MotionSense, WISDM v2, RealWorld2016, UniMiB-SHAR, UMA Fall, and Shoaib Sensors.

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
- 151 engineered features per window
- Subject-level holdout per dataset where possible
- Causal HMM temporal decoding
- Static specialist/refinement for difficult static posture cases
- Final runtime bundle serialized with `joblib`

Live runtime bundle:

```text
src/artifacts/model/inference_bundle.joblib
```

Training runs write metrics, feature schemas, confusion matrices, and transition statistics under `output/...`. The worker loads the committed runtime bundle in `src/app/worker/_steps.py` and runs the live inference flow in `src/app/worker/pipeline.py`.

## Current model metrics

Latest evaluated pocket-only run metrics (`output/model_pocket_ultra_tuned/metrics.json`):

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

The `src/app/` folder contains the live backend, mobile capture page, worker, and dashboard.

```text
src/app/
├── backend/     # FastAPI app, routes, mobile capture page, runtime settings, storage
├── worker/      # queue consumer, feature/inference steps, guards, decoding, serialization
├── ui/          # Streamlit dashboard
├── insights.py  # UI-facing live metrics
└── __init__.py
```

Main components:

- FastAPI backend: session lifecycle, auth, ingest, mobile capture page, signal/inference endpoints
- Streamlit UI: desktop dashboard and QR-code mobile join flow
- Mobile capture page: browser DeviceMotion capture and streaming
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

### 1. Install the project

```powershell
Copy-Item .env.example .env
.venv\Scripts\activate
.venv\Scripts\python.exe -m pip install -e .
```

### 2. Quick local run

Use this mode for the simplest backend + UI setup. In memory mode the backend starts the inference worker inline.

Terminal 1:

```powershell
$env:MLIVE_ALLOW_INSECURE_LOCAL="1"
$env:MLIVE_STORE_BACKEND="memory"
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Terminal 2:

```powershell
$env:MLIVE_BACKEND_BASE_URL="http://localhost:8000"
$env:MLIVE_PUBLIC_BACKEND_URL="http://<LAN-IP>:8000"
.venv\Scripts\python.exe -m streamlit run src/app/ui/streamlit_app.py
```

Open:

```text
http://localhost:8501
```

### 3. Local Redis + dedicated worker

Use this mode when you want the backend and worker in separate processes.

Start Redis in Docker:

```powershell
docker run --name motionlens-redis --rm -p 6379:6379 redis:7-alpine
```

Backend terminal:

```powershell
$env:MLIVE_ALLOW_INSECURE_LOCAL="1"
$env:MLIVE_STORE_BACKEND="redis"
$env:MLIVE_REDIS_URL="redis://localhost:6379/0"
$env:MLIVE_REDIS_PREFIX="mlive"
.venv\Scripts\python.exe -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Worker terminal:

```powershell
$env:MLIVE_STORE_BACKEND="redis"
$env:MLIVE_REDIS_URL="redis://localhost:6379/0"
$env:MLIVE_REDIS_PREFIX="mlive"
.venv\Scripts\python.exe run_worker.py --workers 1 --store-backend redis
```

Streamlit terminal:

```powershell
$env:MLIVE_BACKEND_BASE_URL="http://localhost:8000"
$env:MLIVE_PUBLIC_BACKEND_URL="http://<LAN-IP>:8000"
.venv\Scripts\python.exe -m streamlit run src/app/ui/streamlit_app.py
```

### 4. Phone and browser notes

- For the simplest local setup, use Chrome on Android with a LAN URL.
- Keep the phone and computer on the same network when you use a LAN URL.
- Set `MLIVE_PUBLIC_BACKEND_URL` to the backend address the phone can actually open.
- Android/Chrome can use `http://<LAN-IP>:8000`.
- iPhone requires HTTPS and a user tap to grant motion permission.

For iPhone live capture, run an HTTPS tunnel to the backend:

```powershell
ngrok http 8000
```

Then set the public backend URL to the HTTPS ngrok address before starting Streamlit:

```powershell
$env:MLIVE_PUBLIC_BACKEND_URL="https://<your-ngrok-domain>"
```

The dashboard QR code uses `MLIVE_PUBLIC_BACKEND_URL` to build the phone capture link.

Useful helper scripts:

```text
run_worker.py       # starts worker process
```

## Deployment

The production stack lives under `deployment_artifacts/` and runs five services:

- Caddy for HTTPS and reverse proxy
- Streamlit for the dashboard
- FastAPI for the backend API
- Worker for inference jobs
- Redis for shared queue and session state

### Docker Compose deployment

1. Copy the production environment template:

```powershell
Copy-Item deployment_artifacts/.env.production.example deployment_artifacts/.env.production
```

2. Edit `deployment_artifacts/.env.production` and set at least:

- `ML_HOST`
- `MLIVE_PUBLIC_BACKEND_URL`

3. Build and start the stack:

```powershell
docker compose -f deployment_artifacts/compose.yaml build --pull backend worker streamlit
docker compose -f deployment_artifacts/compose.yaml up -d
docker compose -f deployment_artifacts/compose.yaml ps
```

4. Open `https://<ML_HOST>` after DNS and certificates are ready.

### Local Docker Compose stack

If you want the same containerized app shape locally as on the VM, run the backend, worker, Streamlit, and Redis containers locally without Caddy.

1. Copy the local environment template:

```powershell
Copy-Item deployment_artifacts/.env.local.example deployment_artifacts/.env.local
```

2. If you want phone access from another device on your LAN, edit `deployment_artifacts/.env.local` and set `MLIVE_PUBLIC_BACKEND_URL` to `http://<LAN-IP>:8000`. For desktop-only local use, leave it as `http://localhost:8000`.

3. Build and start the local stack:

```powershell
docker compose -f deployment_artifacts/compose.local.yaml up -d --build
docker compose -f deployment_artifacts/compose.local.yaml ps
```

4. Open `http://localhost:8501` for the dashboard.

5. Stop it with:

```powershell
docker compose -f deployment_artifacts/compose.local.yaml down
```

This is production-like in service layout, but it is intentionally not the public-HTTPS VM setup. It binds backend `8000` and Streamlit `8501` directly on your machine and skips Caddy.

### Ubuntu VM helper script

For a single Ubuntu VM, the repo includes `deployment_artifacts/scripts/deploy_vm.sh`.

With DuckDNS:

```bash
ML_HOST=<duckdns-domain>.duckdns.org DUCKDNS_DOMAIN=<duckdns-domain> DUCKDNS_TOKEN=<duckdns-token> bash deployment_artifacts/scripts/deploy_vm.sh
```

With your own DNS:

```bash
ML_HOST=your.domain.example bash deployment_artifacts/scripts/deploy_vm.sh
```

The script installs Docker if needed, writes `deployment_artifacts/.env.production`, builds the images, and starts Caddy, Redis, the backend, the worker, and Streamlit.

### Updating an existing VM deployment

For normal app updates:

1. Push your local changes.
2. In Cloud Shell, refresh the Cloud Shell checkout.
3. Copy that refreshed checkout to the VM and rerun the deploy helper.

```bash
cd ~/wearable-simulator && \
git pull --ff-only && \
gcloud compute ssh motionlens-vm --zone us-central1-a --command 'rm -rf "$HOME/wearable-simulator"' && \
gcloud compute scp --recurse ~/wearable-simulator motionlens-vm:~/ --zone us-central1-a && \
gcloud compute ssh motionlens-vm --zone us-central1-a --command 'cd "$HOME/wearable-simulator" && ML_HOST=motionlens.duckdns.org DUCKDNS_DOMAIN=motionlens DUCKDNS_TOKEN=<duckdns-token> bash deployment_artifacts/scripts/deploy_vm.sh'
```

If you force-pushed rewritten history, replace `git pull --ff-only` with:

```bash
git fetch origin && git reset --hard origin/main && git clean -fd
```

If you only want deployment files on the VM, stage a minimal bundle first. The bundle includes only `pyproject.toml`, `run_worker.py`, `.dockerignore`, `src/`, and `deployment_artifacts/`.

```bash
cd ~/wearable-simulator
bash deployment_artifacts/scripts/create_vm_bundle.sh
gcloud compute ssh motionlens-vm --zone us-central1-a --command 'rm -rf "$HOME/wearable-simulator"'
gcloud compute scp --recurse .vm-deploy-bundle motionlens-vm:~/wearable-simulator --zone us-central1-a
gcloud compute ssh motionlens-vm --zone us-central1-a --command 'cd "$HOME/wearable-simulator" && ML_HOST=motionlens.duckdns.org DUCKDNS_DOMAIN=motionlens DUCKDNS_TOKEN=<duckdns-token> bash deployment_artifacts/scripts/deploy_vm.sh'
```

This avoids copying `tests/`, `processing/`, `data/`, `output/`, `plans/`, and repo docs to the VM while still preserving the Docker build context the deployment stack expects.

### Clean redeploy

Use this only when the VM is in a bad state, old containers are hanging around, or ports `80` and `443` may still be occupied by an older MotionLens stack.

```bash
cd ~/wearable-simulator && \
git fetch origin && git reset --hard origin/main && git clean -fd && \
gcloud compute ssh motionlens-vm --zone us-central1-a --command '
set -euo pipefail
if [ -d "$HOME/wearable-simulator/deployment_artifacts" ]; then
	sudo docker compose -f "$HOME/wearable-simulator/deployment_artifacts/compose.yaml" down -v --remove-orphans || true
fi
if sudo ss -ltnp | grep -E ":(80|443)\\b" >/dev/null; then
	echo "Ports 80/443 are still in use after stopping the old MotionLens stack:"
	sudo ss -ltnp | grep -E ":(80|443)\\b" || true
	exit 1
fi
rm -rf "$HOME/wearable-simulator"
' && \
gcloud compute scp --recurse ~/wearable-simulator motionlens-vm:~/ --zone us-central1-a && \
gcloud compute ssh motionlens-vm --zone us-central1-a --command '
set -euo pipefail
cd "$HOME/wearable-simulator"
ML_HOST=motionlens.duckdns.org DUCKDNS_DOMAIN=motionlens DUCKDNS_TOKEN=<duckdns-token> bash deployment_artifacts/scripts/deploy_vm.sh
sudo docker compose -f deployment_artifacts/compose.yaml ps
'
```

The clean redeploy removes the previous MotionLens compose stack, its volumes, and the old VM checkout before copying a fresh repo snapshot. It does not remove unrelated host services that may also be bound to ports `80` or `443`.

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
- Do not overclaim signal quality or phone orientation as precise placement classification.
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