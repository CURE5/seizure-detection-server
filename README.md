# Seizure Detection Server

A Raspberry Pi based home monitoring server for people with epilepsy. It watches a
camera and, optionally, a Garmin watch's heart-rate broadcast, and alerts caregivers
when it sees something that looks like a seizure. A companion dashboard lives in
[seizure-detection-app](https://github.com/CURE5/seizure-detection-app).

Published by [CURE5](https://github.com/CURE5) as a reference implementation and a
starting point for families and developers who want to build their own monitor.

> **This is not a medical device.** It has not been clinically validated, it is
> not a substitute for a certified seizure alarm or for supervision, and it will
> miss events and raise false alarms. Use it only as one extra pair of eyes.

## What works, and what does not

Be aware of the current state before relying on any of it:

- **Heart-rate alerting works and is the primary detector.** A Garmin watch
  broadcasting over BLE feeds a rolling-baseline detector that alerts on sustained
  elevated heart rate. This path has caught real events in daily use.
- **Camera-based detection is experimental and mostly does not fire.** The
  motion gate, the pose-sequence model and the rhythmic-pattern logic are all in
  place and documented, but in practice they rarely detect a seizure on their own.
  They were trained on a small, single-subject dataset. Treat the camera path as a
  live video feed with recording, plus a research scaffold.
- **Everything else is solid for a home deployment:** video streaming and
  recording, seizure and medication logs, multi-channel notifications, a
  wake-up detector, a watchdog that reboots the Pi when it loses connectivity,
  and a Hailo-8L accelerated pose backend with a MediaPipe CPU fallback.

## Architecture

```
          ┌────────────┐     frames      ┌──────────────────────────────┐
Pi camera │ picamera2  │ ──────────────▶ │ analysis/                    │
          └────────────┘                 │  motion gate → pose model →  │──▶ alerts
          ┌────────────┐   BLE HR (0x180D)│  pattern detector           │
Garmin    │ garmin_hr  │ ──────────────▶ │  hr_detector (baseline+sustain)│
          └────────────┘                 └──────────────────────────────┘
                                                    │
                       FastAPI + WebSocket  ◀───────┘   notifications.py
                       (server.py)                      email · SMS · push · webhook
                              │
                       logging/   seizure, medication and HR logs (JSON/JSONL)
                       recording/ ffmpeg-backed video recorder
                       utils/     connectivity watchdog, health, logger
```

Package layout:

```
seizure_detection/
├── analysis/       detectors, pose backends (Hailo / MediaPipe), overlays
├── camera/         camera manager (Pi camera, USB/Mac webcam)
├── recording/      video recorder
├── logging/        seizure, medication, HR and data loggers
├── utils/          connectivity watchdog, health checks, logging setup
├── garmin_hr.py    BLE Heart Rate Profile client (optional, needs bleak)
├── notifications.py
└── config.py       every tunable, read from .env
server.py           FastAPI app, WebSocket stream, REST API
scripts/            training, evaluation, Garmin backfill and analysis tools
tests/              pytest suite (unit tests + model-dependent integration tests)
docs/               design and setup documents
```

## Quick start

### Development machine (macOS or Linux)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # fast installer
uv venv venv --python python3.11                     # mediapipe needs 3.11
source venv/bin/activate
uv pip install -r requirements.txt
bash download_models.sh                              # MediaPipe pose/face models
cp .env.example .env                                 # then edit
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/status` to confirm it is up, or run the dashboard app.

### Raspberry Pi 5

See [docs/PI_SETUP.md](docs/PI_SETUP.md) for the full walkthrough (OS image, Hailo,
camera, PM2, Cloudflare tunnel). The short version, once the Pi is prepared:

```bash
git clone https://github.com/CURE5/seizure-detection-server.git ~/seizure-detection-server
cd ~/seizure-detection-server
cp .env.example .env            # edit notification settings etc.
bash deploy.sh                  # venv, deps, models, PM2 process
```

From your workstation, later deploys are one command:

```bash
PI_HOST=pi@my-pi.local ./deploy_remote.sh
```

`deploy.sh`, `deploy_remote.sh` and `ecosystem.config.js` read `PI_HOST`,
`SERVER_DIR`, `APP_DIR` and `VENV_DIR` from the environment, so nothing about your
hostname or paths needs to be edited in the repo.

## Configuration

All settings live in `.env`; `.env.example` documents every key. The ones that
matter most:

| Setting | Purpose |
|---|---|
| `DETECTION_ALERT_NOTIFY_ONLY` | `true` (recommended): detectors only notify; a human confirms via `POST /sz`. |
| `SHADOW_MODE` | Run detectors without sending notifications (for testing). |
| `NOTIFY_*_ENABLED` | Enable email, SMS (Twilio), push (Pushover) or webhook alerts. |
| `ENABLE_GARMIN_HR` | Turn on the BLE heart-rate detector. |
| `HR_DELTA_THRESHOLD`, `HR_ABSOLUTE_THRESHOLD`, `HR_SUSTAIN_SEC` | Heart-rate trigger tuning. |
| `USE_POSE_MODEL`, `POSE_MODEL_THRESHOLD` | Camera-based pose detector (experimental). |
| `HAILO_MODEL_PATH` | Override the Hailo pose model location. |

How the two alert flags combine:

| `SHADOW_MODE` | `DETECTION_ALERT_NOTIFY_ONLY` | Behaviour |
|---|---|---|
| false | false | Notify, auto-log and auto-record |
| false | true | Notify only, no automatic actions (recommended) |
| true | false | Auto-log and auto-record, no notifications |
| true | true | Detection runs silently |

Notification setup is covered in [docs/NOTIFICATION_SETUP.md](docs/NOTIFICATION_SETUP.md).

### Garmin heart-rate monitor

On the watch: Settings → Sensors & Accessories → Wrist Heart Rate → Broadcast
Heart Rate → Broadcast Now, and leave that screen on. The server discovers any
device advertising the HR service, reconnects when the watch comes back into range,
and pauses baseline updates while readings are elevated so an event cannot drag the
baseline up and silence itself. A fixed pairing can be set with `GARMIN_HR_ADDRESS`.

On the Pi the onboard Bluetooth radio shares an antenna with WiFi; a cheap USB
Bluetooth dongle with `dtoverlay=disable-bt` gives a much more stable link.

The `scripts/garmin_backfill.py` and `scripts/garmin_hr_merge.py` tools pull daily
heart-rate history from Garmin Connect to fill gaps when the watch was out of range.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health`, `/status` | Liveness and full state, including the `hr` block and detector stats |
| GET/POST/PUT/DELETE | `/sz`, `/sz/{id}` | Seizure log |
| GET/POST/PUT/DELETE | `/medicine`, `/medicine/{id}` | Medication log |
| POST | `/command` | Start/stop recording, toggle overlay, night mode, notifications |
| GET | `/notifications/stats`, `/notifications/history` | Alert bookkeeping |
| WS | `/ws` | Live frames and status updates |

## Testing

```bash
./run_tests.sh            # picks venv/, sz/ or .venv/
uvx ruff check .          # lint (same config as CI)
```

Unit tests run anywhere. The backend-conformity tests exercise the real pose
pipeline and are skipped unless `mediapipe_models/` has been downloaded.

## Training your own camera model

The shipped ONNX models were trained on a small single-subject dataset and should be
treated as placeholders. [docs/CUSTOM_MODEL_TRAINING.md](docs/CUSTOM_MODEL_TRAINING.md)
and `training_data/README.md` describe how to extract pose sequences from your own
labelled videos and train a replacement. Keep training videos out of version control.

## Privacy

This system records video and health data of a vulnerable person. Everything it
writes (`seizure_logs/`, `medication_logs/`, `hr_logs/`, `data_logs/`,
`recorded_video/`, `garmin_backfill/`) is git-ignored. Do not put the Pi directly on
the internet; use an authenticated tunnel such as Cloudflare Access.

## Documentation

- [docs/ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md) — model vs. server responsibilities
- [docs/PIPELINE_ARCHITECTURE.md](docs/PIPELINE_ARCHITECTURE.md) — multi-stage detection pipeline
- [docs/AI_DETECTION.md](docs/AI_DETECTION.md) — how detections flow into alerts
- [docs/HAILO_SEIZURE_DETECTION_DESIGN.md](docs/HAILO_SEIZURE_DETECTION_DESIGN.md) — Hailo-8L pose backend
- [docs/MEDIAPIPE_33_LANDMARKS.md](docs/MEDIAPIPE_33_LANDMARKS.md) — landmark reference
- [docs/MOTION_DETECTION.md](docs/MOTION_DETECTION.md) — motion gate and motion model
- [docs/NOTIFICATION_SETUP.md](docs/NOTIFICATION_SETUP.md) — email, SMS, push, webhooks
- [docs/PI_SETUP.md](docs/PI_SETUP.md) — Raspberry Pi provisioning

## License

MIT. See [LICENSE](LICENSE).
