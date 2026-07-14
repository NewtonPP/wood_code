# Woodchip Monitor — Development & Deployment Guide

This is the single, authoritative guide for running Woodchip Monitor locally and
deploying it to Google Cloud. It is written so that a new developer can go from
a fresh clone to a working production deployment by reading top to bottom.

- [1. What this application is](#1-what-this-application-is)
- [2. Repository layout](#2-repository-layout)
- [3. Local development](#3-local-development)
- [4. Deploying to Google Cloud](#4-deploying-to-google-cloud)
- [5. Model files](#5-model-files)
- [6. Redeploying after code changes](#6-redeploying-after-code-changes)
- [7. Environment variable reference](#7-environment-variable-reference)
- [8. Troubleshooting](#8-troubleshooting)
- [9. Cost notes](#9-cost-notes)

---

## 1. What this application is

Woodchip Monitor analyses a live camera feed of wood chips and reports particle
sizes, a size histogram, moisture class, and quality alarms. Users open a
website, sign in, grant camera access, and the browser streams downsized JPEG
frames to the cloud; the AI results come back as JSON and the browser draws the
overlay itself. Only small frames go up and small JSON comes down.

```
Browser (any OS + USB camera)
   │  1. loads the website (served by the backend)
   │  2. streams downsized JPEG frames over a secure WebSocket (WSS)
   ▼
Cloud Run: woodchip-backend   (PUBLIC)    serves the React UI + API: auth, events,
   │                                      rules, audit; reads/writes Postgres
   │  POSTs each frame to /infer
   ▼
Cloud Run: woodchip-inference (PRIVATE)   runs the model + post-processing,
   │                                      returns JSON (boxes, sizes, moisture, alarm)
   ▲
Cloud SQL: Postgres                       users, sessions, events, rules, audit log
```

Key properties:

- **The inference service is private** (`--ingress=internal`) — only the backend
  can reach it, never the internet.
- **Inference is real**: the DETR ONNX model (exported from the trained
  checkpoint, Section 5) is baked into the inference image. It detects wood
  chips only — with no chips in view it correctly reports zero detections.
  Millimeter sizing and the oversize alarm activate once the blue reference
  disk is visible (or `DEFAULT_PIXELS_PER_MM` is set).
- **Database is automatic**: if `DATABASE_URL` is set to a `postgresql://…` DSN
  the app uses Postgres; if unset it uses a local SQLite file. The schema is
  created idempotently on first boot — there is no manual migration step.

### The legacy "device" role

The same backend can also run directly on an NVIDIA Jetson with a local USB
camera (`WOODCHIP_ROLE=device`, the default). In that mode it runs an
in-process TensorRT inference loop from `live_cam_trt/` instead of calling the
cloud inference service. That deployment is still supported for on-device
installs, but everything in Section 4 of this guide is about the **cloud** role.
The cloud Docker images do not contain any of the Jetson/TensorRT code.

## 2. Repository layout

| Path | What it is | Part of which deployable |
|---|---|---|
| `app/` | FastAPI backend: auth/RBAC, sessions, events, quality rules, audit, live endpoints, frame-ingest WebSocket | backend image |
| `backend_app.py` | Entrypoint shim (`uvicorn backend_app:app`) | backend image |
| `frontend/` | React + TypeScript + Vite source for the UI (see `frontend/README.md`) | built into the backend image |
| `web/` | Prebuilt frontend bundle, served by FastAPI at `/`. Committed so a device (Jetson) deploy can serve the UI without Node; the Docker build regenerates it from `frontend/` | backend image (rebuilt) |
| `woodchip_core/` | Shared, CUDA-free post-processing library (sizing, calibration, histogram, moisture, alarms). Used by both inference paths | backend + inference images |
| `inference_service/` | Private inference microservice: `POST /infer`, ONNX models (`model.py`), DETR ONNX export script | inference image |
| `live_cam_trt/` | Jetson-only TensorRT camera loop (device role). Not in any cloud image | Jetson install only |
| `deploy/` | `cloudbuild.yaml` (builds both images) and reference Cloud Run service specs | deploy tooling |
| `Dockerfile` | Backend image: Node stage builds the frontend, Python 3.11 stage serves it | backend image |
| `inference_service/Dockerfile` | Inference image (Python 3.11 + headless OpenCV + ONNX Runtime) | inference image |
| `requirements-cloud.txt` | Backend dependencies (cloud role): FastAPI, uvicorn, httpx, psycopg, numpy | backend image |
| `inference_service/requirements.txt` | Inference dependencies: FastAPI, numpy, opencv-headless, onnxruntime | inference image |
| `requirements.txt` | **Legacy Jetson (Python 3.6) pins** — only for on-device installs; TensorRT/pycuda/cv2 come from JetPack, not pip | Jetson install only |
| `run.sh` | Local dev launcher (backend + Vite dev server) | dev only |

## 3. Local development

### Prerequisites

- Python 3.11+
- Node 20+
- (optional) Docker, only if you want to test the exact production images

### First-time setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-cloud.txt
cd frontend && npm install && cd ..
```

Locally the app uses **SQLite** (`woodchip_app.sqlite3` in the repo root, git-
ignored) — no database setup needed. The admin account is provisioned on boot
from `ADMIN_EMAIL` / `ADMIN_PASSWORD` (put them in a local `.env` file, which is
git-ignored, or export them in your shell).

### Run the full cloud-mode stack (recommended)

This mirrors production: browser camera → backend → inference service, all with
the **real DETR model**. You need `inference_service/models/detr_resnet101.onnx`
on disk first — it is git-ignored, so on a fresh clone export it once
(Section 5).

The one-command way:

```bash
./run.sh    # starts inference :9000 + backend :8000 + Vite dev server :5173
```

Or the three processes by hand:

```bash
# Terminal 1: the inference service (real ONNX model)
.venv/bin/python -m uvicorn inference_service.main:app --port 9000
# needs: .venv/bin/pip install -r inference_service/requirements.txt

# Terminal 2: the backend in cloud mode, pointed at the inference service
WOODCHIP_ROLE=cloud INFERENCE_URL=http://localhost:9000 \
  ADMIN_EMAIL=admin@test.com ADMIN_PASSWORD=testpass123 \
  .venv/bin/python -m uvicorn backend_app:app --port 8000

# Terminal 3: the frontend dev server (hot reload; proxies /api to :8000)
cd frontend && npm run dev
```

Open http://localhost:5173, sign in with `admin@test.com` / `testpass123`, go to
**Live**, and grant camera access. Point the camera at wood chips: boxes and
statistics appear, and with the blue reference disk in view they switch from
pixels to mm (enabling the oversize alarm). With no chips in view, zero
detections is correct behavior.

Note: the frontend served at http://localhost:8000 is the **prebuilt** bundle
in `web/`; the dev server at :5173 is the live-reload source. After changing
frontend code, regenerate `web/` with `cd frontend && npm run build`.

### Test the exact production containers (optional)

```bash
docker build -t woodchip-backend:test .
docker build -f inference_service/Dockerfile -t woodchip-inference:test .
docker network create wc-net
docker run -d --name wc-inf --network wc-net woodchip-inference:test
docker run -d --name wc-be --network wc-net -p 8080:8000 \
  -e WOODCHIP_ROLE=cloud -e INFERENCE_URL=http://wc-inf:9000 \
  -e ADMIN_EMAIL=admin@test.com -e ADMIN_PASSWORD=testpass123 woodchip-backend:test
curl http://localhost:8080/ping        # -> {"status":"ok"}
# cleanup:
docker rm -f wc-inf wc-be && docker network rm wc-net
```

## 4. Deploying to Google Cloud

What you end up with: a public HTTPS website on **Cloud Run**
(`woodchip-backend`), a private inference service on Cloud Run
(`woodchip-inference`), a **Cloud SQL Postgres** database, and secrets in
**Secret Manager**. Cloud Build compiles the images, so you do **not** need
Docker or Node locally to deploy.

Run every command from the repo root. Steps 4.1–4.9 are one-time setup;
after that, updates are just Section 6.

### 4.1 Prerequisites

A Google Cloud account with billing enabled and a project. Then:

```bash
gcloud auth login
export PROJECT_ID=your-project-id     # your real GCP project ID
export REGION=us-central1             # pick a region near your users
export REPO=woodchip                  # Artifact Registry repo name
gcloud config set project "$PROJECT_ID"
```

These shell variables are reused by every step below — if you come back in a
new terminal, re-export them first.

### 4.2 Enable the required APIs

```bash
gcloud services enable \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  sqladmin.googleapis.com secretmanager.googleapis.com vpcaccess.googleapis.com
```

### 4.3 Create the image repository (Artifact Registry)

If you are deploying into an **existing** project, first check whether a Docker
repository already exists and reuse its name as `$REPO` instead of creating a
new one (the deployed Cloud Run services must pull from the repo that actually
holds the images):

```bash
gcloud artifacts repositories list --location="$REGION"
```

Otherwise create it:

```bash
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker --location="$REGION"
```

### 4.4 Build and push both images (Cloud Build)

**Prerequisite:** `inference_service/models/detr_resnet101.onnx` (required)
and `inference_service/models/moistnetlite.onnx` (moisture) must exist on the
machine you build from — both are baked into the inference image. They are
git-ignored, so on a fresh clone export them first (Section 5).

Cloud Build reads [`deploy/cloudbuild.yaml`](deploy/cloudbuild.yaml), builds the
`backend` image (which compiles the React frontend inside it) and the
`inference` image, and pushes both to Artifact Registry:

```bash
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_REGION="$REGION",_REPO="$REPO"
```

Expect a few minutes. If it fails immediately, an API from 4.2 is usually
missing, or you are not in the repo root.

### 4.5 Create the database (Cloud SQL for Postgres)

```bash
gcloud sql instances create woodchip-pg \
  --database-version=POSTGRES_16 --tier=db-f1-micro --region="$REGION"

gcloud sql databases create woodchip --instance=woodchip-pg
gcloud sql users create wc_app --instance=woodchip-pg --password='CHANGE_ME_STRONG'

export SQL_CONN=$(gcloud sql instances describe woodchip-pg \
  --format='value(connectionName)')
export DATABASE_URL="postgresql://wc_app:CHANGE_ME_STRONG@/woodchip?host=/cloudsql/${SQL_CONN}"
```

The `?host=/cloudsql/…` form makes psycopg connect over Cloud Run's built-in
Cloud SQL unix socket — the database needs no public IP. You do **not** create
tables manually; the app creates the schema on first boot.

### 4.6 Store secrets (Secret Manager)

```bash
printf '%s' "admin@yourcompany.com"   | gcloud secrets create ADMIN_EMAIL    --data-file=-
printf '%s' "a-strong-admin-password" | gcloud secrets create ADMIN_PASSWORD --data-file=-
printf '%s' "$DATABASE_URL"           | gcloud secrets create DATABASE_URL   --data-file=-

# Allow the Cloud Run runtime service account to read them:
export PROJ_NUM=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
export RUN_SA="${PROJ_NUM}-compute@developer.gserviceaccount.com"
for S in ADMIN_EMAIL ADMIN_PASSWORD DATABASE_URL; do
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:${RUN_SA}" --role=roles/secretmanager.secretAccessor
done
```

The admin account is provisioned from these secrets on the backend's first
boot — there is no "first signup becomes admin" flow.

### 4.7 Deploy the private inference service

The DETR ONNX model is baked into the image (it must have been on disk when
you ran the build in 4.4 — see Section 5). CPU inference takes ~2 s/frame, so
keep one instance warm and the concurrency low:

```bash
gcloud run deploy woodchip-inference \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/inference:latest" \
  --region="$REGION" --port=9000 \
  --ingress=internal --no-allow-unauthenticated \
  --min-instances=1 --max-instances=4 --cpu=4 --memory=4Gi --concurrency=2

export INFERENCE_URL=$(gcloud run services describe woodchip-inference \
  --region="$REGION" --format='value(status.url)')
echo "Inference URL: $INFERENCE_URL"
```

`--ingress=internal` makes the service unreachable from the internet — only
callers routed through your project's VPC can reach it.

**Required:** the backend does not attach IAM identity tokens to its requests,
so you must allow unauthenticated *invocations* (privacy is enforced by the
internal ingress, not by IAM). Without this, every `/infer` call is rejected
with HTTP 403 and the Live page stays on "Collecting…" forever:

```bash
gcloud run services add-iam-policy-binding woodchip-inference \
  --region="$REGION" --member=allUsers --role=roles/run.invoker
```

### 4.8 Private connectivity (backend → inference)

Because the inference service only accepts in-VPC traffic, the backend needs a
**Serverless VPC Access connector** so its outbound requests are routed through
the VPC:

```bash
gcloud compute networks vpc-access connectors create woodchip-conn \
  --region="$REGION" --range=10.8.0.0/28
```

The backend is deployed with `--vpc-egress=all-traffic` (next step), which
sends **all** of its outbound traffic through this connector. That breaks the
backend's access to the public internet unless the VPC has a NAT, so also
create a Cloud Router + Cloud NAT:

```bash
gcloud compute routers create woodchip-router \
  --project="$PROJECT_ID" --network=default --region="$REGION"

gcloud compute routers nats create woodchip-nat \
  --project="$PROJECT_ID" --router=woodchip-router --region="$REGION" \
  --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges
```

### 4.9 Deploy the public backend

```bash
gcloud run deploy woodchip-backend \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/backend:latest" \
  --region="$REGION" --port=8000 \
  --allow-unauthenticated \
  --vpc-connector=woodchip-conn --vpc-egress=all-traffic \
  --add-cloudsql-instances="$SQL_CONN" \
  --set-env-vars=WOODCHIP_ROLE=cloud,WOODCHIP_COOKIE_SECURE=1,INFERENCE_URL="$INFERENCE_URL" \
  --set-secrets=ADMIN_EMAIL=ADMIN_EMAIL:latest,ADMIN_PASSWORD=ADMIN_PASSWORD:latest,DATABASE_URL=DATABASE_URL:latest \
  --min-instances=1 --max-instances=10 --cpu=1 --memory=512Mi
```

On first boot the backend creates the Postgres schema and provisions the admin
account from the secrets. Watch the first-boot logs:

```bash
gcloud run services logs read woodchip-backend --region="$REGION" --limit=50
# look for: "Administrator account ready: <email>"
```

Reference Cloud Run specs for both services (same settings in declarative
form) live in `deploy/backend.service.yaml` and `deploy/inference.service.yaml`.

### 4.10 Verify the deployment

```bash
export BACKEND_URL=$(gcloud run services describe woodchip-backend \
  --region="$REGION" --format='value(status.url)')
echo "$BACKEND_URL"

curl -s "$BACKEND_URL/ping"       # {"status":"ok"}
curl -s "$BACKEND_URL/api/info"   # {"ok":true,"device_id":"device-1","role":"cloud"}
```

Then in a browser:

1. Open `$BACKEND_URL` and sign in with the admin email/password (the secrets).
2. Go to **Live**, allow camera access, and point the camera at wood chips —
   boxes and statistics appear; with the blue reference disk in view, sizes
   switch to mm and oversize chips are highlighted red. (No chips in view →
   zero detections, which is correct.)
3. Confirm the inference service is **not** public: `curl $INFERENCE_URL/healthz`
   from your laptop must be blocked (403/404), not return JSON.
4. Create a second, non-admin user (Sign up, or admin → Manage Users) and
   confirm they only see their own device's data on the Events page.

### 4.11 Custom domain + HTTPS (optional)

Cloud Run already gives you an HTTPS `…run.app` URL. For your own domain:

```bash
gcloud beta run domain-mappings create --service=woodchip-backend \
  --domain=app.yourcompany.com --region="$REGION"
```

Add the DNS records the command prints; a managed TLS certificate is
provisioned automatically (allow up to ~1 hour). Keep `WOODCHIP_COOKIE_SECURE=1`
(already set) so session cookies are HTTPS-only.

## 5. Model files

The inference service (`inference_service/model.py`) loads the models from
`inference_service/models/` at startup. The DETR sizing model is **required**;
the MoistNetLite moisture model is optional (the service runs without it, but
the moisture panel then shows "Collecting…").

**5.1 DETR (sizing) — export once per trained checkpoint.** On your
workstation (needs `torch` and `transformers<5`; these are deliberately NOT in
the serving image):

```bash
python inference_service/export_detr_onnx.py \
  --ckpt /path/to/best-detr-repaired.ckpt \
  --out  inference_service/models/detr_resnet101.onnx
```

The `.onnx` file is git-ignored but IS copied into the inference image (the
Dockerfile copies all of `inference_service/`), so the machine that runs the
image build must have the file on disk. After exporting a new checkpoint,
rebuild (4.4) and roll out the inference service (Section 6).

**Calibration:** measurements are in **mm** (and the oversize alarm / red
boxes work) only when the system has a pixels-per-mm scale — either the blue
reference disk is visible in frame (auto-calibration), or
`DEFAULT_PIXELS_PER_MM` is set on the inference service. Until then the UI
shows an "Uncalibrated" hint and sizes in pixels.

**5.2 Moisture model — export once per trained weights file.** On your
workstation (needs `tensorflow` 2.15 [Keras 2] and `tf2onnx`; deliberately NOT
in the serving image):

```bash
python inference_service/export_moistnet_onnx.py \
  --weights moistnetlite_best_weights.h5 \
  --out inference_service/models/moistnetlite.onnx
```

The script rebuilds the MoistNetLite inference graph, loads and verifies every
weight tensor, exports with a parity check (Keras ↔ ONNX), and — because the
runtime normalizes crops with ImageNet mean/std — prepends a compensation
layer selected by `--train-preproc {raw,unit,imagenet}` (default `raw`:
training fed 0–255 pixels). If deployed predictions look degenerate (always
one class, saturated probabilities), re-export with `--train-preproc unit` or
`imagenet` and roll out again.

Class labels live in `inference_service/models/moistnetlite_classes.txt`
(committed; `dry`/`medium`/`wet`, one per line, in the model's output order).
Like the DETR file, the exported `.onnx` is git-ignored but baked into the
inference image at build time, so the build host must have it on disk. After
exporting, rebuild (4.4) and roll out the inference service (Section 6).

**5.3 GPU (optional, for throughput).** Swap `onnxruntime` for
`onnxruntime-gpu` in `inference_service/requirements.txt`, rebuild on a CUDA
base image, and add a Cloud Run GPU accelerator to the inference service. The
CUDA execution provider is picked up automatically.

## 6. Redeploying after code changes

```bash
# 1. rebuild + push both images
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_REGION="$REGION",_REPO="$REPO"

# 2. roll out the service(s) that changed
gcloud run services update woodchip-backend --region="$REGION" \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/backend:latest"
gcloud run services update woodchip-inference --region="$REGION" \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/inference:latest"
```

Schema changes are applied automatically on boot (idempotent create); there is
no separate migration step for the current schema.

## 7. Environment variable reference

| Variable | Service | Meaning |
|---|---|---|
| `WOODCHIP_ROLE` | backend | `cloud` (browser capture → inference service) or `device` (Jetson in-process loop, the default). Production uses `cloud`. |
| `INFERENCE_URL` | backend | Base URL of the private inference service (cloud role). |
| `DATABASE_URL` | backend | `postgresql://…` DSN → Postgres; unset → SQLite. |
| `WOODCHIP_DB_PATH` | backend | SQLite file path override (default `woodchip_app.sqlite3`). |
| `WOODCHIP_COOKIE_SECURE` | backend | `1` = HTTPS-only session cookie. Set in production. |
| `WOODCHIP_DEVICE_ID` | backend | Deployment identifier shown in the UI (default `device-1`). |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | backend | Provision the admin account on startup (Secret Manager in prod, `.env` in dev). |
| `DETR_ONNX_PATH` | inference | DETR sizing model path (default: `inference_service/models/detr_resnet101.onnx`, baked into the image). |
| `DEFAULT_PIXELS_PER_MM` | inference | Manual mm calibration for fixed-mount cameras. Without it (and with no blue reference disk in view) sizes are in **pixels** and the oversize alarm/red boxes are disabled. |
| `MOISTURE_ONNX_PATH` / `MOISTURE_CLASSES_PATH` | inference | Moisture model + labels file (defaults: `inference_service/models/moistnetlite.onnx` + `moistnetlite_classes.txt`, baked into the image). |

## 8. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `gcloud builds submit` fails immediately | An API isn't enabled (4.2), or you're not in the repo root. |
| Backend boots but 500s on login | `DATABASE_URL` secret wrong, or Cloud SQL not attached — check `--add-cloudsql-instances` and the DSN's `?host=/cloudsql/…` part. |
| "Administrator account was not provisioned" in logs | `ADMIN_EMAIL`/`ADMIN_PASSWORD` secrets missing, or the runtime SA lacks `secretAccessor` (4.6). |
| Live page shows "Connection lost" / no overlays | Backend can't reach the inference service — check the VPC connector (4.8), `INFERENCE_URL`, and that inference is deployed. |
| Everything stays "Collecting…" and inference logs show `POST 403 …/infer` | The invoker grant is missing — run the `add-iam-policy-binding` command in 4.7. |
| Everything stays "Collecting…", no requests in inference logs | Check the WebSocket messages in browser DevTools (Network → `ingest/ws`): `"ok": false` errors name the actual failure. |
| Backend can't reach anything on the public internet | `--vpc-egress=all-traffic` without Cloud NAT — create the router + NAT (4.8). |
| Camera never starts in the browser | The site must be HTTPS (Cloud Run is) and the user must allow camera access; some browsers block it in embedded/incognito contexts. |
| Inference service reachable publicly | It must be `--ingress=internal --no-allow-unauthenticated`; redeploy 4.7. |

## 9. Cost notes

- **Cloud Run** bills per request/CPU time. `min-instances=1` on the backend
  keeps it warm (small always-on cost); inference scales to zero when idle.
- **Cloud SQL** `db-f1-micro` is the cheapest tier; it runs continuously.
- **GPU inference** (5.4) is the biggest cost lever once real models are live —
  right-size the GPU, keep scale-to-zero, and cap the camera FPS.
