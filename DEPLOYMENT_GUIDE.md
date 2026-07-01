# Woodchip Monitor — Detailed Deployment Guide (Google Cloud)

This is the complete, step-by-step procedure to deploy the app to the internet on
Google Cloud. It explains **what each step does** and **what to expect**, so you can
follow it without prior GCP experience. For the terse command-only version see
[`deploy/README.md`](deploy/README.md).

> **What "deployed" means here:** a public HTTPS website (the React UI + FastAPI
> backend) on **Cloud Run**, a **private** GPU-capable inference service on Cloud Run,
> and a **Postgres** database on **Cloud SQL**. Users open the website in any browser,
> grant camera access, and the browser streams frames to the cloud for analysis.

---

## 0. How the pieces fit together

```
Browser (any OS + USB camera)
   │  1. loads the website (served by the backend)
   │  2. streams downsized JPEG frames over a secure WebSocket (WSS)
   ▼
Cloud Run: woodchip-backend  (PUBLIC)         ── serves the UI, auth, events, rules
   │  forwards each frame ▼                       reads/writes Postgres
Cloud Run: woodchip-inference (PRIVATE)        ── runs the model + post-processing
   │  returns JSON (boxes, sizes, moisture, alarm)
   ▲
Cloud SQL: Postgres                            ── users, sessions, events, rules, audit
```

- **The browser draws the AI overlay itself** from the returned JSON, so only small
  frames go up and small JSON comes down.
- **The inference service is private** — only the backend can reach it.
- **Today the model is a "mock"** (returns realistic synthetic detections) because the
  real model files aren't available yet. Everything else is real. Swapping in the real
  model later is one config change to one service (Section 9).

---

## 1. Answers to the common questions first

- **Does it all work?** Yes — verified locally end-to-end: both Docker images **build and
  run**, the backend serves the UI and connects to the inference service, admin login
  works, browser→cloud→overlay data flow works, and it runs on **both** SQLite (dev) and
  **Postgres** (verified against a real Postgres). The one thing not machine-tested is the
  live camera overlay rendering, which needs a physical webcam — the data path behind it
  is verified.
- **Switching to Postgres = just set `DATABASE_URL`?** Yes. If `DATABASE_URL` is set (to a
  `postgresql://…` string) the app uses Postgres; if unset, it uses SQLite. Nothing else to
  change — the schema is created automatically on first startup. `psycopg` (the Postgres
  driver) is already in `requirements-cloud.txt`.
- **Are we using Postgres in Docker right now?** No. The Postgres-in-Docker was a temporary
  **test** container that has been removed. Locally the app uses **SQLite**
  (`woodchip_app.sqlite3`). Postgres is used **in the cloud** via Cloud SQL.
- **Will the deploy be seamless?** The app artifacts are proven (images build + run). The
  GCP part is a sequence of console/CLI steps; it's straightforward but not "one click" —
  budget time for the manual bits (Cloud SQL creation, secrets, the VPC connector). Common
  snags and fixes are in Section 8.

---

## 2. Prerequisites

1. A **Google Cloud account** with billing enabled, and a **project** (note its Project ID).
2. The **gcloud CLI** installed and logged in:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```
3. That's it for the cloud path — **Cloud Build compiles the images for you**, so you do
   **not** need Docker or Node installed locally to deploy. (You only need them if you want
   to test locally first, Section 3.)

Set these shell variables once; the rest of the guide reuses them:
```bash
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-central1        # pick a region near your users
export REPO=woodchip
```

---

## 3. (Optional but recommended) Test locally before deploying

This confirms your machine + code are healthy before touching the cloud. Requires Python
(the repo `.venv`), Node, and optionally Docker.

**Option A — run the three processes directly:**
```bash
# Terminal 1: the (mock) inference service
INFER_BACKEND=mock .venv/bin/python -m uvicorn inference_service.main:app --port 9000

# Terminal 2: the backend in cloud mode, pointed at the inference service
WOODCHIP_ROLE=cloud INFERENCE_URL=http://localhost:9000 \
  ADMIN_EMAIL=admin@test.com ADMIN_PASSWORD=testpass123 \
  .venv/bin/python -m uvicorn backend_app:app --port 8000

# Terminal 3: the frontend dev server (proxies /api to :8000)
cd frontend && npm run dev
```
Open http://localhost:5173, sign in with `admin@test.com` / `testpass123`, and grant camera
access on the Live page. (Uses SQLite locally — no Postgres needed.)

**Option B — run the exact container images (closest to production):**
```bash
docker build -t woodchip-backend:test .
docker build -f inference_service/Dockerfile -t woodchip-inference:test .
docker network create wc-net
docker run -d --name wc-inf --network wc-net -e INFER_BACKEND=mock woodchip-inference:test
docker run -d --name wc-be --network wc-net -p 8080:8000 \
  -e WOODCHIP_ROLE=cloud -e INFERENCE_URL=http://wc-inf:9000 \
  -e ADMIN_EMAIL=admin@test.com -e ADMIN_PASSWORD=testpass123 woodchip-backend:test
curl http://localhost:8080/ping         # -> {"status":"ok"}
# cleanup:
docker rm -f wc-inf wc-be && docker network rm wc-net
```

---

## 4. Google Cloud — one-time project setup

**4.1 Enable the required APIs** (services this deploy uses):
```bash
gcloud services enable \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  sqladmin.googleapis.com secretmanager.googleapis.com vpcaccess.googleapis.com
```

**4.2 Create the image repository** (where your two built images are stored):
```bash
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker --location="$REGION"
```

---

## 5. Build and push the images

Cloud Build reads [`deploy/cloudbuild.yaml`](deploy/cloudbuild.yaml), builds **both** images
(backend + inference), and pushes them to Artifact Registry. This also compiles the React
frontend inside the backend image.
```bash
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_REGION="$REGION",_REPO="$REPO"
```
Expect a few minutes. Success ends with both images listed. If it fails, it's almost always
a missing API (redo 4.1) or a Dockerfile path issue (run from the repo root).

---

## 6. Create the database (Cloud SQL for Postgres)

```bash
# Create the instance (db-f1-micro is the cheapest; size up later if needed)
gcloud sql instances create woodchip-pg \
  --database-version=POSTGRES_16 --tier=db-f1-micro --region="$REGION"

# Create the database and an application user
gcloud sql databases create woodchip --instance=woodchip-pg
gcloud sql users create wc_app --instance=woodchip-pg --password='CHANGE_ME_STRONG'

# Capture the instance connection name and build the DSN Cloud Run will use
export SQL_CONN=$(gcloud sql instances describe woodchip-pg --format='value(connectionName)')
export DATABASE_URL="postgresql://wc_app:CHANGE_ME_STRONG@/woodchip?host=/cloudsql/${SQL_CONN}"
echo "$DATABASE_URL"
```
The `?host=/cloudsql/…` form makes psycopg connect over Cloud Run's built-in Cloud SQL
socket — no public database IP needed. **You do not create any tables manually**; the app
creates them on first boot.

---

## 7. Store secrets

Keep the admin password and DB URL out of plain config by putting them in Secret Manager:
```bash
printf '%s' "admin@yourcompany.com"  | gcloud secrets create ADMIN_EMAIL    --data-file=-
printf '%s' "a-strong-admin-password" | gcloud secrets create ADMIN_PASSWORD --data-file=-
printf '%s' "$DATABASE_URL"           | gcloud secrets create DATABASE_URL   --data-file=-

# Allow the Cloud Run runtime service account to read them
export PROJ_NUM=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
export RUN_SA="${PROJ_NUM}-compute@developer.gserviceaccount.com"
for S in ADMIN_EMAIL ADMIN_PASSWORD DATABASE_URL; do
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:${RUN_SA}" --role=roles/secretmanager.secretAccessor
done
```

---

## 8. Deploy the two services

**8.1 Inference service (private).** `--ingress=internal` + `--no-allow-unauthenticated`
means it is not reachable from the internet — only from inside your project's network.
```bash
gcloud run deploy woodchip-inference \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/inference:latest" \
  --region="$REGION" --port=9000 \
  --ingress=internal --no-allow-unauthenticated \
  --set-env-vars=INFER_BACKEND=mock \
  --min-instances=0 --max-instances=4 --cpu=2 --memory=2Gi

export INFERENCE_URL=$(gcloud run services describe woodchip-inference \
  --region="$REGION" --format='value(status.url)')
echo "Inference URL: $INFERENCE_URL"
```

**8.2 VPC connector** so the public backend can reach the private inference URL:
```bash
gcloud compute networks vpc-access connectors create woodchip-conn \
  --region="$REGION" --range=10.8.0.0/28
```

**8.3 Backend service (public).** Serves the website and talks to Postgres + inference.
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
On first boot the backend **creates the Postgres schema** and **provisions the admin account**
from the secrets. Watch the logs the first time:
```bash
gcloud run services logs read woodchip-backend --region="$REGION" --limit=50
# look for: "Administrator account ready: <email>"
```

---

## 9. Verify the deployment

```bash
export BACKEND_URL=$(gcloud run services describe woodchip-backend \
  --region="$REGION" --format='value(status.url)')
echo "$BACKEND_URL"

curl -s "$BACKEND_URL/ping"       # {"status":"ok"}
curl -s "$BACKEND_URL/api/info"   # {"ok":true,"device_id":"device-1","role":"cloud"}
```
Then in a browser:
1. Open `$BACKEND_URL`, sign in with your admin email/password (the secrets).
2. Go to **Live**, allow camera access → you should see boxes/sizes/alarm overlays plus the
   stats, histogram, and moisture panels updating (driven by the mock model).
3. Confirm the **inference URL is not public**: `curl $INFERENCE_URL/healthz` from your
   laptop should be blocked/unauthorized.

Create a second, non-admin user (Sign up, or admin → Manage Users) and confirm they only see
their own device's data on the Events page — tenant isolation is enforced.

---

## 10. Custom domain + HTTPS (permanent URL)

Cloud Run gives you a `…run.app` URL with HTTPS already. For your own domain:
```bash
gcloud beta run domain-mappings create --service=woodchip-backend \
  --domain=app.yourcompany.com --region="$REGION"
```
Add the DNS records it prints. A managed TLS certificate is provisioned automatically
(can take up to ~1 hour). Keep `WOODCHIP_COOKIE_SECURE=1` (already set) so cookies are
HTTPS-only.

---

## 11. Switching to the REAL model later (no app rebuild of backend/frontend)

The ONNX model backend is already wired. When you have the model files:
1. Export the DETR (sizing) and moisture models to **ONNX**.
2. Make them available to the inference image (bake them in, or mount from a bucket/volume).
3. Rebuild the inference image on a **GPU** base (uncomment `onnxruntime-gpu` in
   `inference_service/requirements.txt`) and redeploy **only** the inference service with:
   ```bash
   gcloud run services update woodchip-inference --region="$REGION" \
     --update-env-vars=INFER_BACKEND=onnx,DETR_ONNX_PATH=/models/detr.onnx,MOISTURE_ONNX_PATH=/models/moist.onnx,MOISTURE_CLASSES_PATH=/models/classes.txt
   ```
No changes to the backend, frontend, database, or user experience.

---

## 12. Updating / redeploying after code changes

```bash
# rebuild + push both images
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_REGION="$REGION",_REPO="$REPO"
# roll out the new image (repeat for woodchip-inference if that changed)
gcloud run deploy woodchip-backend \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/backend:latest" --region="$REGION"
```
Database schema changes are applied automatically on boot (the schema is created
idempotently); there is no separate migration step for the current schema.

---

## 13. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `gcloud builds submit` fails immediately | An API isn't enabled (redo 4.1), or you're not in the repo root. |
| Backend boots but 500s on login | `DATABASE_URL` secret wrong, or Cloud SQL not attached — check `--add-cloudsql-instances` and the DSN's `?host=/cloudsql/…`. |
| "Administrator account was not provisioned" in logs | `ADMIN_EMAIL` / `ADMIN_PASSWORD` secrets missing or SA lacks access (redo Section 7). |
| Live page shows "Connection lost" / no overlays | Backend can't reach the inference service — check the VPC connector (8.2), `INFERENCE_URL`, and that inference is deployed. |
| Camera never starts in the browser | The site must be HTTPS (it is on Cloud Run) and the user must **allow** camera access; some browsers block it in embedded/incognito contexts. |
| Inference reachable publicly | It should be `--ingress=internal --no-allow-unauthenticated`; redeploy 8.1. |

---

## 14. Cost notes (rough)

- **Cloud Run** bills per request/CPU-time; `min-instances=1` on the backend keeps it warm
  (small always-on cost). Inference uses `min-instances=0` (scales to zero when idle).
- **Cloud SQL** `db-f1-micro` is the cheapest tier; it runs continuously.
- **GPU inference** (Section 11) is the biggest cost lever once real models are added —
  right-size the GPU, keep scale-to-zero, and cap camera FPS.

---

### Reference: environment variables

| Variable | Service | Meaning |
|---|---|---|
| `WOODCHIP_ROLE` | backend | `cloud` (browser-capture) or `device` (Jetson all-in-one). Use `cloud`. |
| `INFERENCE_URL` | backend | Base URL of the private inference service. |
| `DATABASE_URL` | backend | Postgres DSN → uses Postgres; unset → SQLite. |
| `WOODCHIP_COOKIE_SECURE` | backend | `1` for HTTPS-only session cookies (cloud). |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | backend | Provision the admin account on startup. |
| `INFER_BACKEND` | inference | `mock` (default, no model files) or `onnx` (real models). |
| `DETR_ONNX_PATH` / `MOISTURE_ONNX_PATH` / `MOISTURE_CLASSES_PATH` | inference | Model files when `INFER_BACKEND=onnx`. |
