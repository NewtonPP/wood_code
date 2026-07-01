# Deployment Plan — Woodchip Monitor as a Cloud-Inference Product

## Context

This codebase is a real-time wood-chip monitoring system that today runs **entirely
on a single Jetson Nano**: FastAPI backend (`app/`), React frontend (`frontend/` →
built into `web/`), SQLite DB, and AI inference (`live_cam_trt/`) all run in **one
process on one device**, reading a USB camera locally and running two TensorRT models
on the Jetson GPU.

**Chosen product direction (directive):** the software should be **platform-independent**
— deployable not just on Jetson Nano (Linux) but on Linux, Windows, and macOS in the
future. To achieve that, **the AI model runs in the cloud (Google Cloud)** and the
**local device acts primarily as the user interface + camera/data-collection
component**. This centralizes the model so updates, maintenance, and scaling across
hardware platforms are handled once, in the cloud, with no per-device model
deployment.

**Consequences of this direction (acknowledged):**
- The Jetson-specific inference (`live_cam_trt/loop.py`, `engines.py`, TensorRT,
  `pycuda`, JetPack/CUDA) is **retired from the local device**. The local device needs
  **no GPU and no CUDA** — this is exactly what makes it platform-independent.
- The model is served centrally on a **cloud GPU**, fed by frames streamed from the
  local device.
- The pure-Python post-processing logic currently inside `live_cam_trt` (sizing,
  reference calibration, alarm, histogram, moisture aggregation) **moves to the cloud**
  alongside the model (it's mostly numpy and is not Jetson-specific).

---

## Target Architecture

```
 Local device — ANY OS (Jetson/Linux/Windows/macOS) — just a browser + USB camera
        │  1) Browser captures the USB camera (getUserMedia / WebRTC)
        │  2) Streams downsized frames  ──────────────► (WSS / WebRTC)
        ▼
 ┌──────────────────────────── GOOGLE CLOUD ────────────────────────────┐
 │  React frontend (web/)  ── served to the browser                      │
 │  FastAPI backend        ── auth, RBAC, sessions, rules, events, audit │
 │       │ forwards frames ▼                                             │
 │  Inference service (GPU) ── model + post-processing                   │
 │       • DETR (size) + Moisture model  (served from your PyTorch files)│
 │       • sizing / calibration / alarm / histogram / moisture aggregation│
 │       └─ returns detections + stats + moisture + histogram (JSON)     │
 │  Postgres DB  ── users, devices, events, rules, audit                 │
 └───────────────────────────────────────────────────────────────────────┘
        ▲
        │  3) Results (boxes + stats) ──► browser overlays them on its own video
```

**Flow:** the local browser opens the website (served from the cloud), accesses the
USB camera, and streams downsized frames up. The cloud backend forwards frames to a
GPU-backed inference service that runs the model **and** the sizing/moisture/alarm
post-processing, then returns compact JSON (bounding boxes, diameters, alarm state,
histogram, moisture). The browser draws the overlay on its **own** local video (so we
only upload frames and download small JSON — no annotated frames sent back). Events
are sampled to Postgres exactly as today.

### Why browser-based capture (recommended)
- **True zero-install platform independence:** any device with a modern browser + a
  USB webcam works — Jetson, Linux, Windows, macOS, even a tablet. No native client,
  no per-OS packaging.
- **Reuses the existing React frontend** heavily; the local side is "just the website."
- An **optional native capture client** (cross-platform Python+OpenCV or Electron) can
  be added later for headless/industrial cameras or unattended kiosks — but it is not
  required for v1.

---

## What we reuse, build, and retire

| | Detail |
|---|---|
| **Reuse** | React frontend (`frontend/`), FastAPI backend structure (auth, `rbac.py`, `db.py`, routers for rules/events/audit/admin), and the **numpy post-processing** in `live_cam_trt/geometry.py`, `reference.py`, `moisture.py`, plus histogram/alarm logic. |
| **Build (new)** | Browser camera capture + frame streaming; a cloud **inference service** (model serving + post-processing); a frame **ingest/stream** endpoint on the backend; browser-side overlay rendering; per-device live-data routing. |
| **Retire from device** | `live_cam_trt/loop.py`, `engines.py`, the in-process inference thread (`app/main.py:57-65`), and all `tensorrt`/`pycuda`/JetPack/CUDA dependencies. The local device runs **no Python inference at all**. |

---

## Model serving on Google Cloud

You have the **original PyTorch model files** — this is what makes cloud serving
straightforward.

1. **Export for portable, fast cloud inference:** convert the PyTorch models
   (DETR ResNet-101 for sizing, the moisture model) to **ONNX**. ONNX runs on any
   cloud GPU (no Jetson/TensorRT lock-in) and can be accelerated with TensorRT or
   ONNX Runtime *on the cloud GPU* if needed.
2. **Serve via a managed/containerized model server.** Recommended options on GCP
   (pick one):
   - **NVIDIA Triton Inference Server** on a **Cloud Run (GPU)** or GKE container —
     supports ONNX/PyTorch/TensorRT backends, built-in **model versioning** (clean
     model updates), dynamic batching for throughput.
   - **Vertex AI Endpoints** — fully managed serving + versioning + autoscaling; least
     ops overhead, aligns with "simplify updates & maintenance."
   - **Custom FastAPI + PyTorch** service on a Compute Engine GPU VM — simplest to
     start, most manual to scale.
3. **Post-processing co-located with the model:** wrap the model server with a thin
   service that runs the migrated `geometry`/`reference`/`moisture`/alarm/histogram
   logic on the raw detections, producing the same JSON shape the frontend already
   consumes (`/api/stats`, `/api/moisture`, `/api/hist`).
4. **Model updates:** push a new model version to the server/endpoint — **no device
   changes, no reflashing**. This is the key maintainability win of this architecture.

---

## Implementation Roadmap

### Phase 1 — Extract platform-independent post-processing
- Move the numpy post-processing out of the Jetson/TensorRT path into a standalone,
  CUDA-free module the cloud can run: sizing/calibration (`geometry.py`,
  `reference.py`), alarm + histogram logic, moisture aggregation (`moisture.py`).
- Keep the exact output schema the frontend expects (so the React app barely changes).

### Phase 2 — Cloud inference service
- Stand up the model server (ONNX export + Triton/Vertex/custom).
- Wrap it with the post-processing from Phase 1; expose an internal
  `infer(frame) → {boxes, diameters, stats, moisture, histogram, alarm}` API.
- Validate accuracy parity against the Jetson output on sample frames.

### Phase 3 — Backend: frame ingest + live routing (cloud)
- In `app/main.py`, gate out the in-process inference thread (it no longer runs in the
  cloud); add a `WOODCHIP_ROLE`/feature flag so the legacy all-in-one Jetson build
  still works during transition (non-destructive migration).
- Add a **frame-ingest** path: a WebSocket (recommended) or `POST /api/ingest/frame`
  that receives a browser's frame, calls the inference service, and updates a
  **per-device latest-data store** (in-memory for a single instance; **Redis** if
  scaled out).
- Point `app/routers/live.py` (`/api/frame`, `/api/stats`, `/api/hist`,
  `/api/moisture`) and `devices.py` at that per-device store, enforcing that a user
  only sees **their own** device.

### Phase 4 — Frontend: capture, stream, overlay
- Add browser camera capture (`getUserMedia`) and frame streaming (WebSocket of
  downsized JPEGs, or WebRTC) in the live page (`frontend/src/pages/LivePage.tsx`,
  `frontend/src/lib/api.ts`).
- Render the AI overlay (boxes + diameters + alarm) **in the browser** on the local
  video, using the returned JSON — so annotated frames are never sent back down.
- Keep the existing stats/moisture/histogram panels; they consume the same JSON shape.

### Phase 5 — Data & multi-tenancy
- Migrate the DB from **SQLite → Postgres** (`app/db.py` is the central layer) for
  concurrent multi-user / multi-device use in the cloud. Port `users`, `sessions`,
  `events`, `quality_rules_versions`, `audit_log`.
- Add a `devices` registry (device ↔ owner) if multiple cameras/customers are in scope.

### Phase 6 — Cloud deploy & model lifecycle
- Containerize backend + built frontend; deploy on GCP (Cloud Run / GKE / Compute
  Engine). Managed Postgres (Cloud SQL). Optional Redis (Memorystore) if multi-instance.
- Wire model versioning + a rollout process for model updates.

---

## Google Cloud deployment steps

1. **Containerize the web app:** a `Dockerfile` (Python **3.11+** — no Jetson 3.6
   limit in the cloud) installing deps, copying `app/` + built `web/`, running
   `uvicorn backend_app:app --host 0.0.0.0 --port 8000`. Guard `import live_cam_trt`
   so the cloud build never loads CUDA.
2. **Deploy the inference service** (Triton/Vertex/custom) on a **GPU** instance;
   expose it privately to the backend.
3. **Provision infra:** Cloud Run/GKE for backend+frontend, **Cloud SQL (Postgres)**,
   optional **Memorystore (Redis)**, a GPU instance/endpoint for the model.
4. **Env vars** (extends `DEPLOYMENT.md`): `ADMIN_EMAIL`, `ADMIN_PASSWORD`,
   `DATABASE_URL`, `WOODCHIP_COOKIE_SECURE=1`, `INFERENCE_URL`, plus the model
   server's config.
5. **Domain + HTTPS:** map a domain, terminate TLS at the load balancer (managed
   certs) → permanent customer URL.
6. **Deploy, run DB migrations, verify admin login**, then connect the first camera.

---

## Platform independence (Linux / Windows / macOS / Jetson)

- **Local requirement is just: a modern browser + a USB camera.** No CUDA, no
  TensorRT, no JetPack, no GPU. Same experience on every OS.
- **Python 3.6 constraint disappears from the critical path:** since the device no
  longer runs the model, the Jetson's old Python/JetPack stack is irrelevant to
  processing. The cloud uses modern Python (3.11+).
- If an **optional native capture client** is built later, keep it cross-platform
  (Python 3.8+ with OpenCV `cv2.VideoCapture`, which works on Win/Mac/Linux, or an
  Electron app). Not needed for v1.

---

## Operational considerations (eyes-open) & cost levers

These are inherent to cloud inference; listing them with mitigations, not as
objections:

- **Bandwidth:** each active camera uploads live frames continuously. *Mitigate:*
  downsize/compress frames, cap to ~5–10 fps, send only-on-change, regional endpoints.
- **Latency:** a cloud round-trip adds delay vs on-device. *Mitigate:* nearest GCP
  region, WebRTC, modest resolution; acceptable for monitoring-rate analysis.
- **GPU cost scales with concurrent streams.** *Mitigate:* dynamic **batching** across
  cameras on one GPU, **scale-to-zero** (Cloud Run GPU / Vertex autoscaling), right-size
  the GPU, lower FPS.
- **Internet dependency:** no connectivity = no processing (unlike on-device). *Mitigate:*
  graceful "offline" UI state; optionally buffer/queue frames; document the uptime
  requirement for customer sites.

---

## Security

- **HTTPS everywhere;** `WOODCHIP_COOKIE_SECURE=1`. Reject frame ingest over plain HTTP.
- **Tenant isolation:** every live/events query filters by the authenticated user's
  owned devices (reuse `app/rbac.py`). #1 multi-tenant risk.
- **Inference service is private** (not internet-exposed); only the backend calls it.
- **Secrets** (admin creds, DB URL, model endpoint) via env/secret manager — never
  committed (`.env` is git-ignored today).

---

## Verification (end-to-end)

1. **Accuracy parity:** run sample frames through the cloud inference service and
   confirm sizing/moisture/alarm outputs match the Jetson's results within tolerance.
2. **Cloud app boots without CUDA:** the backend container starts, serves `web/`, admin
   login works, and never imports TensorRT.
3. **Browser → cloud → browser loop:** open the site on a **Mac** with a USB webcam;
   confirm the live video shows AI overlays (boxes/diameters/alarm), stats, moisture,
   and histogram, all driven by the cloud model.
4. **Cross-platform check:** repeat on **Windows** and on a **Jetson/Linux** browser —
   identical behaviour, no per-OS install.
5. **Model update:** push a new model version in the cloud; confirm clients pick it up
   with **no device changes**.
6. **Multi-tenant + resilience:** two accounts see only their own camera; dropping the
   network shows an offline state and recovers cleanly.

---

## Open decisions (to confirm while building)

- **Model server:** Triton on Cloud Run/GKE vs **Vertex AI Endpoints** (most managed)
  vs custom FastAPI+PyTorch on a GPU VM.
- **Transport:** WebSocket JPEG (simplest) vs **WebRTC** (lowest latency).
- **Overlay rendering:** browser-side (recommended, low bandwidth) vs cloud-rendered
  annotated frames (simpler reuse of existing overlay, more bandwidth).
- **DB:** Cloud SQL Postgres (recommended) vs SQLite-on-volume (MVP only).
- **Scope of v1:** single camera per account vs full multi-tenant fleet from day one.

> Note: This is a product-engineering effort (browser capture, a cloud GPU inference
> service, backend ingest, DB migration, GCP infra), not a one-shot deploy. Phases are
> sequenced so the cloud inference service and accuracy can be validated before the
> browser-capture frontend is wired in.
