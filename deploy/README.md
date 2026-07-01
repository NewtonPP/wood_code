# Deploying Woodchip Monitor to Google Cloud

Ordered, copy-paste runbook. Backend + frontend and the private inference
service run on **Cloud Run**; data on **Cloud SQL (Postgres)**; secrets in
**Secret Manager**. Inference stays on the **mock** backend until model files
exist (Part 9). You run these; the app code + config are already in the repo.

> The build artifacts are two images (`backend`, `inference`) — see
> `deploy/cloudbuild.yaml`. Cloud Run specs: `deploy/{backend,inference}.service.yaml`.

## 0. Prerequisites
```bash
gcloud auth login
export PROJECT_ID=your-project-id
export REGION=us-central1
export REPO=woodchip
gcloud config set project "$PROJECT_ID"
```

## 1. Enable APIs
```bash
gcloud services enable \
  run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  sqladmin.googleapis.com secretmanager.googleapis.com vpcaccess.googleapis.com
```

## 2. Artifact Registry
```bash
gcloud artifacts repositories create "$REPO" \
  --repository-format=docker --location="$REGION"
```

## 3. Build + push both images
```bash
gcloud builds submit --config deploy/cloudbuild.yaml \
  --substitutions=_REGION="$REGION",_REPO="$REPO"
```

## 4. Cloud SQL (Postgres)
```bash
gcloud sql instances create woodchip-pg \
  --database-version=POSTGRES_16 --tier=db-f1-micro --region="$REGION"
gcloud sql databases create woodchip --instance=woodchip-pg
gcloud sql users create wc_app --instance=woodchip-pg --password='CHANGE_ME_STRONG'

export SQL_CONN=$(gcloud sql instances describe woodchip-pg \
  --format='value(connectionName)')
# Unix-socket DSN used by Cloud Run (psycopg understands ?host=/cloudsql/...):
export DATABASE_URL="postgresql://wc_app:CHANGE_ME_STRONG@/woodchip?host=/cloudsql/${SQL_CONN}"
```

## 5. Secrets
```bash
printf '%s' "admin@yourco.com"      | gcloud secrets create ADMIN_EMAIL    --data-file=-
printf '%s' "a-strong-admin-pass"   | gcloud secrets create ADMIN_PASSWORD --data-file=-
printf '%s' "$DATABASE_URL"         | gcloud secrets create DATABASE_URL   --data-file=-

# Let the Cloud Run runtime service account read them:
export PROJ_NUM=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
export RUN_SA="${PROJ_NUM}-compute@developer.gserviceaccount.com"
for S in ADMIN_EMAIL ADMIN_PASSWORD DATABASE_URL; do
  gcloud secrets add-iam-policy-binding "$S" \
    --member="serviceAccount:${RUN_SA}" --role=roles/secretmanager.secretAccessor
done
```

## 6. Deploy the private inference service
```bash
gcloud run deploy woodchip-inference \
  --image="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/inference:latest" \
  --region="$REGION" --port=9000 \
  --ingress=internal --no-allow-unauthenticated \
  --set-env-vars=INFER_BACKEND=mock \
  --min-instances=0 --max-instances=4 --cpu=2 --memory=2Gi

export INFERENCE_URL=$(gcloud run services describe woodchip-inference \
  --region="$REGION" --format='value(status.url)')
```

## 7. Private connectivity (backend → inference)
Internal ingress means only in-VPC callers reach the inference service. Give the
backend a Serverless VPC connector so its egress is routed through the VPC:
```bash
gcloud compute networks vpc-access connectors create woodchip-conn \
  --region="$REGION" --range=10.8.0.0/28
```

## 8. Deploy the public backend
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
First boot runs `db_init` (creates the Postgres schema) and seeds the admin from
the secrets. Allow the inference SA invoker if you later enforce token auth.

## 9. Verify
```bash
export BACKEND_URL=$(gcloud run services describe woodchip-backend \
  --region="$REGION" --format='value(status.url)')
curl -s "$BACKEND_URL/ping"          # {"status":"ok"}
curl -s "$BACKEND_URL/api/info"      # {"ok":true,"device_id":...,"role":"cloud"}
```
Open `$BACKEND_URL` in a browser, sign in with the admin secret, grant camera
access on the Live page, and confirm overlays + stats/histogram/moisture (mock
model). Confirm the inference URL is **not** publicly reachable.

## 10. Domain + TLS (permanent customer URL)
Map a custom domain to the backend service (managed certificate):
```bash
gcloud beta run domain-mappings create --service=woodchip-backend \
  --domain=app.yourco.com --region="$REGION"
```
Then add the DNS records gcloud prints. (Or front it with an external HTTPS load
balancer + serverless NEG for a static IP.)

## 11. Real models (later — no app changes blocked today)
Export DETR + moisture to ONNX, push them with the inference image (or mount via
a volume), then redeploy **only** the inference service:
```bash
gcloud run services update woodchip-inference --region="$REGION" \
  --update-env-vars=INFER_BACKEND=onnx,DETR_ONNX_PATH=/models/detr.onnx,MOISTURE_ONNX_PATH=/models/moist.onnx,MOISTURE_CLASSES_PATH=/models/classes.txt
```
For GPU inference, rebuild the inference image on an `onnxruntime-gpu` base and
add a Cloud Run GPU accelerator. No backend/frontend changes needed.
```
