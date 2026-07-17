# AudioPrism Backend

Render-ready FastAPI backend for AudioPrism stem separation.

See [RENDER_DEPLOYMENT.md](RENDER_DEPLOYMENT.md) for the exact deployment flow.

## Quick Start

For local Windows development, start the backend with:

```bat
copy .env.example .env
start_backend.bat
```

That launches the API on `http://127.0.0.1:8001`, which is what the Vite
frontend proxy expects by default.

Put your MongoDB URI in `.env` before starting:

```text
MONGO_REQUIRED=1
MONGODB_URI=mongodb+srv://...
MONGODB_DB=audioprism
MONGODB_TIMEOUT_MS=15000
JWT_SECRET=replace-with-a-long-random-secret
```

With `MONGO_REQUIRED=1`, auth and history must use MongoDB. If the URI is
missing or invalid, startup fails instead of falling back to local JSON.

To test Mongo before starting the API:

```bash
..\..\.venv\Scripts\python.exe scripts/check_mongo.py
```

If it says port `8001` is already in use, the backend is probably already
running. To intentionally replace the existing local server:

```bat
start_backend.bat -Restart
```

For a temporary local test without Mongo/Supabase:

```bat
start_backend.bat -Restart -LocalJsonAuth
```

Then run a tiny generated-audio inference smoke test:

```bash
..\..\.venv\Scripts\python.exe scripts/smoke_local_infer.py --stems Vocal --duration 1
```

Do not use `python -m uvicorn app:app --port 8000` on this machine. Your global
`python` currently resolves to Python 3.13, and that crashes the installed
FastAPI/Pydantic stack before the API starts.

Manual setup:

```bash
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe scripts/export_render_checkpoint.py path/to/hpc_checkpoint.pth models/model_render.pth
.venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8001
```

Or use the guarded local launcher:

```powershell
.\scripts\start_local.ps1
```

Use Python 3.11 locally. Running global `uvicorn` from Python 3.13 can crash
inside FastAPI/Pydantic before the API starts, which makes the frontend proxy
show `ECONNREFUSED 127.0.0.1:8001`.

## API

```text
GET  /healthz
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
GET  /api/stems
POST /api/infer/segment
GET  /api/infer/results
```

## Model

AudioPrism uses one final UNet checkpoint. It is loaded lazily on the first
inference request and reused by the single Uvicorn worker. The inference lock
prevents concurrent requests from duplicating peak tensor memory on a 512 MB
Render instance.

`INFERENCE_CHUNK_SECONDS=1.0` bounds UNet activation memory independently of
uploaded audio duration. Checkpoints are mmap-loaded directly into model
parameters, and ffmpeg is used as the constant-memory decoder fallback. The
full-resolution spectrogram is processed in overlapping 512-bin frequency
tiles and blended back together to keep the trained FFT scale without the
full-height activation peak.

For deployed frontends, set `ALLOWED_ORIGINS` to the exact browser origin of the
frontend, such as `https://your-frontend.vercel.app`. If the frontend uses Vercel
preview URLs, add `ALLOWED_ORIGIN_REGEX` for those hosts.

Set `MONGODB_URI` to keep auth/history durable on Render. Set `SUPABASE_URL`,
`SUPABASE_SERVICE_ROLE_KEY`, and `SUPABASE_BUCKET` to move generated WAV stems
out of Render's ephemeral filesystem after each run. Set
`STORAGE_BACKEND=supabase` on Render so invalid credentials fail clearly. Local
`/output` serving is development-only.
