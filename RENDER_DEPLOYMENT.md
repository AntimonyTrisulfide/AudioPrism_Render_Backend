# AudioPrism Backend Render Deployment

This backend is configured for a strict 512 MB Render instance:

- one Uvicorn worker
- one Torch CPU thread
- lazy model loading after startup
- MongoDB-backed auth/history
- required Supabase Storage for persistent stems
- upload cap, with optional audio duration cap

## Required Model File

Render should receive a lightweight inference checkpoint at:

```text
models/model_render.pth
```

If your HPC checkpoint includes optimizer/scaler state, strip it before deployment:

```bash
python scripts/export_render_checkpoint.py path/to/epoch_XXXX.pth models/model_render.pth
```

The exported checkpoint keeps only:

- `model_state`
- `train_metadata`
- `config`

The API reads `train_metadata` and `config` from the checkpoint, so source names,
sample rate, FFT size, hop length, chunk size, base channels, and bilinear mode
stay aligned with the HPC training run.

You have two deployment options:

1. Put `model_render.pth` in `models/` before building the Docker image.
2. Host `model_render.pth` somewhere reachable and set `MODEL_URL`; the container
   downloads it to `/models/model_render.pth` during startup.

## Render Settings

Use Docker deployment.

Health check path:

```text
/healthz
```

Important environment variables:

```text
MODEL_FILENAME=model_render.pth
MODEL_URL=https://optional-url-to/model_render.pth
MAX_UPLOAD_MB=25
MAX_AUDIO_SECONDS=60
MAX_AUDIO_SECONDS_AUTOTUNE_FOR_RAM=1
INFERENCE_CHUNK_SECONDS=0.5
INFERENCE_FREQ_TILE_BINS=256
INFERENCE_FREQ_OVERLAP_BINS=64
INFERENCE_AUTOTUNE_FOR_RAM=1
OUTPUT_TTL_MINUTES=120
INFERENCE_JOB_TTL_SECONDS=3600
ALLOWED_ORIGINS=https://your-frontend.vercel.app
ALLOWED_ORIGIN_REGEX=^https://.*\.vercel\.app$
PUBLIC_BASE_URL=https://your-backend.onrender.com
AUTH_REQUIRED=1
MONGODB_URI=mongodb+srv://...
MONGODB_DB=audioprism
MONGO_REQUIRED=1
JWT_SECRET=replace-with-a-long-random-secret
STORAGE_BACKEND=supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
SUPABASE_BUCKET=audioprism-stems
SUPABASE_PUBLIC_BUCKET=0
```

`ALLOWED_ORIGINS` must include the deployed frontend origin, for example the
Vercel URL shown in the browser address bar. The app normalizes trailing slashes,
but do not include paths. Use `ALLOWED_ORIGIN_REGEX` only if you also need branch
or preview deployment URLs.

`PUBLIC_BASE_URL` is optional on Render because `RENDER_EXTERNAL_URL` is also
used when available. Set it explicitly if returned stem links need a fixed host.

`MAX_AUDIO_SECONDS=60` keeps 512 MB deployments away from song-length jobs. Set
it higher only after increasing the Render instance size. The backend still keeps
`MAX_UPLOAD_MB` because compressed uploads can otherwise blow through Render
request and memory limits. With `MAX_AUDIO_SECONDS_AUTOTUNE_FOR_RAM=1`, stale
unlimited values are clamped to 60 seconds whenever `RAM_LIMIT_MB <= 512`.

Keep the three `INFERENCE_*` values above on a 512 MB service. They bound the
time axis and process the original-resolution spectrogram in overlapping
frequency tiles. Increasing either chunk seconds or tile bins raises peak RAM.
The frontend uses async inference jobs, so smaller chunks are safer even though
they can make large files take longer. With `INFERENCE_AUTOTUNE_FOR_RAM=1`, the
app clamps stale/heavier env values back to the 512 MB-safe settings whenever
`RAM_LIMIT_MB <= 512`.

If the browser still reports a CORS-looking error with a `502` during processing,
check the Render logs. That usually means the worker restarted or was killed
while the background job was running, so FastAPI never got a chance to attach
CORS headers. On a 512 MB service, the practical fixes are shorter audio,
a positive `MAX_AUDIO_SECONDS`, smaller `INFERENCE_*` settings, or a larger
Render instance.

Use the backend-only service-role or `sb_secret_...` key, never the browser
publishable key. Generated stems are uploaded to Supabase Storage and local WAVs
are deleted only after a successful upload. Private-bucket object keys are stored
in MongoDB and fresh signed URLs are generated whenever history is loaded.

For production, keep `MONGO_REQUIRED=1`. That makes startup fail clearly if
`MONGODB_URI` is missing or invalid, instead of silently using local JSON files.
For throwaway local-only development, set `MONGO_REQUIRED=0`.

## Endpoints

```text
GET  /healthz
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
GET  /api/stems
POST /infer
POST /api/infer
POST /api/infer/segment
GET  /api/infer/jobs/{job_id}
GET  /api/infer/results
GET  /output/{cache_id}/{stem_file}
```

On Render, keep `STORAGE_BACKEND=supabase`; an invalid storage configuration then
fails the inference request clearly instead of silently writing to ephemeral disk.
Use `STORAGE_BACKEND=local` only for local development.
