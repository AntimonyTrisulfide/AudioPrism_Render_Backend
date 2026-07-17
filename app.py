import asyncio
import base64
import ctypes
import gc
import hashlib
import hmac
import http.client
import json
import os
import pathlib
import re
import shutil
import secrets
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import quote, urlparse
from urllib.error import HTTPError
from urllib.request import Request as UrlRequest, urlopen

import torch
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audiomask.dataset import ExternalPreprocessedDataset
from audiomask.model import UNet
from audiomask.preprocessing import ExternalPreprocessor

try:
    from pymongo import MongoClient
    from pymongo.errors import DuplicateKeyError
    from bson import ObjectId
except ImportError:
    MongoClient = None
    DuplicateKeyError = Exception
    ObjectId = None


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


def _env_float(name: str, default: float, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(minimum, value)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file(path: pathlib.Path = pathlib.Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file()


torch.set_num_threads(_env_int("TORCH_NUM_THREADS", 1))
try:
    torch.set_num_interop_threads(_env_int("TORCH_NUM_INTEROP_THREADS", 1))
except RuntimeError:
    pass


def normalize_cors_origin(origin: str) -> str:
    cleaned = origin.strip()
    if not cleaned or cleaned == "*":
        return cleaned
    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return cleaned.rstrip("/")


def parse_allowed_origins(raw_origins: str | None) -> list[str]:
    origins: list[str] = []
    for raw_origin in (raw_origins or "").split(","):
        origin = normalize_cors_origin(raw_origin)
        if not origin:
            continue
        if origin == "*":
            return ["*"]
        if origin not in origins:
            origins.append(origin)
    return origins


app = FastAPI()

default_allowed_origins = [
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]

configured_origins = parse_allowed_origins(os.getenv("ALLOWED_ORIGINS"))
frontend_origins = parse_allowed_origins(
    ",".join(
        origin
        for origin in [
            os.getenv("FRONTEND_URL"),
            os.getenv("FRONTEND_ORIGIN"),
            os.getenv("FRONTEND_HOST"),
        ]
        if origin
    )
)
allowed_origins = configured_origins or list(default_allowed_origins)
if allowed_origins != ["*"]:
    for origin in frontend_origins:
        if origin not in allowed_origins:
            allowed_origins.append(origin)
allowed_origin_regex = os.getenv("ALLOWED_ORIGIN_REGEX") or os.getenv("CORS_ALLOW_ORIGIN_REGEX") or None

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=allowed_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Accept-Ranges", "Content-Length", "Content-Range"],
)

OUTPUT_DIR = pathlib.Path(os.getenv("OUTPUT_DIR", "reconstructed_audio"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
TEMP_DIR = pathlib.Path(os.getenv("TEMP_DIR", "tmp_uploads"))
TEMP_DIR.mkdir(exist_ok=True, parents=True)
MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 50, minimum=0)
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
UPLOAD_CHUNK_SIZE = _env_int("UPLOAD_CHUNK_SIZE", 1024 * 1024)
MAX_AUDIO_SECONDS = _env_float("MAX_AUDIO_SECONDS", 0.0, minimum=0.0)
OUTPUT_TTL_MINUTES = _env_int("OUTPUT_TTL_MINUTES", 120, minimum=0)
OUTPUT_TTL_SECONDS = OUTPUT_TTL_MINUTES * 60
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or os.getenv("RENDER_EXTERNAL_URL") or "").rstrip("/")
DATA_DIR = pathlib.Path(os.getenv("DATA_DIR", "runtime_data"))
DATA_DIR.mkdir(exist_ok=True, parents=True)
MONGODB_URI = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
MONGODB_DB = os.getenv("MONGODB_DB", "audioprism")
MONGO_REQUIRED = _env_bool("MONGO_REQUIRED", False)
MONGODB_TIMEOUT_MS = _env_int("MONGODB_TIMEOUT_MS", 15000)
RESULT_PERSIST_REQUIRED = _env_bool("RESULT_PERSIST_REQUIRED", False)
RAM_LIMIT_MB = _env_int("RAM_LIMIT_MB", 512)
INFERENCE_CHUNK_SECONDS = _env_float("INFERENCE_CHUNK_SECONDS", 1.0, minimum=0.25)
INFERENCE_FREQ_TILE_BINS = _env_int("INFERENCE_FREQ_TILE_BINS", 512)
INFERENCE_FREQ_OVERLAP_BINS = _env_int("INFERENCE_FREQ_OVERLAP_BINS", 128)
JWT_SECRET = os.getenv("JWT_SECRET", "dev-only-change-me")
TOKEN_TTL_SECONDS = _env_int("TOKEN_TTL_SECONDS", 60 * 60 * 24 * 7)
AUTH_REQUIRED = _env_bool("AUTH_REQUIRED", True)
PASSWORD_ITERATIONS = _env_int("PASSWORD_ITERATIONS", 180000)
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "auto").lower()
SUPABASE_URL = (os.getenv("SUPABASE_URL") or "").rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or ""
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "audioprism-stems")
SUPABASE_PUBLIC_BUCKET = os.getenv("SUPABASE_PUBLIC_BUCKET", "0").lower() in {"1", "true", "yes"}
SUPABASE_SIGNED_URL_SECONDS = _env_int("SUPABASE_SIGNED_URL_SECONDS", 60 * 60 * 2)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
_INFERENCE_LOCK = asyncio.Lock()
INFERENCE_JOB_TTL_SECONDS = _env_int("INFERENCE_JOB_TTL_SECONDS", 60 * 60, minimum=60)
_INFERENCE_JOBS: dict[str, dict] = {}
_INFERENCE_JOBS_LOCK = threading.Lock()
_MODEL_LOCK = threading.Lock()
_MODEL_RUNTIME = None

DEFAULT_SOURCE_NAMES = [
    "Vocal",
    "Guitar",
    "Bass",
    "Drums",
    "Percussion",
    "Piano_Keyboard",
    "Woodwinds",
    "Brass",
    "Strings",
    "Effects_Other",
]

STEM_LABELS = {
    "Vocal": "Vocals",
    "Guitar": "Guitar",
    "Bass": "Bass",
    "Drums": "Drums",
    "Percussion": "Percussion",
    "Piano_Keyboard": "Keys",
    "Woodwinds": "Woodwinds",
    "Brass": "Brass",
    "Strings": "Strings",
    "Effects_Other": "Other",
}


class RegisterPayload(BaseModel):
    username: str
    email: str
    password: str


class LoginPayload(BaseModel):
    email: str
    password: str


def normalize_source_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def stem_display_name(name: str) -> str:
    return STEM_LABELS.get(name, name.replace("_", " "))


def resolve_requested_sources(source_names: list[str], requested_stems: str | None) -> list[str]:
    if not requested_stems:
        return list(source_names)
    requested_keys = {
        normalize_source_key(item)
        for item in re.split(r"[,|]", requested_stems)
        if item.strip()
    }
    if not requested_keys:
        return list(source_names)
    lookup = {normalize_source_key(name): name for name in source_names}
    resolved = [lookup[key] for key in requested_keys if key in lookup]
    if not resolved:
        raise HTTPException(status_code=400, detail="No requested stems match this model.")
    return sorted(resolved, key=source_names.index)


def _mongo_database():
    if not MONGODB_URI or MongoClient is None:
        return None
    client = MongoClient(
        MONGODB_URI,
        serverSelectionTimeoutMS=MONGODB_TIMEOUT_MS,
        connectTimeoutMS=MONGODB_TIMEOUT_MS,
    )
    return client[MONGODB_DB]


_MONGO_CLIENT = None
_MONGO_DB = None
try:
    _MONGO_DB = _mongo_database()
    if _MONGO_DB is not None:
        _MONGO_CLIENT = _MONGO_DB.client
        _MONGO_CLIENT.admin.command("ping")
        _MONGO_DB.users.create_index("email", unique=True)
        _MONGO_DB.results.create_index([("user_id", 1), ("created_at", -1)])
except Exception as mongo_error:
    print(f"[Mongo] Disabled: {mongo_error}")
    _MONGO_DB = None

if MONGO_REQUIRED and _MONGO_DB is None:
    raise RuntimeError("MongoDB is required. Set a valid MONGODB_URI or set MONGO_REQUIRED=0 for local-only dev.")


def _load_local_json(path: pathlib.Path, default):
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def _save_local_json(path: pathlib.Path, payload) -> None:
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    tmp_path.replace(path)


LOCAL_USERS_PATH = DATA_DIR / "users.json"
LOCAL_RESULTS_PATH = DATA_DIR / "results.json"


def normalize_email(email: str) -> str:
    return email.strip().lower()


def public_user(user: dict) -> dict:
    return {
        "id": str(user.get("_id") or user.get("id")),
        "username": user.get("username"),
        "email": user.get("email"),
    }


def hash_password(password: str, salt: str | None = None) -> str:
    if salt is None:
        salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt),
        PASSWORD_ITERATIONS,
    ).hex()
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        scheme, iterations_text, salt, expected = stored_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt),
            int(iterations_text),
        ).hex()
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def _b64url_encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _b64url_decode(payload: str) -> bytes:
    padding = "=" * (-len(payload) % 4)
    return base64.urlsafe_b64decode(payload + padding)


def create_token(user_id: str) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    body = {"sub": user_id, "iat": now, "exp": now + TOKEN_TTL_SECONDS}
    signing_input = ".".join(
        [
            _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64url_encode(json.dumps(body, separators=(",", ":")).encode("utf-8")),
        ]
    )
    signature = hmac.new(JWT_SECRET.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode_token(token: str) -> dict | None:
    try:
        header_b64, body_b64, signature_b64 = token.split(".", 2)
        signing_input = f"{header_b64}.{body_b64}"
        expected_signature = hmac.new(
            JWT_SECRET.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(_b64url_encode(expected_signature), signature_b64):
            return None
        payload = json.loads(_b64url_decode(body_b64))
        if int(payload.get("exp", 0)) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def get_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    return authorization[len(prefix):].strip()


def create_user(username: str, email: str, password: str) -> dict:
    email = normalize_email(email)
    user = {
        "id": uuid.uuid4().hex,
        "username": username.strip(),
        "email": email,
        "password_hash": hash_password(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if _MONGO_DB is not None:
        mongo_user = dict(user)
        mongo_user["_id"] = user["id"]
        try:
            _MONGO_DB.users.insert_one(mongo_user)
        except DuplicateKeyError:
            raise HTTPException(status_code=409, detail="Email is already registered.")
        return mongo_user

    payload = _load_local_json(LOCAL_USERS_PATH, {"users": []})
    if any(existing.get("email") == email for existing in payload["users"]):
        raise HTTPException(status_code=409, detail="Email is already registered.")
    payload["users"].append(user)
    _save_local_json(LOCAL_USERS_PATH, payload)
    return user


def find_user_by_email(email: str) -> dict | None:
    email = normalize_email(email)
    if _MONGO_DB is not None:
        return _MONGO_DB.users.find_one({"email": email})
    payload = _load_local_json(LOCAL_USERS_PATH, {"users": []})
    return next((user for user in payload["users"] if user.get("email") == email), None)


def find_user_by_id(user_id: str) -> dict | None:
    if _MONGO_DB is not None:
        candidates: list[object] = [user_id]
        if ObjectId is not None:
            try:
                candidates.append(ObjectId(user_id))
            except Exception:
                pass
        return _MONGO_DB.users.find_one({"$or": [{"_id": {"$in": candidates}}, {"id": user_id}]})
    payload = _load_local_json(LOCAL_USERS_PATH, {"users": []})
    return next((user for user in payload["users"] if user.get("id") == user_id), None)


def current_user_from_authorization(authorization: str | None, required: bool = False) -> dict | None:
    token = get_bearer_token(authorization)
    payload = decode_token(token) if token else None
    user = find_user_by_id(str(payload["sub"])) if payload and payload.get("sub") else None
    if required and user is None:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return user


@dataclass
class ModelRuntime:
    model: torch.nn.Module
    device: torch.device
    model_path: pathlib.Path
    source_names: list[str]
    sample_rate: int
    n_fft: int
    hop_length: int
    chunk_samples: int
    chunk_duration: float
    base_channels: int
    bilinear: bool
    in_channels: int


def resolve_device() -> torch.device:
    requested = os.getenv("DEVICE", "cpu").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested or "cpu")


def candidate_model_paths() -> list[pathlib.Path]:
    model_path_env = os.getenv("MODEL_PATH")
    model_dir = pathlib.Path(os.getenv("MODEL_DIR", "/models"))
    model_filename = os.getenv("MODEL_FILENAME", "model_render.pth")

    paths: list[pathlib.Path] = []
    if model_path_env:
        paths.append(pathlib.Path(model_path_env))
    paths.extend([model_dir / model_filename, pathlib.Path("models") / model_filename])
    paths.extend(
        [
            model_dir / "model_render.pth",
            model_dir / "epoch_0025.pth",
            pathlib.Path("models/model_render.pth"),
            pathlib.Path("models/epoch_0025.pth"),
            pathlib.Path("app/model_weights.pth"),
            pathlib.Path("model_weights.pth"),
        ]
    )
    return list(dict.fromkeys(paths))


def find_model_path() -> pathlib.Path | None:
    return next((path for path in candidate_model_paths() if path.exists()), None)


def configured_source_names() -> list[str]:
    if _MODEL_RUNTIME is not None:
        return list(_MODEL_RUNTIME.source_names)
    configured = [name.strip() for name in os.getenv("MODEL_SOURCE_NAMES", "").split(",") if name.strip()]
    return configured or list(DEFAULT_SOURCE_NAMES)


def load_checkpoint(path: pathlib.Path):
    kwargs = {"map_location": "cpu"}
    if os.getenv("TORCH_LOAD_MMAP", "1") != "0":
        kwargs["mmap"] = True
    try:
        return torch.load(path, **kwargs)
    except TypeError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)
    except ValueError:
        kwargs.pop("mmap", None)
        return torch.load(path, **kwargs)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    return checkpoint


def inference_chunk_samples(trained_chunk_samples: int, sample_rate: int) -> int:
    requested = int(INFERENCE_CHUNK_SECONDS * sample_rate)
    return min(trained_chunk_samples, requested) if trained_chunk_samples > 0 else requested


def build_unet_from_state(
    state_dict: dict,
    in_channels: int,
    out_channels: int,
    base_channels: int,
    bilinear: bool,
) -> torch.nn.Module:
    try:
        with torch.device("meta"):
            model = UNet(
                in_channels=in_channels,
                out_channels=out_channels,
                base_c=base_channels,
                bilinear=bilinear,
            )
        model.load_state_dict(state_dict, assign=True)
        return model
    except TypeError:
        model = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_c=base_channels,
            bilinear=bilinear,
        )
        model.load_state_dict(state_dict)
        return model


def build_model_runtime() -> ModelRuntime:
    model_path = find_model_path()
    if not model_path:
        searched = ", ".join(str(path) for path in candidate_model_paths())
        raise HTTPException(
            status_code=503,
            detail=f"UNet weights are not installed. Searched: {searched}",
        )

    device = resolve_device()
    checkpoint = load_checkpoint(model_path)
    train_metadata = checkpoint.get("train_metadata", {}) if isinstance(checkpoint, dict) else {}
    checkpoint_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    source_names = list(train_metadata.get("all_source_names", DEFAULT_SOURCE_NAMES))
    sample_rate = int(train_metadata.get("sr", checkpoint_config.get("sample_rate", 16000)))
    n_fft = int(train_metadata.get("n_fft", checkpoint_config.get("n_fft", 2048)))
    hop_length = int(train_metadata.get("hop_length", checkpoint_config.get("hop_length", 512)))
    trained_chunk_samples = int(train_metadata.get("chunk_samples", int(sample_rate * checkpoint_config.get("chunk_duration", 4.0))))
    chunk_samples = inference_chunk_samples(trained_chunk_samples, sample_rate)
    chunk_duration = chunk_samples / sample_rate
    base_channels = int(checkpoint_config.get("base_channels", 64))
    bilinear = bool(checkpoint_config.get("bilinear", False))
    in_channels = int(checkpoint_config.get("in_channels", 1))

    state_dict = extract_state_dict(checkpoint)
    model = build_unet_from_state(
        state_dict,
        in_channels=in_channels,
        out_channels=len(source_names),
        base_channels=base_channels,
        bilinear=bilinear,
    )
    model.to(device)
    model.eval()

    runtime = ModelRuntime(
        model=model,
        device=device,
        model_path=model_path,
        source_names=source_names,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        chunk_samples=chunk_samples,
        chunk_duration=chunk_duration,
        base_channels=base_channels,
        bilinear=bilinear,
        in_channels=in_channels,
    )

    del checkpoint
    del state_dict
    gc.collect()
    return runtime


def current_rss_mb() -> float | None:
    statm_path = pathlib.Path("/proc/self/statm")
    if not statm_path.exists():
        return None
    try:
        resident_pages = int(statm_path.read_text(encoding="ascii").split()[1])
        return round(resident_pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024), 1)
    except (OSError, ValueError, IndexError):
        return None


def trim_process_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if os.name == "posix":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (AttributeError, OSError):
            pass


def get_model_runtime() -> ModelRuntime:
    global _MODEL_RUNTIME
    if _MODEL_RUNTIME is not None:
        return _MODEL_RUNTIME
    with _MODEL_LOCK:
        if _MODEL_RUNTIME is None:
            _MODEL_RUNTIME = build_model_runtime()
    return _MODEL_RUNTIME


def frequency_tile_starts(total_bins: int, tile_bins: int, overlap_bins: int) -> list[int]:
    if total_bins <= tile_bins:
        return [0]
    step = max(1, tile_bins - min(overlap_bins, tile_bins - 1))
    starts = list(range(0, total_bins - tile_bins + 1, step))
    final_start = total_bins - tile_bins
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def predict_masks(model: torch.nn.Module, mix_spec_chunk: torch.Tensor) -> torch.Tensor:
    total_bins = int(mix_spec_chunk.shape[-2])
    tile_bins = min(INFERENCE_FREQ_TILE_BINS, total_bins)
    starts = frequency_tile_starts(total_bins, tile_bins, INFERENCE_FREQ_OVERLAP_BINS)
    if len(starts) == 1:
        return torch.sigmoid(model(mix_spec_chunk)).squeeze(0).cpu()

    blended = None
    weight_sum = torch.zeros((total_bins, 1), dtype=torch.float32)
    overlap = min(INFERENCE_FREQ_OVERLAP_BINS, tile_bins // 2)
    for start in starts:
        end = start + tile_bins
        prediction = torch.sigmoid(model(mix_spec_chunk[..., start:end, :])).squeeze(0).cpu()
        if blended is None:
            blended = torch.zeros(
                (prediction.shape[0], total_bins, prediction.shape[-1]),
                dtype=prediction.dtype,
            )
        weight = torch.ones((tile_bins, 1), dtype=prediction.dtype)
        if overlap and start > 0:
            weight[:overlap] = torch.linspace(0.001, 1.0, overlap).unsqueeze(1)
        if overlap and end < total_bins:
            weight[-overlap:] = torch.linspace(1.0, 0.001, overlap).unsqueeze(1)
        blended[:, start:end, :] += prediction * weight
        weight_sum[start:end, :] += weight
        del prediction, weight

    return blended / weight_sum.clamp_min(0.001).unsqueeze(0)


def _load_preprocessed_chunk(
    track_dir: pathlib.Path,
    chunk_idx: int,
    default_chunk_samples: int,
    fallback_mix_data=None,
):
    chunk_path = track_dir / f"chunk_{chunk_idx:05d}.pt"
    if chunk_path.exists():
        chunk_data = torch.load(chunk_path, map_location="cpu")
        return (
            chunk_data["spectrogram"].float(),
            chunk_data["phase"].float(),
            int(chunk_data.get("valid_samples", default_chunk_samples)),
            fallback_mix_data,
        )

    if fallback_mix_data is None:
        fallback_mix_data = torch.load(track_dir / "mix.pt", map_location="cpu")

    return (
        fallback_mix_data["spectrogram"][chunk_idx].float(),
        fallback_mix_data["phases"][chunk_idx].float(),
        int(fallback_mix_data.get("chunk_samples", default_chunk_samples)),
        fallback_mix_data,
    )


def reconstruct_and_save_audio(
    model,
    dataset,
    preprocessor,
    save_dir=OUTPUT_DIR,
    device="cpu",
    public_base_url: str = "",
    requested_source_names: list[str] | None = None,
    owner_id: str | None = None,
):
    import soundfile as sf
    import torchaudio.transforms as T

    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)

    track_urls = []
    track_payloads = []
    istft = T.InverseSpectrogram(n_fft=preprocessor.n_fft, hop_length=preprocessor.hop_length)

    for track_index, track_dir in enumerate(dataset.track_dirs):
        track_name = track_dir.name
        track_output_dir = save_dir / track_name
        track_output_dir.mkdir(exist_ok=True)
        current_track_stems = []
        metadata = dataset.track_metadata[track_index] if hasattr(dataset, "track_metadata") else {}
        n_chunks = int(metadata.get("n_chunks", 0))
        fallback_data = None
        if not n_chunks and (track_dir / "mix.pt").exists():
            fallback_data = torch.load(track_dir / "mix.pt", map_location="cpu")
            n_chunks = int(fallback_data["spectrogram"].shape[0])

        selected_source_names = requested_source_names or list(dataset.all_source_names)
        selected_sources = [
            (dataset.all_source_names.index(source_name), source_name)
            for source_name in selected_source_names
            if source_name in dataset.all_source_names
        ]
        stem_outputs = []
        stem_writers = []
        try:
            for source_name in selected_source_names:
                save_path = track_output_dir / f"{source_name}_reconstructed.wav"
                writer = sf.SoundFile(
                    save_path,
                    mode="w",
                    samplerate=preprocessor.sr,
                    channels=1,
                    format="WAV",
                    subtype=os.getenv("OUTPUT_WAV_SUBTYPE", "PCM_16"),
                )
                stem_outputs.append((source_name, save_path))
                stem_writers.append(writer)

            for chunk_idx in range(n_chunks):
                mix_magnitude, phase, valid_samples, fallback_data = _load_preprocessed_chunk(
                    track_dir,
                    chunk_idx,
                    preprocessor.chunk_samples,
                    fallback_data,
                )
                mix_spec_chunk = mix_magnitude.unsqueeze(0).unsqueeze(0).to(device)

                with torch.inference_mode():
                    masks = predict_masks(model, mix_spec_chunk)

                for writer_index, (src_idx, _source_name) in enumerate(selected_sources):
                    writer = stem_writers[writer_index]
                    masked_spec = masks[src_idx] * mix_magnitude
                    complex_spec = torch.polar(masked_spec, phase)
                    reconstructed_audio = istft(
                        complex_spec.unsqueeze(0),
                        length=preprocessor.chunk_samples,
                    ).squeeze(0)
                    if valid_samples < preprocessor.chunk_samples:
                        reconstructed_audio = reconstructed_audio[:valid_samples]
                    writer.write(reconstructed_audio.detach().cpu().numpy())

                del mix_magnitude, phase, mix_spec_chunk, masks
                if chunk_idx % 4 == 0:
                    gc.collect()
        finally:
            for writer in stem_writers:
                writer.close()

        for source_name, save_path in stem_outputs:
            if storage_is_supabase_enabled():
                owner_path = quote(owner_id or "anonymous", safe="")
                object_path = f"users/{owner_path}/{track_name}/{save_path.name}"
                url = upload_to_supabase(save_path, object_path)
                save_path.unlink(missing_ok=True)
            else:
                object_path = None
                url = build_output_url(public_base_url, track_name, save_path.name)
            track_urls.append(url)
            current_track_stems.append({
                "name": source_name,
                "label": stem_display_name(source_name),
                "url": url,
                "storageKey": object_path,
            })

        track_payloads.append({"cache_id": track_name, "stems": current_track_stems})

    primary_track = track_payloads[0] if track_payloads else {"cache_id": None, "stems": []}
    return {
        "status": "success",
        "cache_id": primary_track["cache_id"],
        "files": track_urls,
        "stems": primary_track["stems"],
    }


def inference_pipeline(
    temp_input_path,
    runtime: ModelRuntime,
    track_id,
    public_base_url: str,
    requested_source_names: list[str] | None = None,
    owner_id: str | None = None,
):
    output_dir_preprocessed = pathlib.Path("preprocessed_output") / track_id
    output_dir_preprocessed.mkdir(parents=True, exist_ok=True)

    try:
        preprocessor = ExternalPreprocessor(
            temp_input_path,
            output_dir_preprocessed,
            chunk_duration=runtime.chunk_duration,
            sr=runtime.sample_rate,
            n_fft=runtime.n_fft,
            hop_length=runtime.hop_length,
            max_duration_seconds=MAX_AUDIO_SECONDS,
        )
        track_output_dir = preprocessor.preprocess()

        if track_output_dir.name != track_id:
            desired_dir = track_output_dir.parent / track_id
            if desired_dir.exists():
                shutil.rmtree(desired_dir, ignore_errors=True)
            track_output_dir.rename(desired_dir)

        gc.collect()
        dataset = ExternalPreprocessedDataset(output_dir_preprocessed, runtime.source_names)
        result = reconstruct_and_save_audio(
            runtime.model,
            dataset,
            preprocessor,
            device=runtime.device,
            public_base_url=public_base_url,
            requested_source_names=requested_source_names,
            owner_id=owner_id,
        )
        result["cache_id"] = track_id
        return result

    except Exception as error:
        traceback.print_exc()
        return {"status": "failed", "message": str(error), "error": str(error)}

    finally:
        try:
            if temp_input_path.exists():
                temp_input_path.unlink()
                print(f"[Cleanup] Deleted temporary input file: {temp_input_path}")

            if output_dir_preprocessed.exists():
                shutil.rmtree(output_dir_preprocessed, ignore_errors=True)
                print(f"[Cleanup] Deleted preprocessed folder: {output_dir_preprocessed}")

            trim_process_memory()

        except Exception as cleanup_error:
            print(f"[Warning] Cleanup failed: {cleanup_error}")


def build_output_url(public_base_url: str, track_id: str, filename: str) -> str:
    return f"{public_base_url}/output/{quote(track_id, safe='')}/{quote(filename, safe='')}"


def get_public_base_url(request: Request) -> str:
    if PUBLIC_BASE_URL:
        return PUBLIC_BASE_URL
    return str(request.base_url).rstrip("/")


def cleanup_old_outputs() -> None:
    if not OUTPUT_TTL_SECONDS:
        return
    cutoff = time.time() - OUTPUT_TTL_SECONDS
    for child in OUTPUT_DIR.iterdir():
        if not child.is_dir():
            continue
        try:
            if child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
        except OSError:
            continue


def storage_is_supabase_enabled() -> bool:
    if STORAGE_BACKEND == "local":
        return False
    config_error = supabase_config_error()
    if config_error:
        if STORAGE_BACKEND == "supabase":
            raise RuntimeError(config_error)
        return False
    return True


def supabase_config_error() -> str | None:
    if not any([SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, SUPABASE_BUCKET]):
        return "Supabase storage is not configured."
    if not SUPABASE_URL:
        return "SUPABASE_URL is required for Supabase storage."
    parsed = urlparse(SUPABASE_URL)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "SUPABASE_URL must be the Supabase Project URL, for example https://project-ref.supabase.co."
    if parsed.scheme != "https":
        return "SUPABASE_URL must use https."
    if "supabase.co" not in parsed.netloc:
        return "SUPABASE_URL should be the Supabase Project URL ending in supabase.co, not the Postgres database URL."
    if not SUPABASE_SERVICE_ROLE_KEY:
        return "SUPABASE_SERVICE_ROLE_KEY is required for Supabase storage."
    if SUPABASE_SERVICE_ROLE_KEY.startswith("sb_publishable_"):
        return "SUPABASE_SERVICE_ROLE_KEY must be a backend secret/service-role key, not a publishable key."
    if not SUPABASE_BUCKET:
        return "SUPABASE_BUCKET is required for Supabase storage."
    return None


def supabase_request(method: str, path: str, body: bytes | None = None, headers: dict[str, str] | None = None):
    if not storage_is_supabase_enabled():
        raise RuntimeError("Supabase storage is not configured.")
    url = f"{SUPABASE_URL}{path}"
    request_headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }
    if headers:
        request_headers.update(headers)
    request = UrlRequest(url, data=body, headers=request_headers, method=method)
    with urlopen(request, timeout=60) as response:
        data = response.read()
    if not data:
        return None
    return json.loads(data.decode("utf-8"))


def ensure_supabase_bucket() -> None:
    payload = {
        "id": SUPABASE_BUCKET,
        "name": SUPABASE_BUCKET,
        "public": SUPABASE_PUBLIC_BUCKET,
    }
    try:
        supabase_request(
            "POST",
            "/storage/v1/bucket",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        print(f"[Supabase] Created storage bucket: {SUPABASE_BUCKET}")
    except HTTPError as error:
        response_text = error.read().decode("utf-8", errors="replace")
        if error.code in {400, 409} and "already exists" in response_text.lower():
            return
        raise RuntimeError(
            f"Could not create Supabase bucket '{SUPABASE_BUCKET}' ({error.code}): {response_text}"
        ) from error


def supabase_upload_file(local_path: pathlib.Path, object_path: str, allow_bucket_create: bool = True) -> None:
    parsed = urlparse(SUPABASE_URL)
    connection_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    quoted_path = quote(object_path, safe="/")
    request_path = f"/storage/v1/object/{SUPABASE_BUCKET}/{quoted_path}"
    headers = {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
        "Content-Type": "audio/wav",
        "Cache-Control": "3600",
        "x-upsert": "true",
        "Content-Length": str(local_path.stat().st_size),
    }
    connection = connection_cls(parsed.netloc, timeout=120)
    try:
        with local_path.open("rb") as handle:
            connection.request("POST", request_path, body=handle, headers=headers)
            response = connection.getresponse()
            response_body = response.read()
        if response.status >= 400:
            response_text = response_body.decode("utf-8", errors="replace")
            bucket_missing = "bucket not found" in response_text.lower()
            if bucket_missing and allow_bucket_create:
                ensure_supabase_bucket()
                supabase_upload_file(local_path, object_path, allow_bucket_create=False)
                return
            raise RuntimeError(
                f"Supabase upload failed ({response.status}): {response_text}"
            )
    finally:
        connection.close()


def upload_to_supabase(local_path: pathlib.Path, object_path: str) -> str:
    quoted_path = quote(object_path, safe="/")
    supabase_upload_file(local_path, object_path)

    if SUPABASE_PUBLIC_BUCKET:
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{quoted_path}"

    signed = supabase_request(
        "POST",
        f"/storage/v1/object/sign/{SUPABASE_BUCKET}/{quoted_path}",
        body=json.dumps({"expiresIn": SUPABASE_SIGNED_URL_SECONDS}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    signed_url = (signed or {}).get("signedURL") or (signed or {}).get("signedUrl")
    if not signed_url:
        return f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{quoted_path}"
    if signed_url.startswith("http"):
        return signed_url
    if signed_url.startswith("/storage/v1/"):
        return f"{SUPABASE_URL}{signed_url}"
    if signed_url.startswith("/object/"):
        return f"{SUPABASE_URL}/storage/v1{signed_url}"
    return f"{SUPABASE_URL}/{signed_url.lstrip('/')}"


def create_supabase_url(object_path: str) -> str:
    quoted_path = quote(object_path, safe="/")
    if SUPABASE_PUBLIC_BUCKET:
        return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{quoted_path}"
    signed = supabase_request(
        "POST",
        f"/storage/v1/object/sign/{SUPABASE_BUCKET}/{quoted_path}",
        body=json.dumps({"expiresIn": SUPABASE_SIGNED_URL_SECONDS}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    signed_url = (signed or {}).get("signedURL") or (signed or {}).get("signedUrl")
    if not signed_url:
        raise RuntimeError("Supabase did not return a signed URL.")
    if signed_url.startswith("http"):
        return signed_url
    if signed_url.startswith("/storage/v1/"):
        return f"{SUPABASE_URL}{signed_url}"
    if signed_url.startswith("/object/"):
        return f"{SUPABASE_URL}/storage/v1{signed_url}"
    return f"{SUPABASE_URL}/{signed_url.lstrip('/')}"


def refresh_stem_urls(stems: list[dict]) -> list[dict]:
    refreshed = []
    for stem in stems:
        item = dict(stem)
        storage_key = item.get("storageKey") or item.get("storage_key") or extract_supabase_storage_key(item.get("url"))
        if storage_key and storage_is_supabase_enabled():
            try:
                item["url"] = create_supabase_url(str(storage_key))
            except Exception as error:
                print(f"[Supabase] Could not refresh signed URL: {error}")
        refreshed.append(item)
    return refreshed


def extract_supabase_storage_key(url: str | None) -> str | None:
    if not url or not SUPABASE_BUCKET:
        return None
    path = urlparse(url).path
    marker_options = [
        f"/storage/v1/object/public/{SUPABASE_BUCKET}/",
        f"/storage/v1/object/sign/{SUPABASE_BUCKET}/",
        f"/object/sign/{SUPABASE_BUCKET}/",
    ]
    for marker in marker_options:
        if marker in path:
            return path.split(marker, 1)[1]
    return None


def build_cached_payload(track_id, public_base_url: str):
    track_dir = OUTPUT_DIR / track_id
    if not track_dir.exists():
        return None

    stems = []
    for wav_file in sorted(track_dir.glob("*.wav")):
        stem_name = wav_file.stem.replace("_reconstructed", "")
        url = build_output_url(public_base_url, track_id, wav_file.name)
        stems.append({"name": stem_name, "label": stem_display_name(stem_name), "url": url, "storageKey": None})

    if not stems:
        return None

    return {
        "files": [stem["url"] for stem in stems],
        "stems": stems,
    }


def persist_result(user: dict | None, result: dict, input_name: str, selected_stems: list[str]) -> bool:
    if result.get("status") not in {"success", "cached"}:
        return False
    payload = {
        "id": uuid.uuid4().hex,
        "user_id": str(user.get("_id") or user.get("id")) if user else None,
        "inputName": input_name,
        "cacheId": result.get("cache_id"),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "outputUrls": result.get("files", []),
        "stems": result.get("stems", []),
        "selectedStems": selected_stems,
        "selectedStemLabels": [stem_display_name(stem) for stem in selected_stems],
    }
    if _MONGO_DB is not None:
        mongo_payload = dict(payload)
        mongo_payload["_id"] = payload["id"]
        mongo_payload["created_at"] = datetime.now(timezone.utc)
        try:
            _MONGO_DB.results.insert_one(mongo_payload)
            return True
        except Exception as error:
            print(f"[Mongo] Result persistence failed: {error}")
            if RESULT_PERSIST_REQUIRED:
                raise
            return False

    local_payload = _load_local_json(LOCAL_RESULTS_PATH, {"results": []})
    local_payload["results"].append(payload)
    _save_local_json(LOCAL_RESULTS_PATH, local_payload)
    return True


def list_persisted_results(user: dict | None):
    user_id = str(user.get("_id") or user.get("id")) if user else None
    if _MONGO_DB is not None:
        query = {"user_id": user_id} if user_id else {}
        cursor = _MONGO_DB.results.find(query).sort("created_at", -1).limit(50)
        results = []
        for item in cursor:
            created = item.get("createdAt") or item.get("created_at")
            if hasattr(created, "isoformat"):
                created = created.isoformat()
            results.append(
                {
                    "inputName": item.get("inputName"),
                    "createdAt": created,
                    "cacheId": item.get("cacheId"),
                    "outputUrls": item.get("outputUrls", []),
                    "stems": refresh_stem_urls(item.get("stems", [])),
                    "selectedStems": item.get("selectedStems", []),
                    "selectedStemLabels": item.get("selectedStemLabels", []),
                }
            )
        return results

    payload = _load_local_json(LOCAL_RESULTS_PATH, {"results": []})
    results = payload["results"]
    if user_id:
        results = [item for item in results if item.get("user_id") == user_id]
    return list(reversed(results[-50:]))


def user_identifier(user: dict | None) -> str | None:
    return str(user.get("_id") or user.get("id")) if user else None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def wants_async_inference(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def public_inference_job(job: dict) -> dict:
    payload = {
        "jobId": job["id"],
        "status": job["status"],
        "cache_id": job.get("cache_id"),
        "createdAt": job.get("createdAt"),
        "updatedAt": job.get("updatedAt"),
        "inputName": job.get("inputName"),
        "detail": job.get("detail"),
        "message": job.get("message"),
        "historySaved": job.get("historySaved"),
        "files": job.get("files", []),
        "stems": job.get("stems", []),
        "selectedStems": job.get("selectedStems", []),
        "selectedStemLabels": job.get("selectedStemLabels", []),
    }
    if job.get("error"):
        payload["error"] = job["error"]
    return payload


def cleanup_old_jobs() -> None:
    cutoff = time.time() - INFERENCE_JOB_TTL_SECONDS
    with _INFERENCE_JOBS_LOCK:
        for job_id, job in list(_INFERENCE_JOBS.items()):
            timestamp = float(job.get("updated_ts") or job.get("created_ts") or 0)
            if timestamp < cutoff:
                _INFERENCE_JOBS.pop(job_id, None)


def create_inference_job(
    user: dict | None,
    input_name: str,
    track_id: str,
    requested_stems: str | None,
) -> dict:
    cleanup_old_jobs()
    now = utc_now_iso()
    job = {
        "id": uuid.uuid4().hex,
        "ownerId": user_identifier(user),
        "status": "queued",
        "cache_id": track_id,
        "createdAt": now,
        "updatedAt": now,
        "created_ts": time.time(),
        "updated_ts": time.time(),
        "inputName": input_name,
        "requestedStemsRaw": requested_stems,
        "detail": "Upload received. Waiting for the inference worker.",
        "files": [],
        "stems": [],
        "selectedStems": [],
        "selectedStemLabels": [],
    }
    with _INFERENCE_JOBS_LOCK:
        _INFERENCE_JOBS[job["id"]] = job
    return job


def update_inference_job(job_id: str, **updates) -> None:
    updates["updatedAt"] = utc_now_iso()
    updates["updated_ts"] = time.time()
    with _INFERENCE_JOBS_LOCK:
        job = _INFERENCE_JOBS.get(job_id)
        if job is not None:
            job.update(updates)


def get_inference_job(job_id: str) -> dict | None:
    with _INFERENCE_JOBS_LOCK:
        job = _INFERENCE_JOBS.get(job_id)
        return dict(job) if job is not None else None


def inference_job_counts() -> dict[str, int]:
    with _INFERENCE_JOBS_LOCK:
        counts: dict[str, int] = {}
        for job in _INFERENCE_JOBS.values():
            status = str(job.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        return counts


async def run_inference_job(
    job_id: str,
    temp_input_path: pathlib.Path,
    track_id: str,
    public_base_url: str,
    requested_stems: str | None,
    user: dict | None,
    input_name: str,
) -> None:
    owner_id = user_identifier(user)
    try:
        update_inference_job(job_id, status="running", detail="Loading model and preparing inference.")
        async with _INFERENCE_LOCK:
            runtime = await asyncio.to_thread(get_model_runtime)
            requested_source_names = resolve_requested_sources(runtime.source_names, requested_stems)
            update_inference_job(
                job_id,
                status="running",
                detail="Separating audio stems.",
                selectedStems=requested_source_names,
                selectedStemLabels=[stem_display_name(stem) for stem in requested_source_names],
            )
            result = await asyncio.to_thread(
                inference_pipeline,
                temp_input_path,
                runtime,
                track_id,
                public_base_url,
                requested_source_names,
                owner_id,
            )

        if result.get("status") == "failed":
            update_inference_job(
                job_id,
                status="failed",
                detail=result.get("message") or "Audio processing failed.",
                message=result.get("message"),
                error=result.get("error"),
            )
            return

        history_saved = await asyncio.to_thread(
            persist_result,
            user,
            result,
            input_name or track_id,
            result.get("selectedStems") or requested_source_names,
        )
        update_inference_job(
            job_id,
            status=result.get("status", "success"),
            detail="Separated stems are ready.",
            message=result.get("message"),
            cache_id=result.get("cache_id") or track_id,
            files=result.get("files", []),
            stems=result.get("stems", []),
            historySaved=history_saved,
        )
    except Exception as error:
        traceback.print_exc()
        temp_input_path.unlink(missing_ok=True)
        update_inference_job(
            job_id,
            status="failed",
            detail=str(error) or "Audio processing failed.",
            message=str(error),
            error=str(error),
        )


_CONTENT_TYPE_EXTENSION_MAP = {
    "audio/mpeg": ".mp3",
    "audio/mp3": ".mp3",
    "audio/wav": ".wav",
    "audio/x-wav": ".wav",
    "audio/wave": ".wav",
    "audio/flac": ".flac",
    "audio/x-flac": ".flac",
    "audio/ogg": ".ogg",
    "audio/vorbis": ".ogg",
    "audio/mp4": ".m4a",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
}


def resolve_upload_suffix(upload: UploadFile) -> str:
    filename = upload.filename or ""
    suffix = pathlib.Path(filename).suffix.lower()
    if suffix:
        sanitized = re.sub(r"[^a-z0-9.]+", "", suffix)
        if sanitized.startswith(".") and len(sanitized) > 1:
            return sanitized

    content_type = (upload.content_type or "").lower()
    return _CONTENT_TYPE_EXTENSION_MAP.get(content_type, ".wav")


def resolve_track_id(cache_id: Optional[str]) -> str:
    if cache_id:
        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "", cache_id)[:96]
        if sanitized:
            return sanitized
    return uuid.uuid4().hex


async def write_upload_to_disk(upload: UploadFile, destination: pathlib.Path) -> int:
    total_bytes = 0
    try:
        with destination.open("wb") as buffer:
            while True:
                chunk = await upload.read(UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break

                total_bytes += len(chunk)
                if MAX_UPLOAD_BYTES and total_bytes > MAX_UPLOAD_BYTES:
                    limit_text = f"{MAX_UPLOAD_MB} MB" if MAX_UPLOAD_MB else "the configured"
                    raise HTTPException(status_code=413, detail=f"Uploaded file exceeds {limit_text} limit.")

                buffer.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return total_bytes


@app.post("/api/auth/register")
def register(payload: RegisterPayload):
    username = payload.username.strip()
    email = normalize_email(payload.email)
    password = payload.password
    if not username:
        raise HTTPException(status_code=400, detail="Username is required.")
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(status_code=400, detail="Valid email is required.")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters.")
    user = create_user(username, email, password)
    public = public_user(user)
    return {"message": "Account created", "token": create_token(public["id"]), "user": public}


@app.post("/api/auth/login")
def login(payload: LoginPayload):
    user = find_user_by_email(payload.email)
    if user is None or not verify_password(payload.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    public = public_user(user)
    return {"token": create_token(public["id"]), "user": public}


@app.get("/api/auth/me")
def me(authorization: str | None = Header(default=None)):
    user = current_user_from_authorization(authorization, required=True)
    return public_user(user)


@app.post("/infer")
@app.post("/api/infer")
@app.post("/api/infer/segment")
async def infer_audio(
    request: Request,
    file: UploadFile = File(...),
    cache_id: Optional[str] = Form(None),
    stems: Optional[str] = Form(None),
    selected_stems: Optional[str] = Form(None),
    async_inference: Optional[str] = Form(None),
    authorization: str | None = Header(default=None),
):
    print("Received file:", file.filename if file else "No file")
    if not file:
        raise HTTPException(status_code=400, detail="No file received")

    cleanup_old_outputs()
    user = current_user_from_authorization(authorization, required=AUTH_REQUIRED)
    track_id = resolve_track_id(cache_id)
    public_base_url = get_public_base_url(request)

    if cache_id:
        cached_payload = build_cached_payload(track_id, public_base_url)
        if cached_payload:
            return JSONResponse({
                "status": "cached",
                "cache_id": track_id,
                **cached_payload,
            })

    temp_input_path = TEMP_DIR / f"{track_id}-{uuid.uuid4().hex}{resolve_upload_suffix(file)}"
    file_size = await write_upload_to_disk(file, temp_input_path)
    print("File size:", file_size, "bytes")

    requested_stems_raw = selected_stems or stems
    if wants_async_inference(async_inference):
        job = create_inference_job(user, file.filename or track_id, track_id, requested_stems_raw)
        asyncio.create_task(
            run_inference_job(
                job["id"],
                temp_input_path,
                track_id,
                public_base_url,
                requested_stems_raw,
                user,
                file.filename or track_id,
            )
        )
        return JSONResponse(public_inference_job(job), status_code=202)

    async with _INFERENCE_LOCK:
        try:
            runtime = get_model_runtime()
            requested_source_names = resolve_requested_sources(runtime.source_names, requested_stems_raw)
            result = inference_pipeline(
                temp_input_path,
                runtime,
                track_id,
                public_base_url,
                requested_source_names=requested_source_names,
                owner_id=user_identifier(user),
            )
        except Exception:
            temp_input_path.unlink(missing_ok=True)
            raise
    if result.get("status") == "failed":
        return JSONResponse(result, status_code=400)
    result["historySaved"] = persist_result(user, result, file.filename or track_id, requested_source_names)
    return JSONResponse(result)


@app.get("/api/infer/jobs/{job_id}")
def infer_job(job_id: str, authorization: str | None = Header(default=None)):
    user = current_user_from_authorization(authorization, required=AUTH_REQUIRED)
    cleanup_old_jobs()
    job = get_inference_job(job_id)
    if job is None or job.get("ownerId") != user_identifier(user):
        raise HTTPException(status_code=404, detail="Inference job was not found.")
    return public_inference_job(job)


@app.get("/api/infer/results")
def infer_results(authorization: str | None = Header(default=None)):
    user = current_user_from_authorization(authorization, required=AUTH_REQUIRED)
    return {"results": list_persisted_results(user)}


@app.get("/api/stems")
def available_stems():
    if find_model_path() is None:
        raise HTTPException(status_code=503, detail="UNet model is not available yet.")
    source_names = configured_source_names()
    return {
        "model_loaded": _MODEL_RUNTIME is not None,
        "stems": [
            {
                "name": source_name,
                "label": stem_display_name(source_name),
            }
            for source_name in source_names
        ],
    }

@app.get("/healthz")
def healthz():
    model_path = find_model_path()
    supabase_error = supabase_config_error()
    supabase_enabled = STORAGE_BACKEND != "local" and supabase_error is None
    return {
        "status": "ok",
        "model_available": model_path is not None,
        "model_loaded": _MODEL_RUNTIME is not None,
        "model_type": "UNet",
        "memory_rss_mb": current_rss_mb(),
        "memory_limit_mb": RAM_LIMIT_MB,
        "inference_chunk_seconds": INFERENCE_CHUNK_SECONDS,
        "inference_frequency_tile_bins": INFERENCE_FREQ_TILE_BINS,
        "inference_frequency_overlap_bins": INFERENCE_FREQ_OVERLAP_BINS,
        "model_path": str(model_path) if model_path else None,
        "max_upload_mb": MAX_UPLOAD_MB,
        "max_audio_seconds": MAX_AUDIO_SECONDS,
        "duration_cap_enabled": MAX_AUDIO_SECONDS > 0,
        "output_ttl_minutes": OUTPUT_TTL_MINUTES,
        "inference_job_ttl_seconds": INFERENCE_JOB_TTL_SECONDS,
        "inference_jobs": inference_job_counts(),
        "auth_required": AUTH_REQUIRED,
        "mongo_required": MONGO_REQUIRED,
        "mongo_enabled": _MONGO_DB is not None,
        "mongodb_timeout_ms": MONGODB_TIMEOUT_MS,
        "result_persist_required": RESULT_PERSIST_REQUIRED,
        "auth_store": "mongodb" if _MONGO_DB is not None else "local_json_dev_fallback",
        "cors_allowed_origins": allowed_origins,
        "cors_allowed_origin_regex": allowed_origin_regex,
        "storage_backend": "supabase" if supabase_enabled else "local",
        "supabase_enabled": supabase_enabled,
        "supabase_config_error": supabase_error if not supabase_enabled else None,
    }
