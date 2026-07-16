from __future__ import annotations

import argparse
import json
import math
import pathlib
import struct
import time
import uuid
import wave
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local AudioPrism API inference smoke test.")
    parser.add_argument("--api", default="http://127.0.0.1:8001", help="Backend API root.")
    parser.add_argument("--email", default="localtest@example.com", help="Test account email.")
    parser.add_argument("--password", default="password123", help="Test account password.")
    parser.add_argument("--stems", default="Vocal", help="Comma-separated stems to request.")
    parser.add_argument("--duration", type=float, default=1.0, help="Generated WAV duration in seconds.")
    parser.add_argument("--no-auth", action="store_true", help="Skip auth headers.")
    return parser.parse_args()


def request_json(url: str, payload: dict | None = None, token: str | None = None) -> dict:
    headers = {"Accept": "application/json"}
    body = None
    method = "GET"
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} failed with HTTP {error.code}: {text}") from error


def register_or_login(api: str, email: str, password: str) -> str:
    username = email.split("@", 1)[0] or "localtest"
    try:
        data = request_json(
            f"{api}/api/auth/register",
            {"username": username, "email": email, "password": password},
        )
        token = data.get("token")
        if token:
            return token
    except RuntimeError as error:
        if "409" not in str(error):
            raise

    data = request_json(f"{api}/api/auth/login", {"email": email, "password": password})
    token = data.get("token")
    if not token:
        raise RuntimeError("Login response did not include a token.")
    return token


def create_test_wav(path: pathlib.Path, duration: float, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_frames = max(1, int(duration * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for index in range(total_frames):
            t = index / sample_rate
            sample = 0.22 * math.sin(2 * math.pi * 220 * t) + 0.08 * math.sin(2 * math.pi * 440 * t)
            frames.extend(struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32767)))
        handle.writeframes(bytes(frames))


def post_multipart(api: str, wav_path: pathlib.Path, stems: str, token: str | None) -> dict:
    boundary = f"----AudioPrismSmoke{uuid.uuid4().hex}"
    file_bytes = wav_path.read_bytes()
    parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="selected_stems"\r\n\r\n'
            f"{stems}\r\n"
        ).encode("utf-8"),
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{wav_path.name}"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
        ).encode("utf-8"),
        file_bytes,
        f"\r\n--{boundary}--\r\n".encode("utf-8"),
    ]
    body = b"".join(parts)
    headers = {
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "Content-Length": str(len(body)),
        "Accept": "application/json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = Request(f"{api}/api/infer/segment", data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=300) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        text = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Inference failed with HTTP {error.code}: {text}") from error


def main() -> None:
    args = parse_args()
    api = args.api.rstrip("/")

    health = request_json(f"{api}/healthz")
    print("Health:", json.dumps(health, indent=2))
    if not health.get("model_available"):
        raise RuntimeError("No model found. Put model_render.pth in backend/models or set MODEL_PATH.")

    token = None if args.no_auth else register_or_login(api, args.email, args.password)
    wav_path = pathlib.Path("tmp_uploads") / f"smoke-{int(time.time())}.wav"
    create_test_wav(wav_path, args.duration)
    print(f"Generated test WAV: {wav_path} ({wav_path.stat().st_size} bytes)")

    try:
        result = post_multipart(api, wav_path, args.stems, token)
    finally:
        wav_path.unlink(missing_ok=True)
    print("Inference result:", json.dumps(result, indent=2))
    stems = result.get("stems") or []
    if not stems:
        raise RuntimeError("Inference response did not include stems.")
    print("OK:", ", ".join(stem.get("label") or stem.get("name") or stem.get("url", "") for stem in stems))


if __name__ == "__main__":
    main()
