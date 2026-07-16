from __future__ import annotations

import os
import pathlib
import sys
from urllib.parse import urlparse


def load_env(path: pathlib.Path = pathlib.Path(".env")) -> None:
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


def main() -> int:
    load_env()
    storage_backend = os.getenv("STORAGE_BACKEND", "auto").lower()
    supabase_url = (os.getenv("SUPABASE_URL") or "").rstrip("/")
    service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY") or ""
    bucket = os.getenv("SUPABASE_BUCKET", "audioprism-stems")

    print("Storage backend:", storage_backend)
    print("Bucket:", bucket or "<empty>")

    if storage_backend == "local":
        print("Supabase disabled: STORAGE_BACKEND=local")
        return 0

    parsed = urlparse(supabase_url)
    host = parsed.hostname or "<invalid>"
    if parsed.port:
        host = f"{host}:{parsed.port}"
    print("Supabase URL host:", host)

    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        print("ERROR: SUPABASE_URL must be the Project URL, e.g. https://project-ref.supabase.co")
        print("Do not use the postgresql://... database connection string here.")
        return 1
    if parsed.scheme != "https":
        print("ERROR: SUPABASE_URL must use https.")
        return 1
    if "supabase.co" not in parsed.netloc:
        print("ERROR: SUPABASE_URL should end in supabase.co.")
        return 1
    if not service_key:
        print("ERROR: SUPABASE_SERVICE_ROLE_KEY is empty.")
        return 1
    if service_key.startswith("sb_publishable_"):
        print("ERROR: This is a publishable key. Use a backend secret/service-role key.")
        return 1
    if not bucket:
        print("ERROR: SUPABASE_BUCKET is empty.")
        return 1

    print("Supabase config shape looks OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
