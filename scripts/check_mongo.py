from __future__ import annotations

import os
import pathlib
import re
import sys
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError


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


def redact_uri(uri: str) -> str:
    try:
        parts = urlsplit(uri)
        netloc = parts.netloc
        if "@" in netloc:
            _userinfo, host = netloc.rsplit("@", 1)
            netloc = f"***:***@{host}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    except Exception:
        return re.sub(r"//[^@]+@", "//***:***@", uri)


def main() -> int:
    load_env()
    uri = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI")
    db_name = os.getenv("MONGODB_DB", "audioprism")
    timeout_ms = int(os.getenv("MONGODB_TIMEOUT_MS", "15000"))

    if not uri:
        print("MONGODB_URI is empty.")
        return 1

    print("Mongo URI:", redact_uri(uri))
    print("Database:", db_name)
    print("Timeout ms:", timeout_ms)

    if "<" in uri or ">" in uri:
        print("ERROR: MONGODB_URI still contains < or > placeholder characters.")
        print("Atlas examples show <password> as a placeholder; remove the angle brackets.")
        return 1

    if "mongodb+srv://" not in uri and "mongodb://" not in uri:
        print("ERROR: MONGODB_URI must start with mongodb+srv:// or mongodb://")
        return 1

    client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms, connectTimeoutMS=timeout_ms)
    try:
        print("Pinging MongoDB...")
        client.admin.command("ping")

        db = client[db_name]
        print("Testing write access...")
        db["_audioprism_connection_test"].update_one(
            {"_id": "local-check"},
            {"$set": {"checked_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        print("MongoDB OK: primary reachable and write access works.")
        return 0
    except ServerSelectionTimeoutError as error:
        print("ERROR: Could not select a writable MongoDB primary.")
        print(str(error))
        print("\nCheck Atlas Network Access/IP whitelist, cluster status, and that your URI targets the cluster, not a single secondary.")
        return 1
    except PyMongoError as error:
        print("ERROR: MongoDB command failed.")
        print(str(error))
        print("\nIf this is Authentication failed, re-check username/password and URL-encode special characters in the password.")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
