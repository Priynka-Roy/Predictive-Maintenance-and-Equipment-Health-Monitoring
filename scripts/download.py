"""Download the official archive intact; extract originals without modification."""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"


def main():
    raw = ROOT / "data/raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "metropt3.zip"
    if not archive.exists():
        partial = archive.with_suffix(".partial")
        with requests.get(URL, stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open("wb") as target:
                for chunk in response.iter_content(1024 * 1024):
                    target.write(chunk)
        partial.replace(archive)
    with ZipFile(archive) as zipped:
        for entry in zipped.infolist():
            destination = (raw / entry.filename).resolve()
            if not destination.is_relative_to(raw.resolve()):
                raise ValueError("Unsafe archive member")
        zipped.extractall(raw)
    records = []
    for path in sorted(raw.rglob("*")):
        if path.is_file() and path.name != ".gitkeep":
            records.append(
                {
                    "file": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                }
            )
    previous = ROOT / "reports/provenance.json"
    retrieved = json.loads(previous.read_text()).get("retrieved_utc") if previous.exists() else None
    metadata = {
        "url": URL,
        "retrieved_utc": retrieved or datetime.now(timezone.utc).isoformat(),
        "doi": "10.24432/C5VW3R",
        "license": "CC BY 4.0",
        "files": records,
    }
    (ROOT / "reports/provenance.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
