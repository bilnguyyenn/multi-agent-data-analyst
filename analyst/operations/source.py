"""Streaming conversion of a pinned, attributed UCI workbook. No source rows are dropped."""

from __future__ import annotations

import csv
import json
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

from analyst.operations.metrics import COLUMNS
from analyst.operations.service import file_hash

URL = "https://archive.ics.uci.edu/static/public/352/online%2Bretail.zip"
SOURCE_SHA256 = "f5385cbb54bbebf7196389109c6b0621faab0c304e3702548165e71c84aede8b"
ATTRIBUTION = "Chen, D. (2015). Online Retail. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 (CC BY 4.0)."
PERIODS = [
    ("development", "2011-11-01", "2011-11-08"),
    ("refresh", "2011-11-08", "2011-11-15"),
    ("evaluation_a", "2011-11-15", "2011-11-22"),
    ("evaluation_b", "2011-11-22", "2011-11-29"),
]


def import_uci(cache: Path, output: Path, download: bool = False) -> dict:
    from openpyxl import load_workbook

    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / "online-retail.zip"
    if not archive.exists():
        if not download:
            raise FileNotFoundError(
                "source missing; pass --download to fetch the public UCI archive"
            )
        temporary = archive.with_suffix(".partial")
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open("wb") as target:
            while chunk := response.read(1024 * 1024):
                target.write(chunk)
        if file_hash(temporary) != SOURCE_SHA256:
            temporary.unlink()
            raise ValueError("downloaded source hash differs from the pinned UCI release")
        temporary.replace(archive)
    if file_hash(archive) != SOURCE_SHA256:
        raise ValueError("UCI source archive fingerprint mismatch")
    workbook = cache / "Online Retail.xlsx"
    with zipfile.ZipFile(archive) as zipped:
        # Never extract untrusted archive paths.
        with zipped.open("Online Retail.xlsx") as source, workbook.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
    output.mkdir(parents=True, exist_ok=True)
    paths = {name: output / name / "transactions.csv" for name, _, _ in PERIODS}
    paths["full"] = cache / "full" / "transactions.csv"
    handles, writers, counts = {}, {}, {name: 0 for name in paths}
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        handles[name] = path.open("w", newline="", encoding="utf-8")
        writers[name] = csv.writer(handles[name], lineterminator="\n")
        writers[name].writerow(COLUMNS)
    book = load_workbook(workbook, read_only=True, data_only=True)
    try:
        iterator = book.active.iter_rows(values_only=True)
        if list(next(iterator)) != COLUMNS:
            raise ValueError("source workbook schema changed")
        for row in iterator:
            values = [
                value.isoformat(sep=" ", timespec="seconds")
                if isinstance(value, datetime)
                else ""
                if value is None
                else str(int(value))
                if isinstance(value, float) and value.is_integer()
                else str(value)
                for value in row
            ]
            writers["full"].writerow(values)
            counts["full"] += 1
            for name, start, end in PERIODS:
                if start <= values[4][:10] < end:
                    writers[name].writerow(values)
                    counts[name] += 1
    finally:
        book.close()
        for handle in handles.values():
            handle.close()
    if counts["full"] != 541909:
        raise ValueError("source row count differs from the pinned release")
    manifest = {
        "source_url": URL,
        "source_sha256": SOURCE_SHA256,
        "workbook_sha256": file_hash(workbook),
        "attribution": ATTRIBUTION,
        "conversion": "openpyxl read-only; all rows retained; nulls become empty CSV cells; ISO local timestamps; integral IDs preserved",
        "timezone": "unspecified source-local",
        "splits": {
            name: {"rows": counts[name], "sha256": file_hash(path), "bytes": path.stat().st_size}
            for name, path in paths.items()
        },
    }
    (output / "SOURCE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
