"""Dataset manifest: what agents are told about the data.

content_hash is computed over the raw bytes of every CSV (sorted by name) — it's
the staleness guard the memory layer (phase d) will filter on: a cached sub-result
is only reusable if the data it came from is byte-identical."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from analyst.graph.state import DatasetManifest, TableInfo
from analyst.quality import profile_dataset


def build_manifest(root: str | Path) -> DatasetManifest:
    root = Path(root)
    csvs = sorted(root.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files in {root}")

    h = hashlib.sha256()
    tables = []
    frames: dict[str, pd.DataFrame] = {}
    for f in csvs:
        h.update(f.name.encode())
        h.update(f.read_bytes())
        frame = pd.read_csv(f)
        frames[f.stem] = frame
        tables.append(
            TableInfo(
                name=f.stem,
                path=str(f),
                columns=[str(column) for column in frame.columns],
                dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
                row_count=len(frame),
            )
        )
    return DatasetManifest(
        root=str(root),
        tables=tables,
        content_hash=h.hexdigest(),
        quality_report=profile_dataset(frames),
    )
