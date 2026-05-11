from __future__ import annotations

import csv
import os
import sqlite3
from typing import List

from .storage import Storage


TABLES = ["events", "markets", "snapshots", "trades", "features", "scores", "outcomes", "collector_state"]


def export_db(db_path: str, out_dir: str, fmt: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    for table in TABLES:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        if fmt == "csv":
            _export_csv(rows, os.path.join(out_dir, f"{table}.csv"))
        elif fmt == "parquet":
            _export_parquet(rows, os.path.join(out_dir, f"{table}.parquet"))


def _export_csv(rows: List[sqlite3.Row], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _export_parquet(rows: List[sqlite3.Row], path: str) -> None:
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Parquet export requires pandas + pyarrow. Install optional deps.") from exc

    if not rows:
        return
    df = pd.DataFrame([dict(r) for r in rows])
    df.to_parquet(path, index=False)

