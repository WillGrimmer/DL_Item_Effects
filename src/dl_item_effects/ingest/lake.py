"""The deadlock-api data lake: manifest, base files, and which partitions cover a date window.

See docs/data_dumps.md. The manifest is regenerated hourly and a base file's key changes when
its partition is rebuilt, so the manifest is fetched fresh on every run and never cached.
Partition date ranges are cached by file URL, which changes whenever the file does.

Dates are UTC (``extract.connect`` sets the DuckDB session time zone to UTC).
"""
from __future__ import annotations

import json
import urllib.request
from datetime import date
from pathlib import Path

import duckdb

from .. import config


def fetch_manifest(url: str | None = None, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(url or config.MANIFEST_URL,
                                 headers={"User-Agent": "dl-item-effects"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def base_files(manifest: dict, table: str = "match_player",
               base_url: str | None = None) -> dict[int, str]:
    """Partition -> URL of its base parquet. Delta files are ignored: they are hourly
    increments that get folded into the base files, and reading both would double count."""
    root = (base_url or config.LAKE_BASE).rstrip("/")
    return {int(f["partition"]): f"{root}/{f['key']}"
            for f in manifest["tables"][table]["files"] if f["kind"] == "base"}


def partition_dates(con: duckdb.DuckDBPyConnection, files: dict[int, str],
                    partitions: list[int],
                    cache_path: str | Path | None = None) -> dict[int, tuple[date, date]]:
    """First and last ``start_time`` date (UTC) of each partition. Reads one column."""
    cache_file = Path(cache_path or (config.INTERIM_DIR / "partition_dates.json"))
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    out = {}
    for p in partitions:
        url = files[p]
        if url not in cache:
            lo, hi = con.execute(f"SELECT min(start_time)::DATE, max(start_time)::DATE "
                                 f"FROM read_parquet('{url}')").fetchone()
            cache[url] = [lo.isoformat(), hi.isoformat()]
        out[p] = tuple(date.fromisoformat(d) for d in cache[url])
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(cache, indent=1))
    return out


def partitions_for_window(con: duckdb.DuckDBPyConnection, files: dict[int, str],
                          start: date, end: date,
                          cache_path: str | Path | None = None) -> list[int]:
    """Partitions holding any match that started between ``start`` and ``end`` (inclusive).

    match_id grows with time, so walk down from the newest partition and stop at the first
    one that ends before ``start``.
    """
    keep = []
    for p in sorted(files, reverse=True):
        lo, hi = partition_dates(con, files, [p], cache_path)[p]
        if hi < start:
            break
        if lo <= end:
            keep.append(p)
    return sorted(keep)
