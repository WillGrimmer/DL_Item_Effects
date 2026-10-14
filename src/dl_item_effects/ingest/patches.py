"""Patch labels and patch windows from the deadlock-api changelog list.

A patch is named by the date in its changelog title ("09-29-2026", or "09-16-2026 Update").
Matches are labelled with the latest patch that started on or before their UTC start date.
A window skips the release day, because matches that day can come from either side of it.
Hotfixes appended to a patch thread do not start a new window.
"""
from __future__ import annotations

import json
import re
from datetime import date, timedelta
from pathlib import Path

from .. import config

TITLE_DATE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")


def parse_patches(records: list[dict]) -> list[tuple[str, date]]:
    """(name, start date) for every changelog thread whose title carries a date, oldest first."""
    out = {}
    for r in records:
        m = TITLE_DATE.search(r.get("title", ""))
        if m:
            mm, dd, yyyy = m.groups()
            out[f"{mm}-{dd}-{yyyy}"] = date(int(yyyy), int(mm), int(dd))
    return sorted(out.items(), key=lambda kv: kv[1])


def load_patches(api=None, refresh: bool = False,
                 cache_path: str | Path | None = None) -> list[tuple[str, date]]:
    cache = Path(cache_path or (config.INTERIM_DIR / "patches.json"))
    if refresh or not cache.exists():
        if api is None:
            from ..api import DeadlockAPI
            api = DeadlockAPI()
        records = api.patches()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(records))
    else:
        records = json.loads(cache.read_text())
    return parse_patches(records)


def patch_window(patches: list[tuple[str, date]], name: str) -> tuple[date, date | None]:
    """First and last full day of a patch (inclusive). The last day is None for the
    current patch."""
    names = [n for n, _ in patches]
    i = names.index(name)
    start = patches[i][1] + timedelta(days=1)
    end = patches[i + 1][1] - timedelta(days=1) if i + 1 < len(patches) else None
    return start, end
