"""Objective and mid-boss events per match, for the decision frame's game-state covariates.

``objectives.*`` and ``mid_boss.*`` are match-level arrays repeated on every player row.
Each objective entry belongs to a team and has ``destroyed_time_s`` (0 while standing), so a
destroyed entry is a loss for that team. Mid-boss entries carry the team that claimed the
reward. Output: one row per event with ``match_id``, ``team`` (the team that lost the
objective, or claimed the mid boss), ``kind`` and ``t`` (seconds):

* ``t1``: lane guardians (Tier1Lane*), ``t2``: walkers (Tier2Lane*),
* ``base``: base guardians, shield generators, patron and core,
* ``mid``: mid boss claimed.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb

from .. import config

KINDS = ("t1", "t2", "base", "mid")


def build_objectives(source_urls: list[str], out_path: str | Path,
                     match_modes: tuple[str, ...] = ("Ranked",), start: date | None = None,
                     end: date | None = None,
                     con: duckdb.DuckDBPyConnection | None = None) -> Path:
    if con is None:
        from .extract import connect
        con = connect()
    urls = "[" + ", ".join(f"'{u}'" for u in source_urls) + "]"
    modes = ", ".join(f"'{m}'" for m in match_modes)
    window = ""
    if start:
        window += f" AND start_time >= '{start.isoformat()}'::TIMESTAMPTZ"
    if end:
        window += f" AND start_time < '{(end + timedelta(days=1)).isoformat()}'::TIMESTAMPTZ"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""COPY (
    WITH m AS (
        SELECT match_id,
               any_value("objectives.destroyed_time_s") AS od,
               any_value("objectives.team_objective")::VARCHAR[] AS oo,
               any_value("objectives.team")::VARCHAR[] AS ot,
               any_value("mid_boss.destroyed_time_s") AS md,
               any_value("mid_boss.team_claimed")::VARCHAR[] AS mc
        FROM read_parquet({urls})
        WHERE match_mode IN ({modes}) AND game_mode = 'Normal' {window}
        GROUP BY match_id
    ),
    obj AS (
        SELECT match_id, u[3] AS team,
               CASE WHEN u[2] LIKE 'Tier1%' THEN 't1' WHEN u[2] LIKE 'Tier2%' THEN 't2'
                    ELSE 'base' END AS kind,
               u[1]::INTEGER AS t
        FROM m, UNNEST(list_zip(od, oo, ot)) AS _(u)
        WHERE u[1] > 0
    ),
    mid AS (
        SELECT match_id, u[2] AS team, 'mid' AS kind, u[1]::INTEGER AS t
        FROM m, UNNEST(list_zip(md, mc)) AS _(u)
        WHERE u[1] > 0 AND u[2] IN ('Team0', 'Team1')
    )
    SELECT * FROM obj UNION ALL SELECT * FROM mid ORDER BY match_id, t
    ) TO '{out.as_posix()}' (FORMAT PARQUET)""")
    return out


def extract_objectives_window(patch: str, end: date | None = None,
                              out_dir: str | Path | None = None,
                              con: duckdb.DuckDBPyConnection | None = None,
                              patches: list[tuple[str, date]] | None = None) -> Path:
    """``_objectives.parquet`` for one patch window, next to its purchase events."""
    from . import lake
    from .extract import connect
    from .patches import load_patches, patch_window

    con = con or connect()
    patches = patches or load_patches()
    start, last = patch_window(patches, patch)
    end = end or last or (datetime.now(timezone.utc).date() - timedelta(days=1))
    out_dir = Path(out_dir or (config.PROCESSED_DIR / "events" / f"patch_{patch}"))
    out = out_dir / "_objectives.parquet"
    tmp = out.with_suffix(".tmp")
    files = lake.base_files(lake.fetch_manifest())
    parts = lake.partitions_for_window(con, files, start, end)
    for attempt in range(3):          # rebuilt partitions 404 under their old key
        try:
            build_objectives([files[p] for p in parts], tmp, start=start, end=end, con=con)
            break
        except duckdb.HTTPException:
            if attempt == 2:
                raise
            files = lake.base_files(lake.fetch_manifest())
    tmp.replace(out)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract objective events for one patch window.")
    ap.add_argument("--patch", required=True)
    ap.add_argument("--end", type=date.fromisoformat)
    a = ap.parse_args(argv)
    print(extract_objectives_window(a.patch, end=a.end))


if __name__ == "__main__":
    main()
