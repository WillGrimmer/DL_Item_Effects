"""Purchase-event extraction from the deadlock-api data lake (DuckDB over HTTPS).

One row per shop purchase in Ranked, Normal-mode matches with exactly 12 player rows
(PLAN.md §3.1). Column semantics of the source are in docs/data_dumps.md. Output columns:

* identity, label and patch: match/account/slot/team/hero, ``won``, ``start_time``,
  ``patch``, ``days_since_patch``, ``match_has_abandon``
* skill: ``badge_level`` (match average), ``rank_level`` (own, NULL while in calibration),
  ``in_calibration``, ``calibration_games``, ``win_streak``, ``team_rank_mean``,
  ``enemy_rank_mean``, ``own_minus_team``, ``team_rank_gap``, ``assigned_lane``,
  ``has_build``. Rank levels use the contiguous scale ``6 * tier + subrank``.
* the item: id/name/slot/tier/``cost`` (list), ``price_paid`` (list minus the components
  this purchase consumed), ``full_price``, ``sold_time_s``, ``consumed_by_upgrade`` (this
  row's ``sold_time_s`` is when an upgrade consumed it, not a sale)
* economy at the last snapshot at or before the purchase: ``net_worth_at_buy``, both teams'
  net worth and the gap, ``level``, K/D/A, ``snapshot_age_s``. Before the first snapshot
  (180 s) this is the start-of-game state (``config.START_SOULS``), never a later snapshot.
* ``n_prior_items``: shop purchases strictly earlier. ``ally_heroes``, ``enemy_heroes``.
* negative-control outcomes (PLAN.md §4), never covariates: ``nc_prior_souls`` and
  ``nc_prior_deaths`` over the snapshot interval that ends at that snapshot
  (``nc_prior_len_s`` long; NULL before the first snapshot), and ``nc_prev_won``, the result
  of the player's previous Ranked match ``prev_gap_h`` hours earlier.
* ``duration_s``, for diagnostics only: it is measured after every decision.

Same-click component rows are dropped. Buying an upgrade outright writes a row for its
component that is bought and consumed in the same second; that row is not a decision.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb

from .. import config
from ..assets import ItemDecoder
from . import lake
from .patches import load_patches, patch_window

SHOP_COSTS = (800, 1600, 3200, 6400)


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs; SET TimeZone = 'UTC';")
    return con


def _sql_list(xs) -> str:
    return "[" + ", ".join(f"'{x}'" for x in xs) + "]"


def _register_items(con: duckdb.DuckDBPyConnection, decoder: ItemDecoder) -> None:
    shop = [i for i in decoder.shop_items() if i.cost in SHOP_COSTS]
    ids = {i.id for i in shop}
    con.execute("CREATE OR REPLACE TEMP TABLE item_meta(item_id UBIGINT, name VARCHAR, "
                "slot VARCHAR, tier INTEGER, cost INTEGER)")
    con.executemany("INSERT INTO item_meta VALUES (?,?,?,?,?)",
                    [(i.id, i.name, i.slot, i.tier, i.cost) for i in shop])
    con.execute("CREATE OR REPLACE TEMP TABLE comp_map(item_id UBIGINT, component_id UBIGINT)")
    pairs = [(a, c) for a, c in decoder.component_pairs() if a in ids and c in ids]
    if pairs:
        con.executemany("INSERT INTO comp_map VALUES (?,?)", pairs)


def _register_patches(con: duckdb.DuckDBPyConnection,
                      patches: list[tuple[str, date]]) -> None:
    con.execute("CREATE OR REPLACE TEMP TABLE patch_starts(patch VARCHAR, starts DATE)")
    if patches:
        con.executemany("INSERT INTO patch_starts VALUES (?,?)", patches)


def build_history(source_urls: list[str], out_path: str | Path,
                  match_modes: tuple[str, ...] = ("Ranked",),
                  con: duckdb.DuckDBPyConnection | None = None) -> Path:
    """Slim (account_id, match_id, start_time, won) table for the previous-match link."""
    con = con or connect()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    modes = ", ".join(f"'{m}'" for m in match_modes)
    con.execute(f"""COPY (
        SELECT account_id, match_id, start_time, won FROM read_parquet({_sql_list(source_urls)})
        WHERE match_mode IN ({modes}) AND game_mode = 'Normal' AND team IN ('Team0', 'Team1')
    ) TO '{out.as_posix()}' (FORMAT PARQUET)""")
    return out


def build_purchase_events(
    source_urls: list[str],
    out_path: str | Path,
    history: str | Path | None = None,
    patches: list[tuple[str, date]] | None = None,
    match_modes: tuple[str, ...] = ("Ranked",),
    start: date | None = None,
    end: date | None = None,
    max_matches: int | None = None,
    con: duckdb.DuckDBPyConnection | None = None,
    decoder: ItemDecoder | None = None,
) -> Path:
    """Extract shop-purchase events from ``source_urls`` (match_player parquet files or URLs).

    ``history`` is a parquet from :func:`build_history` covering at least the matches
    before ``start``; without it the previous match is looked up in ``source_urls`` only.
    ``start`` and ``end`` are inclusive UTC dates on ``start_time``.
    """
    con = con or connect()
    decoder = decoder or ItemDecoder.load()
    _register_items(con, decoder)
    _register_patches(con, patches if patches is not None else load_patches())

    urls = _sql_list(source_urls)
    modes = ", ".join(f"'{m}'" for m in match_modes)
    window = ""
    if start:
        window += f" AND start_time >= '{start.isoformat()}'::TIMESTAMPTZ"
    if end:
        window += f" AND start_time < '{(end + timedelta(days=1)).isoformat()}'::TIMESTAMPTZ"
    limit = f"LIMIT {int(max_matches)}" if max_matches else ""
    if history:
        hist = f"SELECT account_id, start_time, won FROM read_parquet('{Path(history).as_posix()}')"
    else:
        hist = (f"SELECT account_id, start_time, won FROM read_parquet({urls}) "
                f"WHERE match_mode IN ({modes}) AND game_mode = 'Normal' "
                f"AND team IN ('Team0', 'Team1')")

    sql = f"""
    WITH raw AS (
        SELECT match_id, account_id, player_slot, team, hero_id, won, start_time, duration_s,
               average_badge::INTEGER AS ab,
               player_rank_initial_display_rank::INTEGER AS rank_code,
               player_rank_initial_calibration_games::INTEGER AS calibration_games,
               player_rank_initial_win_streak::INTEGER AS win_streak,
               assigned_lane::INTEGER AS assigned_lane,
               hero_build_id IS NOT NULL AS has_build,
               abandon_match_time_s,
               "items.game_time_s" AS buy_t, "items.item_id" AS buy_id,
               "items.sold_time_s" AS sold_t, "items.flags" AS buy_flags,
               "stats.time_stamp_s" AS st, "stats.net_worth" AS nw, "stats.level" AS lvl,
               "stats.kills" AS kls, "stats.deaths" AS dth, "stats.assists" AS ast
        FROM read_parquet({urls})
        WHERE match_mode IN ({modes}) AND game_mode = 'Normal'
          AND team IN ('Team0', 'Team1') {window}
    ),
    full12 AS (SELECT match_id, max(abandon_match_time_s) > 0 AS match_has_abandon
               FROM raw GROUP BY 1 HAVING count(*) = 12),
    sel AS (SELECT * FROM full12 ORDER BY match_id DESC {limit}),
    b AS (
        SELECT raw.*, sel.match_has_abandon,
               CASE WHEN rank_code > 0 THEN 6 * (rank_code // 10) + rank_code % 10 END
                   AS rank_level,
               CASE WHEN ab > 0 THEN 6 * (ab // 10) + ab % 10 END AS badge_level,
               CASE WHEN team = 'Team0' THEN 'Team1' ELSE 'Team0' END AS enemy_team
        FROM raw JOIN sel USING (match_id)
    ),
    team_rank AS (SELECT match_id, team, avg(rank_level) AS rank_mean FROM b GROUP BY 1, 2),
    comp AS (SELECT match_id, team, list(hero_id ORDER BY player_slot) AS heroes
             FROM b GROUP BY 1, 2),

    -- economy snapshots on the match's shared grid, plus the start-of-game state at t=0
    snap AS (
        SELECT match_id, account_id, team, u[1]::INTEGER AS ts, u[2]::INTEGER AS nw,
               u[3]::INTEGER AS level, u[4]::INTEGER AS kills, u[5]::INTEGER AS deaths,
               u[6]::INTEGER AS assists
        FROM b, UNNEST(list_zip(st, nw, lvl, kls, dth, ast)) AS _(u)
        UNION ALL
        SELECT match_id, account_id, team, 0, {config.START_SOULS}, {config.START_LEVEL}, 0, 0, 0
        FROM b
    ),
    snapd AS (
        SELECT *, nw - lag(nw) OVER w AS nc_prior_souls,
               deaths - lag(deaths) OVER w AS nc_prior_deaths,
               ts - lag(ts) OVER w AS nc_prior_len_s
        FROM snap WINDOW w AS (PARTITION BY match_id, account_id ORDER BY ts)
    ),
    team_ts AS (SELECT match_id, team, ts, sum(nw) AS team_nw FROM snap GROUP BY 1, 2, 3),

    -- purchases: shop items only, minus same-click component rows
    purch AS (
        SELECT match_id, account_id, player_slot, team, enemy_team, hero_id, won, start_time,
               duration_s, match_has_abandon, badge_level, rank_level, rank_code,
               calibration_games, win_streak, assigned_lane, has_build,
               u[1]::INTEGER AS game_time_s, u[2]::UBIGINT AS item_id,
               u[3]::INTEGER AS sold_time_s, u[4]::INTEGER AS flags
        FROM b, UNNEST(list_zip(buy_t, buy_id, sold_t, buy_flags)) AS _(u)
    ),
    shop AS (
        SELECT p.*, m.name AS item_name, m.slot, m.tier, m.cost
        FROM purch p JOIN item_meta m USING (item_id)
        WHERE NOT (p.flags = 1 AND p.sold_time_s = p.game_time_s)
    ),
    -- components held before this purchase and consumed by it
    disc AS (
        SELECT s.match_id, s.account_id, s.item_id, sum(c.cost) AS discount
        FROM shop s
        JOIN comp_map cm ON cm.item_id = s.item_id
        JOIN shop c ON c.match_id = s.match_id AND c.account_id = s.account_id
                   AND c.item_id = cm.component_id AND c.flags = 1
                   AND c.sold_time_s = s.game_time_s AND c.game_time_s < s.game_time_s
        GROUP BY 1, 2, 3
    ),

    -- state at the last snapshot at or before the purchase (ASOF: ts <= game_time_s)
    e_nw AS (
        SELECT s.*, snp.ts AS snap_ts, snp.nw AS net_worth_at_buy, snp.level, snp.kills,
               snp.deaths, snp.assists, snp.nc_prior_souls, snp.nc_prior_deaths,
               snp.nc_prior_len_s
        FROM shop s ASOF LEFT JOIN snapd snp
          ON s.match_id = snp.match_id AND s.account_id = snp.account_id
         AND s.game_time_s >= snp.ts
    ),
    e_my AS (
        SELECT e.*, tt.team_nw AS my_team_nw
        FROM e_nw e ASOF LEFT JOIN team_ts tt
          ON e.match_id = tt.match_id AND e.team = tt.team AND e.game_time_s >= tt.ts
    ),
    e_en AS (
        SELECT e.*, tt.team_nw AS enemy_team_nw
        FROM e_my e ASOF LEFT JOIN team_ts tt
          ON e.match_id = tt.match_id AND e.enemy_team = tt.team AND e.game_time_s >= tt.ts
    ),
    -- purchases strictly earlier; same-second purchases do not count each other
    e_prior AS (
        SELECT *, count(*) OVER (
            PARTITION BY match_id, account_id ORDER BY game_time_s
            RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior_items
        FROM e_en
    ),

    -- the player's previous Ranked match (negative-control outcome)
    hist AS ({hist}),
    players AS (SELECT DISTINCT match_id, account_id, start_time FROM b),
    prev AS (
        SELECT p.match_id, p.account_id, h.won AS nc_prev_won,
               date_diff('second', h.start_time, p.start_time) / 3600.0 AS prev_gap_h
        FROM players p ASOF LEFT JOIN hist h
          ON p.account_id = h.account_id AND p.start_time > h.start_time
    ),
    pm AS (SELECT DISTINCT match_id, start_time FROM b),
    pl AS (
        SELECT pm.match_id, ps.patch,
               date_diff('day', ps.starts, pm.start_time::DATE) AS days_since_patch
        FROM pm ASOF LEFT JOIN patch_starts ps ON pm.start_time::DATE >= ps.starts
    )
    SELECT
        e.match_id, e.account_id, e.player_slot, e.team, e.hero_id, e.won, e.start_time,
        pl.patch, pl.days_since_patch, e.match_has_abandon,
        e.badge_level, e.rank_level, coalesce(e.rank_code, 0) = 0 AS in_calibration,
        e.calibration_games, e.win_streak,
        tr.rank_mean AS team_rank_mean, er.rank_mean AS enemy_rank_mean,
        e.rank_level - tr.rank_mean AS own_minus_team, tr.rank_mean - er.rank_mean AS team_rank_gap,
        e.assigned_lane, e.has_build,
        e.game_time_s, e.item_id, e.item_name, e.slot, e.tier, e.cost,
        e.cost - coalesce(d.discount, 0) AS price_paid, d.discount IS NULL AS full_price,
        e.sold_time_s, e.flags = 1 AS consumed_by_upgrade,
        e.net_worth_at_buy, e.my_team_nw, e.enemy_team_nw,
        e.my_team_nw - e.enemy_team_nw AS team_nw_diff_at_buy,
        e.level, e.kills, e.deaths, e.assists, e.game_time_s - e.snap_ts AS snapshot_age_s,
        e.n_prior_items,
        list_filter(ally.heroes, h -> h <> e.hero_id) AS ally_heroes,
        enemy.heroes AS enemy_heroes,
        e.nc_prior_souls, e.nc_prior_deaths, e.nc_prior_len_s, pv.nc_prev_won, pv.prev_gap_h,
        e.duration_s
    FROM e_prior e
    LEFT JOIN disc d ON d.match_id = e.match_id AND d.account_id = e.account_id
                    AND d.item_id = e.item_id
    LEFT JOIN team_rank tr ON tr.match_id = e.match_id AND tr.team = e.team
    LEFT JOIN team_rank er ON er.match_id = e.match_id AND er.team = e.enemy_team
    LEFT JOIN comp ally  ON ally.match_id = e.match_id AND ally.team = e.team
    LEFT JOIN comp enemy ON enemy.match_id = e.match_id AND enemy.team = e.enemy_team
    LEFT JOIN prev pv ON pv.match_id = e.match_id AND pv.account_id = e.account_id
    LEFT JOIN pl ON pl.match_id = e.match_id
    ORDER BY e.match_id, e.account_id, e.game_time_s
    """

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{out.as_posix()}' (FORMAT PARQUET)")
    return out


def extract_window(patch: str, end: date | None = None, out_dir: str | Path | None = None,
                   lookback_days: int = 21, max_matches: int | None = None,
                   con: duckdb.DuckDBPyConnection | None = None,
                   decoder: ItemDecoder | None = None,
                   patches: list[tuple[str, date]] | None = None) -> list[Path]:
    """Extract one patch window to one parquet per partition, plus ``_sources.json``.

    Resumable: a partition whose output already exists is skipped. The window runs from the
    day after the patch to the day before the next one (or ``end``, default yesterday UTC).
    """
    con = con or connect()
    decoder = decoder or ItemDecoder.load()
    patches = patches or load_patches()
    start, last = patch_window(patches, patch)
    end = end or last or (datetime.now(timezone.utc).date() - timedelta(days=1))
    out_dir = Path(out_dir or (config.PROCESSED_DIR / "events" / f"patch_{patch}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = lake.fetch_manifest()
    files = lake.base_files(manifest)
    parts = lake.partitions_for_window(con, files, start, end)
    hist_parts = lake.partitions_for_window(con, files, start - timedelta(days=lookback_days), end)
    (out_dir / "_sources.json").write_text(json.dumps({
        "patch": patch, "start": start.isoformat(), "end": end.isoformat(),
        "git_commit": _git_commit(), "manifest_version": manifest.get("version"),
        "partitions": parts, "history_partitions": hist_parts,
        "files": {p: files[p] for p in sorted(set(parts) | set(hist_parts))},
    }, indent=1))

    hist = out_dir / "_history.parquet"
    if not hist.exists():
        t0 = time.monotonic()
        build_history([files[p] for p in hist_parts], hist, con=con)
        print(f"history: partitions {hist_parts[0]}-{hist_parts[-1]}, "
              f"{time.monotonic() - t0:.0f} s", flush=True)
    outs = []
    for p in parts:
        out = out_dir / f"part-{p}.parquet"
        if not out.exists():
            t0 = time.monotonic()
            tmp = out.with_suffix(".tmp")
            build_purchase_events([files[p]], tmp, history=hist, patches=patches, start=start,
                                  end=end, max_matches=max_matches, con=con, decoder=decoder)
            tmp.replace(out)
            n = con.execute(f"SELECT count(*), count(DISTINCT match_id) "
                            f"FROM read_parquet('{out.as_posix()}')").fetchone()
            print(f"part-{p}: {n[0]:,} rows, {n[1]:,} matches, "
                  f"{time.monotonic() - t0:.0f} s", flush=True)
        outs.append(out)
    return outs


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Extract one patch window of purchase events.")
    ap.add_argument("--patch", required=True, help='patch name, e.g. "09-29-2026"')
    ap.add_argument("--end", type=date.fromisoformat, help="last UTC day to include")
    ap.add_argument("--out-dir")
    ap.add_argument("--max-matches", type=int, help="per partition; for smoke tests")
    a = ap.parse_args(argv)
    for p in extract_window(a.patch, end=a.end, out_dir=a.out_dir, max_matches=a.max_matches):
        print(p)


if __name__ == "__main__":
    main()
