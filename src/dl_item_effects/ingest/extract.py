"""Bulk purchase-event extraction from the data dumps (DuckDB over S3).

Ported from DL_Decision_Analytics (where it ran on shard 94) and EXTENDED for v2 with the
state the risk-set design needs: ``sold_time_s`` (so inventory-at-t is exact), the enemy
team's badge, and the player's level / kills / deaths / assists at the last snapshot.
The extensions have NOT yet been run against a live dump: PLAN.md Phase 1 verifies them.

Reads one or more ``match_player`` parquet shards straight from the deadlock-api data-dump
bucket and produces a slim, one-row-per-shop-purchase feature table. Replaces the old
rate-limited per-match API crawl. See docs/data_dumps.md and PLAN.md §2.

Design notes:
* ``stats.time_stamp_s`` is a shared 180 s grid across a team, so team net worth at a time
  is ``sum(net_worth)`` per (match, team, ts) — cheap. We look up the value at/just-before a
  purchase with an **ASOF JOIN** (fast; no correlated subqueries).
* Shop vs ability split uses the dump's own ``upgrade_items_dict`` (canonical), cross-checked
  by our ItemDecoder for names/slot/tier/cost.
* NOTE: the public matchmaking queue is labelled ``Unranked`` (the ``Ranked`` enum is empty in
  the dumps) but carries full badge data — filter on match_mode + non-null badge for MMR.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from .. import config
from ..assets import ItemDecoder

BUCKET = "https://s3-cache.deadlock-api.com/db-snapshot/public"


def shard_url(n: int) -> str:
    return f"{BUCKET}/match_player/match_player_{n}.parquet"


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL httpfs; LOAD httpfs;")
    return con


def build_purchase_events(
    shards: list[int],
    out_path: str | Path,
    match_modes: tuple[str, ...] = ("Unranked",),
    require_badge: bool = True,
    max_matches: int | None = 20_000,
    con: duckdb.DuckDBPyConnection | None = None,
    decoder: ItemDecoder | None = None,
    source_urls: list[str] | None = None,
    upgrade_dict_url: str | None = None,
) -> Path:
    """Extract shop-purchase events from ``shards`` to a parquet at ``out_path``.

    One row per shop purchase, with: identity (match/account/slot/team/hero), label (won),
    MMR (avg_badge, enemy_avg_badge), timing (game_time_s, sold_time_s), the item
    (id/name/slot/tier/cost), player state at the last snapshot (level/K/D/A), economy
    (net_worth_at_buy, my/enemy team net worth + differential at buy), inventory context
    (n_prior_items), and composition (ally_heroes, enemy_heroes).
    """
    con = con or connect()
    decoder = decoder or ItemDecoder.load()  # cached assets; for names/slot/tier/cost

    # item metadata table for enrichment (id -> name/slot/tier/cost), shop items only
    meta_rows = [
        (i.id, i.name, i.slot, i.tier, i.cost)
        for i in decoder.shop_items()
    ]
    con.execute("CREATE OR REPLACE TEMP TABLE item_meta(item_id UBIGINT, name VARCHAR, "
                "slot VARCHAR, tier INTEGER, cost INTEGER)")
    con.executemany("INSERT INTO item_meta VALUES (?,?,?,?,?)", meta_rows)

    # source_urls / upgrade_dict_url let tests point the same SQL at small local parquet files.
    urls = source_urls or [shard_url(n) for n in shards]
    updict = upgrade_dict_url or f"{BUCKET}/upgrade_items_dict.parquet"
    modes = ", ".join(f"'{m}'" for m in match_modes)
    badge_filter = "AND average_badge_team0 IS NOT NULL" if require_badge else ""
    limit = f"LIMIT {int(max_matches)}" if max_matches else ""

    sql = f"""
    WITH base AS (
        SELECT match_id, account_id, player_slot, team, hero_id, won,
               average_badge_team0, average_badge_team1, duration_s,
               "items.game_time_s" AS buy_t, "items.item_id" AS buy_id,
               "items.sold_time_s" AS sold_t,
               "stats.time_stamp_s" AS st, "stats.net_worth" AS nw,
               "stats.level" AS lvl, "stats.kills" AS kls, "stats.deaths" AS dth,
               "stats.assists" AS ast
        FROM read_parquet({urls!r})
        WHERE match_mode IN ({modes}) {badge_filter}
          AND team IN ('Team0','Team1')
    ),
    sel AS (SELECT DISTINCT match_id FROM base ORDER BY match_id DESC {limit}),
    b AS (SELECT base.* FROM base JOIN sel USING (match_id)),

    -- per-team hero composition
    comp AS (SELECT match_id, team, list(hero_id) AS heroes FROM b GROUP BY 1, 2),

    -- per-player economy snapshots (shared 180s grid)
    snap AS (SELECT match_id, account_id, team, u[1] AS ts, u[2] AS nw,
                    u[3] AS level, u[4] AS kills, u[5] AS deaths, u[6] AS assists
             FROM b, UNNEST(list_zip(st, nw, lvl, kls, dth, ast)) AS _(u)),
    team_ts AS (SELECT match_id, team, ts, sum(nw) AS team_nw
                FROM snap GROUP BY 1, 2, 3),
    -- earliest snapshot per player/team, to backfill buys made before the first (180s) snapshot
    first_snap AS (SELECT match_id, account_id, arg_min(nw, ts) AS nw0 FROM snap GROUP BY 1, 2),
    first_team AS (SELECT match_id, team, arg_min(team_nw, ts) AS tnw0 FROM team_ts GROUP BY 1, 2),

    -- explode purchases, keep shop items only
    purch AS (SELECT match_id, account_id, player_slot, team, hero_id, won,
                     -- a player's skill covariate is THEIR OWN team's avg badge, not always
                     -- team0's (that bug gave every Team1 player team0's badge).
                     CASE WHEN team = 'Team0' THEN average_badge_team0
                          ELSE average_badge_team1 END AS avg_badge,
                     CASE WHEN team = 'Team0' THEN average_badge_team1
                          ELSE average_badge_team0 END AS enemy_avg_badge, duration_s,
                     CASE WHEN team='Team0' THEN 'Team1' ELSE 'Team0' END AS enemy_team,
                     u[1] AS game_time_s, u[2] AS item_id, u[3] AS sold_time_s
              FROM b, UNNEST(list_zip(buy_t, buy_id, sold_t)) AS _(u)),
    shop AS (SELECT p.* FROM purch p JOIN read_parquet('{updict}') d ON d.id = p.item_id),

    -- buyer net worth at buy (ASOF: latest snapshot with ts <= game_time_s)
    e_nw AS (
        SELECT s.*, snp.nw AS net_worth_at_buy, snp.level, snp.kills, snp.deaths, snp.assists
        FROM shop s ASOF LEFT JOIN snap snp
          ON s.match_id = snp.match_id AND s.account_id = snp.account_id
         AND s.game_time_s >= snp.ts
    ),
    -- own-team net worth at buy
    e_my AS (
        SELECT e.*, tt.team_nw AS my_team_nw
        FROM e_nw e ASOF LEFT JOIN team_ts tt
          ON e.match_id = tt.match_id AND e.team = tt.team AND e.game_time_s >= tt.ts
    ),
    -- enemy-team net worth at buy
    e_en AS (
        SELECT e.*, tt.team_nw AS enemy_team_nw
        FROM e_my e ASOF LEFT JOIN team_ts tt
          ON e.match_id = tt.match_id AND e.enemy_team = tt.team AND e.game_time_s >= tt.ts
    ),
    -- inventory context: how many shop items this player already bought before now
    e_prior AS (
        SELECT *, count(*) OVER (
            PARTITION BY match_id, account_id ORDER BY game_time_s
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prior_items
        FROM e_en
    )
    SELECT
        e.match_id, e.account_id, e.player_slot, e.team, e.hero_id, e.won, e.avg_badge,
        e.enemy_avg_badge, e.duration_s, e.game_time_s, e.item_id, e.sold_time_s,
        e.level, e.kills, e.deaths, e.assists,
        m.name AS item_name, m.slot, m.tier, m.cost,
        coalesce(e.net_worth_at_buy, fs.nw0)       AS net_worth_at_buy,
        coalesce(e.my_team_nw, ftm.tnw0)           AS my_team_nw,
        coalesce(e.enemy_team_nw, fte.tnw0)        AS enemy_team_nw,
        coalesce(e.my_team_nw, ftm.tnw0) - coalesce(e.enemy_team_nw, fte.tnw0)
                                                   AS team_nw_diff_at_buy,
        e.n_prior_items,
        list_filter(ally.heroes, h -> h <> e.hero_id) AS ally_heroes,
        enemy.heroes AS enemy_heroes
    FROM e_prior e
    LEFT JOIN item_meta m USING (item_id)
    LEFT JOIN first_snap fs  ON fs.match_id = e.match_id AND fs.account_id = e.account_id
    LEFT JOIN first_team ftm ON ftm.match_id = e.match_id AND ftm.team = e.team
    LEFT JOIN first_team fte ON fte.match_id = e.match_id AND fte.team = e.enemy_team
    LEFT JOIN comp ally  ON ally.match_id = e.match_id AND ally.team = e.team
    LEFT JOIN comp enemy ON enemy.match_id = e.match_id AND enemy.team = e.enemy_team
    ORDER BY e.match_id, e.account_id, e.game_time_s
    """

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY ({sql}) TO '{out.as_posix()}' (FORMAT PARQUET)")
    return out
