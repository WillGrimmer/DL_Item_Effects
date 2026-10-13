# Deadlock API data dumps — reference & working queries

Source: <https://deadlock-api.com/data-dumps> · Bucket:
`https://s3-cache.deadlock-api.com/db-snapshot/public/` · Format: **Parquet** (data) +
**ClickHouse SQL** (schema) · Cadence: **daily** · Access: DuckDB `httpfs` / DuckLake.

This is the **primary training-corpus source** — it replaces crawling the rate-limited
per-match API. See `PLAN.md` §2 for context and caveats.

## Listing the bucket
```bash
curl -s "https://s3-cache.deadlock-api.com/db-snapshot/public/?list-type=2"   # S3 XML
```
Schemas are human-readable: `curl .../public/match_player.sql`.

## The one table that matters: `match_player`
One row per **player-per-match**, 388 columns in parquet, partitioned by
`intDiv(match_id, 1_000_000)` → shards `match_player/match_player_{0..95}.parquet`.
Recent match_ids live in high-numbered shards (5–7 GB each); whole table ≈ 300 GB —
**never pull it all**: prune columns + filter shard/date/mode.

Key list (array) columns actually present in the parquet:
- `items.game_time_s`, `items.item_id`, `items.sold_time_s`, `items.upgrade_id`,
  `items.imbued_ability_id` — purchase arrays (mix shop items + ability-point spends)
- `stats.time_stamp_s`, `stats.net_worth`, `stats.level`, `stats.kills/deaths/assists`, … —
  economy time series (~180 s cadence)
- `won` (label), `average_badge_team0/1` (MMR), `objectives.*`, `mid_boss.*`,
  `death_details.*`, `x_pos`/`y_pos`/`health` (positions)

**NOT in the parquet** (ClickHouse materialized cols — compute yourself, see below):
`items.net_worth_at_buy`, `upgrades.item_id/game_time_s/net_worth_at_buy`.

Helper dicts: `upgrade_items_dict.parquet` (col `id`, 173 shop items),
`ability_items_dict.parquet`, `items.parquet` (`id`,`name`,…).

## Working query — purchase-event table straight from the dump
Builds one row per **shop** purchase with net-worth-at-buy, labels, hero, MMR.
Verified: ~100k events from `match_player_4` (9 MB) in ~3.7 s.

```python
import duckdb
con = duckdb.connect(); con.execute("INSTALL httpfs; LOAD httpfs;")
BASE = "https://s3-cache.deadlock-api.com/db-snapshot/public"
shard  = f"{BASE}/match_player/match_player_4.parquet"   # pick recent shards for MVP
updict = f"{BASE}/upgrade_items_dict.parquet"

q = f"""
WITH mp AS (
  SELECT match_id, account_id, hero_id, won, average_badge_team0,
         "items.game_time_s" AS buy_t, "items.item_id" AS buy_id,
         "stats.time_stamp_s" AS st, "stats.net_worth" AS nw
  FROM read_parquet('{shard}')
  WHERE match_mode IN ('Ranked','Unranked')
),
ev AS (   -- one row per purchase (list_zip fields are positional: u[1],u[2])
  SELECT match_id, account_id, hero_id, won, average_badge_team0,
         u[1] AS game_time_s, u[2] AS item_id, st, nw
  FROM mp, UNNEST(list_zip(buy_t, buy_id)) AS t(u)
),
evn AS ( -- net worth at buy = last stats snapshot with time_stamp_s <= game_time_s
  SELECT * EXCLUDE (st, nw),
         list_aggregate(list_transform(list_zip(st, nw),
            x -> CASE WHEN x[1] <= game_time_s THEN x[2] ELSE NULL END), 'max')
         AS net_worth_at_buy
  FROM ev
)
SELECT e.* FROM evn e
JOIN read_parquet('{updict}') d ON d.id = e.item_id   -- keep SHOP items only
"""
df = con.execute(q).df()
```

## Notes / gotchas
- `list_zip(...)` yields **unnamed** structs → access by index `u[1]`, `u[2]` (not `.field`).
- To build 10-hero ally/enemy **comp** state, group all 12 rows of a `match_id` together
  (they're all in the same shard since partitioning is by match_id).
- Old shards (low match_id) have `average_badge_team0 = NULL` (predate badge tracking) —
  filter to recent shards / non-null badge for MMR bucketing.
- Map match_id ranges → `start_time` to pick a patch window (partition is by id, not date).
- For live/real-time scoring later, use the API (`/v1/matches/{id}/metadata`); the dump is a
  daily static snapshot.
```
