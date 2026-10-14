# Deadlock API data lake: reference and working query

Source: <https://deadlock-api.com/data-dumps>. Verified 2026-10-13 against manifest version 692.
Numbers are from partition 116 (Normal games, 25k matches, 5.0M shop rows) unless another
partition is named.

**The layout this file used to describe is gone.** `s3-cache.deadlock-api.com/db-snapshot/public/`
no longer resolves, there is no `upgrade_items_dict` table, and `match_player` has 157 columns
instead of 388. `ingest/extract.py` still points at the old bucket.

## Where the data is
- **Manifest:** `https://data.deadlock-api.com/v1/manifest.json`, regenerated hourly. Each table
  lists its columns (with comments) and its files. File URL = `https://data.deadlock-api.com/` +
  `files[].key`.
- **`match_player`:** one row per player per match. `kind: "base"` is one parquet per partition
  (`match_id // 1_000_000`, 0 to 116 today). `kind: "delta"` files are hourly increments that get
  folded into the base files. Read base files only, or dedupe on `(match_id, account_id)`.
- **Size:** a full partition is 2 to 5 GB and covers about 2 days now (5 to 6 days in August).
  Never `SELECT *`. A
  column-pruned aggregate over a 3 GB partition takes about 10 s; unnesting the item arrays of
  one 0.5 GB partition takes about 30 s.
- **Other tables:** `leaderboard`, `hero_leaderboard`, `match_salts`, `steam_profiles`,
  `steam_profile_observed_names`. None are needed.
- **Shop items:** `https://api.deadlock-api.com/v1/assets/items`, entries with
  `type == "upgrade"` (250). Shop tiers cost 800 / 1600 / 3200 / 6400, with 23 / 43 / 46 / 44
  `shopable` items. Tier 5 entries (cost 9999, such as Frostbite Charm) are bought only in Street
  Brawl and never appear in Normal games. `component_items` lists what an item upgrades from (65
  entries have one).
- **Patches:** `https://api.deadlock-api.com/v1/patches`. The `title` is the patch date; `pub_date`
  moves when hotfixes are appended to the thread.
- A DuckLake catalog also exists (`v1/catalog/<version>.ducklake`, built with DuckDB 1.5.6). We
  read the parquet files directly, which works on DuckDB 1.4.

## Which rows to use
- `game_mode = 'Normal'`. Without this, 27% of "Unranked" matches are 4v4 Street Brawl.
- `match_mode = 'Ranked'` for anything that needs a skill covariate (see below).
- Drop matches without exactly 12 player rows (0.7% of Ranked) and decide what to do with
  matches that have an abandon (`abandon_match_time_s > 0`, 5% of Ranked matches; identical to
  `player_match_outcome = 'Penalized'`).
- The `not_scored` flag is almost always false. `player_match_outcome = 'NotScored'` is the real
  marker (1.7% of Ranked players).

## Skill columns changed on 2026-07-30
Ranked mode launched that day. Since then:

| | Unranked | Ranked |
|---|---|---|
| `average_badge_team0/1` | 0 (not NULL) | 0 (not NULL) |
| `average_badge` (one value per match) | NULL | populated |
| `player_rank_initial_display_rank` | NULL | populated; 0 for the 5.5% still in calibration |

Before 2026-07-30 Unranked rows had team badges (mean about 60) and nothing per player.
`average_badge_team0 IS NOT NULL` therefore filters nothing today.

Imputing a rank onto Unranked players from their own Ranked games covers 22% of player rows, and
0.6% of matches have 10 or more of 12 players covered. Unranked is not usable for adjusted
estimates.

Ranked lobbies are narrow. On a linear scale of 66 rank levels (`(tier-1)*6 + (sub-1)` from the
`tier*10+sub` code; SD across all players 16.9), the within-team SD is 1.6 levels and the gap
between the two teams' mean rank is 0.3 levels. `average_badge` correlates 0.999 with the mean
of the 12 player ranks. A player's own rank adds about 1.2 to 1.6 levels beyond it.

`party` is always NULL. `hero_build_id` is present for about 11% of Ranked players.
`assigned_lane` is always present (values 1, 4, 6).

## Item arrays
`items.game_time_s`, `items.item_id`, `items.sold_time_s`, `items.flags` are parallel arrays.
They also contain ability-point spends; keep ids that are shop items.

- **Not time-sorted** for 14% of players. Always sort by `game_time_s`.
- **One row per (player, item).** No item id ever appears twice for a player, in seven
  partitions checked. A sell and re-buy cannot be seen.
- **`sold_time_s = 0` means never sold.** It is never NULL and never earlier than the purchase.
- **`flags = 1` means consumed by an upgrade.** When a player buys upgrade U while holding its
  component C, C's row gets `sold_time_s` = U's purchase time and `flags = 1`. This is 32% of all
  shop rows. Real sales are 5%.
- **30.5% of all shop purchases paid less than list price**, because a held component was
  consumed (74% of purchases of items that have a component). The price actually paid is not in
  the data; infer it as list cost minus the consumed component's cost.
- **Same-click component rows are not decisions.** Buying U without holding C writes a C row
  with `game_time_s = sold_time_s` and `flags = 1` in the same second (69k rows). Drop them.
- "Owned just before time t" must be `bought < t AND (sold = 0 OR sold >= t)`. With `sold > t`
  the consumed component disappears from the inventory at the moment of the upgrade.
- `items.upgrade_id` and `items.upgrade_info` are constant on shop rows. `items.imbued_ability_id`
  is set on nine items that target an ability.

## Stats snapshots
`stats.*` arrays share `stats.time_stamp_s`. The grid is the same for every player in a match:
180, 360, 540, 720, 900, then every 300 s, plus a final snapshot at exactly `duration_s`. State
looked up at a purchase is up to 179 s stale early and up to 299 s stale later. 7.4% of shop
purchases come before the first snapshot. The final snapshot is the end of the match, so never
use "latest snapshot" without the `ts <= t` condition.

Also in the table now: `objectives.destroyed_time_s`, `mid_boss.destroyed_time_s`, and about 40
more `stats.*` series (souls by source, damage, healing, `max_health`, `weapon_power`,
`tech_power`).

## Volume (Ranked, Normal)
About 15k to 16k matches a day (range 14k to 18k). The two most recent patch windows:

| Patch | Days used | Matches | Player-matches per hero: median / min / max |
|---|---|---|---|
| 09-16-2026 | 09-17 to 09-28 | 176k | 56k / 22k (Mirage) / 112k (Haze) |
| 09-29-2026 | 09-30 to 10-12 | 205k | 57k / 21k / 152k (Rat King) |

Infernus, Lash and Bebop sit near 90k to 97k per window.

## Extracting a patch window
`src/dl_item_effects/ingest/` does this end to end: `lake.py` reads the manifest and finds the
partitions covering a date range, `patches.py` turns `/v1/patches` into windows, and
`extract.py` writes one parquet per partition plus `_history.parquet` and `_sources.json`:

```bash
python -m dl_item_effects.ingest.extract --patch 09-29-2026 --end 2026-10-12
```

The 09-16-2026 window (09-17 to 09-28) is in partitions 106 to 108. The 09-29-2026 window
(09-30 to 10-12) is in partitions 108 to 115. Output is about 0.5 GB per 100k matches.

## Working query
Shop-purchase rows for one partition, Ranked and Normal only. Runs in about 30 s on the newest
partition (2.3M rows before the shop-item filter).

```python
import json, urllib.request
import duckdb

BASE = "https://data.deadlock-api.com/"
req = urllib.request.Request(BASE + "v1/manifest.json", headers={"User-Agent": "dl-item-effects"})
man = json.load(urllib.request.urlopen(req, timeout=30))
base = {f["partition"]: BASE + f["key"]
        for f in man["tables"]["match_player"]["files"] if f["kind"] == "base"}

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
q = f"""
WITH mp AS (
  SELECT match_id, account_id, hero_id, won, start_time, average_badge,
         player_rank_initial_display_rank AS own_rank,
         "items.game_time_s" AS t, "items.item_id" AS id,
         "items.sold_time_s" AS sold, "items.flags" AS flags
  FROM read_parquet('{base[max(base)]}')
  WHERE match_mode = 'Ranked' AND game_mode = 'Normal'
)
SELECT match_id, account_id, hero_id, won, start_time, average_badge, own_rank,
       u[1] AS game_time_s, u[2] AS item_id, u[3] AS sold_time_s, u[4] = 1 AS consumed_by_upgrade
FROM mp, UNNEST(list_zip(t, id, sold, flags)) AS _(u)
WHERE NOT (u[4] = 1 AND u[3] = u[1])     -- same-click component rows are not decisions
"""
df = con.execute(q).df()                  # then keep item_id in the shop-item ids from the assets API
```

## Gotchas
- **File URLs go stale within about two hours.** Recent partitions are rebuilt roughly hourly
  under a new key and the old file then returns 404. Never cache a file URL; re-read the
  manifest. `extract_window` retries with a fresh manifest and records the file actually
  used per partition in `_sources.json`.
- Column names contain dots and must be double-quoted: `"items.item_id"`.
- `list_zip(...)` yields unnamed structs. Access fields by index: `u[1]`, `u[2]`.
- The rank and badge columns are `UInt32`. Cast before subtracting or DuckDB raises an overflow.
- All 12 rows of a match are in the same partition, so team composition can be built per file.
- The newest partition is still filling, and the most recent day can look low.
