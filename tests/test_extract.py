"""Run the real extraction SQL against a tiny hand-built match_player parquet.

The v1 project never tested its SQL, and that SQL is where its biases lived. This fixture is
one 4-player match with known purchases, sells and snapshots, so every derived column can be
checked by hand.
"""
import duckdb
import pandas as pd
import pytest

from dl_item_effects.assets import ItemDecoder
from dl_item_effects.ingest import extract

FIX = "tests/fixtures/assets_items_sample.json"
A800, B800, C1600 = 2462046703, 1009965641, 84321454
ABILITY = 1999680326


@pytest.fixture()
def tiny_dump(tmp_path):
    con = duckdb.connect()
    rows = [
        # match, acct, slot, team, hero, won, buys(t,id,sold), snapshots(ts,nw,lvl,k,d,a)
        (1, 11, 0, "Team0", 7, True,
         [(100, A800, 0), (200, ABILITY, 0), (400, B800, 700), (500, C1600, 0)]),
        (1, 12, 1, "Team0", 8, True, [(150, B800, 0)]),
        (1, 21, 2, "Team1", 9, False, [(120, A800, 0), (390, C1600, 0)]),
        (1, 22, 3, "Team1", 10, False, []),
    ]
    snaps = {11: [(180, 1000, 3, 1, 0, 0), (360, 2500, 6, 2, 1, 1)],
             12: [(180, 900, 3, 0, 0, 1), (360, 2000, 5, 0, 1, 2)],
             21: [(180, 800, 2, 0, 1, 0), (360, 1500, 4, 1, 2, 0)],
             22: [(180, 700, 2, 0, 0, 0), (360, 1200, 4, 0, 1, 0)]}
    recs = []
    for m, a, slot, team, hero, won, buys in rows:
        sn = snaps[a]
        recs.append({
            "match_id": m, "account_id": a, "player_slot": slot, "team": team,
            "hero_id": hero, "won": won, "match_mode": "Unranked",
            "average_badge_team0": 95, "average_badge_team1": 101, "duration_s": 1800,
            "items.game_time_s": [b[0] for b in buys], "items.item_id": [b[1] for b in buys],
            "items.sold_time_s": [b[2] for b in buys],
            "stats.time_stamp_s": [s[0] for s in sn], "stats.net_worth": [s[1] for s in sn],
            "stats.level": [s[2] for s in sn], "stats.kills": [s[3] for s in sn],
            "stats.deaths": [s[4] for s in sn], "stats.assists": [s[5] for s in sn]})
    df = pd.DataFrame(recs)
    shard = tmp_path / "mp.parquet"
    con.register("df", df)
    con.execute(f"""COPY (SELECT * REPLACE (
            "items.game_time_s"::UINTEGER[] AS "items.game_time_s",
            "items.item_id"::UBIGINT[] AS "items.item_id",
            "items.sold_time_s"::UINTEGER[] AS "items.sold_time_s")
        FROM df) TO '{shard}' (FORMAT PARQUET)""")
    updict = tmp_path / "updict.parquet"
    con.execute(f"COPY (SELECT unnest([{A800}, {B800}, {C1600}])::UBIGINT AS id) "
                f"TO '{updict}' (FORMAT PARQUET)")
    out = tmp_path / "events.parquet"
    extract.build_purchase_events([], out, con=con, decoder=ItemDecoder.from_json(FIX),
                                  source_urls=[str(shard)], upgrade_dict_url=str(updict))
    return pd.read_parquet(out)


def test_shop_only_and_row_count(tiny_dump):
    assert ABILITY not in set(tiny_dump["item_id"])
    assert len(tiny_dump) == 6


def test_badges_are_own_and_enemy_team(tiny_dump):
    t0 = tiny_dump[tiny_dump.team == "Team0"].iloc[0]
    t1 = tiny_dump[tiny_dump.team == "Team1"].iloc[0]
    assert (t0.avg_badge, t0.enemy_avg_badge) == (95, 101)
    assert (t1.avg_badge, t1.enemy_avg_badge) == (101, 95)


def test_sold_time_and_snapshot_state(tiny_dump):
    r = tiny_dump[(tiny_dump.account_id == 11) & (tiny_dump.item_id == B800)].iloc[0]
    assert r.sold_time_s == 700
    # bought at 400s -> last snapshot is 360s
    assert (r.net_worth_at_buy, r.level, r.kills, r.deaths) == (2500, 6, 2, 1)
    assert r.n_prior_items == 1          # only the 800 at 100s; the ability spend is excluded


def test_pre_first_snapshot_buy_is_backfilled(tiny_dump):
    r = tiny_dump[(tiny_dump.account_id == 11) & (tiny_dump.item_id == A800)].iloc[0]
    assert r.net_worth_at_buy == 1000     # bought at 100s, before the 180s snapshot
    assert r.team_nw_diff_at_buy == (1000 + 900) - (800 + 700)
