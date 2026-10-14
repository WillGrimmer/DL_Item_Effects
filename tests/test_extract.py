"""Run the real extraction SQL against a tiny hand-built match_player parquet.

The v1 project never tested its SQL, and that SQL is where its biases lived. The fixture is
one complete Ranked match with known purchases, sells, component upgrades and snapshots, so
every derived column can be checked by hand. Around it sit matches that must be filtered out
(11 players; Street Brawl) and an earlier match that only feeds the previous-match link.
"""
from datetime import date

import pandas as pd
import pytest

from dl_item_effects.assets import ItemDecoder
from dl_item_effects.frame.risk_set import STATE_COLUMNS
from dl_item_effects.ingest import extract

FIX = "tests/fixtures/assets_items_sample.json"
GOOSE, MONSTER, QUICK, CULTIST, LIFESTEAL = (2462046703, 1009965641, 84321454, 709540378,
                                             876563814)
ABILITY = 1999680326
PATCHES = [("09-16-2026", date(2026, 9, 16)), ("09-29-2026", date(2026, 9, 29))]
T = pd.Timestamp


def _player(match, acct, slot, team, hero, won, rank, buys=(), snaps=None, mode="Ranked",
            game="Normal", start="2026-10-01 12:00", streak=0, calib=0, lane=1, build=None,
            abandon=0):
    """buys: (t, item, sold, flags); snaps: (ts, nw, level, k, d, a)."""
    snaps = snaps if snaps is not None else [(180, 900, 3, 0, 0, 0), (360, 2000, 5, 1, 1, 1),
                                             (540, 3500, 7, 1, 2, 2)]
    return {
        "match_id": match, "account_id": acct, "player_slot": slot, "team": team,
        "hero_id": hero, "won": won, "start_time": T(start, tz="UTC"), "duration_s": 1800,
        "match_mode": mode, "game_mode": game, "average_badge": 95 if mode == "Ranked" else None,
        "player_rank_initial_display_rank": rank if mode == "Ranked" else None,
        "player_rank_initial_calibration_games": calib, "player_rank_initial_win_streak": streak,
        "assigned_lane": lane, "hero_build_id": build, "abandon_match_time_s": abandon,
        "items.game_time_s": [b[0] for b in buys], "items.item_id": [b[1] for b in buys],
        "items.sold_time_s": [b[2] for b in buys], "items.flags": [b[3] for b in buys],
        "stats.time_stamp_s": [s[0] for s in snaps], "stats.net_worth": [s[1] for s in snaps],
        "stats.level": [s[2] for s in snaps], "stats.kills": [s[3] for s in snaps],
        "stats.deaths": [s[4] for s in snaps], "stats.assists": [s[5] for s in snaps],
    }


def _rows():
    rows = [
        # match 0: earlier Ranked match, one row only. Feeds account 11's previous match.
        _player(0, 11, 0, "Team0", 7, False, 96, start="2026-10-01 10:00"),
        # match 3: Unranked Street Brawl; account 12's game here must not count as "previous".
        _player(3, 12, 0, "Team0", 8, True, None, buys=[(100, GOOSE, 0, 0)], mode="Unranked",
                game="StreetBrawl", start="2026-10-01 11:00"),
    ]
    # match 1: the complete Ranked match, 2026-10-01 12:00 UTC
    rows += [
        _player(1, 11, 0, "Team0", 7, True, 96, streak=2, build=123, buys=[
            (100, GOOSE, 0, 0),          # before the first snapshot
            (200, ABILITY, 0, 0),        # ability point, not a purchase
            (250, MONSTER, 700, 1),      # consumed by Cultist Sacrifice at 700
            (400, QUICK, 0, 0),
            (700, CULTIST, 0, 0),        # pays 3200 - 800
        ], snaps=[(180, 1000, 3, 1, 0, 0), (360, 2500, 6, 2, 1, 1), (540, 4000, 8, 3, 1, 2)]),
        _player(1, 12, 1, "Team0", 8, True, 0, calib=4, lane=4, buys=[(150, MONSTER, 0, 0)]),
        _player(1, 13, 2, "Team0", 9, True, 94, buys=[
            (600, MONSTER, 600, 1),      # same-click component row: not a decision
            (600, CULTIST, 0, 0),        # bought outright, full price
        ]),
        _player(1, 14, 3, "Team0", 10, True, 95),
        _player(1, 15, 4, "Team0", 11, True, 95),
        _player(1, 16, 5, "Team0", 12, True, 96),
        _player(1, 21, 6, "Team1", 13, False, 101, buys=[
            (120, GOOSE, 0, 0), (390, QUICK, 0, 0), (390, LIFESTEAL, 0, 0)]),  # same-second pair
        _player(1, 22, 7, "Team1", 14, False, 96),
        _player(1, 23, 8, "Team1", 15, False, 95),
        _player(1, 24, 9, "Team1", 16, False, 95),
        _player(1, 25, 10, "Team1", 17, False, 94),
        _player(1, 26, 11, "Team1", 18, False, 93, abandon=1500),
    ]
    # match 2: Ranked but only 11 player rows, so excluded
    rows += [_player(2, 100 + i, i, "Team0" if i < 6 else "Team1", 30 + i, i < 6, 95,
                     buys=[(300, QUICK, 0, 0)], start="2026-10-01 13:00") for i in range(11)]
    return rows


@pytest.fixture(scope="module")
def events(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("lake")
    con = extract.connect()
    con.register("df", pd.DataFrame(_rows()))
    shard = tmp / "mp.parquet"
    con.execute(f"""COPY (SELECT * REPLACE (
            "items.game_time_s"::UINTEGER[] AS "items.game_time_s",
            "items.item_id"::UINTEGER[] AS "items.item_id",
            "items.sold_time_s"::UINTEGER[] AS "items.sold_time_s",
            "items.flags"::UINTEGER[] AS "items.flags",
            average_badge::UINTEGER AS average_badge,
            player_rank_initial_display_rank::UINTEGER AS player_rank_initial_display_rank,
            hero_build_id::UBIGINT AS hero_build_id)
        FROM df) TO '{shard.as_posix()}' (FORMAT PARQUET)""")
    hist = extract.build_history([shard.as_posix()], tmp / "hist.parquet", con=con)
    out = extract.build_purchase_events([shard.as_posix()], tmp / "events.parquet",
                                        history=hist, patches=PATCHES, con=con,
                                        decoder=ItemDecoder.from_json(FIX))
    return pd.read_parquet(out)


def _row(ev, acct, item):
    return ev[(ev.account_id == acct) & (ev.item_id == item)].iloc[0]


def test_filters_and_row_count(events):
    assert set(events.match_id) == {1}          # 11-player and Street Brawl matches dropped
    assert ABILITY not in set(events.item_id)
    # 11: Goose, Monster, Quicksilver, Cultist; 12: Monster; 13: Cultist; 21: three items
    assert len(events) == 9


def test_same_click_component_row_is_dropped(events):
    p13 = events[events.account_id == 13]
    assert list(p13.item_id) == [CULTIST]
    assert p13.iloc[0].price_paid == 3200 and p13.iloc[0].full_price
    assert p13.iloc[0].n_prior_items == 0


def test_price_paid_after_component_upgrade(events):
    r = _row(events, 11, CULTIST)
    assert (r.cost, r.price_paid, r.full_price) == (3200, 2400, False)
    m = _row(events, 11, MONSTER)
    assert m.consumed_by_upgrade and m.sold_time_s == 700
    assert not _row(events, 11, QUICK).consumed_by_upgrade


def test_start_of_game_state_before_first_snapshot(events):
    r = _row(events, 11, GOOSE)
    assert (r.net_worth_at_buy, r.level, r.kills, r.deaths, r.assists) == (600, 1, 0, 0, 0)
    assert (r.my_team_nw, r.enemy_team_nw, r.team_nw_diff_at_buy) == (3600, 3600, 0)
    assert r.snapshot_age_s == 100
    assert pd.isna(r.nc_prior_souls) and pd.isna(r.nc_prior_deaths)


def test_snapshot_state_and_team_net_worth(events):
    r = _row(events, 11, QUICK)                  # bought at 400 -> snapshot 360
    assert (r.net_worth_at_buy, r.level, r.kills, r.deaths, r.assists) == (2500, 6, 2, 1, 1)
    assert r.snapshot_age_s == 40
    assert (r.my_team_nw, r.enemy_team_nw) == (2500 + 5 * 2000, 6 * 2000)
    c = _row(events, 11, CULTIST)                # bought at 700 -> snapshot 540
    assert (c.net_worth_at_buy, c.snapshot_age_s) == (4000, 160)


def test_negative_control_prior_interval(events):
    m = _row(events, 11, MONSTER)                # snapshot 180, previous is the t=0 state
    assert (m.nc_prior_souls, m.nc_prior_deaths, m.nc_prior_len_s) == (1000 - 600, 0, 180)
    q = _row(events, 11, QUICK)                  # snapshot 360, previous 180
    assert (q.nc_prior_souls, q.nc_prior_deaths, q.nc_prior_len_s) == (1500, 1, 180)


def test_prior_items_strictly_earlier(events):
    assert list(events[events.account_id == 11].n_prior_items) == [0, 1, 2, 3]
    p21 = events[events.account_id == 21].set_index("item_id")
    assert p21.loc[QUICK, "n_prior_items"] == 1 and p21.loc[LIFESTEAL, "n_prior_items"] == 1


def test_skill_columns(events):
    r = _row(events, 11, GOOSE)
    assert r.badge_level == 6 * 9 + 5 and r.rank_level == 6 * 9 + 6
    assert (r.win_streak, r.calibration_games, r.assigned_lane) == (2, 0, 1)
    assert r.has_build and not r.in_calibration
    team = (60 + 58 + 59 + 59 + 60) / 5          # account 12 is in calibration: excluded
    enemy = (61 + 60 + 59 + 59 + 58 + 57) / 6
    assert r.team_rank_mean == pytest.approx(team)
    assert r.enemy_rank_mean == pytest.approx(enemy)
    assert r.own_minus_team == pytest.approx(60 - team)
    assert r.team_rank_gap == pytest.approx(team - enemy)
    c = _row(events, 12, MONSTER)
    assert c.in_calibration and pd.isna(c.rank_level) and not c.has_build
    assert c.assigned_lane == 4


def test_patch_and_abandon(events):
    r = events.iloc[0]
    assert (r.patch, r.days_since_patch) == ("09-29-2026", 2)
    assert events.match_has_abandon.all()


def test_previous_ranked_match(events):
    r = _row(events, 11, GOOSE)
    assert r.nc_prev_won == False and r.prev_gap_h == pytest.approx(2.0)  # noqa: E712
    assert pd.isna(_row(events, 12, MONSTER).nc_prev_won)   # Street Brawl does not count


def test_composition(events):
    r = _row(events, 11, GOOSE)
    assert list(r.ally_heroes) == [8, 9, 10, 11, 12]
    assert list(r.enemy_heroes) == [13, 14, 15, 16, 17, 18]


def test_negative_controls_are_never_covariates(events):
    assert not [c for c in STATE_COLUMNS if c.startswith("nc_") or c == "duration_s"]
    assert set(STATE_COLUMNS) <= set(events.columns)
