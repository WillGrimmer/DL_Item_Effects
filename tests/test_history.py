"""Player-history covariates: strictly earlier matches only, tie order, the cap, 24 h window."""
import numpy as np
import pandas as pd
import pytest

from dl_item_effects.frame import (HISTORY_COLUMNS, add_player_history, covariate_columns,
                                   history_features)

T0 = pd.Timestamp("2026-10-01 00:00", tz="UTC")


def _hist():
    rows = []
    # account 1: 14 matches, one every 2 hours; wins on even indices
    for i in range(14):
        rows.append((1, 100 + i, T0 + pd.Timedelta(hours=2 * i), i % 2 == 0))
    # account 2: two matches at the same second (match_id breaks the tie), then one 23 h
    # later and one 25 h after the first pair
    rows += [(2, 202, T0, True), (2, 201, T0, False),
             (2, 203, T0 + pd.Timedelta(hours=23), True),
             (2, 204, T0 + pd.Timedelta(hours=25), False)]
    return pd.DataFrame(rows, columns=["account_id", "match_id", "start_time", "won"])


@pytest.fixture(scope="module")
def feats():
    return history_features(_hist()).set_index(["account_id", "match_id"])


def test_first_match_has_no_history(feats):
    r = feats.loc[(1, 100)]
    assert np.isnan(r.prev_won) and np.isnan(r.wr_2_6) and np.isnan(r.wr_2_11)
    assert r.n_hist == 0 and r.games_24h == 0
    assert r.nc_next_won == 0.0


def test_previous_result_and_two_back_windows(feats):
    r = feats.loc[(1, 101)]          # one earlier match: no "two back" yet
    assert r.prev_won == 1.0 and np.isnan(r.wr_2_6) and r.n_hist == 1
    r = feats.loc[(1, 102)]          # two earlier: match 100 is two back
    assert r.prev_won == 0.0 and r.wr_2_6 == 1.0 and r.wr_2_11 == 1.0 and r.n_hist == 2
    r = feats.loc[(1, 113)]          # index 13: previous is 12 (won)
    assert r.prev_won == 1.0
    # matches 2..6 back are indices 11..7: wins at 10 and 8 -> 2/5
    assert r.wr_2_6 == pytest.approx(2 / 5)
    # matches 2..11 back are indices 11..2: wins at 10, 8, 6, 4, 2 -> 5/10
    assert r.wr_2_11 == pytest.approx(5 / 10)
    assert r.n_hist == 11            # 13 earlier, capped
    assert np.isnan(r.nc_next_won)   # last match


def test_current_match_never_counts(feats):
    # if the current result leaked in, wr_2_6 at index 6 (won) would include it
    r = feats.loc[(1, 106)]          # matches 2..6 back: indices 4..0, wins at 4, 2, 0
    assert r.wr_2_6 == pytest.approx(3 / 5)


def test_games_in_24h(feats):
    assert feats.loc[(1, 113)].games_24h == 12   # 24, 22, ..., 2 hours back (24 h included)
    assert feats.loc[(1, 112)].games_24h == 12   # 24, 22, ..., 2 hours back
    assert feats.loc[(1, 101)].games_24h == 1


def test_ties_order_by_match_id_and_peers_do_not_count(feats):
    a, b = feats.loc[(2, 201)], feats.loc[(2, 202)]
    assert np.isnan(a.prev_won) and b.prev_won == 0.0   # 201 sorts first
    assert a.games_24h == 0 and b.games_24h == 0        # same-second peer is not earlier
    assert feats.loc[(2, 203)].games_24h == 2           # both, 23 h back
    assert feats.loc[(2, 204)].games_24h == 1           # only 203 (2 h back)
    assert feats.loc[(2, 204)].n_hist == 3


def test_accounts_filter():
    f = history_features(_hist(), accounts=[2])
    assert set(f["account_id"]) == {2}


def test_add_to_frame(feats):
    df = pd.DataFrame({"match_id": [113, 999], "account_id": [1, 1], "prev_gap_h": [2.0, np.nan],
                       "level": [5, 6]})
    df.attrs = {"arms": [], "own_cols": []}
    out = add_player_history(df, feats.reset_index())
    assert out.loc[0, "prev_won"] == 1.0 and out.loc[0, "n_hist"] == 11
    assert np.isnan(out.loc[1, "prev_won"]) and out.loc[1, "n_hist"] == 0
    cols = covariate_columns(out)
    assert set(HISTORY_COLUMNS) <= set(cols)
    assert "nc_next_won" not in cols and "nc_prev_won" not in cols
