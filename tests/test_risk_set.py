"""Risk-set decision frame: the v1 control-arm flaw must be impossible by construction, and
the Phase 2 rules (visits, price paid, component eligibility) must hold on hand-built rows."""
import numpy as np
import pandas as pd

from dl_item_effects.frame import (add_composition, add_enemy_inventory, decision_frame,
                                   owned_matrix, support_arms)
from dl_item_effects.frame.risk_set import assign_visits, covariate_columns

X, Y, Z, CHEAP, UP = 101, 102, 103, 900, 104     # UP (3200) upgrades from CHEAP (800)


def _events():
    rows = [
        # match, acct, t, item, cost, sold
        (1, 1, 100, X, 3200, 0),        # player 1 buys X early...
        (1, 1, 900, Y, 3200, 0),        # ...so at 900s X is NOT an option for them
        (1, 2, 300, CHEAP, 800, 500),   # player 2 sells the cheap item at 500s
        (1, 2, 900, X, 3200, 0),
        (1, 3, 400, Z, 3200, 950),      # player 3 owns Z until 950s
        (1, 3, 900, Y, 3200, 0),
        (1, 3, 1000, Z, 3200, 0),       # bought again after selling
        (2, 4, 900, X, 3200, 0),
        (2, 4, 900, Y, 3200, 0),        # same visit, two arms at one price: left out
    ]
    df = pd.DataFrame(rows, columns=["match_id", "account_id", "game_time_s", "item_id",
                                     "cost", "sold_time_s"])
    df["won"] = [1, 1, 0, 0, 1, 1, 1, 0, 0]
    df["hero_id"] = 7
    return df


def test_owned_excludes_future_and_sold():
    ev = _events()
    own = owned_matrix(ev, [X, Z, CHEAP])
    p1_late = ev.index[(ev.account_id == 1) & (ev.game_time_s == 900)][0]
    assert own.loc[p1_late, f"own_{X}"] == 1
    p2_late = ev.index[(ev.account_id == 2) & (ev.game_time_s == 900)][0]
    assert own.loc[p2_late, f"own_{CHEAP}"] == 0          # sold at 500 < 900
    p3_y = ev.index[(ev.account_id == 3) & (ev.item_id == Y)][0]
    assert own.loc[p3_y, f"own_{Z}"] == 1                 # held until 950
    p3_rebuy = ev.index[(ev.account_id == 3) & (ev.game_time_s == 1000)][0]
    assert own.loc[p3_rebuy, f"own_{Z}"] == 0
    p4 = ev.index[ev.account_id == 4]
    assert own.loc[p4, f"own_{X}"].sum() == 0


def test_item_sold_or_consumed_at_t_is_still_owned_at_t():
    ev = pd.DataFrame({"match_id": [1, 1], "account_id": [1, 1], "game_time_s": [100, 600],
                       "item_id": [CHEAP, UP], "sold_time_s": [600, 0]})
    own = owned_matrix(ev, [CHEAP])
    assert own.loc[1, f"own_{CHEAP}"] == 1                # consumed by the upgrade at 600


def test_owner_of_x_is_never_eligible_for_x():
    df = decision_frame(_events(), 3200, arms=[X, Y, Z])
    k = df.attrs["arms"].index(X)
    p1 = df[(df.account_id == 1) & (df.item_id == Y)].iloc[0]
    assert p1[f"elig_{k}"] == 0
    assert (df[df.arm == k][f"elig_{k}"] == 1).all()
    # 8 buys at 3200, minus player 4's two-arm visit; the 800 buy is not a 3200 decision
    assert len(df) == 6 and df.attrs["n_multi_arm_visits"] == 1


def test_support_arms_thresholds():
    ev = _events()
    assert set(support_arms(ev, 3200, min_share=0.0, min_buys=2)) == {X, Y, Z}
    assert support_arms(ev, 3200, min_share=0.0, min_buys=3) == [X, Y]


def _visit_events():
    """Player 1: X and the cheap item in one visit at 100/110, then Y alone at 500.
    Player 2: holds CHEAP, so UP would be discounted for them; buys Y at full price.
    Player 3: buys UP while holding CHEAP (discounted, not a trial), then X."""
    rows = [
        (1, 1, 100, X, 3200, 3200, True, 0),
        (1, 1, 110, CHEAP, 800, 800, True, 0),
        (1, 1, 500, Y, 3200, 3200, True, 0),
        (1, 2, 50, CHEAP, 800, 800, True, 0),
        (1, 2, 700, Y, 3200, 3200, True, 0),
        (1, 2, 1400, X, 3200, 3200, True, 0),
        (1, 3, 60, CHEAP, 800, 800, True, 300),
        (1, 3, 300, UP, 3200, 2400, False, 0),
        (1, 3, 800, X, 3200, 3200, True, 0),
    ]
    df = pd.DataFrame(rows, columns=["match_id", "account_id", "game_time_s", "item_id",
                                     "cost", "price_paid", "full_price", "sold_time_s"])
    df["won"] = [1, 1, 1, 0, 0, 0, 1, 1, 1]
    df["net_worth_at_buy"] = df["game_time_s"] * 10           # stands in for snapshot state
    df["assigned_lane"] = [1, 1, 1, 4, 4, 4, 6, 6, 6]
    return df


def test_visits_group_purchases_within_gap():
    ev = assign_visits(_visit_events(), gap_s=30)
    p1 = ev[ev.account_id == 1]
    assert list(p1.visit_id) == [1, 1, 2] and list(p1.visit_start) == [100, 100, 500]


def test_component_holder_cannot_choose_the_upgrade_at_full_price():
    comps = {UP: (CHEAP,)}
    df = decision_frame(_visit_events(), 3200, arms=[X, Y, UP], components=comps)
    k_up = df.attrs["arms"].index(UP)
    p2 = df[(df.account_id == 2) & (df.item_id == Y)].iloc[0]
    assert p2[f"elig_{k_up}"] == 0                        # holds CHEAP: UP is not 3200 for them
    assert not ((df.account_id == 3) & (df.item_id == UP)).any()   # discounted: not a trial
    p3 = df[(df.account_id == 3) & (df.item_id == X)].iloc[0]
    assert p3[f"elig_{k_up}"] == 0                        # owns UP by then


def test_state_comes_from_the_start_of_the_visit_and_crossover_times():
    df = decision_frame(_visit_events(), 3200, arms=[X, Y], components={})
    p1 = df[(df.account_id == 1) & (df.item_id == X)].iloc[0]
    assert p1.game_time_s == 100 and p1.net_worth_at_buy == 1000
    kx, ky = df.attrs["arms"].index(X), df.attrs["arms"].index(Y)
    assert p1[f"next_{ky}"] == 400                        # bought Y 400 s later
    assert np.isnan(p1[f"next_{kx}"])                     # the chosen arm itself
    p2 = df[(df.account_id == 2) & (df.item_id == Y)].iloc[0]
    assert p2[f"next_{kx}"] == 700
    assert (p2.lane_4, p2.lane_1) == (1, 0)
    assert {"lane_1", "lane_4", "lane_6", "net_worth_at_buy"} <= set(covariate_columns(df))


def test_enemy_inventory_counts_owned_items_at_decision_time():
    dec = pd.DataFrame({"match_id": [1, 1], "team": ["Team0", "Team0"],
                        "game_time_s": [500, 1000]})
    ev = pd.DataFrame({
        "match_id": [1, 1, 1, 1], "team": ["Team1", "Team1", "Team1", "Team0"],
        "item_id": [X, X, Y, X], "game_time_s": [100, 400, 600, 100],
        "sold_time_s": [0, 800, 0, 0]})
    out = add_enemy_inventory(dec, ev, items=[X, Y])
    assert list(out[f"enemy_{X}"]) == [2, 1] and list(out[f"enemy_{Y}"]) == [0, 1]
    assert out.attrs["enemy_cols"] == [f"enemy_{X}", f"enemy_{Y}"]


def test_composition_multi_hot():
    df = pd.DataFrame({"ally_heroes": [[1, 2], [3]], "enemy_heroes": [[3, 4], [1, 2]]})
    out = add_composition(df, [1, 2, 3, 4])
    assert list(out.ally_h1) == [1, 0] and list(out.enemy_h1) == [0, 1]
    assert list(out.enemy_h4) == [1, 0] and len(out.attrs["comp_cols"]) == 8
