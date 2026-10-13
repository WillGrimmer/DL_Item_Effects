"""Risk-set decision frame: the v1 control-arm flaw must be impossible by construction."""
import pandas as pd

from dl_item_effects.frame import decision_frame, owned_matrix, support_arms

X, Y, Z, CHEAP = 101, 102, 103, 900


def _events():
    rows = [
        # match, acct, t, item, cost, sold
        (1, 1, 100, X, 3200, 0),        # player 1 buys X early...
        (1, 1, 900, Y, 3200, 0),        # ...so at 900s X is NOT an option for them
        (1, 2, 300, CHEAP, 800, 500),   # player 2 sells the cheap item at 500s
        (1, 2, 900, X, 3200, 0),
        (1, 3, 400, Z, 3200, 950),      # player 3 owns Z until 950s
        (1, 3, 900, Y, 3200, 0),
        (1, 3, 1000, Z, 3200, 0),       # re-buy after selling: Z eligible again
        (2, 4, 900, X, 3200, 0),
        (2, 4, 900, Y, 3200, 0),        # same-second buys do not see each other
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


def test_owner_of_x_is_never_eligible_for_x():
    df = decision_frame(_events(), 3200, arms=[X, Y, Z])
    k = df.attrs["arms"].index(X)
    p1 = df[(df.account_id == 1) & (df.item_id == Y)].iloc[0]
    assert p1[f"elig_{k}"] == 0
    assert (df[df.arm == k][f"elig_{k}"] == 1).all()
    assert len(df) == 8                                   # the 800-cost buy is not a 3200 decision


def test_support_arms_thresholds():
    ev = _events()
    assert set(support_arms(ev, 3200, min_share=0.0, min_buys=2)) == {X, Y, Z}
    assert support_arms(ev, 3200, min_share=0.0, min_buys=3) == [X, Y]
