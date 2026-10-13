"""Decision frame for sequential target-trial emulation (PLAN.md §3.1).

Every shop purchase is a *decision*: the player has chosen to spend ``cost`` souls right now
and picks ONE item at that price. We emulate a trial at each decision:

* **Population**: decisions by this hero at this cost.
* **Arms**: the on-support items at that cost (enough buys to estimate).
* **Eligibility**: an arm is available to a player only if they do not currently own it.
  Owned = bought strictly before ``t`` and not sold by ``t``. This is the fix for the v1 flaw
  where players who already owned X sat in X's control group.
* **Outcome**: match win.

Covariates are the player's state at the decision (economy, level, K/D/A, badges, team net
worth gap, time) plus what they already own (``own_<item_id>`` multi-hot). Composition is
added by :func:`add_composition`.

Pure pandas so it is testable on hand-built fixtures; run it per hero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATE_COLUMNS = [
    "avg_badge", "enemy_avg_badge", "net_worth_at_buy", "my_team_nw", "enemy_team_nw",
    "team_nw_diff_at_buy", "n_prior_items", "level", "kills", "deaths", "assists",
    "game_time_s",
]


def owned_matrix(events: pd.DataFrame, items: list[int]) -> pd.DataFrame:
    """For each purchase row, which of ``items`` the player owns just before that purchase.

    Owned at t means bought at time < t and (never sold, or sold at time > t). Ties in
    ``game_time_s`` (two buys in the same second) do not count as owned, so simultaneous
    buys never see each other.
    """
    ev = events[["match_id", "account_id", "game_time_s"]].copy()
    ev["_row"] = np.arange(len(events))
    out = np.zeros((len(events), len(items)), dtype=np.uint8)
    item_pos = {int(i): k for k, i in enumerate(items)}

    buys = events[events["item_id"].isin(item_pos)][
        ["match_id", "account_id", "item_id", "game_time_s", "sold_time_s"]]
    if not buys.empty:
        # Join each decision to that player's purchases of tracked items, then test the
        # owned-interval condition. Per hero this is small (a player buys ~15-25 items).
        j = ev.merge(buys, on=["match_id", "account_id"], suffixes=("", "_b"))
        sold = j["sold_time_s"].fillna(0)
        own = (j["game_time_s_b"] < j["game_time_s"]) & ((sold == 0) | (sold > j["game_time_s"]))
        j = j[own]
        out[j["_row"].to_numpy(), j["item_id"].map(item_pos).to_numpy()] = 1
    return pd.DataFrame(out, columns=[f"own_{i}" for i in items], index=events.index)


def support_arms(events: pd.DataFrame, cost: int, min_share: float = 0.02,
                 min_buys: int = 500) -> list[int]:
    """On-support arms at ``cost``: items with at least ``min_share`` of that cost's decisions
    and at least ``min_buys`` buys. Fixed per (hero, cost) BEFORE looking at outcomes."""
    at = events[events["cost"] == cost]
    if at.empty:
        return []
    n = at.groupby("item_id").size()
    keep = n[(n / n.sum() >= min_share) & (n >= min_buys)]
    return [int(i) for i in keep.sort_values(ascending=False).index]


def decision_frame(events: pd.DataFrame, cost: int, arms: list[int] | None = None,
                   own_items: list[int] | None = None, min_share: float = 0.02,
                   min_buys: int = 500) -> pd.DataFrame:
    """One row per decision at ``cost`` whose choice is an on-support arm.

    Adds ``arm`` (index into ``df.attrs['arms']``), ``elig_<k>`` (1 if arm k is not owned),
    and ``own_<item>`` for ``own_items`` (default: every item this hero buys).
    """
    events = events.reset_index(drop=True)
    arms = arms if arms is not None else support_arms(events, cost, min_share, min_buys)
    if len(arms) < 2:
        raise ValueError(f"cost {cost}: fewer than 2 on-support arms ({arms})")
    own_items = own_items if own_items is not None else sorted(
        int(i) for i in events["item_id"].unique())

    own = owned_matrix(events, sorted(set(own_items) | set(arms)))
    sel = (events["cost"] == cost) & events["item_id"].isin(arms)
    df = events.loc[sel].copy()
    own = own.loc[sel]

    arm_pos = {a: k for k, a in enumerate(arms)}
    df["arm"] = df["item_id"].map(arm_pos).astype(int)
    for k, a in enumerate(arms):
        df[f"elig_{k}"] = (1 - own[f"own_{a}"]).astype(np.uint8)
    # A row whose chosen item is already owned is a data error (duplicate buy); drop it.
    E = df[[f"elig_{k}" for k in range(len(arms))]].to_numpy()
    df = df[E[np.arange(len(df)), df["arm"].to_numpy()] == 1]
    own = own.loc[df.index, [f"own_{i}" for i in own_items]]
    df = pd.concat([df, own], axis=1).reset_index(drop=True)
    df["won"] = df["won"].astype(int)
    df.attrs["arms"] = arms
    df.attrs["cost"] = cost
    df.attrs["own_cols"] = list(own.columns)
    return df


def add_composition(df: pd.DataFrame, heroes: list[int]) -> pd.DataFrame:
    """Ally/enemy hero multi-hot from the ``ally_heroes`` / ``enemy_heroes`` list columns."""
    out = df.copy()
    cols = []
    for h in heroes:
        out[f"enemy_h{h}"] = out["enemy_heroes"].map(lambda xs, h=h: int(h in list(xs)))
        out[f"ally_h{h}"] = out["ally_heroes"].map(lambda xs, h=h: int(h in list(xs)))
        cols += [f"enemy_h{h}", f"ally_h{h}"]
    out.attrs = {**df.attrs, "comp_cols": cols}
    return out


def covariate_columns(df: pd.DataFrame) -> list[str]:
    """Every pre-decision covariate in the frame: state, inventory, composition."""
    state = [c for c in STATE_COLUMNS if c in df.columns]
    return state + list(df.attrs.get("own_cols", [])) + list(df.attrs.get("comp_cols", []))
