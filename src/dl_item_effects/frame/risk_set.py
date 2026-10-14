"""Decision frame for sequential target-trial emulation (PLAN.md §3.2).

A trial is a shop visit at one price. The player is about to spend ``price`` souls and picks
ONE item at that price:

* **Visits**: a player's purchases at most ``visit_gap_s`` apart are one visit. State and
  inventory are taken at the start of the visit, so purchases in the same visit never see
  each other. A visit that buys two or more arms at the price is counted and left out.
* **Price paid, not list price**: only full-price purchases are trials. A purchase that
  consumes a held component costs that player less, and nothing else costs the same for them.
* **Arms**: the full-price items at that price this hero buys often enough to estimate.
  Fixed per (hero, price, patch) BEFORE looking at outcomes.
* **Eligibility**: an arm is available on a row only if the player does not own it and holds
  none of its components (at any depth of the upgrade chain). Owned at t = bought before t
  and not sold or consumed before t (``sold_time_s = 0 or sold_time_s >= t``).
* **Outcome**: match win.

Covariates are the player's state at the start of the visit (``STATE_COLUMNS``), lane,
what they already own (``own_<item>``), and optionally team composition
(:func:`add_composition`) and enemy inventories (:func:`add_enemy_inventory`).
``next_<k>`` holds the seconds from the decision until the player buys arm k later in the
match (NaN if never), for the crossover shares reported with every effect.

Pure pandas (plus DuckDB for the enemy join) so it is testable on hand-built fixtures; run it
per hero.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Pre-decision state from ingest/extract.py. The nc_* columns there are negative-control
# outcomes and must never appear here. rank_level and own_minus_team are NULL for players
# still in calibration; the learners impute them with an indicator.
STATE_COLUMNS = [
    "badge_level", "rank_level", "own_minus_team", "team_rank_gap", "in_calibration",
    "win_streak", "has_build", "net_worth_at_buy", "my_team_nw", "enemy_team_nw",
    "team_nw_diff_at_buy", "n_prior_items", "level", "kills", "deaths", "assists",
    "snapshot_age_s", "game_time_s",
]
LANES = (1, 4, 6)
KEY = ["match_id", "account_id"]


def assign_visits(events: pd.DataFrame, gap_s: float = 30) -> pd.DataFrame:
    """Add ``visit_id`` (per player) and ``visit_start``: a new visit starts when the gap
    from the player's previous purchase is more than ``gap_s`` seconds."""
    ev = events.sort_values(KEY + ["game_time_s"], kind="stable")
    gap = ev.groupby(KEY)["game_time_s"].diff()
    ev = ev.assign(visit_id=(gap.isna() | (gap > gap_s)).groupby([ev[k] for k in KEY]).cumsum())
    ev["visit_start"] = ev.groupby(KEY + ["visit_id"])["game_time_s"].transform("min")
    return ev.loc[events.index]


def owned_matrix(events: pd.DataFrame, items: list[int],
                 decisions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Which of ``items`` each decision's player owns just before its time ``t``.

    ``decisions`` has match_id, account_id and t (default: every event row at its own
    ``game_time_s``). Owned at t = bought strictly before t and not sold or consumed before
    t. A purchase in the same second as t is not owned, so simultaneous buys never see each
    other.
    """
    if decisions is None:
        decisions = events[KEY].assign(t=events["game_time_s"])
    dec = decisions[KEY + ["t"]].copy()
    dec["_row"] = np.arange(len(dec))
    out = np.zeros((len(dec), len(items)), dtype=np.uint8)
    item_pos = {int(i): k for k, i in enumerate(items)}

    buys = events[events["item_id"].isin(item_pos)][KEY + ["item_id", "game_time_s",
                                                         "sold_time_s"]]
    if not buys.empty:
        j = dec.merge(buys, on=KEY)
        sold = j["sold_time_s"].fillna(0)
        own = (j["game_time_s"] < j["t"]) & ((sold == 0) | (sold >= j["t"]))
        j = j[own]
        out[j["_row"].to_numpy(), j["item_id"].astype(int).map(item_pos).to_numpy()] = 1
    return pd.DataFrame(out, columns=[f"own_{i}" for i in items], index=decisions.index)


def _with_price(events: pd.DataFrame) -> pd.DataFrame:
    """Events without price columns (synthetic data) are treated as full-price buys."""
    if "price_paid" in events and "full_price" in events:
        return events
    return events.assign(price_paid=events["cost"], full_price=True)


def support_arms(events: pd.DataFrame, price: int, min_share: float = 0.02,
                 min_buys: int = 2500) -> list[int]:
    """On-support arms at ``price``: full-price items with at least ``min_share`` of that
    price's full-price purchases and at least ``min_buys`` buys (2,500 gives a 95% half-width
    of about 2pp at best). Fixed per (hero, price, patch) BEFORE looking at outcomes."""
    ev = _with_price(events)
    at = ev[ev["full_price"].astype(bool) & (ev["price_paid"] == price)]
    if at.empty:
        return []
    n = at.groupby("item_id").size()
    keep = n[(n / n.sum() >= min_share) & (n >= min_buys)]
    return [int(i) for i in keep.sort_values(ascending=False).index]


def decision_frame(events: pd.DataFrame, price: int, arms: list[int] | None = None,
                   own_items: list[int] | None = None,
                   components: dict[int, tuple[int, ...]] | None = None,
                   visit_gap_s: float = 30, min_share: float = 0.02,
                   min_buys: int = 2500) -> pd.DataFrame:
    """One row per shop visit that buys exactly one on-support arm at full ``price``.

    ``events`` is one hero's purchase table (all prices). ``components`` maps an item to
    every component it upgrades from, at any depth (``ItemDecoder.component_closure``).
    Adds ``arm`` (index into ``df.attrs['arms']``), ``elig_<k>``, ``own_<item>`` for
    ``own_items`` (default: every item this hero buys), ``next_<k>``, and ``lane_<n>``.
    """
    events = assign_visits(_with_price(events).reset_index(drop=True), visit_gap_s)
    events["item_id"] = events["item_id"].astype("int64")   # parquet gives uint64
    arms = arms if arms is not None else support_arms(events, price, min_share, min_buys)
    if len(arms) < 2:
        raise ValueError(f"price {price}: fewer than 2 on-support arms ({arms})")
    components = components or {}
    own_items = own_items if own_items is not None else sorted(
        int(i) for i in events["item_id"].unique())

    visit = KEY + ["visit_id"]
    cand = events[events["full_price"].astype(bool) & (events["price_paid"] == price)
                  & events["item_id"].isin(arms)]
    n_in_visit = cand.groupby(visit)["item_id"].transform("size")
    n_multi = int(cand[n_in_visit > 1].drop_duplicates(visit).shape[0])
    df = cand[n_in_visit == 1].copy()

    # state at the start of the visit: the visit's first purchase, at any price. The in-game
    # negative-control outcomes are aligned the same way (they are never covariates).
    first = events.sort_values("game_time_s", kind="stable").drop_duplicates(visit)
    state = [c for c in STATE_COLUMNS + ["assigned_lane", "nc_prior_souls", "nc_prior_deaths",
                                         "nc_prior_len_s"] if c in events.columns]
    df = df.drop(columns=state).merge(first[visit + state], on=visit, how="left")
    df["game_time_s"] = df["visit_start"]

    comp_items = {c for a in arms for c in components.get(a, ())}
    tracked = sorted(set(own_items) | set(arms) | comp_items)
    own = owned_matrix(events, tracked, df[KEY].assign(t=df["visit_start"]))

    arm_pos = {a: k for k, a in enumerate(arms)}
    df["arm"] = df["item_id"].astype(int).map(arm_pos).astype(int)
    for k, a in enumerate(arms):
        blocked = own[f"own_{a}"].to_numpy().astype(bool)
        for c in components.get(a, ()):
            blocked |= own[f"own_{c}"].to_numpy().astype(bool)
        df[f"elig_{k}"] = (~blocked).astype(np.uint8)
    # The chosen arm must be buyable at full price (else the row contradicts the data)
    E = df[[f"elig_{k}" for k in range(len(arms))]].to_numpy()
    ok = E[np.arange(len(df)), df["arm"].to_numpy()] == 1
    n_bad = int((~ok).sum())
    df, own = df[ok], own[ok]

    # seconds from the decision until the player buys each arm later (crossover)
    buys = events[events["item_id"].isin(arms)].pivot_table(
        index=KEY, columns="item_id", values="game_time_s", aggfunc="min")
    later = df[KEY].merge(buys, left_on=KEY, right_index=True, how="left")
    for k, a in enumerate(arms):
        tk = later[a].to_numpy(dtype=float) if a in later else np.full(len(df), np.nan)
        dt = tk - df["visit_start"].to_numpy(dtype=float)
        dt[(dt < 0) | (df["arm"].to_numpy() == k)] = np.nan
        df[f"next_{k}"] = dt

    lane_cols = []
    if "assigned_lane" in df:
        for v in LANES:
            df[f"lane_{v}"] = (df["assigned_lane"] == v).astype(np.uint8)
            lane_cols.append(f"lane_{v}")

    own = own[[f"own_{i}" for i in own_items]]
    df = pd.concat([df.reset_index(drop=True), own.reset_index(drop=True)], axis=1)
    df["won"] = df["won"].astype(int)
    df.attrs.update(arms=arms, price=price, own_cols=list(own.columns), lane_cols=lane_cols,
                    next_cols=[f"next_{k}" for k in range(len(arms))],
                    n_multi_arm_visits=n_multi, n_chosen_ineligible=n_bad,
                    visit_gap_s=visit_gap_s)
    return df


def add_composition(df: pd.DataFrame, heroes: list[int]) -> pd.DataFrame:
    """Ally/enemy hero multi-hot from the ``ally_heroes`` / ``enemy_heroes`` list columns."""
    out = df.copy()
    cols = []
    for side in ("enemy", "ally"):
        long = out[f"{side}_heroes"].explode().dropna().astype(int)
        hit = pd.crosstab(long.index, long.to_numpy())
        for h in heroes:
            col = f"{side}_h{h}"
            out[col] = (hit[h].reindex(out.index, fill_value=0).astype(np.uint8)
                        if h in hit.columns else np.uint8(0))
            cols.append(col)
    out.attrs = {**df.attrs, "comp_cols": cols}
    return out


def add_enemy_inventory(df: pd.DataFrame, events: pd.DataFrame,
                        items: list[int] | None = None) -> pd.DataFrame:
    """``enemy_<item>``: how many enemy players own each item at the decision time.

    ``events`` must hold every player's purchases in these matches (all heroes), with
    ``team``. Owned uses the same rule as :func:`owned_matrix`.
    """
    import duckdb

    items = items if items is not None else sorted(int(i) for i in events["item_id"].unique())
    pos = {i: k for k, i in enumerate(items)}
    con = duckdb.connect()
    con.register("d", pd.DataFrame({"match_id": df["match_id"].to_numpy(),
                                    "team": df["team"].astype(str).to_numpy(),
                                    "t": df["game_time_s"].to_numpy(),
                                    "_row": np.arange(len(df))}))
    con.register("e", pd.DataFrame({"match_id": events["match_id"].to_numpy(),
                                    "team": events["team"].astype(str).to_numpy(),
                                    "item_id": events["item_id"].astype("int64").to_numpy(),
                                    "bought": events["game_time_s"].to_numpy(),
                                    "sold": events["sold_time_s"].fillna(0).to_numpy()}))
    con.register("it", pd.DataFrame({"item_id": np.array(items, dtype="int64")}))
    cnt = con.execute("""
        SELECT d._row, e.item_id, count(*) AS n
        FROM d JOIN e ON e.match_id = d.match_id AND e.team <> d.team
        WHERE e.bought < d.t AND (e.sold = 0 OR e.sold >= d.t)
          AND e.item_id IN (SELECT item_id FROM it)
        GROUP BY 1, 2""").df()
    mat = np.zeros((len(df), len(items)), dtype=np.uint8)
    if len(cnt):
        mat[cnt["_row"].to_numpy(), cnt["item_id"].map(pos).to_numpy()] = cnt["n"].to_numpy()
    cols = [f"enemy_{i}" for i in items]
    out = pd.concat([df, pd.DataFrame(mat, columns=cols, index=df.index)], axis=1)
    out.attrs = {**df.attrs, "enemy_cols": cols}
    return out


def add_objectives(df: pd.DataFrame, objectives: pd.DataFrame) -> pd.DataFrame:
    """Objectives lost and mid bosses claimed before the decision, for each side:
    ``own_lost_<kind>`` / ``enemy_lost_<kind>`` for t1, t2 and base, and ``own_mid`` /
    ``enemy_mid``. ``objectives`` comes from ``ingest.objectives.build_objectives``."""
    import duckdb

    con = duckdb.connect()
    con.register("d", pd.DataFrame({"match_id": df["match_id"].to_numpy(),
                                    "team": df["team"].astype(str).to_numpy(),
                                    "t": df["game_time_s"].to_numpy(),
                                    "_row": np.arange(len(df))}))
    con.register("o", objectives[["match_id", "team", "kind", "t"]])
    cnt = con.execute("""
        SELECT d._row, (o.team = d.team) AS own, o.kind, count(*) AS n
        FROM d JOIN o ON o.match_id = d.match_id AND o.t < d.t
        GROUP BY 1, 2, 3""").df()
    cols = [f"{side}_lost_{k}" for side in ("own", "enemy") for k in ("t1", "t2", "base")]
    cols += ["own_mid", "enemy_mid"]
    out = df.copy()
    for c in cols:
        out[c] = np.uint8(0)
    if len(cnt):
        side = np.where(cnt["own"], "own", "enemy")
        name = np.where(cnt["kind"] == "mid", side + "_mid", side + "_lost_" + cnt["kind"])
        for c in set(name):
            sel = name == c
            vals = np.zeros(len(df), dtype=np.uint8)
            vals[cnt.loc[sel, "_row"].to_numpy()] = cnt.loc[sel, "n"].to_numpy()
            out[c] = vals
    out.attrs = {**df.attrs, "objective_cols": cols}
    return out


def covariate_columns(df: pd.DataFrame) -> list[str]:
    """Every pre-decision covariate in the frame: state, player history, lane, inventory,
    composition, enemy inventories, objectives."""
    state = [c for c in STATE_COLUMNS if c in df.columns]
    return state + [c for key in ("history_cols", "lane_cols", "own_cols", "comp_cols",
                                  "enemy_cols", "objective_cols")
                    for c in df.attrs.get(key, [])]
