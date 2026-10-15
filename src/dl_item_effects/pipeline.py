"""Load one hero's patch window and build full decision frames (PLAN.md §3.2).

The frame carries every pre-decision covariate: state at the start of the visit, the player's
Ranked history before the match, lane, own inventory, both teams' heroes, enemy inventories
and objectives. ``is_dev`` marks the dev
set (PLAN.md §3.1): a fixed splitmix64 hash of ``match_id``, modulo 5, equal to 0. Raw match ids
are not uniform modulo 5 (31k to 40k matches per residue in the 09-16 window) and the residue
shifts mean rank slightly, so the hash is needed for a representative 20%.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import config
from .frame import (add_composition, add_enemy_inventory, add_objectives, add_player_history,
                    covariate_columns, decision_frame, history_features)

ENEMY_COLUMNS = ["match_id", "team", "item_id", "game_time_s", "sold_time_s"]
# Populations by minimum match-average badge on the linear scale 6 * tier + sub (PLAN.md §8,
# 2026-10-15). Oracle is tier 8, so Oracle 1 is 49.
POPULATIONS = {"all": None, "oracle_plus": 49}


@dataclass
class HeroWindow:
    patch: str
    hero_id: int
    events: pd.DataFrame        # this hero's purchases
    match_events: pd.DataFrame  # every player's purchases in this hero's matches (slim)
    objectives: pd.DataFrame
    heroes: list[int]           # every hero seen in the window, for composition columns
    history: pd.DataFrame | None = None   # history_features for this hero's accounts


def is_dev(match_id) -> np.ndarray:
    """Dev-set membership: splitmix64(match_id) % 5 == 0. Stable across machines and
    library versions."""
    x = np.asarray(match_id, dtype=np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))
    return x % np.uint64(5) == 0


def window_dir(patch: str) -> Path:
    return config.PROCESSED_DIR / "events" / f"patch_{patch}"


def load_hero_window(patch: str, hero_id: int, events_dir: str | Path | None = None,
                     min_badge: int | None = None) -> HeroWindow:
    """One hero's window. ``min_badge`` keeps only matches whose average badge (linear
    ``badge_level``) is at least that; player history still covers all of a player's matches."""
    d = Path(events_dir or window_dir(patch))
    parts = sorted(d.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no extracted partitions in {d}")
    src = "read_parquet([" + ", ".join(f"'{p.as_posix()}'" for p in parts) + "])"
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    badge = f" AND badge_level >= {int(min_badge)}" if min_badge is not None else ""
    con.execute(f"CREATE TEMP TABLE ev AS SELECT * FROM {src} "
                f"WHERE hero_id = {int(hero_id)}{badge}")
    events = con.execute("SELECT * FROM ev").df()
    match_events = con.execute(
        f"SELECT {', '.join(ENEMY_COLUMNS)} FROM {src} "
        f"WHERE match_id IN (SELECT DISTINCT match_id FROM ev)").df()
    heroes = [int(h) for (h,) in con.execute(
        f"SELECT DISTINCT hero_id FROM {src} ORDER BY 1").fetchall()]
    obj_path = d / "_objectives.parquet"
    objectives = (con.execute(
        f"SELECT * FROM read_parquet('{obj_path.as_posix()}') "
        f"WHERE match_id IN (SELECT DISTINCT match_id FROM ev)").df()
        if obj_path.exists() else pd.DataFrame(columns=["match_id", "team", "kind", "t"]))
    hist_path = d / "_history.parquet"
    history = (history_features(hist_path.as_posix(), accounts=events["account_id"], con=con)
               if hist_path.exists() else None)
    return HeroWindow(patch, int(hero_id), events, match_events, objectives, heroes, history)


def component_closure(decoder) -> dict[int, tuple[int, ...]]:
    out: dict[int, list[int]] = {}
    for item, comp, _ in decoder.component_closure():
        out.setdefault(item, []).append(comp)
    return {k: tuple(v) for k, v in out.items()}


def build_frame(w: HeroWindow, price: int, decoder, arms: list[int] | None = None,
                **frame_kw) -> pd.DataFrame:
    """Decision frame at ``price`` with every covariate block attached."""
    df = decision_frame(w.events, price, arms=arms, components=component_closure(decoder),
                        **frame_kw)
    df = add_composition(df, w.heroes)
    items = sorted(int(i) for i in w.match_events["item_id"].unique())
    df = add_enemy_inventory(df, w.match_events, items=items)
    if len(w.objectives):
        df = add_objectives(df, w.objectives)
    if w.history is not None:
        df = add_player_history(df, w.history)
    df["dev"] = is_dev(df["match_id"])
    return df


def design_matrix(df: pd.DataFrame):
    """Covariates, chosen arm, outcome, eligibility and clusters for ``fit_multiarm``."""
    K = len(df.attrs["arms"])
    X = df[covariate_columns(df)]
    X = X.astype({c: "float32" for c in X.columns if X[c].dtype == bool})
    return (X, df["arm"].to_numpy(), df["won"].to_numpy(),
            df[[f"elig_{k}" for k in range(K)]].to_numpy().astype(bool),
            df["match_id"].to_numpy())
