"""Design-stage diagnostics for one hero: arms, crossover and expected precision per price.

Uses no outcomes, so it can run on all data without breaking the dev/confirm split
(PLAN.md §3.1). This is the Phase 2 exit table: which arms each price has, how often the
comparison group buys the arm later (the crossover label, PLAN.md §3.2), and the 95%
half-width the headline can reach, approximated as ``1.96 * sqrt(0.25/n_k + 0.25/n_rest)``
(optimistic: it ignores clustering and nuisance estimation).

    python -m dl_item_effects.frame.diagnostics --hero Haze --patch 09-29-2026
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..estimate.aipw import crossover_label
from .risk_set import decision_frame

PRICES = (800, 1600, 3200, 6400)


def arm_table(df: pd.DataFrame, within_s: float = 600.0) -> pd.DataFrame:
    """One row per arm of a decision frame."""
    arms = df.attrs["arms"]
    A = df["arm"].to_numpy()
    E = df[[f"elig_{k}" for k in range(len(arms))]].to_numpy().astype(bool)
    NB = df[df.attrs["next_cols"]].to_numpy(dtype=float)
    rows = []
    for k, a in enumerate(arms):
        n_k = int((A == k).sum())
        comp = E[:, k] & (A != k)
        nb = NB[comp, k]
        by_end = float(np.mean(~np.isnan(nb))) if comp.any() else float("nan")
        rows.append(dict(
            item_id=a, n_chose=n_k, share=n_k / len(df), n_comparison=int(comp.sum()),
            cross_within=float(np.mean(nb <= within_s)) if comp.any() else float("nan"),
            cross_by_end=by_end, crossover=crossover_label(by_end),
            half_width_pp=100 * 1.96 * np.sqrt(0.25 / max(n_k, 1) + 0.25 / max(len(df) - n_k, 1))))
    return pd.DataFrame(rows)


def hero_events(patch: str, hero_id: int, events_dir: str | Path | None = None) -> pd.DataFrame:
    import duckdb

    d = Path(events_dir or (config.PROCESSED_DIR / "events" / f"patch_{patch}"))
    files = sorted(d.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no extracted partitions in {d}")
    paths = "[" + ", ".join(f"'{f.as_posix()}'" for f in files) + "]"
    con = duckdb.connect()
    con.execute("SET enable_progress_bar = false")
    return con.execute(
        f"SELECT * FROM read_parquet({paths}) WHERE hero_id = {int(hero_id)}").df()


def design_report(patch: str, hero_id: int, decoder=None, prices=PRICES,
                  events: pd.DataFrame | None = None) -> pd.DataFrame:
    from ..assets import ItemDecoder

    decoder = decoder or ItemDecoder.load()
    closure: dict[int, list[int]] = {}
    for item, comp, _ in decoder.component_closure():
        closure.setdefault(item, []).append(comp)
    ev = events if events is not None else hero_events(patch, hero_id)
    out = []
    for price in prices:
        try:
            df = decision_frame(ev, price, components={k: tuple(v) for k, v in closure.items()})
        except ValueError as e:                     # fewer than two arms at this price
            print(f"{price}: {e}")
            continue
        t = arm_table(df)
        at = ev[ev["cost"] == price] if "cost" in ev else ev.iloc[:0]
        t.insert(0, "price", price)
        t.insert(2, "item", [decoder.name(i) for i in t["item_id"]])
        t["rows"] = len(df)
        t["multi_arm_visits"] = df.attrs["n_multi_arm_visits"]
        t["chosen_ineligible"] = df.attrs["n_chosen_ineligible"]
        t["list_price_buys"] = len(at)
        t["full_price_buys"] = int((ev["full_price"] & (ev["price_paid"] == price)).sum())
        out.append(t)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv: list[str] | None = None) -> None:
    from ..api import DeadlockAPI

    ap = argparse.ArgumentParser(description="Arms, crossover and precision per price.")
    ap.add_argument("--hero", required=True, help="hero name or id")
    ap.add_argument("--patch", required=True)
    a = ap.parse_args(argv)
    if a.hero.isdigit():
        hero_id, name = int(a.hero), a.hero
    else:
        heroes = {h["name"].lower(): h["id"] for h in DeadlockAPI().assets_heroes()}
        hero_id, name = heroes[a.hero.lower()], a.hero
    rep = design_report(a.patch, hero_id)
    out = config.PROCESSED_DIR / "reports" / f"design_{name.lower().replace(' ', '_')}_{a.patch}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(rep[["price", "item", "n_chose", "share", "cross_within", "cross_by_end",
                   "crossover", "half_width_pp"]].round(3).to_string(index=False))
    print(out)


if __name__ == "__main__":
    main()
