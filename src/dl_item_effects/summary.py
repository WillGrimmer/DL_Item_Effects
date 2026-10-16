"""One combined item table for several heroes (PLAN.md §8, 2026-10-16).

    python -m dl_item_effects.summary --heroes 13 72 --name haze_billy_oracle_plus

Reads each hero's confirm tables (every window and price that ran) and the logistic
agreement table when it exists, and writes ``results/<name>.csv`` with effects in pp and:

* ``significant``: q < 0.05 within the item's tier family (core or extended).
* ``p_better``: Phi(effect_cal / se_cal), the chance the item beats what players actually
  bought, under the calibrated normal approximation.
* ``promising``: not significant, but p_better >= 0.9 (or <= 0.1 for harmful). Extended-tier
  items are reported this way unless they pass their own family.
* ``replicates``: the item ran in both windows with the same calibrated sign.
* ``agree`` / ``counts``: from ``confirm_agreement.csv`` (LightGBM vs logistic), if present.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import norm

from . import config
from .phase3 import PRICES, results_dir

HERO_NAMES = {13: "Haze", 72: "Billy"}
PATCHES = ("09-16-2026", "09-29-2026")
PP = ("effect", "se", "effect_cal", "se_cal", "ci_lo_cal", "ci_hi_cal", "eb_effect", "eb_sd")


def combined(heroes, patches=PATCHES, population: str = "oracle_plus",
             learner: str = "lgbm") -> pd.DataFrame:
    rows = []
    for h in heroes:
        for patch in patches:
            out = results_dir(h, patch, population)
            ag = out / "confirm_agreement.csv"
            agree = (pd.read_csv(ag)[["price", "arm", "agree", "counts"]] if ag.exists()
                     else None)
            for price in PRICES:
                f = out / f"confirm_{learner}_{price}_effects.csv"
                if not f.exists():
                    continue
                t = pd.read_csv(f).assign(hero_id=h, hero=HERO_NAMES.get(h, str(h)),
                                          window=patch[:5], price=price)
                if agree is not None:
                    t = t.merge(agree, on=["price", "arm"], how="left")
                rows.append(t)
    t = pd.concat(rows, ignore_index=True)
    if "tier" not in t:
        t["tier"] = "core"
    t["tier"] = t["tier"].fillna("core")
    t["p_better"] = norm.cdf(t["effect_cal"] / t["se_cal"])
    t["significant"] = t["q"] < 0.05
    t["promising"] = ~t["significant"] & ((t["p_better"] >= 0.9) | (t["p_better"] <= 0.1))
    sign = t.pivot_table(index=["hero_id", "price", "arm"], columns="window",
                         values="effect_cal").apply(np.sign)
    rep = (sign.notna().sum(1) == len(patches)) & (sign.nunique(1) == 1)
    t = t.merge(rep.rename("replicates").reset_index(), on=["hero_id", "price", "arm"],
                how="left")
    for c in PP:
        t[c] = t[c] * 100
    front = ["hero", "window", "price", "item", "tier", "effect_cal", "ci_lo_cal", "ci_hi_cal",
             "p_better", "q", "significant", "promising", "replicates"]
    front += [c for c in ("agree", "counts") if c in t]
    return t[front + [c for c in t.columns if c not in front]].sort_values(
        ["hero", "price", "window", "eb_effect"], ascending=[True, True, True, False])


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Combined item table for several heroes.")
    ap.add_argument("--heroes", type=int, nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--population", default="oracle_plus")
    a = ap.parse_args(argv)
    t = combined(a.heroes, population=a.population)
    f = config.PROCESSED_DIR / "results" / f"{a.name}.csv"
    t.to_csv(f, index=False)
    print(f"{f}: {len(t)} rows; significant {int(t['significant'].sum())}, "
          f"promising {int(t['promising'].sum())}, replicating {int(t['replicates'].sum())}")


if __name__ == "__main__":
    main()
