"""Phase 3 runner: one hero end to end, in the order PLAN.md §3.1 allows.

    python -m dl_item_effects.phase3 --hero 13 --patch 09-16-2026 --stage dev --learner logistic

Stages (results under data/processed/results/hero_<id>/patch_<patch>/):

* ``dev``: effects on the dev set only (20% of matches), for tuning and diagnostics.
* ``falsify``: PLAN.md §4.1 to 4.4 on all matches. Only the checks' statistics are written:
  negative-control effects (with per-match score sums for the bootstrap), SE coverage, and
  split-half z-scores, never the win effects. Also the next-match diagnostic. Ends with the
  window-level pooled rules (``pooled_falsify_<learner>.json``).
* ``confirm``: the reported tables on the other 80%, run once the design is frozen: headline
  per arm with crossover label, pairwise matrix, curves over game time, OVB bounds benchmarked
  against the skill covariates, and BH q-values. Every effect is also given calibrated
  (PLAN.md §8, 2026-10-15: shifted by the next-match control's mean, widened by its
  between-arm SD; ``*_cal`` columns), and p, q and shrinkage use the calibrated values. It
  needs ``pooled_falsify_lgbm.json`` and refuses to run without it.
* ``shuffle``: §4.1 on a shuffled previous-match result (true effect zero), to tell an
  estimator artifact from real confounding. Writes ``shuffle_<learner>_<price>_nc_player.csv``
  and pools it like ``falsify``.
* ``synth``: §4.1 (and the next-match diagnostic) on a semi-synthetic outcome drawn from
  the covariates alone: true effect zero, realistic covariate signal. Pooled like ``falsify``.
* ``pool``: recompute the pooled rules and bootstrap from the saved falsify/shuffle files.
* ``compare``: after ``confirm`` has run with both learners, the agreement table (PLAN.md §3.3:
  a result counts only if logistic and LightGBM agree on sign and significance at q < 0.05).

Propensities use no outcome, so they are fitted once per (price, learner, covariate set) on
all rows and cached as .npy files keyed by the frame's size. Covariate-set tags start with
``h_`` since the player-history covariates were added (2026-10-14); older ``all``/``nostreak``
files belong to the design before that.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import analysis, config
from .assets import ItemDecoder
from .estimate import benjamini_hochberg, dr_curve, effect_table, pairwise_table, two_sided_p
from .estimate.heterogeneity import empirical_bayes
from .pipeline import POPULATIONS, build_frame, load_hero_window

PRICES = (800, 1600, 3200, 6400)
NC_OUTCOMES = ("nc_player", "nc_next")
CURVE_GRID_S = np.arange(300, 2701, 150)


def results_dir(hero_id: int, patch: str, population: str = "all") -> Path:
    d = config.PROCESSED_DIR / "results" / f"hero_{hero_id}" / f"patch_{patch}"
    if population != "all":
        d = d / f"pop_{population}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_propensities(df: pd.DataFrame, out: Path, learner: str, tag: str,
                        drop: tuple[str, ...] = ()) -> np.ndarray:
    f = out / f"prop_{learner}_{df.attrs['price']}_{tag}_{len(df)}.npy"
    if f.exists():
        return np.load(f)
    e = analysis.propensities(df, learner, drop=drop)
    np.save(f, e)
    return e


def _names(decoder, ids) -> list[str]:
    return [decoder.name(int(i)) for i in ids]


def run_price(df: pd.DataFrame, stage: str, learner: str, out: Path, decoder) -> dict:
    price = df.attrs["price"]
    dev = analysis.dev_rows(df)
    allr = np.ones(len(df), bool)
    tag = f"{stage}_{learner}_{price}"
    t0 = time.monotonic()
    e_all = cached_propensities(df, out, learner, "h_all")
    summary = dict(price=price, stage=stage, learner=learner, rows=len(df), dev_rows=int(dev.sum()),
                   arms=_names(decoder, df.attrs["arms"]),
                   multi_arm_visits=df.attrs["n_multi_arm_visits"],
                   chosen_ineligible=df.attrs["n_chosen_ineligible"],
                   propensity_s=round(time.monotonic() - t0))

    if stage == "dev":
        fit = analysis.fit_rows(df, dev, learner, e_hat=e_all[dev])
        tab = effect_table(fit, analysis.next_buy(df, dev))
        tab.insert(1, "item", _names(decoder, tab["arm"]))
        tab.to_csv(out / f"{tag}_effects.csv", index=False)
        summary.update(trim_share=dict(zip(tab["item"], tab["trim_share"].round(3))),
                       fold_absent=len(fit.fold_absent))

    elif stage == "falsify":
        e_nc = cached_propensities(df, out, learner, "h_nc", drop=analysis.NC_PLAYER_DROP)
        p = analysis.nc_player(df, allr, learner, e_all=e_nc)
        _save_nc(p, out, tag, "nc_player", decoder)
        nx = analysis.nc_next(df, allr, learner, e_all=e_all)
        _save_nc(nx, out, tag, "nc_next", decoder)
        g = analysis.nc_ingame(df, allr, learner, e_all=e_all)
        for col in ("nc_prior_souls", "nc_prior_deaths"):
            g[col]["table"].assign(item=_names(decoder, g[col]["table"]["arm"])).to_csv(
                out / f"{tag}_{col}.csv", index=False)
        dev_fit = analysis.fit_rows(df, dev, learner, e_hat=e_all[dev])
        c = analysis.se_calibration(dev_fit, reps=200, seed=price)
        s = analysis.split_half(df, allr, learner, seed=price)
        s["z"].assign(item=_names(decoder, s["z"]["arm"])).to_csv(
            out / f"{tag}_split_half_z.csv", index=False)
        summary.update(
            nc_player=dict(mu=p["mu"], tau=p["tau"], passed=p["passed"],
                           null_fail_rate=p["null_fail_rate"]),
            nc_next=dict(mu=nx["mu"], tau=nx["tau"], passed=nx["passed"]),
            nc_ingame={k: dict(share_within=v["share_within"], mean_abs=v["mean_abs"],
                               passed=v["passed"]) for k, v in g.items() if isinstance(v, dict)},
            se_calibration=dict(arm=decoder.name(int(c["arm"])), coverage=c["coverage"],
                                z_sd=c["z_sd"], passed=c["passed"]),
            split_half=dict(share_beyond=s["share_beyond"], z_sd=s["z_sd"],
                            passed=s["passed"]),
            passed=bool(p["passed"] and g["passed"] and c["passed"] and s["passed"]))

    elif stage == "shuffle":
        e_nc = cached_propensities(df, out, learner, "h_nc", drop=analysis.NC_PLAYER_DROP)
        p = analysis.nc_player_shuffled(df, allr, learner, e_all=e_nc, seed=1)
        _save_nc(p, out, tag, "nc_player", decoder)
        summary.update(nc_player_shuffled=dict(mu=p["mu"], tau=p["tau"]))

    elif stage == "synth":
        e_nc = cached_propensities(df, out, learner, "h_nc", drop=analysis.NC_PLAYER_DROP)
        p = analysis.nc_player_semisynthetic(df, allr, learner, e_all=e_nc, seed=1)
        _save_nc(p, out, tag, "nc_player", decoder)
        nx = analysis.nc_player_semisynthetic(df, allr, learner, e_all=e_all, seed=2,
                                              outcome="nc_next_won", drop=())
        _save_nc(nx, out, tag, "nc_next", decoder)
        summary.update(nc_player_synth=dict(mu=p["mu"], tau=p["tau"], p_sd=p["p_sd"]),
                       nc_next_synth=dict(mu=nx["mu"], tau=nx["tau"], p_sd=nx["p_sd"]))

    elif stage == "confirm":
        cal = analysis.load_calibration(out, "lgbm")
        conf = ~dev
        fit = analysis.fit_rows(df, conf, learner, e_hat=e_all[conf])
        tab = effect_table(fit, analysis.next_buy(df, conf))
        tab["p_raw"] = two_sided_p(tab["effect"], tab["se"])
        _add_cal(tab, cal)
        tab["p"] = two_sided_p(tab["effect_cal"], tab["se_cal"])
        eb = empirical_bayes(tab["effect_cal"].to_numpy(), tab["se_cal"].to_numpy())
        tab["eb_effect"], tab["eb_sd"] = eb["post_mean"], eb["post_sd"]
        e_ns = cached_propensities(df, out, learner, "h_noskill",
                                   drop=tuple(analysis.SKILL_COLUMNS))
        short = analysis.fit_rows(df, conf, learner, drop=tuple(analysis.SKILL_COLUMNS),
                                  e_hat=e_ns[conf])
        sens = []
        for k in range(len(fit.arms)):
            o = analysis.ovb(fit, k)
            b = analysis.ovb_benchmark(fit, short, k)
            at_skill = analysis.ovb(fit, k, cf_y=max(b["cf_y"], 1e-9), cf_d=max(b["cf_d"], 1e-9))
            sens.append(dict(rv=o["rv"], rva=o["rva"], skill_cf_y=b["cf_y"],
                             skill_cf_d=b["cf_d"], bias_if_as_strong_as_skill=at_skill["bias"]))
        tab = pd.concat([tab, pd.DataFrame(sens)], axis=1)
        tab.insert(1, "item", _names(decoder, tab["arm"]))
        tab.to_csv(out / f"{tag}_effects.csv", index=False)
        pw = pairwise_table(fit)
        pw.insert(1, "item", _names(decoder, pw["arm"]))
        pw.insert(3, "vs_item", _names(decoder, pw["vs"]))
        _add_cal(pw, cal, pairwise=True)
        pw.to_csv(out / f"{tag}_pairwise.csv", index=False)
        t = df.loc[conf, "game_time_s"].to_numpy()
        curves = pd.concat([dr_curve(fit, k, t, CURVE_GRID_S).assign(arm=fit.arms[k])
                            for k in range(len(fit.arms))])
        _add_cal(curves, cal)
        curves.to_csv(out / f"{tag}_curves.csv", index=False)
        summary.update(confirm_rows=int(conf.sum()), fold_absent=len(fit.fold_absent),
                       calibration=cal)

    summary["wall_s"] = round(time.monotonic() - t0)
    return summary


def _add_cal(t: pd.DataFrame, cal: dict, pairwise: bool = False) -> None:
    t["effect_cal"], t["se_cal"] = analysis.calibrate(t["effect"], t["se"], cal, pairwise)
    t["ci_lo_cal"] = t["effect_cal"] - 1.96 * t["se_cal"]
    t["ci_hi_cal"] = t["effect_cal"] + 1.96 * t["se_cal"]


def _save_nc(p: dict, out: Path, tag: str, name: str, decoder) -> None:
    p["table"].assign(item=_names(decoder, p["table"]["arm"])).to_csv(
        out / f"{tag}_{name}.csv", index=False)
    p["scores"].to_parquet(out / f"{tag}_{name}_scores.parquet", index=False)


def pool(out: Path, stage: str, learner: str, prices=PRICES, reps: int = 1000) -> dict:
    """The window-level rules (PLAN.md §8, 2026-10-14): §4.1 (and the next-match diagnostic)
    over every arm at every price, with match- and account-clustered bootstraps, and §4.4
    over every price's split-half z."""
    res: dict = dict(stage=stage, learner=learner, prices=list(prices))
    for name in NC_OUTCOMES:
        files = [(p, out / f"{stage}_{learner}_{p}_{name}") for p in prices]
        if not all(f.with_suffix(".csv").exists() for _, f in files):
            continue
        tabs = pd.concat([pd.read_csv(f.with_suffix(".csv")).assign(price=p) for p, f in files],
                         ignore_index=True)
        scores = pd.concat([pd.read_parquet(f"{f}_scores.parquet").assign(price=p)
                            for p, f in files], ignore_index=True)
        r = analysis.pooled_nc(tabs)
        r["bootstrap"] = [analysis.bootstrap_nc(scores, tabs, cluster=c, reps=reps)
                          for c in ("match_id", "account_id")]
        res[name] = r
    zf = [out / f"{stage}_{learner}_{p}_split_half_z.csv" for p in prices]
    if all(f.exists() for f in zf):
        res["split_half"] = analysis.pooled_split_half(
            pd.concat([pd.read_csv(f) for f in zf], ignore_index=True))
    (out / f"pooled_{stage}_{learner}.json").write_text(json.dumps(res, indent=1, default=float))
    return res


def compare(out: Path, prices=PRICES, alpha: float = 0.05) -> pd.DataFrame:
    rows = []
    for price in prices:
        a = pd.read_csv(out / f"confirm_logistic_{price}_effects.csv")
        b = pd.read_csv(out / f"confirm_lgbm_{price}_effects.csv")
        m = a.merge(b, on=["arm", "item"], suffixes=("_logit", "_lgbm"))
        m.insert(0, "price", price)
        rows.append(m)
    m = pd.concat(rows, ignore_index=True)
    sig_a, sig_b = m["q_logit"] < alpha, m["q_lgbm"] < alpha
    same_sign = np.sign(m["effect_cal_logit"]) == np.sign(m["effect_cal_lgbm"])
    m["agree"] = (sig_a & sig_b & same_sign) | (~sig_a & ~sig_b)
    m["counts"] = sig_a & sig_b & same_sign
    m.to_csv(out / "confirm_agreement.csv", index=False)
    return m


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Phase 3: one hero end to end.")
    ap.add_argument("--hero", type=int, required=True)
    ap.add_argument("--patch", required=True)
    ap.add_argument("--stage", choices=("dev", "falsify", "shuffle", "synth", "confirm", "compare",
                                        "pool"),
                    required=True)
    ap.add_argument("--learner", choices=("logistic", "lgbm"), default="logistic")
    ap.add_argument("--prices", type=int, nargs="*", default=list(PRICES))
    ap.add_argument("--population", choices=tuple(POPULATIONS), default="oracle_plus",
                    help="match population (PLAN.md §8, 2026-10-15: Oracle 1 and up)")
    a = ap.parse_args(argv)

    out = results_dir(a.hero, a.patch, a.population)
    if a.stage == "pool":
        for st in ("falsify", "shuffle", "synth"):
            print(json.dumps(pool(out, st, a.learner, a.prices), indent=1, default=float))
        return
    if a.stage == "compare":
        m = compare(out, a.prices)
        print(f"{int(m['agree'].sum())} of {len(m)} arms agree; {int(m['counts'].sum())} count "
              f"(both significant, same sign)")
        return
    decoder = ItemDecoder.load()
    t0 = time.monotonic()
    w = load_hero_window(a.patch, a.hero, min_badge=POPULATIONS[a.population])
    print(f"loaded hero {a.hero} {a.patch} ({a.population}): {len(w.events):,} purchases "
          f"({time.monotonic() - t0:.0f} s)", flush=True)
    summaries = []
    for price in a.prices:
        t1 = time.monotonic()
        df = build_frame(w, price, decoder)
        s = run_price(df, a.stage, a.learner, out, decoder)
        s["frame_s"] = round(time.monotonic() - t1 - s["wall_s"])
        summaries.append(s)
        print(json.dumps(s, default=float), flush=True)
    if a.stage == "confirm":
        # Benjamini-Hochberg across the whole family for this hero and patch (every price), on
        # the calibrated p-values
        files = [out / f"confirm_{a.learner}_{p}_effects.csv" for p in a.prices]
        tabs = [pd.read_csv(f) for f in files]
        q = benjamini_hochberg(np.concatenate([t["p"].to_numpy() for t in tabs]))
        i = 0
        for f, t in zip(files, tabs):
            t["q"] = q[i:i + len(t)]
            i += len(t)
            t.to_csv(f, index=False)
    sfile = out / f"summary_{a.stage}_{a.learner}.json"
    sfile.write_text(json.dumps(summaries, indent=1, default=float))
    print(sfile)
    if a.stage in ("falsify", "shuffle", "synth"):
        print(json.dumps(pool(out, a.stage, a.learner, a.prices), default=float), flush=True)


if __name__ == "__main__":
    main()
