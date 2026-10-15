"""Phase 3 analyses for one hero and price: effects, falsification and sensitivity.

* :func:`fit_rows`: the multi-arm AIPW on a subset of a decision frame.
* Falsification (PLAN.md §4), each returning a table and a pass/fail against the thresholds
  fixed there:
  - :func:`nc_player` (§4.1): the player's previous Ranked match result, fitted without win
    streak or the previous result itself (``NC_PLAYER_DROP``; the history covariates start two
    matches back). Pass: |mu| <= 0.2pp and tau <= 0.3pp across arms, applied once per hero
    and window over every price (:func:`pooled_nc`, PLAN.md §8 2026-10-14).
  - :func:`nc_next` (diagnostic, not a gate): the player's next Ranked match result, with the
    full production covariates.
  - :func:`nc_ingame` (§4.2): souls gained and deaths over the snapshot interval before the
    decision, in SD units. Pass: |d| <= 0.1 SD for at least 90% of arms and mean |d| <= 0.05.
  - :func:`se_calibration` (§4.3): random halves of one arm's buyers as fake arms, 200 times.
    Pass: 92% to 98% of 95% intervals cover zero.
  - :func:`split_half` (§4.4): two random halves of the matches. Pass: at most 10% of arms
    differ by more than 1.96 combined SEs, pooled over a window's prices
    (:func:`pooled_split_half`).
* :func:`bootstrap_nc`: honest pooled SEs and the §4.1 rule's real chance-fail rate, from
  per-cluster score sums resampled over matches or accounts (shared across a window's
  prices, so arms that share rows stay correlated).
* :func:`ovb` (§3.6): omitted-variable-bias bounds for DML (Chernozhukov, Cinelli, Newey,
  Sharma & Syrgkanis 2022), in DoubleML's parametrization: the bias is at most
  ``S * |rho| * sqrt(cf_y * cf_d / (1 - cf_d))`` with ``S^2 = sigma^2 * nu^2``. The Riesz
  representer respects per-row eligibility, which DoubleML cannot express.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .estimate.aipw import (MultiArmFit, _clustered_se, effect_table, fit_multiarm,
                            pairwise, vs_actual, vs_actual_scores)
from .estimate.heterogeneity import empirical_bayes
from .frame import covariate_columns
from .pipeline import is_dev

SKILL_COLUMNS = ["badge_level", "rank_level", "own_minus_team", "team_rank_gap",
                 "in_calibration", "win_streak"]
NC_PLAYER_DROP = ("win_streak", "prev_won")
THRESHOLDS = dict(nc_player_mu=0.002, nc_player_tau=0.003, nc_ingame_arm=0.10,
                  nc_ingame_share=0.90, nc_ingame_mean=0.05, se_cov_lo=0.92, se_cov_hi=0.98,
                  split_half_share=0.10)


def fit_rows(df: pd.DataFrame, rows: np.ndarray, learner: str = "logistic",
             outcome: str = "won", drop: tuple[str, ...] = (), continuous: bool = False,
             e_hat: np.ndarray | None = None, seed: int = 0) -> MultiArmFit:
    rows = np.asarray(rows, dtype=bool)
    arms = df.attrs["arms"]
    cols = [c for c in covariate_columns(df) if c not in drop]
    sub = df.loc[rows]
    X = sub[cols]
    X = X.astype({c: "float32" for c in cols if X[c].dtype == bool})
    elig = sub[[f"elig_{k}" for k in range(len(arms))]].to_numpy().astype(bool)
    return fit_multiarm(X, sub["arm"].to_numpy(), sub[outcome].to_numpy(dtype=float), elig,
                        sub["match_id"].to_numpy(), arms=arms, learner=learner,
                        continuous=continuous, e_hat=e_hat, seed=seed)


def propensities(df: pd.DataFrame, learner: str = "logistic", drop: tuple[str, ...] = (),
                 seed: int = 0) -> np.ndarray:
    """Out-of-fold propensities for every row of the frame (no outcome involved)."""
    from .estimate.aipw import fit_propensity

    cols = [c for c in covariate_columns(df) if c not in drop]
    X = df[cols].astype({c: "float32" for c in cols if df[c].dtype == bool})
    elig = df[[f"elig_{k}" for k in range(len(df.attrs["arms"]))]].to_numpy().astype(bool)
    return fit_propensity(X, df["arm"].to_numpy(), elig, df["match_id"].to_numpy(), learner,
                          seed=seed)


def next_buy(df: pd.DataFrame, rows: np.ndarray) -> np.ndarray:
    return df.loc[np.asarray(rows, dtype=bool), df.attrs["next_cols"]].to_numpy(dtype=float)


# -- falsification ------------------------------------------------------------------------

def _sub(e_all: np.ndarray | None, rows: np.ndarray):
    return None if e_all is None else e_all[np.asarray(rows, dtype=bool)]


def nc_player(df: pd.DataFrame, rows: np.ndarray, learner: str = "logistic",
              e_all: np.ndarray | None = None, outcome: str = "nc_prev_won",
              drop: tuple[str, ...] = NC_PLAYER_DROP) -> dict:
    """§4.1: effect of the choice on the player's previous Ranked match result.
    ``e_all``: propensities for every frame row fitted without ``drop``. ``scores`` holds the
    per-(arm, match) sums behind each estimate, for :func:`bootstrap_nc`."""
    r = np.asarray(rows, dtype=bool) & df[outcome].notna().to_numpy()
    fit = fit_rows(df, r, learner, outcome=outcome, drop=drop, e_hat=_sub(e_all, r))
    tab = effect_table(fit).dropna(subset=["effect", "se"])
    eb = empirical_bayes(tab["effect"].to_numpy(), tab["se"].to_numpy())
    ok = _nc_player_rule(eb)
    return dict(table=tab, mu=eb["prior_mean"], tau=eb["tau"], passed=bool(ok),
                null_fail_rate=null_fail_rate(tab["se"].to_numpy()),
                scores=cluster_scores(fit, df.loc[r, "account_id"].to_numpy(),
                                      keep=set(tab["arm"])))


def nc_next(df: pd.DataFrame, rows: np.ndarray, learner: str = "logistic",
            e_all: np.ndarray | None = None) -> dict:
    """Diagnostic: effect of the choice on the player's NEXT Ranked match result, with the
    full production covariates (``e_all`` fitted on all of them). If §4.1 fails but this is
    about zero, the link is to the last result itself, which production adjusts for."""
    return nc_player(df, rows, learner, e_all=e_all, outcome="nc_next_won", drop=())


def cluster_scores(fit: MultiArmFit, account: np.ndarray, keep=None) -> pd.DataFrame:
    """Per (arm, match): the sums of ``psi_k - Y`` (``d``) and ``1{A != k}`` (``o``) behind the
    headline, so that ``sum d / sum o`` over any resample of matches is its estimate."""
    out = []
    for k in range(fit.psi.shape[1]):
        if keep is not None and fit.arms[k] not in keep:
            continue
        rows, d, other = vs_actual_scores(fit, k)
        g = pd.DataFrame(dict(match_id=fit.groups[rows], account_id=account[rows], d=d,
                              o=other))
        g = g.groupby(["match_id", "account_id"], sort=False, as_index=False).sum()
        out.append(g.assign(arm=fit.arms[k]))
    return pd.concat(out, ignore_index=True)[["arm", "match_id", "account_id", "d", "o"]]


def pooled_nc(tables: pd.DataFrame) -> dict:
    """§4.1 applied once per hero and window: one mean and between-arm SD over every arm at
    every price (``tables``: the per-price effect tables stacked)."""
    eb = empirical_bayes(tables["effect"].to_numpy(), tables["se"].to_numpy())
    w = 1 / tables["se"].to_numpy() ** 2
    return dict(mu=eb["prior_mean"], tau=eb["tau"], passed=bool(_nc_player_rule(eb)),
                n_arms=len(tables), fe_mean=float(np.sum(w * tables["effect"]) / np.sum(w)),
                fe_se=float(np.sqrt(1 / np.sum(w))),
                null_fail_rate=null_fail_rate(tables["se"].to_numpy()))


def bootstrap_nc(scores: pd.DataFrame, tables: pd.DataFrame, cluster: str = "match_id",
                 reps: int = 1000, seed: int = 0) -> dict:
    """Poisson bootstrap over clusters (``match_id`` or ``account_id``) of a window's §4.1
    estimates, all prices at once. ``scores`` and ``tables`` carry a ``price`` column.

    Returns the bootstrap SE of the pooled mean (``mu_se``; ``fe_se`` for the fixed-effect
    mean), the median ratio of bootstrap to nominal per-arm SEs, and ``chance_fail``: how
    often the rule fails on draws centred at the estimates (a null with the real dependence
    between arms and prices, scored with the nominal SEs as the rule is). Nuisance models are
    held fixed, so this is the second-stage variance only."""
    key = ["price", "arm"]
    t = tables.set_index(key)
    gi = t.index.get_indexer(pd.MultiIndex.from_frame(scores[key]))
    s = scores[gi >= 0]
    gi = gi[gi >= 0]
    ci, uniq = pd.factorize(s[cluster])
    d, o = s["d"].to_numpy(), s["o"].to_numpy()
    G = len(t)
    theta = np.bincount(gi, d, G) / np.bincount(gi, o, G)
    se = t["se"].to_numpy()
    w_fe = 1 / se ** 2
    rng = np.random.default_rng(seed)
    draws = np.empty((reps, G))
    for b in range(reps):
        w = rng.poisson(1.0, len(uniq))[ci]
        draws[b] = np.bincount(gi, w * d, G) / np.bincount(gi, w * o, G)
    mus = np.array([empirical_bayes(x, se)["prior_mean"] for x in draws])
    fes = draws @ w_fe / w_fe.sum()
    fails = [not _nc_player_rule(empirical_bayes(x, se)) for x in draws - theta]
    return dict(cluster=cluster, reps=reps, n_clusters=len(uniq), mu_se=float(mus.std()),
                fe_se=float(fes.std()),
                se_ratio_median=float(np.median(draws.std(0) / se)),
                chance_fail=float(np.mean(fails)),
                max_abs_point_gap=float(np.max(np.abs(theta - t["effect"].to_numpy()))))


def nc_player_shuffled(df: pd.DataFrame, rows: np.ndarray, learner: str = "logistic",
                       e_all: np.ndarray | None = None, seed: int = 1) -> dict:
    """§4.1 on a shuffled previous-match result: one value per (match, account), permuted
    across player-matches that have a previous match. The true effect is exactly zero and no
    confounding can exist, so a mu clearly away from zero here would be an estimator artifact.
    Haze, 2026-10-14: shuffled +0.05pp and +0.11pp per window against real -0.34 and -0.28."""
    rng = np.random.default_rng(seed)
    key = df["match_id"].astype("int64") * 10**10 + df["account_id"].astype("int64")
    per_player = df.groupby(key)["nc_prev_won"].first().dropna()
    shuffled = pd.Series(rng.permutation(per_player.to_numpy()), index=per_player.index)
    return nc_player(df.assign(nc_prev_won=key.map(shuffled)), rows, learner, e_all=e_all)


def nc_player_semisynthetic(df: pd.DataFrame, rows: np.ndarray, learner: str = "logistic",
                            e_all: np.ndarray | None = None, seed: int = 1,
                            outcome: str = "nc_prev_won",
                            drop: tuple[str, ...] = NC_PLAYER_DROP) -> dict:
    """§4.1 on a semi-synthetic outcome: one draw per (match, account) from Bernoulli(p), with
    p the out-of-fold prediction of ``outcome`` from the covariates alone (one pooled model,
    no arm), averaged over that player-match's rows. The true effect of every arm is zero, but
    the covariates predict the outcome as strongly as they predict the real one, so unlike
    the shuffle this catches bias from the nuisance models' errors."""
    from sklearn.model_selection import GroupKFold

    from .estimate.aipw import fit_outcome

    r = np.asarray(rows, dtype=bool) & df[outcome].notna().to_numpy()
    cols = [c for c in covariate_columns(df) if c not in drop]
    sub = df.loc[r]
    X = sub[cols].astype({c: "float32" for c in cols if sub[c].dtype == bool}).reset_index(drop=True)
    y = sub[outcome].to_numpy(dtype=int)
    p = np.zeros(len(sub))
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, sub["match_id"])):
        m = fit_outcome(learner, seed + 1000 + fold, X.iloc[tr], y[tr],
                        sub["match_id"].to_numpy()[tr], n_jobs=-1)
        p[te] = m.predict_proba(X.iloc[te])[:, 1]
    key = (sub["match_id"].astype("int64") * 10**10 + sub["account_id"].astype("int64")).to_numpy()
    pm = pd.Series(p).groupby(key).mean()
    rng = np.random.default_rng(seed)
    draw = pd.Series((rng.random(len(pm)) < pm.to_numpy()).astype(float), index=pm.index)
    y_star = np.full(len(df), np.nan)
    y_star[r] = draw.reindex(key).to_numpy()
    res = nc_player(df.assign(_y_star=y_star), rows, learner, e_all=e_all, outcome="_y_star",
                    drop=drop)
    res["p_sd"] = float(pm.std())
    return res


def _nc_player_rule(eb: dict) -> bool:
    return (abs(eb["prior_mean"]) <= THRESHOLDS["nc_player_mu"]
            and eb["tau"] <= THRESHOLDS["nc_player_tau"])


def null_fail_rate(se: np.ndarray, draws: int = 2000, seed: int = 0) -> float:
    """How often the §4.1 rule fails when every true effect is exactly zero, given only the
    arms' standard errors (treated as independent, so approximate). A property of the rule at
    this sample size; it does not look at the estimates."""
    rng = np.random.default_rng(seed)
    se = np.asarray(se, dtype=float)
    fails = [not _nc_player_rule(empirical_bayes(rng.normal(0, se), se)) for _ in range(draws)]
    return float(np.mean(fails))


def nc_ingame(df: pd.DataFrame, rows: np.ndarray, learner: str = "logistic",
              e_all: np.ndarray | None = None) -> dict:
    """§4.2: effect of the choice on souls gained and deaths in the interval before it."""
    out, ok = {}, True
    for col in ("nc_prior_souls", "nc_prior_deaths"):
        r = np.asarray(rows, dtype=bool) & df[col].notna().to_numpy()
        sd = float(df.loc[r, col].std())
        d = df.assign(_nc=df[col] / sd)
        tab = effect_table(fit_rows(d, r, learner, outcome="_nc", continuous=True,
                                    e_hat=_sub(e_all, r)))
        a = tab["effect"].abs().dropna()
        share = float((a <= THRESHOLDS["nc_ingame_arm"]).mean())
        passed = share >= THRESHOLDS["nc_ingame_share"] and a.mean() <= THRESHOLDS["nc_ingame_mean"]
        out[col] = dict(table=tab, share_within=share, mean_abs=float(a.mean()), sd=sd,
                        passed=bool(passed))
        ok &= passed
    out["passed"] = bool(ok)
    return out


def se_calibration(fit: MultiArmFit, k: int | None = None, reps: int = 200,
                   seed: int = 0) -> dict:
    """§4.3: split arm k's buyers at random into two fake arms. Their true difference is
    zero; the fake arms' propensities are e_k / 2 and their outcome model is mu_k."""
    k = int(np.bincount(fit.A).argmax()) if k is None else k
    rng = np.random.default_rng(seed)
    rows = fit.support[:, k]
    A, Y, g = fit.A[rows], fit.Y[rows], fit.groups[rows]
    e2, mu = fit.e[rows, k] / 2, fit.mu[rows, k]
    chose = A == k
    covers, z = [], []
    for _ in range(reps):
        half = chose & (rng.random(len(A)) < 0.5)
        d = (half * (Y - mu) - (chose & ~half) * (Y - mu)) / e2
        m = d.mean()
        se = _clustered_se(d - m, g)
        covers.append(abs(m) <= 1.96 * se)
        z.append(m / se)
    cov = float(np.mean(covers))
    return dict(arm=fit.arms[k], coverage=cov, z_sd=float(np.std(z)), reps=reps,
                passed=THRESHOLDS["se_cov_lo"] <= cov <= THRESHOLDS["se_cov_hi"])


def split_half(df: pd.DataFrame, rows: np.ndarray, learner: str = "logistic",
               seed: int = 0) -> dict:
    """§4.4: estimate on two random halves of the matches and compare per arm. Each half fits
    its own propensities, so the halves share nothing (reusing propensities fitted on both
    halves made them agree more than their SEs allow). Only the standardized differences are
    returned, never the effects themselves."""
    rows = np.asarray(rows, dtype=bool)
    matches = df.loc[rows, "match_id"].unique()
    rng = np.random.default_rng(seed)
    in_a = set(matches[rng.random(len(matches)) < 0.5])
    half_a = rows & df["match_id"].isin(in_a).to_numpy()
    ta = effect_table(fit_rows(df, half_a, learner)).set_index("arm")
    tb = effect_table(fit_rows(df, rows & ~half_a, learner)).set_index("arm")
    z = ((ta["effect"] - tb["effect"]) / np.sqrt(ta["se"] ** 2 + tb["se"] ** 2)).dropna()
    share = float((z.abs() > 1.96).mean())
    return dict(z=z.rename("z").reset_index(), share_beyond=share, z_sd=float(z.std()),
                passed=share <= THRESHOLDS["split_half_share"])


# -- sensitivity --------------------------------------------------------------------------

def _riesz(fit: MultiArmFit, k: int, j: int | None):
    """Rows, Riesz representer alpha, E[alpha^2 | X] and the estimate for the headline
    (j None: arm k instead of what was actually bought) or the pairwise contrast k vs j."""
    A = fit.A
    if j is None:
        rows = fit.support[:, k]
        p = float((A[rows] != k).mean())
        ek = fit.e[rows, k]
        alpha = (A[rows] == k) / ek / p          # E[Y] needs no adjustment
        m_alpha = 1 / ek / p ** 2
        theta = vs_actual(fit, k)["effect"]
    else:
        rows = fit.support[:, k] & fit.support[:, j]
        ek, ej = fit.e[rows, k], fit.e[rows, j]
        alpha = (A[rows] == k) / ek - (A[rows] == j) / ej
        m_alpha = 1 / ek + 1 / ej
        theta = pairwise(fit, k, j)["effect"]
    return rows, alpha, m_alpha, theta


def ovb(fit: MultiArmFit, k: int, j: int | None = None, cf_y: float = 0.03,
        cf_d: float = 0.03, rho: float = 1.0) -> dict:
    """Bounds on the headline (j None) or pairwise effect under a hidden confounder of the
    given strength, and the robustness value: the strength (cf_y = cf_d) that would move the
    estimate to zero. ``rva`` is the same for the edge of the 95% interval, holding the SE
    fixed (an approximation)."""
    rows, alpha, m_alpha, theta = _riesz(fit, k, j)
    mu_obs = fit.mu[rows, fit.A[rows]]
    sigma2 = float(np.mean((fit.Y[rows] - mu_obs) ** 2))
    nu2 = float(max(np.mean(2 * m_alpha - alpha ** 2), 0.0))
    S = np.sqrt(sigma2 * nu2)
    bias = S * abs(rho) * np.sqrt(cf_y * cf_d / (1 - cf_d))
    se = (vs_actual(fit, k) if j is None else pairwise(fit, k, j))["se"]

    def rv(target):
        if S == 0 or target <= 0:
            return 0.0
        a = (target / S) ** 2
        return float((-a + np.sqrt(a * a + 4 * a)) / 2)

    return dict(arm=fit.arms[k], vs=None if j is None else fit.arms[j], effect=theta,
                sigma2=sigma2, nu2=nu2, bias=float(bias), lower=theta - bias,
                upper=theta + bias, rv=rv(abs(theta)), rva=rv(abs(theta) - 1.96 * se))


def ovb_benchmark(long: MultiArmFit, short: MultiArmFit, k: int,
                  j: int | None = None) -> dict:
    """cf_y and cf_d implied by a set of observed covariates (e.g. skill): compare the full
    fit with one that leaves them out. A hidden confounder "as strong as rank" has these."""
    lo, sh = ovb(long, k, j), ovb(short, k, j)
    var_y = float(np.var(long.Y))
    r2_long, r2_short = 1 - lo["sigma2"] / var_y, 1 - sh["sigma2"] / var_y
    cf_y = float(np.clip((r2_long - r2_short) / (1 - r2_long), 0, 1))
    r2_riesz = sh["nu2"] / lo["nu2"] if lo["nu2"] > 0 else 1.0
    cf_d = float(np.clip((1 - r2_riesz) / r2_riesz, 0, 1)) if r2_riesz > 0 else 1.0
    return dict(cf_y=cf_y, cf_d=cf_d, delta=sh["effect"] - lo["effect"])


def pooled_split_half(z: pd.DataFrame) -> dict:
    """§4.4 applied once per hero and window: ``z`` stacks every price's split-half z."""
    share = float((z["z"].abs() > 1.96).mean())
    return dict(share_beyond=share, z_sd=float(z["z"].std()), n_arms=len(z),
                passed=share <= THRESHOLDS["split_half_share"])


# -- calibration --------------------------------------------------------------------------

def load_calibration(out, learner: str = "lgbm") -> dict:
    """Systematic error N(mu, tau^2) for a hero and window (PLAN.md §8, 2026-10-15): the pooled
    next-match control from ``pooled_falsify_<learner>.json``, the only control fitted with
    the production covariates. ``mu_se`` is its match-clustered bootstrap SE."""
    import json
    from pathlib import Path

    r = json.loads((Path(out) / f"pooled_falsify_{learner}.json").read_text())["nc_next"]
    boot = {b["cluster"]: b for b in r["bootstrap"]}
    return dict(source="nc_next", learner=learner, mu=float(r["mu"]), tau=float(r["tau"]),
                mu_se=float(boot["match_id"]["mu_se"]))


def calibrate(effect, se, cal: dict, pairwise: bool = False):
    """Empirical calibration, shift and widen (Schuemie et al. 2018). A headline effect is
    shifted by -mu and its SE widened by tau and se(mu); a pairwise contrast is not shifted (a
    common bias cancels) and is widened by sqrt(2) tau."""
    effect, se = np.asarray(effect, dtype=float), np.asarray(se, dtype=float)
    if pairwise:
        return effect, np.sqrt(se ** 2 + 2 * cal["tau"] ** 2)
    return effect - cal["mu"], np.sqrt(se ** 2 + cal["tau"] ** 2 + cal["mu_se"] ** 2)


def dev_rows(df: pd.DataFrame) -> np.ndarray:
    return is_dev(df["match_id"].to_numpy())
