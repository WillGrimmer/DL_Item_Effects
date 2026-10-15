"""Multi-arm cross-fitted AIPW with per-row eligibility (PLAN.md §3.2).

Treatment is the item chosen among K same-cost arms. For each decision row i and arm k the
doubly-robust score is

    psi_ik = mu_k(x_i) + 1{A_i = k} (Y_i - mu_k(x_i)) / e_k(x_i)

where ``e_k`` is the generalized propensity P(A=k | x) restricted to the arms the player
can actually buy (not already owned) and renormalized, and ``mu_k`` is P(win | x, A=k).
Nuisances are fit out-of-fold with folds grouped by match (double ML). Averages of
``psi_k - psi_j`` over the rows where both arms are supported give the pairwise effect; the
match-clustered variance of that average is the SE.

Overlap: an arm is *supported* on a row if it is eligible AND its propensity is at least
``trim``. Contrasts are restricted to rows where every arm involved is supported (a
covariate-defined overlap population, as in Crump et al. 2009). Rows are dropped, never
clipped: clipping propensities biases AIPW, which was one of the v1 problems.

The per-item headline is :func:`vs_actual`: choosing arm k instead of what the player
actually bought, averaged over supported rows that bought something else (PLAN.md §3.3).
"""
from __future__ import annotations

import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Crossover labels for "X now vs Y now" (PLAN.md §3.2): share of the comparison group that
# buys X later in the match.
CROSSOVER_LABELS = ((0.25, "choice"), (0.50, "mixed"), (1.01, "order"))

# Fits run in parallel threads (LightGBM and BLAS release the GIL), each with a share of the
# cores: on 16 cores, 4 fits x 4 threads ran 8 arm models 2.7x faster than one at a time on
# 16 threads, because a single small fit scales badly across threads. DL_FIT_WORKERS
# overrides the worker count.
FIT_WORKERS = int(os.environ.get("DL_FIT_WORKERS", 4))
FIT_THREADS = max(1, (os.cpu_count() or 4) // FIT_WORKERS)

# LightGBM outcome models stop early on a held-out 10% of the training matches (patience 50,
# at most 1000 trees). With a fixed 400 trees they overfit: on Haze's dev set the held-out
# log-loss for the previous-match outcome was 0.715 against 0.693 for a constant, and 0.691
# with early stopping (PLAN.md §8, 2026-10-14). Propensities keep 400 trees: early stopping
# did not improve them.
ES_MAX_TREES, ES_PATIENCE, ES_VALID_SHARE, ES_MIN_ROWS = 1000, 50, 0.10, 2000


def _parallel(fn, tasks, workers: int = FIT_WORKERS) -> list:
    if workers <= 1 or len(tasks) <= 1:
        return [fn(t) for t in tasks]
    with ThreadPoolExecutor(min(workers, len(tasks))) as ex:
        return list(ex.map(fn, tasks))


def fit_outcome(kind: str, seed: int, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                regression: bool = False, n_jobs: int = FIT_THREADS):
    """Fit an outcome model; LightGBM stops early on held-out training matches."""
    if kind != "lgbm" or len(y) < ES_MIN_ROWS:
        return make_learner(kind, seed, regression, n_jobs=n_jobs).fit(X, y)
    import lightgbm as lgb
    um = np.unique(groups)
    val = np.isin(groups, um[np.random.default_rng(seed).random(len(um)) < ES_VALID_SHARE])
    if not regression and (np.unique(y[val]).size < 2 or np.unique(y[~val]).size < 2):
        return make_learner(kind, seed, regression, n_jobs=n_jobs).fit(X, y)
    m = make_learner(kind, seed, regression, n_jobs=n_jobs, n_estimators=ES_MAX_TREES)
    return m.fit(X[~val], y[~val], eval_set=[(X[val], y[val])],
                 callbacks=[lgb.early_stopping(ES_PATIENCE, verbose=False)])


def make_learner(kind: str, seed: int, regression: bool = False, n_jobs: int = -1,
                 n_estimators: int = 400):
    """Classifier for propensities (multiclass) and binary outcomes, or a regressor for
    continuous outcomes (the in-game negative controls). Missing values are median-imputed
    with indicators on the logistic/linear path; LightGBM handles them natively and picks
    binary or multiclass from the labels it sees."""
    if kind == "logistic":
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = Ridge(alpha=1.0) if regression else LogisticRegression(max_iter=2000, C=1.0)
        return make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                             StandardScaler(), model)
    if kind == "lgbm":
        import lightgbm as lgb
        cls = lgb.LGBMRegressor if regression else lgb.LGBMClassifier
        return cls(
            n_estimators=n_estimators, learning_rate=0.05, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=n_jobs, random_state=seed, verbosity=-1)
    raise ValueError(f"unknown learner {kind!r}")


@dataclass
class MultiArmFit:
    """Cross-fitted nuisances and scores for one decision frame."""
    psi: np.ndarray        # (n, K) doubly-robust scores, nan where the arm is unsupported
    e: np.ndarray          # (n, K) eligibility-restricted propensities
    mu: np.ndarray         # (n, K) outcome predictions
    support: np.ndarray    # (n, K) bool: eligible and e >= trim
    groups: np.ndarray     # (n,) cluster ids (match)
    A: np.ndarray
    Y: np.ndarray
    arms: list
    elig: np.ndarray       # (n, K) bool: arm buyable on the row
    fold_absent: list = field(default_factory=list)   # (fold, arm) with no training rows

    @property
    def trim_share(self) -> np.ndarray:
        """Per arm: share of eligible rows that are unsupported (weak overlap, or the arm was
        missing from that row's training fold)."""
        return 1 - self.support.sum(0) / np.maximum(self.elig.sum(0), 1)


def fit_propensity(X: pd.DataFrame, A: np.ndarray, elig: np.ndarray, groups: np.ndarray,
                   learner: str = "logistic", n_splits: int = 5, seed: int = 0) -> np.ndarray:
    """Out-of-fold generalized propensities, restricted to buyable arms and renormalized.
    Uses no outcome, so it can be fitted once on all rows and reused for any outcome."""
    from sklearn.model_selection import GroupKFold

    X = X.reset_index(drop=True)
    A = np.asarray(A, dtype=int)
    elig = np.asarray(elig, dtype=bool)
    n, K = elig.shape
    e = np.zeros((n, K))
    folds = list(GroupKFold(n_splits=n_splits).split(X, A, groups))
    threads = max(1, (os.cpu_count() or 4) // min(FIT_WORKERS, len(folds)))

    def one(f):
        fold, (tr, te) = f
        prop = make_learner(learner, seed + fold, n_jobs=threads).fit(X.iloc[tr], A[tr])
        return te, prop.classes_, prop.predict_proba(X.iloc[te])

    for te, classes, p in _parallel(one, list(enumerate(folds))):
        for col, k in enumerate(classes):
            e[te, int(k)] = p[:, col]
    e = np.where(elig, e, 0.0)
    return e / np.maximum(e.sum(1, keepdims=True), 1e-12)


def fit_multiarm(X: pd.DataFrame, A: np.ndarray, Y: np.ndarray, elig: np.ndarray,
                 groups: np.ndarray, arms: list | None = None, learner: str = "logistic",
                 n_splits: int = 5, trim: float = 0.01, seed: int = 0,
                 continuous: bool = False, e_hat: np.ndarray | None = None) -> MultiArmFit:
    """Cross-fitted multi-arm AIPW. ``continuous`` fits regressors for the outcome (for
    negative-control outcomes that are not win/loss). ``e_hat`` reuses out-of-fold
    propensities (:func:`fit_propensity`) for these rows, so only outcome models are fitted."""
    from sklearn.model_selection import GroupKFold

    X = X.reset_index(drop=True)
    A = np.asarray(A, dtype=int)
    Y = np.asarray(Y, dtype=float)
    elig = np.asarray(elig, dtype=bool)
    groups = np.asarray(groups)
    n, K = elig.shape
    mu = np.full((n, K), np.nan)
    fold_absent = []
    e = (fit_propensity(X, A, elig, groups, learner, n_splits, seed) if e_hat is None
         else np.asarray(e_hat, dtype=float).copy())

    tasks = []
    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, A, groups)):
        for k in range(K):
            tk = tr[A[tr] == k]
            if len(tk) == 0:
                fold_absent.append((fold, k))
                continue
            yk = Y[tk]
            if yk.min() == yk.max():
                mu[te, k] = yk[0]
                continue
            tasks.append((fold, k, tk, te))

    def one(t):
        fold, k, tk, te = t
        m = fit_outcome(learner, seed + 100 * fold + k, X.iloc[tk], Y[tk], groups[tk],
                        regression=continuous)
        return k, te, (m.predict(X.iloc[te]) if continuous
                       else m.predict_proba(X.iloc[te])[:, 1])

    for k, te, pred in _parallel(one, tasks):
        mu[te, k] = pred

    # Restrict the generalized propensity to buyable arms and renormalize (idempotent).
    e = np.where(elig, e, 0.0)
    e = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
    support = elig & (e >= trim) & ~np.isnan(mu)
    onehot = np.eye(K)[A]
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = mu + onehot * (Y[:, None] - mu) / e
    psi = np.where(support, psi, np.nan)
    if fold_absent:
        warnings.warn(f"arms missing from a training fold (fold, arm): {fold_absent}; "
                      "their rows in that fold are unsupported", stacklevel=2)
    return MultiArmFit(psi=psi, e=e, mu=mu, support=support, groups=groups, A=A, Y=Y,
                       arms=list(arms) if arms is not None else list(range(K)),
                       elig=elig, fold_absent=fold_absent)


def _clustered_mean(d: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(d.mean())
    return m, _clustered_se(d - m, groups), n


def _clustered_se(infl: np.ndarray, groups: np.ndarray) -> float:
    """SE of a mean from its per-row influence values, clustered by ``groups``."""
    cl = pd.Series(infl).groupby(groups).sum().to_numpy()
    return float(np.sqrt(np.sum(cl ** 2)) / len(infl))


def crossover_label(share_by_end: float) -> str:
    for cut, name in CROSSOVER_LABELS:
        if share_by_end < cut:
            return name
    return "order"


def pairwise(fit: MultiArmFit, k: int, j: int) -> dict:
    """Effect of choosing arm k instead of arm j, on rows where both are supported."""
    rows = fit.support[:, k] & fit.support[:, j]
    d = fit.psi[rows, k] - fit.psi[rows, j]
    m, se, n = _clustered_mean(d, fit.groups[rows])
    return dict(arm=fit.arms[k], vs=fit.arms[j], effect=m, se=se, n=n,
                ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se)


def vs_alternatives_scores(fit: MultiArmFit, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Row mask and pseudo-outcome for 'arm k vs the average supported alternative'."""
    others = np.delete(np.arange(fit.psi.shape[1]), k)
    sup_o = fit.support[:, others]
    rows = fit.support[:, k] & sup_o.any(1)
    base = np.nanmean(np.where(sup_o[rows], fit.psi[rows][:, others], np.nan), axis=1)
    return rows, fit.psi[rows, k] - base


def vs_alternatives(fit: MultiArmFit, k: int) -> dict:
    """Effect of choosing arm k instead of a uniformly random supported alternative.

    This is the headline per-item number: a common, explicit baseline, so arms at the same
    cost can be ranked. Its expectation is not forced to zero across arms, but there is no
    extra skew from unequal weighting: one row per decision in every arm.
    """
    rows, d = vs_alternatives_scores(fit, k)
    m, se, n = _clustered_mean(d, fit.groups[rows])
    return dict(arm=fit.arms[k], effect=m, se=se, n=n, n_chose=int((fit.A[rows] == k).sum()),
                trim_share=float(fit.trim_share[k]),
                ci_lo=m - 1.96 * se, ci_hi=m + 1.96 * se)


def vs_actual_scores(fit: MultiArmFit, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row mask, ``psi_k - Y`` and ``1{A != k}`` on rows where arm k is supported."""
    rows = fit.support[:, k]
    return rows, fit.psi[rows, k] - fit.Y[rows], (fit.A[rows] != k).astype(float)


def vs_actual(fit: MultiArmFit, k: int, next_buy_s: np.ndarray | None = None,
              within_s: float = 600.0) -> dict:
    """Headline: choosing arm k instead of what the player actually bought, averaged over
    supported rows whose player bought something else.

    E[Y(k) - Y | A != k] = E[psi_k - Y] / P(A != k) on rows where k is supported. The SE comes
    from the ratio's influence function, clustered by match.

    ``next_buy_s`` (n, K): seconds from the decision until the player buys each arm later
    (NaN if never). With it, the crossover shares of the comparison group are added.
    """
    rows, d, other = vs_actual_scores(fit, k)
    n = int(rows.sum())
    out = dict(arm=fit.arms[k], effect=float("nan"), se=float("nan"), n=n,
               n_chose=int((fit.A[rows] == k).sum()), n_other=int(other.sum()),
               trim_share=float(fit.trim_share[k]))
    if n and other.sum():
        p = other.mean()
        theta = d.mean() / p
        se = _clustered_se((d - theta * other) / p, fit.groups[rows])
        out.update(effect=float(theta), se=se, ci_lo=theta - 1.96 * se,
                   ci_hi=theta + 1.96 * se)
    if next_buy_s is not None and other.sum():
        nb = np.asarray(next_buy_s, dtype=float)[rows][other == 1, k]
        by_end = float(np.mean(~np.isnan(nb)))
        out.update(cross_within=float(np.mean(nb <= within_s)), cross_by_end=by_end,
                   crossover=crossover_label(by_end))
    return out


def effect_table(fit: MultiArmFit, next_buy_s: np.ndarray | None = None) -> pd.DataFrame:
    """One headline row per arm (:func:`vs_actual`)."""
    return pd.DataFrame([vs_actual(fit, k, next_buy_s) for k in range(fit.psi.shape[1])])


def pairwise_table(fit: MultiArmFit) -> pd.DataFrame:
    K = fit.psi.shape[1]
    return pd.DataFrame([pairwise(fit, k, j) for k in range(K) for j in range(K) if k != j])
