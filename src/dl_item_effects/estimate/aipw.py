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
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def make_learner(kind: str, seed: int, multiclass: bool = False):
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            objective="multiclass" if multiclass else "binary",
            n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=-1, random_state=seed, verbosity=-1)
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

    @property
    def trim_share(self) -> np.ndarray:
        """Per arm: share of eligible rows dropped for weak overlap."""
        elig = ~np.isnan(self.e) & (self.e > 0)
        return 1 - self.support.sum(0) / np.maximum(elig.sum(0), 1)


def fit_multiarm(X: pd.DataFrame, A: np.ndarray, Y: np.ndarray, elig: np.ndarray,
                 groups: np.ndarray, arms: list | None = None, learner: str = "logistic",
                 n_splits: int = 5, trim: float = 0.01, seed: int = 0) -> MultiArmFit:
    from sklearn.model_selection import GroupKFold

    X = X.reset_index(drop=True)
    A = np.asarray(A, dtype=int)
    Y = np.asarray(Y, dtype=float)
    elig = np.asarray(elig, dtype=bool)
    groups = np.asarray(groups)
    n, K = elig.shape
    e = np.zeros((n, K))
    mu = np.full((n, K), np.nan)

    for fold, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, A, groups)):
        prop = make_learner(learner, seed + fold, multiclass=True)
        prop.fit(X.iloc[tr], A[tr])
        p = prop.predict_proba(X.iloc[te])
        for col, k in enumerate(prop.classes_):
            e[te, int(k)] = p[:, col]
        for k in range(K):
            tk = tr[A[tr] == k]
            if len(tk) == 0:
                continue
            yk = Y[tk]
            if yk.min() == yk.max():
                mu[te, k] = yk[0]
                continue
            m = make_learner(learner, seed + 100 * fold + k)
            m.fit(X.iloc[tk], yk)
            mu[te, k] = m.predict_proba(X.iloc[te])[:, 1]

    # Restrict the generalized propensity to buyable arms and renormalize.
    e = np.where(elig, e, 0.0)
    e = e / np.maximum(e.sum(1, keepdims=True), 1e-12)
    support = elig & (e >= trim) & ~np.isnan(mu)
    onehot = np.eye(K)[A]
    with np.errstate(divide="ignore", invalid="ignore"):
        psi = mu + onehot * (Y[:, None] - mu) / e
    psi = np.where(support, psi, np.nan)
    return MultiArmFit(psi=psi, e=e, mu=mu, support=support, groups=groups, A=A, Y=Y,
                       arms=list(arms) if arms is not None else list(range(K)))


def _clustered_mean(d: np.ndarray, groups: np.ndarray) -> tuple[float, float, int]:
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(d.mean())
    cl = pd.Series(d - m).groupby(groups).sum().to_numpy()
    return m, float(np.sqrt(np.sum(cl ** 2)) / n), n


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


def effect_table(fit: MultiArmFit) -> pd.DataFrame:
    return pd.DataFrame([vs_alternatives(fit, k) for k in range(fit.psi.shape[1])])


def pairwise_table(fit: MultiArmFit) -> pd.DataFrame:
    K = fit.psi.shape[1]
    return pd.DataFrame([pairwise(fit, k, j) for k in range(K) for j in range(K) if k != j])
