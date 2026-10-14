"""Effect curves over game time, and shrinkage for rankings (PLAN.md §3.3-3.4).

``dr_curve`` is the DR-learner second stage (Kennedy 2023; Semenova & Chernozhukov 2021):
regress the doubly-robust pseudo-outcome on a spline basis in game time, with a
match-clustered sandwich covariance. That gives one smooth curve with pointwise bands,
replacing v1's overlapping windows (whose per-point maximum, the "ideal timing", was biased
upward). For the headline estimand ("vs what they actually bought") the curve is a ratio of
two spline fits on the same basis, ``E[psi_k - Y | t] / P(A != k | t)``, with a delta-method
band from their joint clustered sandwich.

``empirical_bayes`` shrinks a family of noisy estimates toward their precision-weighted mean
(normal-normal, DerSimonian-Laird variance). Rankings are read off the shrunk values, so a
noisy item cannot top a list on luck alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aipw import MultiArmFit, vs_actual_scores, vs_alternatives_scores


def _spline_basis(t: np.ndarray, knots: np.ndarray, degree: int = 3):
    from sklearn.preprocessing import SplineTransformer
    st = SplineTransformer(knots=knots.reshape(-1, 1), degree=degree, include_bias=True)
    st.fit(t.reshape(-1, 1))
    return st


def dr_curve(fit: MultiArmFit, k: int, t: np.ndarray, grid: np.ndarray,
             n_knots: int = 6, estimand: str = "vs_actual") -> pd.DataFrame:
    """Smooth effect of arm k as a function of ``t`` (e.g. game_time_s).

    ``estimand``: "vs_actual" (the headline) or "vs_alternatives" (a uniformly random
    supported alternative).
    """
    if estimand == "vs_actual":
        rows, d, other = vs_actual_scores(fit, k)
        ys = np.column_stack([d, other])
    elif estimand == "vs_alternatives":
        rows, d = vs_alternatives_scores(fit, k)
        ys = d[:, None]
    else:
        raise ValueError(f"unknown estimand {estimand!r}")
    tt = np.asarray(t, dtype=float)[rows]
    g = fit.groups[rows]
    knots = np.unique(np.quantile(tt, np.linspace(0, 1, n_knots)))
    st = _spline_basis(tt, knots)
    B = st.transform(tt.reshape(-1, 1))
    BtB_inv = np.linalg.pinv(B.T @ B)
    beta = BtB_inv @ B.T @ ys                       # (p, m): one column per regression
    resid = ys - B @ beta
    # stacked clustered sandwich over all m regressions (they share rows and clusters)
    scores = np.hstack([B * resid[:, [j]] for j in range(ys.shape[1])])
    S = pd.DataFrame(scores).groupby(g).sum().to_numpy()
    bread = np.kron(np.eye(ys.shape[1]), BtB_inv)
    V = bread @ (S.T @ S) @ bread
    Bg = st.transform(np.clip(grid, knots[0], knots[-1]).reshape(-1, 1))
    if ys.shape[1] == 1:
        est = Bg @ beta[:, 0]
        grad = Bg
    else:
        num, den = Bg @ beta[:, 0], Bg @ beta[:, 1]
        est = num / den
        grad = np.hstack([Bg / den[:, None], -Bg * (num / den ** 2)[:, None]])
    se = np.sqrt(np.einsum("ij,jk,ik->i", grad, V, grad))
    return pd.DataFrame(dict(t=grid, effect=est, se=se,
                             ci_lo=est - 1.96 * se, ci_hi=est + 1.96 * se))


def empirical_bayes(effect: np.ndarray, se: np.ndarray) -> dict:
    """Normal-normal shrinkage. Returns prior mean, between-item sd, and posterior means/sds."""
    y = np.asarray(effect, dtype=float)
    s2 = np.asarray(se, dtype=float) ** 2
    w = 1 / s2
    m_fe = np.sum(w * y) / np.sum(w)
    Q = np.sum(w * (y - m_fe) ** 2)
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (Q - (len(y) - 1)) / c) if c > 0 else 0.0
    wr = 1 / (s2 + tau2)
    m = np.sum(wr * y) / np.sum(wr)
    B = tau2 / (tau2 + s2) if tau2 > 0 else np.zeros_like(y)
    post = m + B * (y - m)
    post_sd = np.sqrt(B * s2) if tau2 > 0 else np.sqrt(1 / np.sum(wr)) * np.ones_like(y)
    return dict(prior_mean=float(m), tau=float(np.sqrt(tau2)), post_mean=post, post_sd=post_sd)
