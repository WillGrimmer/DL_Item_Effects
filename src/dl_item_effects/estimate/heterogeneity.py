"""Effect curves over game time, and shrinkage for rankings (PLAN.md §3.3-3.4).

``dr_curve`` is Kennedy's DR-learner second stage: regress the doubly-robust pseudo-outcome
for "arm k vs alternatives" on a spline basis in game time, with a match-clustered sandwich
covariance. That gives one smooth curve with honest pointwise bands, replacing v1's
overlapping windows (whose per-point maximum, the "ideal timing", was biased upward).

``empirical_bayes`` shrinks a family of noisy estimates toward their precision-weighted mean
(normal-normal, DerSimonian-Laird variance). Rankings are read off the shrunk values, so a
noisy item cannot top a list on luck alone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .aipw import MultiArmFit, vs_alternatives_scores


def _spline_basis(t: np.ndarray, knots: np.ndarray, degree: int = 3):
    from sklearn.preprocessing import SplineTransformer
    st = SplineTransformer(knots=knots.reshape(-1, 1), degree=degree, include_bias=True)
    st.fit(t.reshape(-1, 1))
    return st


def dr_curve(fit: MultiArmFit, k: int, t: np.ndarray, grid: np.ndarray,
             n_knots: int = 6) -> pd.DataFrame:
    """Smooth effect of arm k vs alternatives as a function of ``t`` (e.g. game_time_s)."""
    rows, d = vs_alternatives_scores(fit, k)
    tt = np.asarray(t, dtype=float)[rows]
    g = fit.groups[rows]
    knots = np.quantile(tt, np.linspace(0, 1, n_knots))
    knots = np.unique(knots)
    st = _spline_basis(tt, knots)
    B = st.transform(tt.reshape(-1, 1))
    BtB_inv = np.linalg.pinv(B.T @ B)
    beta = BtB_inv @ B.T @ d
    resid = d - B @ beta
    S = pd.DataFrame(B * resid[:, None]).groupby(g).sum().to_numpy()
    V = BtB_inv @ (S.T @ S) @ BtB_inv
    Bg = st.transform(np.clip(grid, knots[0], knots[-1]).reshape(-1, 1))
    est = Bg @ beta
    se = np.sqrt(np.einsum("ij,jk,ik->i", Bg, V, Bg))
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
