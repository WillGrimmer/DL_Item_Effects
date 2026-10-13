"""Multiple-testing control (ported from DL_Decision_Analytics).

Benjamini-Hochberg q-values across a whole family of effects (every hero x cost x arm), so
"significant" means the expected false-discovery rate is at most alpha.
"""
from __future__ import annotations

import numpy as np


def benjamini_hochberg(p: np.ndarray | list[float]) -> np.ndarray:
    """Benjamini–Hochberg step-up FDR q-values for a family of p-values.

    ``q_i = min_{j >= i} ( n * p_(j) / j )`` over the ascending-sorted p-values, then mapped
    back to the input order. Rejecting every ``q <= alpha`` controls the false-discovery rate
    at ``alpha`` under independence or positive dependence.
    """
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]      # enforce monotonicity (step-up)
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


def two_sided_p(effect: np.ndarray, se: np.ndarray) -> np.ndarray:
    """Two-sided normal p-value for ``effect/se``. Zero/invalid SE → p = 1 (untestable)."""
    from scipy.special import erfc

    effect = np.asarray(effect, dtype=float)
    se = np.asarray(se, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where((se > 0) & np.isfinite(se), effect / se, 0.0)
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    return erfc(np.abs(z) / np.sqrt(2.0))
