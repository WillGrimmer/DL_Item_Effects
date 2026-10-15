"""Multi-arm AIPW recovers known effects on a confounded synthetic DGP with ownership."""
import numpy as np
import pandas as pd
import pytest

import warnings

from dl_item_effects.estimate import (benjamini_hochberg, dr_curve, empirical_bayes,
                                      effect_table, fit_multiarm, pairwise, vs_actual,
                                      vs_alternatives)

TRUE = np.array([0.04, 0.0, -0.02])     # win-prob shift from choosing arm 0 / 1 / 2


def simulate(n=60000, seed=1, time_varying=False):
    rng = np.random.default_rng(seed)
    skill = rng.normal(size=n)
    t = rng.uniform(0, 1, n)
    own0 = (rng.random(n) < 0.3 + 0.2 * (skill > 0)).astype(int)   # skilled players own arm 0
    # skilled players prefer arm 0 (confounding); owners cannot pick arm 0
    logits = np.stack([1.5 * skill, np.zeros(n), -0.5 * skill], 1)
    logits[own0 == 1, 0] = -np.inf
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    A = np.array([rng.choice(3, p=pi) for pi in p])
    eff0 = 0.08 * (t - 0.5) + TRUE[0] if time_varying else np.full(n, TRUE[0])
    eff = np.select([A == 0, A == 1, A == 2], [eff0, TRUE[1], TRUE[2]])
    pw = 0.5 + 0.15 * np.tanh(skill) + 0.03 * own0 + eff
    Y = (rng.random(n) < pw).astype(int)
    X = pd.DataFrame(dict(skill=skill, t=t, own0=own0))
    elig = np.stack([own0 == 0, np.ones(n, bool), np.ones(n, bool)], 1)
    return X, A, Y, elig, np.arange(n) // 6, t


@pytest.fixture(scope="module")
def fit():
    X, A, Y, elig, g, _ = simulate()
    return fit_multiarm(X, A, Y, elig, g, learner="logistic")


def test_naive_is_biased(fit):
    naive = fit.Y[fit.A == 0].mean() - fit.Y[fit.A == 1].mean()
    assert naive - (TRUE[0] - TRUE[1]) > 0.02


def test_pairwise_recovers_truth(fit):
    for k, j in [(0, 1), (2, 1), (0, 2)]:
        r = pairwise(fit, k, j)
        assert r["ci_lo"] <= TRUE[k] - TRUE[j] <= r["ci_hi"], r


def test_vs_alternatives_ordering(fit):
    effs = [vs_alternatives(fit, k)["effect"] for k in range(3)]
    assert effs[0] > effs[1] > effs[2]


def test_dr_curve_tracks_time_varying_effect():
    X, A, Y, elig, g, t = simulate(time_varying=True, seed=3)
    f = fit_multiarm(X, A, Y, elig, g, learner="logistic")
    cur = dr_curve(f, 0, t, np.array([0.15, 0.85]), estimand="vs_alternatives")
    # arm 0 vs average of arms 1,2 = eff0 - (-0.01); slope 0.08 over t
    truth = 0.08 * (np.array([0.15, 0.85]) - 0.5) + 0.04 + 0.01
    assert (cur.ci_lo.to_numpy() <= truth).all() and (truth <= cur.ci_hi.to_numpy()).all()
    assert cur.effect.iloc[1] > cur.effect.iloc[0]


def test_empirical_bayes_shrinks_noisy_extremes():
    y = np.array([0.10, 0.01, 0.0, -0.01, 0.02])
    se = np.array([0.08, 0.005, 0.005, 0.005, 0.005])
    eb = empirical_bayes(y, se)
    assert abs(eb["post_mean"][0]) < 0.05          # the noisy +10% is pulled in hard
    assert abs(eb["post_mean"][1] - 0.01) < 0.005  # precise estimates barely move


def test_bh_monotone_and_bounded():
    q = benjamini_hochberg([0.001, 0.04, 0.03, 0.5])
    assert (q >= np.array([0.001, 0.04, 0.03, 0.5])).all() and q.max() <= 1


def _vs_actual_truth(fit, k=0):
    """E[Y(k) - Y | A != k] on supported rows: effects are constant shifts in this DGP."""
    rows = fit.support[:, k] & (fit.A != k)
    return TRUE[k] - TRUE[fit.A[rows]].mean()


def test_vs_actual_recovers_truth(fit):
    for k in range(3):
        r = vs_actual(fit, k)
        assert r["ci_lo"] <= _vs_actual_truth(fit, k) <= r["ci_hi"], (k, r)
    tab = effect_table(fit)
    assert list(tab.arm) == [0, 1, 2] and (tab.n_other > 0).all()


def test_vs_actual_curve_tracks_time_varying_effect():
    X, A, Y, elig, g, t = simulate(time_varying=True, seed=3)
    f = fit_multiarm(X, A, Y, elig, g, learner="logistic")
    grid = np.array([0.15, 0.85])
    cur = dr_curve(f, 0, t, grid)
    rows = f.support[:, 0] & (f.A != 0)
    truth = 0.08 * (grid - 0.5) + TRUE[0] - TRUE[f.A[rows]].mean()
    assert (cur.ci_lo.to_numpy() <= truth).all() and (truth <= cur.ci_hi.to_numpy()).all()


def test_crossover_shares_and_label(fit):
    n = len(fit.A)
    nb = np.full((n, 3), np.nan)
    nb[:, 0] = np.where(np.arange(n) % 10 < 3, 120.0, np.nan)   # 30% buy arm 0 later
    r = vs_actual(fit, 0, next_buy_s=nb)
    assert abs(r["cross_by_end"] - 0.3) < 0.02 and r["crossover"] == "mixed"
    assert r["cross_within"] == pytest.approx(r["cross_by_end"])


def test_lgbm_handles_two_arms_and_missing_values():
    X, A, Y, elig, g, _ = simulate(n=6000, seed=5)
    keep = A < 2
    X = X[keep].reset_index(drop=True)
    X.loc[::7, "skill"] = np.nan
    for learner in ("lgbm", "logistic"):
        f = fit_multiarm(X, A[keep], Y[keep], elig[keep][:, :2], g[keep], learner=learner)
        assert np.isfinite(pairwise(f, 0, 1)["effect"])


def test_arm_missing_from_a_fold_counts_as_trimmed():
    X, A, Y, elig, g, _ = simulate(n=6000, seed=7)
    A = A.copy()
    rare = np.flatnonzero(A == 2)[:3]
    A[A == 2] = 1
    A[rare] = 2                        # three rows of arm 2: some folds will have none
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        f = fit_multiarm(X, A, Y, elig, g, learner="logistic")
    assert f.fold_absent and any("missing from a training fold" in str(x.message) for x in w)
    assert f.trim_share[2] > 0.5


def test_lgbm_outcome_models_stop_early_and_recover_truth():
    from dl_item_effects.estimate.aipw import fit_outcome
    X, A, Y, elig, g, _ = simulate(seed=11)
    noise = np.random.default_rng(0).integers(0, 2, len(Y))
    m = fit_outcome("lgbm", 0, X, noise, g)
    assert m.best_iteration_ < 200          # nothing to learn: stops long before 1000
    f = fit_multiarm(X, A, Y, elig, g, learner="lgbm")
    for k, j in [(0, 1), (2, 1)]:
        r = pairwise(f, k, j)
        assert r["ci_lo"] <= TRUE[k] - TRUE[j] <= r["ci_hi"], r
