"""Multi-arm AIPW recovers known effects on a confounded synthetic DGP with ownership."""
import numpy as np
import pandas as pd
import pytest

from dl_item_effects.estimate import (benjamini_hochberg, dr_curve, empirical_bayes,
                                      fit_multiarm, pairwise, vs_alternatives)

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
    cur = dr_curve(f, 0, t, np.array([0.15, 0.85]))
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
