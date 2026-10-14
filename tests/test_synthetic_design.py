"""Synthetic recovery through the real frame and estimator (PLAN.md §4.6).

Each test simulates purchase events with a known truth, builds the decision frame and fits
the multi-arm AIPW, so the design choices are checked end to end:

* crossover: "X now vs Y now" equals the ownership effect times (1 - crossover), and the
  reported crossover share is the one in the data;
* same-visit baskets: grouping purchases into visits removes click-order artifacts, and
  without it the estimate is biased (docs/design_checks_2026-10-13.md §2);
* component holders: discounted upgrades are not trials and holders cannot "choose" the
  upgrade at full price, so their higher win rate does not leak into the contrast.
"""
import numpy as np
import pandas as pd
import pytest

from dl_item_effects.estimate import fit_multiarm, pairwise, vs_actual
from dl_item_effects.frame import covariate_columns, decision_frame

X, Y, Z, U, C = 11, 12, 13, 14, 20          # U (3200) upgrades from C (800)
OWN_X = 0.03                                # owning X at the end is worth +3pp


def _sim(n=100_000, p_cross=0.0, q_basket=0.0, h_holder=0.0, seed=0):
    rng = np.random.default_rng(seed)
    s = rng.normal(size=n)
    holder = rng.random(n) < h_holder
    t0 = rng.uniform(300, 900, n).round()
    arms = np.array([X, Y, Z, U])
    logits = np.stack([0.8 * s, np.zeros(n), -0.5 * s, 0.3 * s + 1.0 * holder], 1)
    if not h_holder:
        logits[:, 3] = -np.inf                      # no U in the crossover/basket designs
    p = np.exp(logits - logits.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    first = arms[(p.cumsum(1) > rng.random((n, 1))).argmax(1)]

    rows, owns_x, chose_u = [], np.zeros(n, bool), np.zeros(n, bool)
    for i in range(n):
        if holder[i]:                               # component bought early, full price
            sold = t0[i] if first[i] == U else 0
            rows.append((i, 100, C, 800, 800, True, sold))
        if first[i] == U and holder[i]:
            rows.append((i, t0[i], U, 3200, 2400, False, 0))
        else:
            rows.append((i, t0[i], first[i], 3200, 3200, True, 0))
        chose_u[i] = first[i] == U
        owns_x[i] = first[i] == X
        if first[i] in (X, Y):
            other = Y if first[i] == X else X
            if rng.random() < q_basket:             # both in one visit, random click order
                if rng.random() < 0.5:
                    rows[-1] = (i, t0[i] + rng.integers(1, 4), first[i], 3200, 3200, True, 0)
                    rows.append((i, t0[i], other, 3200, 3200, True, 0))
                else:
                    rows.append((i, t0[i] + rng.integers(1, 4), other, 3200, 3200, True, 0))
                owns_x[i] = True
            elif rng.random() < p_cross:            # buys the other one later
                rows.append((i, t0[i] + rng.integers(60, 600), other, 3200, 3200, True, 0))
                owns_x[i] = True
    win_p = 0.5 + 0.1 * np.tanh(s) + OWN_X * owns_x + 0.03 * chose_u + 0.05 * holder
    won = rng.random(n) < win_p
    ev = pd.DataFrame(rows, columns=["match_id", "game_time_s", "item_id", "cost",
                                     "price_paid", "full_price", "sold_time_s"])
    ev["account_id"] = ev["match_id"]
    ev["won"] = won[ev["match_id"]]
    ev["net_worth_at_buy"] = s[ev["match_id"]]     # the observed confounder
    return ev


def _fit(ev, arms, **kw):
    df = decision_frame(ev, 3200, arms=arms, components={U: (C,)}, **kw)
    K = len(arms)
    fit = fit_multiarm(df[covariate_columns(df)], df["arm"].to_numpy(), df["won"].to_numpy(),
                       df[[f"elig_{k}" for k in range(K)]].to_numpy(), df["match_id"].to_numpy(),
                       arms=arms, learner="logistic")
    return df, fit


@pytest.mark.parametrize("p_cross", [0.0, 0.6])
def test_crossover_scales_the_order_effect(p_cross):
    ev = _sim(p_cross=p_cross, seed=1)
    df, fit = _fit(ev, [X, Y, Z])
    r = pairwise(fit, 0, 1)
    truth = OWN_X * (1 - p_cross)
    assert r["ci_lo"] <= truth <= r["ci_hi"], r
    if p_cross:
        assert r["ci_hi"] < OWN_X                    # clearly not the ownership effect
    # the reported crossover share is the data's: Y- or Z-first players who later buy X
    h = vs_actual(fit, 0, next_buy_s=df[df.attrs["next_cols"]].to_numpy())
    rows = fit.support[:, 0] & (fit.A != 0)
    direct = np.mean(~np.isnan(df[df.attrs["next_cols"][0]].to_numpy()[rows]))
    assert h["cross_by_end"] == pytest.approx(direct)


def test_visits_remove_click_order_artifacts():
    ev = _sim(q_basket=0.5, seed=2)
    df, fit = _fit(ev, [X, Y, Z])
    r = pairwise(fit, 0, 1)
    assert r["ci_lo"] <= OWN_X <= r["ci_hi"], r      # truth among single-arm visits
    assert df.attrs["n_multi_arm_visits"] > 10_000
    _, naive = _fit(ev, [X, Y, Z], visit_gap_s=0)   # every click its own decision
    assert pairwise(naive, 0, 1)["ci_hi"] < OWN_X


def test_component_holders_do_not_leak_into_the_contrast():
    ev = _sim(h_holder=0.4, seed=3)
    df, fit = _fit(ev, [X, Y, Z, U])
    k_u = 3
    assert not df.loc[df.arm == k_u, "account_id"].isin(
        ev.loc[ev.item_id == C, "account_id"]).any()   # holders' discounted U: not trials
    holders = df.account_id.isin(ev.loc[ev.item_id == C, "account_id"])
    assert (df.loc[holders, f"elig_{k_u}"] == 0).all()
    r = pairwise(fit, k_u, 1)                          # U vs Y: +3pp, X ownership 0 for both
    assert r["ci_lo"] <= 0.03 <= r["ci_hi"], r
