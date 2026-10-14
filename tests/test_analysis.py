"""Falsification checks and sensitivity bounds on synthetic data with known structure."""
import numpy as np
import pytest

from dl_item_effects import analysis
from dl_item_effects.frame import decision_frame

from test_synthetic_design import C, U, X, Y, Z, _sim


@pytest.fixture(scope="module")
def frame():
    ev = _sim(n=60_000, seed=11)
    df = decision_frame(ev, 3200, arms=[X, Y, Z], components={U: (C,)})
    rng = np.random.default_rng(5)
    s = df["net_worth_at_buy"].to_numpy()
    # negative controls: nothing the choice can cause. Souls depend on the observed state
    # (which also drives the choice), so only an adjusted estimate is near zero.
    df["nc_prev_won"] = (rng.random(len(df)) < 0.5).astype(float)
    df["nc_prior_souls"] = 500 + 200 * s + rng.normal(0, 150, len(df))
    df["nc_prior_deaths"] = rng.poisson(0.5, len(df)).astype(float)
    return df


@pytest.fixture(scope="module")
def fit(frame):
    return analysis.fit_rows(frame, np.ones(len(frame), bool))


def test_negative_controls_pass_without_confounding(frame):
    rows = np.ones(len(frame), bool)
    p = analysis.nc_player(frame, rows)
    assert p["passed"], (p["mu"], p["tau"])
    g = analysis.nc_ingame(frame, rows)
    assert g["passed"], {k: (v["share_within"], v["mean_abs"]) for k, v in g.items()
                         if isinstance(v, dict)}


def test_se_calibration_near_nominal(fit):
    r = analysis.se_calibration(fit, reps=200, seed=1)
    assert 0.90 <= r["coverage"] <= 0.99 and 0.8 < r["z_sd"] < 1.2


def test_split_half_is_stable(frame):
    r = analysis.split_half(frame, np.ones(len(frame), bool))
    assert set(r["z"].columns) == {"arm", "z"} and "effect" not in r["z"]
    assert r["share_beyond"] <= 1 / 3


def test_robustness_value_solves_its_equation(fit):
    for j in (None, 1):
        o = analysis.ovb(fit, 0, j)
        at_rv = analysis.ovb(fit, 0, j, cf_y=o["rv"], cf_d=o["rv"])
        assert at_rv["bias"] == pytest.approx(abs(o["effect"]), rel=1e-6)
        assert o["sigma2"] > 0 and o["nu2"] > 0 and 0 <= o["rva"] <= o["rv"] < 1


def test_benchmark_from_dropping_the_confounder(frame, fit):
    short = analysis.fit_rows(frame, np.ones(len(frame), bool), drop=("net_worth_at_buy",))
    b = analysis.ovb_benchmark(fit, short, 0, 1)
    assert 0 < b["cf_y"] <= 1 and 0 < b["cf_d"] <= 1
    assert b["delta"] > 0          # leaving out the confounder inflates arm 0's effect
