# DL_Item_Effects

This project estimates what a **Deadlock** item choice is worth. When a player spends souls,
it asks how much picking item X instead of another item at the same price changes their chance
of winning, separately for each hero and over the course of the game.

It replaces `DL_Decision_Analytics`. The v1 control group included players who already owned
the item, which biased effects toward zero. See [`PLAN.md`](PLAN.md) §2 for the full list.

## Method
- **Sequential target-trial emulation:** every shop visit is a small trial among the
  same-price items the player doesn't already own (`frame/risk_set.py`).
- **Multi-arm cross-fitted AIPW**, a doubly robust estimator with match-clustered SEs and
  overlap trimming (`estimate/aipw.py`).
- **DR-learner spline curves** that show the effect over game time
  (`estimate/heterogeneity.py`).
- **Empirical-Bayes shrinkage** for rankings and **Benjamini-Hochberg FDR** for significance.
- **Falsification checks with thresholds fixed in advance:** negative-control outcomes,
  standard-error calibration, split-half stability, patch diff-in-diff, and a WPA baseline
  (PLAN §4).

The estimate is "X now instead of Y now". It is published with a crossover rate, because for
the most-bought items it is a question about build order and not about the value of owning
the item (PLAN §1).

## Status
Phase 0 is done: the design, the core estimators, and ported data plumbing, with 21 passing
tests. The design was then checked against the live data
([`docs/design_checks_2026-10-13.md`](docs/design_checks_2026-10-13.md)), which changed the
plan: Ranked matches only, shop visits as the unit, full-price purchases as the trials. The
next step is Phase 1: repoint the extraction at the new data lake.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

Data: the deadlock-api.com data lake (see `docs/data_dumps.md`). Not affiliated with Valve.
