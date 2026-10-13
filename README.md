# DL_Item_Effects

This project estimates what a **Deadlock** item choice is worth. When a player spends souls,
it asks how much picking item X instead of another item at the same price changes their chance
of winning, separately for each hero and over the course of the game.

It replaces `DL_Decision_Analytics`. The v1 control group included players who already owned
the item, which biased effects toward zero. See [`PLAN.md`](PLAN.md) §2 for the full list.

## Method
- **Sequential target-trial emulation:** every shop purchase is a small trial among the
  same-cost items the player doesn't already own (`frame/risk_set.py`).
- **Multi-arm cross-fitted AIPW**, a doubly robust estimator with match-clustered SEs and
  overlap trimming (`estimate/aipw.py`).
- **DR-learner spline curves** that show the effect over game time
  (`estimate/heterogeneity.py`).
- **Empirical-Bayes shrinkage** for rankings and **Benjamini-Hochberg FDR** for significance.
- **Falsification checks fixed in advance:** negative-control outcomes, placebo arms,
  week-to-week replication, patch diff-in-diff, and a WPA baseline (PLAN §4).

## Status
Phase 0 is done: the design, the core estimators, and ported data plumbing, with 21 passing
tests. The next step is Phase 1: run the extended extraction against the live data dumps.

## Setup
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

Data: deadlock-api.com data dumps (see `docs/data_dumps.md`). Not affiliated with Valve.
