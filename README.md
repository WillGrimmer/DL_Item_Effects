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
Phases 0 to 2 are done. The design was checked against the live data
([`docs/design_checks_2026-10-13.md`](docs/design_checks_2026-10-13.md)), which changed the
plan: Ranked matches only, shop visits as the unit, full-price purchases as the trials. Two
patch windows are extracted (380k Ranked matches), and the frame and estimator pass 59 tests,
including end-to-end synthetic checks. Phase 3 (Haze end to end) is in progress.

## Running it
```bash
python -m dl_item_effects.ingest.extract --patch 09-29-2026 --end 2026-10-12
python -m dl_item_effects.ingest.objectives --patch 09-29-2026 --end 2026-10-12
python -m dl_item_effects.frame.diagnostics --hero Haze --patch 09-29-2026
python -m dl_item_effects.phase3 --hero 13 --patch 09-29-2026 --stage falsify
```
Set `PYTHONPATH=src` or `pip install -e .` first. Outputs go to `data/processed/` (git-ignored).

## Setup
```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m pytest -q
```

Data: the deadlock-api.com data lake (see `docs/data_dumps.md`). Not affiliated with Valve.
