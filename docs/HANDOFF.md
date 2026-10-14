# Handoff: DL_Item_Effects

Written 2026-10-14 for an agent starting with a fresh context. Read this first, then
[`PLAN.md`](../PLAN.md). Everything below was checked against the repo and data on disk at the
time of writing.

## 1. Where things stand

| Phase | State |
|---|---|
| 0, 0b (scaffold, design checks) | done |
| 1 (data) | done: two Ranked patch windows extracted, 175,188 and 204,837 matches |
| 2 (frame and estimator) | done: 61 tests pass |
| 3 (Haze end to end) | **in progress.** Falsification ran (results in §6). The confirm stage, which computes the real win effects, has **not** run and is blocked on a decision (§7) |
| 4 to 8 | not started |

- **Repo:** `C:\Users\Will\Code\Projects\DL_Item_Effects`, GitHub `WillGrimmer/DL_Item_Effects`.
- **Branch:** `plan-revision-and-phase-1` (pushed). `main` still holds only Phase 0. No PR has
  been opened.
- **Uncommitted changes:** none, apart from this file once it is written.

## 2. What the project is, in one paragraph

It estimates, from Deadlock match data, how much buying shop item X instead of another item at
the same price changes a player's chance of winning, per hero, per patch and over game time.
The design is a sequential target-trial emulation: every shop visit is a small trial among the
same-price items the player can buy. Estimation is cross-fitted multi-arm AIPW with
match-clustered SEs. Falsification thresholds are fixed in advance (PLAN §4), and
omitted-variable-bias bounds are reported (PLAN §3.6). The number answers "X now instead of
what you actually bought", and every effect carries a crossover label, because for the
most-bought items the comparison group usually buys X later anyway. The v1 project
(`DL_Decision_Analytics`) was replaced for the reasons in
[`docs/v1_review_2026-10-13.md`](v1_review_2026-10-13.md).

## 3. Read in this order

1. [`PLAN.md`](../PLAN.md): the design (§3), the falsification rules (§4), the phases (§5) and
   the decision log (§8, newest first). The log records why every choice was made.
2. [`docs/design_checks_2026-10-13.md`](design_checks_2026-10-13.md): the evidence behind the
   design, including crossover rates, simulations, power and DoubleML's limits.
3. [`docs/data_dumps.md`](data_dumps.md): the data lake and every column quirk that matters.

## 4. Environment and how to run things

- **Python:** global Python 3.14 at `C:\Users\Will\AppData\Local\Programs\Python\Python314`,
  no venv. Installed: duckdb 1.4.4, pandas 3.0, numpy 2.4, scikit-learn 1.9, lightgbm 4.7,
  scipy, pyarrow, pytest. DoubleML is **not** installed globally; one was installed in a
  throwaway scratch venv only for checking its source.
- **Imports:** run from the repo root with `PYTHONPATH=src`, or `pip install -e .` first.
- **Tests:** `python -m pytest -q` gives 61 passed, in about 20 s.
- **Machine:** Windows 10, 16 logical CPUs, 31 GB RAM, about 190 GB free on C:.
- **Network:** `data.deadlock-api.com` and `api.deadlock-api.com` work. The old hosts
  `s3-cache.` and `assets.deadlock-api.com` no longer resolve.

Pipeline commands, all from the repo root with `PYTHONPATH=src`:

```bash
python -m dl_item_effects.ingest.extract --patch 09-29-2026 --end 2026-10-12      # ~15 min per window
python -m dl_item_effects.ingest.objectives --patch 09-29-2026 --end 2026-10-12   # ~3 min
python -m dl_item_effects.frame.diagnostics --hero Haze --patch 09-29-2026        # ~1 min, no outcomes
python -m dl_item_effects.phase3 --hero 13 --patch 09-29-2026 --stage falsify --learner lgbm
python -m dl_item_effects.phase3 --hero 13 --patch 09-29-2026 --stage confirm --learner lgbm
python -m dl_item_effects.phase3 --hero 13 --patch 09-29-2026 --stage compare     # after both learners
```

**Run heavy jobs one at a time.** Two LightGBM or logistic jobs in parallel thrash the CPU:
one run stalled for 30 minutes on a fit that takes 90 seconds alone.

## 5. Code map (`src/dl_item_effects/`)

| Module | What it does |
|---|---|
| `ingest/lake.py` | Fetches the hourly manifest, maps partitions to base-file URLs, and finds the partitions covering a date window (dates cached by file URL) |
| `ingest/patches.py` | Patch names and windows from `/v1/patches`. A window skips the release day |
| `ingest/extract.py` | The purchase-event SQL (see its docstring for every column), plus `build_history` and `extract_window`. The window runner is resumable, retries when a partition's file has rotated, and records sources in `_sources.json` |
| `ingest/objectives.py` | Objective losses (t1, t2, base) and mid-boss claims per match |
| `assets/decoder.py` | Item metadata from the assets API, components and their full upgrade-chain closure |
| `frame/risk_set.py` | The decision frame: visits, price-paid trials, component-aware eligibility, `sold >= t` ownership, crossover times `next_<k>`, lanes, composition, enemy inventories, objectives |
| `frame/diagnostics.py` | Design table per hero and price: arms, crossover and optimistic half-width. Uses no outcomes |
| `estimate/aipw.py` | `fit_propensity` and `fit_multiarm` (cross-fitted, eligibility-restricted, trimmed at 1%), `vs_actual` (the headline), `pairwise`, `effect_table` |
| `estimate/heterogeneity.py` | `dr_curve` (spline over game time, ratio form for the headline) and `empirical_bayes` |
| `estimate/fdr.py` | Benjamini-Hochberg q-values and p-values |
| `pipeline.py` | `load_hero_window`, `build_frame` (all covariate blocks, about 417 columns), and `is_dev` (the splitmix64 dev split) |
| `analysis.py` | The falsification checks (PLAN §4.1 to 4.4) with `THRESHOLDS`, `null_fail_rate`, `ovb` and `ovb_benchmark` |
| `phase3.py` | Runner with the dev, falsify, confirm and compare stages. Propensities are cached as `.npy` |

## 6. Phase 3 falsification results (Haze, LightGBM)

Run 2026-10-14, 20:54 to 22:15 UTC, at commit `e1ebf38`: 8 frames (2 windows x 4 prices), 77
minutes. Per-frame files are in `data/processed/results/hero_13/patch_*/`
(`summary_falsify_lgbm.json`, `falsify_lgbm_<price>_*.csv`). The combined table is
`data/processed/results/hero_13/falsify_lgbm_summary.csv`. Only the checks' statistics were
computed; no win effect has been looked at outside the dev set.

| Window, price | Arms | 4.1 μ (τ), pp | Rule's null false-fail | 4.2 souls / deaths (SD) | 4.3 coverage | 4.4 share beyond (z SD) |
|---|---|---|---|---|---|---|
| 09-16, 800 | 9 | −0.30 (0.00) ✗ | 0.47 | 0.005 / 0.009 ✓ | 0.975 ✓ | 0.00 (0.58) ✓ |
| 09-16, 1600 | 13 | −0.24 (0.36) ✗ | 0.49 | 0.009 / 0.016 ✓ | 0.945 ✓ | 0.00 (0.78) ✓ |
| 09-16, 3200 | 10 | −0.74 (0.00) ✗ | 0.64 | 0.014 / 0.026 ✓ | 0.970 ✓ | 0.00 (0.71) ✓ |
| 09-16, 6400 | 10 | −0.24 (0.35) ✗ | 0.51 | 0.016 / 0.023 ✓ | 0.950 ✓ | 0.20 (1.47) ✗ |
| 09-29, 800 | 7 | −0.23 (0.41) ✗ | 0.44 | 0.004 / 0.006 ✓ | 0.915 ✗ | 0.14 (1.72) ✗ |
| 09-29, 1600 | 13 | −0.34 (0.74) ✗ | 0.46 | 0.008 / 0.014 ✓ | 0.985 ✗ | 0.00 (0.84) ✓ |
| 09-29, 3200 | 9 | −0.30 (0.81) ✗ | 0.67 | 0.019 / 0.028 ✓ | 0.940 ✓ | 0.00 (0.72) ✓ |
| 09-29, 6400 | 9 | 0.00 (0.48) ✗ | 0.49 | 0.017 / 0.017 ✓ | 0.990 ✗ | 0.00 (0.67) ✓ |

The previous-match control pooled across prices (fixed effect, treating arms as independent,
so the SE is somewhat optimistic):

| Window | Pooled μ | SE |
|---|---|---|
| 09-16 | −0.34pp | 0.10 |
| 09-29 | −0.28pp | 0.10 |
| Both | −0.31pp | 0.07 |

What this says:

- **§4.2 (in-game momentum) is clean everywhere.** Recent souls and deaths are balanced after
  adjustment, with adjusted differences of at most 0.03 SD.
- **§4.1 fails, and not only through noise.** Applied per frame, the frozen rule
  (|μ| ≤ 0.2pp and τ ≤ 0.3pp) fails 44% to 67% of the time even with no confounding at all,
  because μ's own SE is about 0.2pp. But the pooled estimates fail it as well, at about 3 SEs
  from zero, in both windows independently. Players' item choices are associated with whether
  they won their previous Ranked match, after adjusting for everything except win streak (which
  this check leaves out on purpose), by about −0.3pp. The sign is the same in 7 of 8 frames.
- **§4.3 and §4.4 are mixed, and their rules are also coarse per frame.** SE coverage averages
  0.96 but ranges from 0.915 to 0.990. That points both ways, so the SEs are roughly right on
  average and vary by frame. For 4.4, with 7 to 9 arms a single arm beyond 1.96 fails the
  rule; that happens 30% to 37% of the time by chance. The real flags are 09-16 at 6400
  (Siphon Bullets z −2.86, Cursed Relic −2.19, z SD 1.47) and 09-29 at 800 (z SD 1.72).
- **Phase 3's exit (§4.1 to 4.4 pass on Haze) is not met.** The confirm stage has not run,
  deliberately.

## 7. Open decisions (for Will)

These are the user's calls because the §4 rules were frozen when Phase 3 started (PLAN log,
2026-10-13), and the results have now been seen.

1. **What level the rules apply at.** The code applies §4.1 and §4.4 per (window, price) frame,
   where they fail by chance 30% to 67% of the time. PLAN §4.1's wording ("across a hero's
   arms") supports one pooled test per hero and window. That is a reading of the text, not
   a new threshold, but it was noticed after the results came in, so it needs an explicit yes.
   Under either reading §4.1 still fails.
2. **What to do about the previous-match shift.** This depends on the permutation diagnostic
   below:
   - If the shift is real player-level confounding, the planned response (PLAN §4) is a design
     change. Add player-history covariates built from `_history.parquet` (for example win rate
     over the previous 5 and 10 Ranked matches, and games in the last 24 hours), then test with
     a deeper control (the result two matches back), because the previous result itself would
     then be a covariate.
   - Alternatively, accept it and widen or shift every reported interval by the measured
     systematic error (PLAN §4.1 already says intervals are widened by τ; Schuemie et al.
     2018).
   - If the shuffle test shows an estimator artifact, fix the estimator first.

**Permutation diagnostic: the shift is real, not an estimator artifact.** The previous-match
result was shuffled across player-matches, keeping one value per player-match and the same
cached propensities, so the true effect is exactly zero and no confounding can exist. The same
§4.1 estimate was then rerun on every frame, one shuffle each, on 2026-10-14. To rerun it:
`--stage shuffle`. Results are in `data/processed/results/hero_13/patch_*/diag_nc_player_permuted.csv`.

| Frame | Real | Shuffled | Real minus shuffled |
|---|---|---|---|
| 09-16 · 800 | −0.30 | +0.24 | −0.54 |
| 09-16 · 1600 | −0.27 | −0.00 | −0.27 |
| 09-16 · 3200 | −0.74 | +0.05 | −0.79 |
| 09-16 · 6400 | −0.25 | −0.11 | −0.14 |
| 09-29 · 800 | −0.36 | +0.06 | −0.42 |
| 09-29 · 1600 | −0.43 | +0.47 | −0.90 |
| 09-29 · 3200 | −0.16 | −0.37 | +0.21 |
| 09-29 · 6400 | −0.09 | +0.00 | −0.09 |

Values are fixed-effect means in pp. Pooled, the shuffled values are +0.05pp (09-16) and +0.11pp (09-29),
against the real −0.34pp and −0.28pp. Real sits below shuffled in 7 of 8 frames, by 0.37pp
on average. The shuffled values spread with an SD of about 0.25pp per frame, wider than the
nominal fixed-effect SEs (0.18 to 0.27), because a frame's arms share rows and are not
independent. Even against that empirical noise, the gap is about 3 SEs.

## 8. Data on disk (all git-ignored, under `data/`)

| Path | Contents |
|---|---|
| `processed/events/patch_09-16-2026/` | 3 partitions (106 to 108), 35.6M purchase rows, 175,188 matches, 0.9 GB. `_history.parquet` (previous-match link), `_objectives.parquet`, `_sources.json` (commit `12afc9c`, manifest 693, file URLs) |
| `processed/events/patch_09-29-2026/` | 8 partitions (108 to 115), 41.3M purchase rows, 204,837 matches, 1.1 GB, 09-30 to 10-12. Same side files; `_sources.json` records the exact file per partition (commit `471cdc6`, manifest 694) |
| `processed/reports/design_haze_*.csv` | Phase 2 design tables |
| `processed/results/hero_13/patch_*/` | Phase 3 outputs and cached propensities (`prop_<learner>_<price>_<set>_<rows>.npy`) |
| `interim/` | Cached assets, patches and partition dates |

Source files on the lake rotate about hourly, so re-extracting gives slightly different counts
at the edges. `_sources.json` says what was used.

## 9. Things that will bite you

- **File URLs on the lake go stale within about two hours.** Always re-read the manifest.
- **Team badges are 0, not NULL, since 2026-07-30.** Only Ranked rows carry skill: `average_badge`
  per match and per-player `player_rank_initial_display_rank`, which is 0 during calibration.
- **Filter `game_mode = 'Normal'`.** Without it Street Brawl gets in.
- **One row per (player, item).** Re-buys are invisible. Item arrays are not time-sorted.
- **Components.** A consumed component gets `flags = 1` and `sold_time_s` set to the upgrade's
  purchase time. Buying an upgrade outright writes a same-second component row, which the
  extractor drops. Price paid is credited through whole upgrade chains, once per component.
- **Snapshots** come every 180 s up to 900 s, then every 300 s, plus a final one at match end.
  Before 180 s the state is the start-of-game state: 600 souls, level 1.
- **Item ids arrive as uint64.** The frame casts them to int64.
- **DuckDB:** set the progress bar off with `SET enable_progress_bar = false` after connecting
  (it cannot be a connect-time config). The rank and badge columns are UInt32, so cast them
  before subtracting.
- **Learners:** LightGBM is faster and better than logistic for propensities at this size.
- **Raw `match_id % 5` is not uniform.** Use `pipeline.is_dev`.
- **The dev/confirm rule (PLAN §3.1):** tuning may look at dev-set effects and at
  falsification statistics only. The confirm stage computes the real win effects on the other
  80% once, after the design is frozen, and that commit goes in the PLAN log.

## 10. Next steps

1. Get Will's answers to §7.
2. If the shift is real: add the player-history covariates (in `ingest/extract.py`'s history
   join or as a frame step), add the deeper negative control, and rerun `--stage falsify`
   for Haze. Cached propensities are keyed by covariate set and frame size, so delete the
   stale `.npy` files or use new tags.
3. Look at SE calibration per frame (more placebo reps, the weight distribution at 6400 and
   800) and the two split-half flags.
4. Once §4.1 to 4.4 pass: `--stage confirm` with `lgbm` and with `logistic`, then
   `--stage compare`, then write the Haze tables up and record the freeze commit in the PLAN
   log. Expect about 1 hour for LightGBM and 1.5 to 2 hours for logistic (logistic
   propensities are about 4 times slower).
5. **Compute budget:** Haze is the largest hero. Falsification took 77 minutes for 8 frames,
   and confirm adds about 1 to 3 hours. A median hero has about 45% of Haze's data, so the
   roster (about 42 heroes, PLAN Phase 5) is roughly 25 to 40 hours of serial compute for
   falsify plus confirm. Run it one job at a time.
6. Phase 4 (a per-hero falsification report and the WPA baseline), Phase 5 (roster),
   Phase 6 (pages), Phase 7 (patch diff-in-diff across 09-16 and 09-29, both windows already
   extracted).
7. Merge `plan-revision-and-phase-1` into `main` (or open a PR) when Will wants; nothing has
   been merged.

## 11. Not in the repo

Agent and scratch scripts (crossover measurement, simulations, the column re-derivation, the
gap diagnosis, benchmarks) live in the session scratchpad, which is temporary. Their findings
are recorded in `docs/design_checks_2026-10-13.md` and the PLAN log. The synthetic designs that
matter were ported into `tests/test_synthetic_design.py`.
