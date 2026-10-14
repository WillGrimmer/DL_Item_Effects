# DL_Item_Effects: plan

Owner: Will (gh: WillGrimmer) · Started 2026-10-13 · Successor to `DL_Decision_Analytics` (v1)

Keep this file short. Record decisions in the log at the bottom, one or two lines each. The
v1 repo's PLAN.md turned into a 760-line diary, and that made it hard to see what the design
actually was. Evidence goes in `docs/`, not here.

## 1. The question

At the moment a player spends souls on a shop item, does picking item X instead of another
item at the same price change their chance of winning? If it does, by how much, for which
hero, and when in the game?

Raw win rate and WPA can't answer this. Players pick items because of the game state, so an
item's win rate mixes "the item is good" with "players who are ahead buy it". WPA compares
win probability before and after a moment, so it measures how the game moved, not whether a
different choice would have done better.

**What the answer is, precisely.** "X now instead of Y now, with everything the player buys
afterwards left as it happened." That is the right number for "what should I buy next". It is
**not** "what is owning X worth": for the most-bought items, most players who pick Y buy X
later in the same match, and the effect is then about order (§3.2, and
[`docs/design_checks_2026-10-13.md`](docs/design_checks_2026-10-13.md) §1-2). Every number is
published with its crossover rate so the two readings are never confused.

Not covered: buying versus saving, selling, component upgrades (§3.2), and the value of owning
an item (Phase 8, only if the rest holds up).

## 2. Why v1 is being replaced

The full review is [`docs/v1_review_2026-10-13.md`](docs/v1_review_2026-10-13.md). Ranked:

1. **Its control group included players who already owned X** (or bought it later). That pulls
   every effect toward zero or below once you're past an item's earliest buy window. A
   synthetic test with v1's own estimator turned a true +3pp into +1pp "not significant".
2. **Thin confounder set.** Team-average badge only, no enemy badge, no inventory, no
   level/K/D/A, no enemy builds.
3. **Uneven weighting.** One row per treated player but every purchase for control players.
4. **No falsification test that could detect confounding**, and the arm-building SQL had no
   tests at all.
5. **The estimand changed six times** based on whether the output looked plausible.
6. **Smaller issues:** the "ideal timing" was a max over noisy points (winner's curse),
   propensities were clipped instead of trimmed, page-level significance had no FDR, and the
   data was one week with no patch ID.

v2 removes the "already owned X" half of flaw 1 by construction. The "bought it later" half is
still there. It is now measured and reported instead of hidden.

## 3. The statistical design

### 3.1 Data and population
Source and column semantics: [`docs/data_dumps.md`](docs/data_dumps.md).

- **Ranked, `game_mode = 'Normal'`, exactly 12 player rows.** Unranked has had no skill data
  since 2026-07-30 and cannot be adjusted.
- **No filter on anything that happens after a decision.** Matches with an abandon stay in.
- **Patch windows.** Every table is for one patch. In hand now: 09-16-2026 (09-17 to 09-28,
  176k matches) and 09-29-2026 (09-30 onward, 205k matches by 10-12).
- **Dev and confirm split.** Matches whose splitmix64 hash of `match_id` is 0 modulo 5 are
  the dev set (`pipeline.is_dev`). Raw match ids are not uniform modulo 5. Design tuning looks
  only at the dev set and at falsification outputs. Win-outcome tables on the other 80% are
  computed once, after the design is frozen, and the commit is recorded in the log.

### 3.2 Decision frame (`frame/risk_set.py`; built, changes pending in Phase 2)
Every shop visit is a small trial (Hernán & Robins' target-trial framework, applied
sequentially as in Hernán et al. 2008 and Danaei et al. 2013).

- **Unit: a shop visit at one price.** Purchases 30 s or less apart are one visit. State and
  inventory are taken at the start of the visit. A visit that buys exactly one arm at that
  price is a row. Visits that buy two or more arms at the same price are counted and left out.
- **Price paid, not list price.** A purchase that consumes a held component pays less than
  list. A trial is a full-price purchase. Component upgrades (30.5% of purchases) are not
  trials, because nothing else costs the same for that player; the component still counts as
  inventory.
- **Arms:** the full-price items at that price that hero H buys often enough to estimate:
  at least 2% share and at least 2,500 buys in the window (a 2pp half-width at best). Fixed
  per (hero, price, patch) **before** anyone looks at outcomes.
- **Eligibility:** an arm is available on a row only if the player does not own it and holds
  none of its components. Owned means bought before the visit and not sold or consumed before
  it (`sold = 0 OR sold >= t`).
- **Outcome:** match win.
- **Covariates,** all known at the start of the visit:
  - skill: match average badge, own rank, own rank minus team mean, calibration flag, win
    streak entering the match
  - assigned lane, whether a build is equipped
  - net worth, both teams' net worth and the gap, level, kills, deaths, assists, and the age
    of the snapshot they come from (up to 299 s)
  - game time, purchases so far, **current inventory** (multi-hot)
  - hero composition of both teams
  - Phase 2 adds enemy inventories by slot, objectives destroyed and mid-boss state.

  Before the first snapshot (180 s) the state is the start-of-game state, never a later
  snapshot. Match duration is never a covariate.
- **Crossover, reported with every effect:** the share of the comparison group that buys X
  within 10 minutes and by the end of the match. Labels are fixed now: under 25% by match end
  is "choice", 25% to 50% is "mixed", 50% or more is "order".

### 3.3 Multi-arm doubly robust estimation (`estimate/aipw.py`; built, changes pending)
The treatment has K values (which item), not two. We use cross-fitted AIPW for multi-valued
treatments: double machine learning (Chernozhukov et al. 2018) with average potential outcomes
as in Cattaneo 2010.

- **Propensity model:** a multinomial classifier for P(choose k | state), set to zero for arms
  the player can't buy and then renormalized.
- **Outcome model:** one per arm, P(win | state, chose k).
- **Cross-fitting:** 5 folds grouped by match. Standard errors are match-clustered.
- **Overlap:** an arm counts on a row only if it is eligible there with propensity ≥ 1%. Rows
  are **dropped**, never clipped, and the dropped share is reported per arm. Each contrast
  therefore has its own population. Joint trimming (Yang et al. 2016) is not usable here: it
  would drop every row where the player already owns any arm.
- **Outputs:**
  - `pairwise(k, j)`: X versus Y directly, on rows where both are supported. Used by the shop
    tab.
  - **Headline per item: X instead of what the player actually bought,** averaged over
    supported players who bought something else. Computed from the same scores as
    `(mean(psi_k) − mean(Y)) / (1 − share_k)`. This replaces "X versus a uniformly random
    alternative", which no player does and which gives a 2% item the weight of the default.
- **Learners:** LightGBM for the reported numbers and the falsification checks; logistic as
  the cross-check. A result counts only if the two agree on sign and significance.
  Disagreement gets flagged.

### 3.4 Effect over game time (`estimate/heterogeneity.py`, built)
Regress the doubly robust pseudo-outcome on a cubic spline in game time with a clustered
sandwich variance (Kennedy 2023; Semenova & Chernozhukov 2021). The bands are pointwise.
"Best timing" is reported only from a uniform band (cluster bootstrap, Phase 3) that clearly
separates. We never take the max over grid points.

### 3.5 Rankings and multiple testing (built)
- **Empirical-Bayes shrinkage** (normal-normal, DerSimonian-Laird variance) across items within
  a hero. Lists use the shrunk values and are called "buy next" lists, not tier lists.
- **Benjamini-Hochberg FDR** across the whole family (hero × price × arm), applied everywhere an
  item gets called "significant", including the pages.

### 3.6 Sensitivity to unmeasured confounding (Phase 3)
Omitted-variable-bias bounds for DML (Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022),
implemented here for the headline and pairwise contrasts. The robustness value answers: how
strong would a hidden confounder need to be to erase this effect? Benchmark it against observed
covariates ("as strong as rank"). DoubleML 0.11.4 cannot cluster its multi-level model or
restrict levels per row, so it is a cross-check on a synthetic case, not the engine.

### 3.7 Power
Per 13-day patch window, the median item's 95% half-width is about 1.3pp for the least-played
hero, 0.8pp for a median hero and 0.5pp for the most-played. Detectable effects at 80% power
are about 1.4 times that. These figures are optimistic (design_checks §3). Windows can be
pooled for items a patch did not touch, with patch as a covariate.

## 4. Falsification: how we'll know if it's wrong

The thresholds below are fixed before any hero results exist. A design change is accepted when
it passes them, not when the output "looks right". **There is no "items must matter"
principle.** An effect near zero is an allowed answer.

1. **Negative control, player level.** Outcome: the same player's previous Ranked match result.
   The true effect is zero, it is on the same scale as the real outcome, and it catches
   skill or form that rank misses. Run it with win streak left out of the covariates. Across a
   hero's arms, fit a mean μ and between-arm SD τ. **Pass: |μ| ≤ 0.2pp and τ ≤ 0.3pp.** Reported
   intervals are then widened by τ (Schuemie et al. 2018).
2. **Negative control, in-game.** Outcomes: souls gained and deaths over the snapshot interval
   before the visit. Neither is ever a covariate. **Pass: adjusted difference ≤ 0.1 SD for at
   least 90% of arms, and ≤ 0.05 SD on average.**
3. **Standard-error calibration.** Randomly split one item's buyers into two fake arms, 200
   times. **Pass: 92% to 98% of intervals cover zero.** This checks the SEs. It cannot detect
   confounding.
4. **Split-half stability.** Estimate on two random halves of a window.
   **Pass: at most 10% of arms differ by more than 1.96 combined SEs.** This is stability, not
   validity: confounding replicates too.
5. **Patch diff-in-diff.** For each item a patch changed: the change in its effect minus the
   mean change for untouched arms at the same hero and price. Buff or nerf direction is coded
   from the patch notes before estimating. **Pass: sign agreement ≥ 70%** among changed items
   with a half-width of 2pp or less. The direction is known from outside the data, so this is
   the strongest check.
6. **Synthetic recovery** with known truth in `tests/`, including crossover, same-visit and
   component cases.
7. **WPA baseline.** A Statlocker-style WPA on the same data, so "better than WPA" becomes a
   measured comparison.

## 5. Phases

| # | Phase | Output | Exit criteria |
|---|---|---|---|
| 0 | Scaffold, design, ported plumbing | this repo; risk-set frame, multi-arm AIPW, DR curves, EB, BH; 21 tests | ✅ done 2026-10-13 |
| 0b | Design checks against live data | `docs/design_checks_2026-10-13.md`, `docs/data_dumps.md`, this revision | ✅ done 2026-10-13 |
| 1 | **Data** | `extract.py` on the new data lake: manifest-driven URLs, §3.1 filters, shop items from the assets API, drop same-click component rows, consumed flag and price paid, skill and lane columns, start-of-game state before 180 s, patch label from `start_time`, previous-match link, prior-interval souls and deaths. Extract both patch windows. | ✅ done 2026-10-13: 175,188 and 204,837 matches (−0.4% and −0.2% vs the volume table) |
| 2 | **Frame and estimator fixes** | visits, price paid, component eligibility, `sold >= t`; headline estimand; LightGBM with two arms; NaN handling; fold-absent arms counted in `trim_share`; explain the 0.2pp synthetic gap; enemy inventories and objectives | ✅ done 2026-10-13: 59 tests; Haze tables in `data/processed/reports/` |
| 3 | **First hero end to end** (Haze: most data in both windows) | effect table, pairwise matrix and DR curves at every price, each with its crossover label; logistic vs LightGBM; OVB bounds; wall time per (hero, price) | §4.1 to 4.4 pass on Haze; roster compute budget written down. **Falsification run 2026-10-14: §4.1 fails, see log; confirm not run** |
| 4 | **Falsification suite** as code | `validate/` module: both negative controls, SE calibration, split-half, WPA baseline | runs per hero, writes a pass/fail report |
| 5 | Roster | all heroes, FDR across the family, EB-shrunk "buy next" lists; Mirage reported as the low-power case | §4.1 to 4.4 pass for ≥80% of heroes; failures listed, not hidden |
| 6 | Pages | port v1's hero-page template: curves with bands, shop tab from `pairwise`, crossover label and sensitivity shown per item | every number on a page traces to a cached table |
| 7 | Patch diff-in-diff | §4.5 across 09-16 → 09-29 and each later patch | sign agreement reported |
| 8 | Ownership estimand (optional) | "own X by minute T" by clone-censor-weight with a grace period and time-varying covariates | its own falsification set, written before any results |

## 6. Code status
- `assets/decoder.py`, `api/client.py`: ported; the decoder now resolves components and the
  client fetches patches.
- `ingest/lake.py`, `ingest/patches.py`, `ingest/extract.py`: Phase 1 extraction on the new
  data lake, tested on a hand-built 12-player fixture and smoke-tested on 2,000 live matches.
- `ingest/objectives.py`: objective losses and mid-boss claims per match.
- `frame/risk_set.py`: visits, price-paid trials, component eligibility, crossover times,
  enemy inventories, objectives. `frame/diagnostics.py`: the per-hero design table.
- `estimate/aipw.py`: the headline, reusable propensities, continuous outcomes.
  `estimate/heterogeneity.py`, `estimate/fdr.py`: built.
- `pipeline.py`: loads a hero's window and builds full frames; `is_dev` is the split.
- `analysis.py`: §4.1 to 4.4 and the OVB bounds. `phase3.py`: the dev, falsify and confirm
  stages for one hero. Phase 4 generalizes these into a per-hero report and adds the WPA
  baseline.

Not ported from v1: the global win-prob model, the benchmark, the windowed decision frame, and
every cached table and page.

## 7. Known limits we accept
- Covariates are observed, so effects are "deconfounded on observables", backed by the
  sensitivity bounds and the negative controls. We don't claim more than that.
- Ranked players only. Party membership is not in the data.
- Component upgrades and multi-item visits are outside the comparison (about 30% and 17% of
  purchases).
- State comes from snapshots up to 5 minutes old. Inventory is exact.
- A sell and re-buy of the same item cannot be seen.
- Effects are specific to a patch. Every table is labelled with its patch.

## 8. Decision log (newest first)
- **2026-10-14** (Will accepted the recommendation in `docs/HANDOFF.md` §7): §4.1 and §4.4 are
  applied once per hero and window, pooled over prices, as their wording says (per frame they
  fail by chance in 95% and 55% to 76% of windows). Decided after seeing Haze's falsification
  statistics but before any confirm result, for all heroes. It turns Haze's §4.4 into a pass;
  per-frame flags stay reported. Thresholds are unchanged. For the previous-match shift (a
  shuffle of the previous result gives +0.05pp and +0.11pp against the real -0.34pp and
  -0.28pp, so it is not an estimator artifact): add player-history covariates (previous
  result, win rate over matches 2 to 6 and 2 to 11 back, history count, games in the previous
  24 hours, plus the existing gap since the previous match). §4.1 keeps its outcome and runs
  without win streak or the previous result; history starts two matches back, so the outcome
  stays outside it. The next Ranked match's result, with the full production covariates, is a
  diagnostic, not a gate. Pooled SEs and the rule's real chance-fail rate come from a
  bootstrap over matches and over accounts. Calibration (widen by τ, never shift) is a
  fallback only if the rerun fails, and Will chooses it. Hero experience is not added:
  `_history.parquet` has no hero_id.
- **2026-10-14**: Haze falsification (LightGBM, 8 frames = 2 windows x 4 prices, 77 min).
  §4.2 passes everywhere (adjusted differences at most 0.03 SD). §4.1 fails everywhere: the
  previous-match control is negative in 7 of 8 frames, -0.34pp (SE 0.10) pooled in 09-16 and
  -0.28pp (SE 0.10) in 09-29. Applied per frame, the rule fails 44% to 67% of the time with no
  confounding at all, but the pooled estimates fail it too. §4.3 passes in 5 of 8 frames
  (coverage 0.915 to 0.990) and §4.4 in 6 of 8 (one late-game frame has Siphon Bullets at
  z -2.86). Phase 3 exit is not met. The confirm stage has not run. Open questions are in
  `docs/HANDOFF.md` §7.
- **2026-10-14**: LightGBM, not logistic, runs the falsification checks. On the largest Haze
  frame (260k training rows, 417 covariates, 9 arms) one propensity fold takes 17 s with
  LightGBM against 63 s for logistic, and held-out log-loss is 1.020 against 1.113. The
  checks should test the learner the reported numbers use. Logistic stays as the agreement
  check. Heavy jobs run one at a time: two in parallel stalled each other.
- **2026-10-13**: Phase 3 started on Haze. The §4 thresholds are frozen from this commit; they
  are encoded in `analysis.THRESHOLDS`.
- **2026-10-13**: Phase 2 done. Haze design tables: 42 arms in the 09-16 window (31 "choice",
  9 "mixed", 2 "order") and 38 in the 09-29 window (24, 12, 2). Burst Fire and Swift Striker
  are "order" in both. Optimistic 95% half-widths are 0.36pp to 1.5pp. The 0.2pp simulation gap
  was Monte Carlo noise from shared seeds, not bias.
- **2026-10-13**: Dev split is a splitmix64 hash of `match_id`, not `match_id % 5`. Raw ids are
  not uniform modulo 5 (31k to 40k matches per residue) and the residue shifts mean rank.
- **2026-10-13**: Phase 1 done. An independent re-derivation of 2,000 live matches matched
  every column. It found that components two levels down an upgrade chain were not credited
  (547 of 391k purchases); fixed before the full extraction.
- **2026-10-13**: Phase 1 code done: manifest-driven extraction (`ingest/lake.py`,
  `ingest/patches.py`, `ingest/extract.py`), price paid, start-of-game state of 600 souls and
  level 1 (measured), previous-match link, prior-interval negative controls. 12-player Ranked
  matches per window are within 0.4% of the volume table (175,188 and 204,573).
- **2026-10-13**: Thresholds in §4 written down with numbers. They are proposals until Phase 3
  starts and frozen from then.
- **2026-10-13**: Added the previous-match negative control and the dev/confirm split. Relabelled
  placebo arms as SE calibration and replication as stability.
- **2026-10-13**: Headline changed to "X instead of what the player actually bought". The
  uniform-alternative baseline was not a choice anyone makes.
- **2026-10-13**: Estimand named as "X now vs Y now" and published with a crossover label.
  Crossover by match end is 50% or more for 24 of 120 top Mirage pairs; at 90% the estimator
  returns about zero for a +3pp item. Ownership estimand deferred to Phase 8.
- **2026-10-13**: Unit is a shop visit; trials are full-price purchases; arms need 2,500 buys.
  17% of purchases share a visit with a same-cost purchase and 30.5% are component upgrades.
- **2026-10-13**: Population is Ranked only. Team badges have been 0 since Ranked launched on
  2026-07-30; Ranked rows carry per-player rank and win streak.
- **2026-10-13**: First hero is Haze, not Mirage. Mirage is the least-played Ranked hero (23k
  player-matches per window against 129k), and v1's Mirage numbers are not a usable benchmark.
- **2026-10-13**: Data source moved to `data.deadlock-api.com`; `docs/data_dumps.md` rewritten.
- **2026-10-13**: v2 started. Design: sequential target trial, eligibility-restricted
  multi-arm AIPW, DR-learner time curves, EB plus BH, DML OVB sensitivity. Falsification
  criteria fixed before any results (§4).

## References
- Hernán & Robins (2016), *Using big data to emulate a target trial when a randomized trial is not available*, AJE.
- Hernán et al. (2008), observational studies analyzed like a sequence of randomized trials, Epidemiology.
- Danaei et al. (2013), emulated trials with time-varying treatment, *Stat Methods Med Res*.
- Cain et al. (2010), dynamic strategies with grace periods (clone-censor-weight), Int J Biostat.
- Chernozhukov et al. (2018), *Double/debiased machine learning*, Econometrics Journal.
- Cattaneo (2010), efficient semiparametric estimation of multi-valued treatment effects, J. Econometrics.
- Yang, Imbens, Cui, Faries & Kadziola (2016), propensity methods with multi-level treatments, Biometrics.
- Crump, Hotz, Imbens & Mitnik (2009), dealing with limited overlap, Biometrika.
- Kennedy (2023), *Towards optimal doubly robust estimation of heterogeneous causal effects*, EJS ([arXiv 2004.14497](https://arxiv.org/abs/2004.14497)).
- Semenova & Chernozhukov (2021), debiased ML of conditional average treatment effects and other causal functions, Econometrics Journal ([arXiv 1702.06240](https://arxiv.org/abs/1702.06240)).
- Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), *Long story short: omitted variable bias in causal machine learning*, NBER w30302 ([arXiv 2112.13398](https://arxiv.org/abs/2112.13398)).
- Lipsitch, Tchetgen Tchetgen & Cohen (2010), negative controls, Epidemiology.
- Schuemie et al. (2018), empirical confidence interval calibration, PNAS.
- Bach et al. (2022), *DoubleML: an object-oriented implementation of double machine learning in Python*, JMLR ([arXiv 2104.03220](https://arxiv.org/abs/2104.03220)).
