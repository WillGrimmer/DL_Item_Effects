# DL_Item_Effects: plan

Owner: Will (gh: WillGrimmer) · Started 2026-10-13 · Successor to `DL_Decision_Analytics` (v1)

Keep this file short. Record decisions in the log at the bottom, one or two lines each. The
v1 repo's PLAN.md turned into a 760-line diary, and that made it hard to see what the design
actually was.

## 1. The question

At the moment a player spends souls on a shop item, does picking item X instead of another
item at the same price change their chance of winning? If it does, by how much, for which
hero, and when in the game?

Raw win rate and WPA can't answer this. Players pick items because of the game state, so an
item's win rate mixes "the item is good" with "players who are ahead buy it". WPA compares
win probability before and after a moment, so it measures how the game moved, not whether a
different choice would have done better.

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

## 3. The statistical design (chosen)

### 3.1 Sequential target-trial emulation (`frame/risk_set.py`, built)
Every shop purchase is a small trial (Hernán & Robins' target-trial framework, applied
sequentially as in Danaei et al. 2013).

- **Who is in the trial:** a decision by hero H where the player spends exactly cost C. Buying
  something at C proves they could afford anything at C, so affordability holds by
  construction. This is the principled version of v1's cost matching.
- **Arms:** the items at cost C that hero H buys often enough to estimate (default ≥2% share
  and ≥500 buys). They are fixed per (hero, cost) **before** anyone looks at outcomes.
- **Eligibility:** an arm is only available if the player doesn't currently own it, meaning
  they never bought it, or they sold it. This removes v1 flaw #1 by construction.
- **Outcome:** match win.
- **Covariates:** everything known at the moment of the decision:
  - own badge, enemy badge
  - net worth, team net worth for both teams and the gap
  - level, kills, deaths, assists
  - game time, number of items bought so far
  - **current inventory** (owned-items multi-hot)
  - hero composition of both teams

  Phase 2 adds enemy inventories by slot and objective state. Nothing measured after the
  decision is used. Match duration is never a covariate.

**What the answer means:** "Given a player is spending C now, choosing X instead of an
alternative changes their win probability by τ." It does **not** cover buy-versus-save
decisions or selling. Those are separate questions and we don't claim them.

### 3.2 Multi-arm doubly robust estimation (`estimate/aipw.py`, built)
The treatment has K values (which item), not two. We use cross-fitted AIPW for multi-valued
treatments. This is the double machine learning approach of Chernozhukov et al. 2018, with
average potential outcomes as in Cattaneo 2010 and the DoubleML `DoubleMLAPOS` model.

- **Propensity model:** a multinomial classifier for P(choose k | state), set to zero for arms
  the player can't buy and then renormalized.
- **Outcome model:** one per arm, P(win | state, chose k). Separate per-arm models stop the item
  signal getting starved out, which is what happened to v1's global model.
- **Cross-fitting:** 5 folds grouped by match. Standard errors are match-clustered.
- **Overlap:** a row only counts toward a comparison if every arm in that comparison has
  propensity ≥ 1% on that row. Rows are **dropped**, never clipped. We report the trimmed share
  for each arm.
- **Outputs:**
  - `pairwise(k, j)`: X versus Y directly, used by the shop tab.
  - `vs_alternatives(k)`: X versus a randomly chosen buyable alternative. This is the single
    per-item headline, with the same baseline for every arm.
- **Learners:** logistic for speed, LightGBM for the reported numbers. A result counts only if
  the two agree on sign and significance. Disagreement gets flagged.

### 3.3 Effect over game time (`estimate/heterogeneity.py`, built)
Kennedy's DR-learner (2023): regress the doubly robust pseudo-outcome on a cubic spline in game
time, with a clustered sandwich variance. That gives one smooth curve per item with honest
confidence bands, instead of v1's overlapping windows. "Best timing" is reported only when a
band clearly separates. We never take the max over grid points.

### 3.4 Rankings and multiple testing (built)
- **Empirical-Bayes shrinkage** (normal-normal, DerSimonian-Laird variance) across items within
  a hero. Rankings and tier lists use the shrunk values, so a noisy item can't reach the top on
  luck.
- **Benjamini-Hochberg FDR** across the whole family (hero × cost × arm), applied everywhere an
  item gets called "significant", including the pages.

### 3.5 Sensitivity to unmeasured confounding (Phase 3)
No observational design can rule out confounders we don't measure, such as individual skill
within a team. For every headline effect we report omitted-variable-bias bounds for DML
(Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis 2022, available in DoubleML's
`sensitivity_analysis`). The robustness value answers: how strong would a hidden confounder
need to be to erase this effect? Benchmark it against observed covariates, for example
"as strong as badge".

## 4. Falsification: how we'll know if it's wrong

These checks are decided before any hero results exist. A design change is accepted when it
passes them, not when the output "looks right". **There is no "items must matter" principle.**
An effect near zero is an allowed answer.

1. **Negative-control outcomes.** Run the same estimator on outcomes that happened *before* the
   decision: net-worth gain and deaths in the 3 minutes before it. The true effect is zero by
   construction. A non-zero result means confounding is leaking through. This is the main test
   v1 never had.
2. **Placebo arms.** Randomly split one item's buyers into two fake arms. Their difference has
   to be about zero, with roughly 95% of confidence intervals covering zero.
3. **Week-to-week replication.** Estimate on week A, then re-estimate on week B of the same
   patch. Report sign agreement and the correlation of effects.
4. **Patch diff-in-diff.** When an item is buffed or nerfed, its effect should move the right
   way relative to items that weren't changed. This doesn't depend on overlap, so it is the
   strongest external check.
5. **Synthetic recovery tests** of the estimator with known truth, already in `tests/`.
6. **WPA baseline.** Implement a Statlocker-style WPA on the same data, so "better than WPA"
   becomes a measured comparison.

## 5. Phases

| # | Phase | Output | Exit criteria |
|---|---|---|---|
| 0 | Scaffold, design, ported plumbing | this repo; risk-set frame, multi-arm AIPW, DR curves, EB, BH; 21 tests | ✅ done 2026-10-13 |
| 1 | **Data** | run the extended `extract.py` on live dumps: verify `sold_time_s`, enemy badge, level/K/D/A columns; add patch/date (map match_id→start_time); pull **2+ weeks of one patch** and one week either side of a balance patch | column checks pass; corpus has patch IDs |
| 2 | **State enrichment** | enemy inventory by slot at t, objectives destroyed, pre-decision 3-min deltas (for negative controls); interpolate net worth between 180s snapshots | fixture tests for each new column |
| 3 | **First hero end to end** (Mirage, so it can be compared with v1) | effect table + pairwise matrix + DR curves at every cost; logistic vs lgbm; OVB sensitivity | all of §4.1-4.2 pass on Mirage |
| 4 | **Falsification suite** as code | `validate/` module: negative controls, placebo arms, week replication, WPA baseline | runs per hero, writes a pass/fail report |
| 5 | Roster | all heroes, FDR across the family, EB-shrunk rankings | §4.1-4.3 pass for ≥80% of heroes; failures listed, not hidden |
| 6 | Pages | port v1's hero-page template: curves with bands, shop tab from `pairwise`, sensitivity shown per item | every number on a page traces to a cached table |
| 7 | Patch diff-in-diff | DiD for every buffed or nerfed item across one patch | sign agreement reported |

## 6. Ported from v1, and what changed
- `assets/decoder.py`, `api/client.py`: unchanged.
- `ingest/extract.py`: now also extracts `sold_time_s`, `enemy_avg_badge`, and
  level/kills/deaths/assists at the last snapshot. It accepts local source paths, so the SQL is
  tested against a hand-built fixture (`tests/test_extract.py`).
- `estimate/fdr.py`: Benjamini-Hochberg, unchanged.
- v1's `crossfit_aipw` was replaced by the multi-arm version. Its cross-fit and clustered-SE
  logic carries over.

Not ported: the global win-prob model, the benchmark, the windowed decision frame, and every
cached table and page.

## 7. Known limits we accept
- Covariates are observed, so effects are "deconfounded on observables", backed by the
  sensitivity bounds. We don't claim more than that.
- Economy snapshots come every 180s. Interpolation (Phase 2) helps, but the timing isn't exact.
- Effects are specific to a patch. Every table is labelled with its patch.

## 8. Decision log (newest first)
- **2026-10-13**: v2 started. Design: sequential target trial, eligibility-restricted
  multi-arm AIPW, DR-learner time curves, EB plus BH, DML OVB sensitivity. Falsification
  criteria fixed before any results (§4).

## References
- Hernán & Robins (2016), *Using big data to emulate a target trial when a randomized trial is not available*, AJE.
- Danaei et al. (2013), sequential trials for time-varying treatments, *Stat Methods Med Res*.
- Chernozhukov et al. (2018), *Double/debiased machine learning*, Econometrics Journal.
- Cattaneo (2010), efficient semiparametric estimation of multi-valued treatment effects, J. Econometrics.
- Kennedy (2023), *Towards optimal doubly robust estimation of heterogeneous causal effects*, EJS ([arXiv 2004.14497](https://arxiv.org/abs/2004.14497)).
- Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), *Long story short: omitted variable bias in causal machine learning*.
- Crump, Hotz, Imbens & Mitnik (2009), dealing with limited overlap, Biometrika.
- Lipsitch, Tchetgen Tchetgen & Cohen (2010), negative controls, Epidemiology.
- Bach et al. (2022), *DoubleML: an object-oriented implementation of double machine learning in Python*, JMLR ([arXiv 2104.03220](https://arxiv.org/abs/2104.03220)).
