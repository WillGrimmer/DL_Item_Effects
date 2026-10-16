# Method review: what is computed, what is wrong with it, what is left

Written 2026-10-16 for an agent picking this project up with little context. It describes every
calculation the pipeline makes, from the data lake to the reported numbers, logs every known or
suspected problem with evidence, and lists what is done and what is not. Read it with
[`HANDOFF.md`](HANDOFF.md) (the running log, newest section first) and [`PLAN.md`](../PLAN.md)
(the design and the decision log, §8, newest first). Where this document and the code disagree,
the code is right; file and function names are given so you can check.

**The question.** For a hero in a patch, how much does buying shop item X, instead of what the
player actually bought at the same price, change their chance of winning? The answer is wanted
per item, per hero, for the **latest patch**, including items that are bought rarely but may be
good (Will's stated goal, 2026-10-16).

---

## 1. Where things stand (2026-10-16)

| Area | State |
|---|---|
| Data | Two Ranked patch windows extracted: 09-16-2026 (09-17 to 09-28) and 09-29-2026 (09-30 to 10-12). 09-29 is still the latest patch (checked 2026-10-16); the lake has about 4 more days of it than we extracted. |
| Population | Ranked, match-average badge **Oracle 1 or higher** (`--population oracle_plus`, the default). |
| Items | Two tiers per hero, window and price: **core** (≥ 2% share and ≥ 2,500 full-price buys) and **extended / "rare"** (500 to 2,499 buys). |
| Heroes run | **Haze (13)** and **Billy (72)**, both windows, LightGBM. Falsify, calibration and confirm done. |
| Logistic agreement check | Only Haze 09-16 at 800 and 1600 finished. Full logistic is far too slow (see issue F1). A targeted version is proposed, not built. |
| Results | `data/processed/results/haze_billy_oracle_plus.csv` (235 item rows). Dashboard (fixed copy): https://claude.ai/artifact/8F558Frfg4Mq6zvqLehn8w |
| Tests | 77 pass (`python -m pytest -q`, about 25 s). |
| Branch | `plan-revision-and-phase-1`, pushed. `main` holds only Phase 0. No PR. |

**Headline findings so far.**
- Haze, common items: real signal. The z-scores of item effects spread 1.59 against about 1.07
  for fake outcomes. The two windows agree (r 0.63 over 30 items). LightGBM and logistic agree
  (r 0.80 over 19 items). Two items are significant after BH (09-16, 1600): Bullet Resist
  Shredder +3.6pp and Active Reload −5.0pp.
- Haze rare items and all of Billy: no detectable signal yet. Their spread equals the fake
  outcomes', the windows do not agree (rare r ≈ −0.1), and LightGBM vs logistic agree poorly on
  rare items (r 0.45, same sign 9 of 16).
- The design carries a residual confounding bias of about −0.25pp (Haze) and −0.10pp (Billy) on
  outcomes an item cannot affect. Results are calibrated for it (§4.9).

**Decisions Will has made, newest first** (all logged in PLAN §8):
1. Future runs should use **only the latest patch** (discussed 2026-10-16; not yet logged or
   implemented; see To-do T1).
2. Two item tiers, BH per tier, "promising" defined (2026-10-16).
3. Population Oracle+ (2026-10-15).
4. Calibration: shift and widen; Oracle+ borrows the all-ranks next-match calibration, pooled over
   windows (2026-10-15).
5. Falsification rules §4.1 and §4.4 apply pooled per hero and window (2026-10-14).
6. Player-history covariates (2026-10-14).
7. LightGBM outcome models stop early; fits run in parallel (2026-10-14).

---

## 2. Data: what is extracted (`ingest/`)

Source: the deadlock-api data lake (`data.deadlock-api.com`), hourly-rotating parquet files of
the `match_player` table. Read through DuckDB over HTTPS. Column quirks: `docs/data_dumps.md`.

**Match filter** (`ingest/extract.py`, `build_purchase_events`): `match_mode = 'Ranked'`,
`game_mode = 'Normal'`, team in Team0/Team1, **exactly 12 player rows**. Abandons stay in
(`match_has_abandon` is recorded). The window is `start_time` from the day after the patch
release (the release day is skipped) to the end date, inclusive.

**One row per shop purchase** (unnested `items.*` arrays), keeping shop items whose list cost is
800, 1,600, 3,200 or 6,400. Columns, all computed in SQL:

- **Identity and label:** match, account, slot, team, hero, `won`, `start_time`, `patch`,
  `days_since_patch`.
- **Skill:** `badge_level` = match `average_badge` on the linear scale `6·tier + sub`
  (Oracle 1 = 49); `rank_level` = the player's own rank on the same scale (NULL in calibration);
  `in_calibration`; `calibration_games`; `win_streak` (entering the match);
  `own_minus_team` = own rank − team mean rank; `team_rank_gap` = team mean − enemy mean.
- **Item:** id, name, slot, tier, `cost` (list price), `price_paid` = list price minus the cost
  of held components this purchase consumed. Components are credited through the whole upgrade
  chain, once each; when two same-second upgrades could have consumed a component, the direct
  (cheaper) one gets it. `full_price` = `price_paid == cost`. `sold_time_s`;
  `consumed_by_upgrade` (a component's `sold_time_s` is the upgrade time, not a sale).
  **Same-click component rows are dropped:** buying an upgrade outright writes a row for a
  component bought and consumed in the same second.
- **Economy at the last snapshot at or before the purchase** (ASOF join): `net_worth_at_buy`,
  both teams' net worth and the gap, `level`, K/D/A, `snapshot_age_s`. Snapshots come every
  180 s up to 900 s, then every 300 s, plus one at match end. **Before 180 s the state is the
  start-of-game state** (600 souls, level 1), never a later snapshot.
- `n_prior_items` (shop purchases strictly earlier), `ally_heroes`, `enemy_heroes` (lists).
- **Negative-control outcomes, never covariates:** `nc_prior_souls` and `nc_prior_deaths` =
  the change in net worth and deaths over the snapshot interval ending at that snapshot
  (`nc_prior_len_s` long; NULL before the first snapshot); `nc_prev_won` = the player's previous
  Ranked match result, `prev_gap_h` hours earlier (ASOF on `_history.parquet`).
- `duration_s`: diagnostics only, never a covariate.

**Side files per window** (`data/processed/events/patch_<patch>/`):
- `_history.parquet`: (account, match, start, won) for every Ranked Normal match in the history
  partitions. These cover about 2 weeks before the window and the window itself. **No hero_id.**
- `_objectives.parquet`: objective losses (t1, t2, base) and mid-boss claims per match
  (`ingest/objectives.py`).
- `_sources.json`: partitions, exact file URLs, manifest version, git commit.

**Sizes.** 09-16: 175,188 matches, 35.6M purchase rows. 09-29: 204,837 matches, 41.3M rows.
Oracle+ keeps 26% (09-16) and 29% (09-29) of matches; Haze is in 30,732 and 39,682 of them,
Billy in 14,182 and 13,830.

---

## 3. The decision frame (`frame/risk_set.py`, `frame/history.py`, `pipeline.py`)

Built per (hero, window, price) by `pipeline.build_frame`. One price is one "experiment".

1. **Load** (`pipeline.load_hero_window`): the hero's purchases, filtered to `badge_level ≥ 49`
   for Oracle+; every player's purchases in those matches (for enemy inventories); objectives;
   history features for the hero's accounts.
2. **Visits** (`assign_visits`): a player's purchases at most **30 s** apart are one visit.
   `visit_start` = its first purchase time. State is taken from the visit's first purchase, at
   any price, so purchases in one visit never see each other.
3. **Arms** (`arm_tiers`): full-price purchases at this price, counted per item. Core = share
   ≥ 2% of the price's full-price buys and ≥ 2,500 buys; extended = any other item with ≥ 500
   buys. Uses purchase counts only. A price with fewer than 2 arms is skipped.
4. **Rows:** visits that buy **exactly one** arm at **full price** at this price. Visits buying
   two or more arms are counted (`n_multi_arm_visits`) and dropped. Component upgrades
   (`price_paid < cost`) are never rows.
5. **Eligibility** (`elig_k`): arm k is buyable on a row unless the player owns it or holds any
   of its components (at any depth). Owned at t = bought before t and not sold or consumed
   before t (`sold_time_s = 0 or ≥ t`). Rows whose chosen arm is ineligible are dropped
   (`n_chosen_ineligible`, about 0 to 230 per frame).
6. **Crossover** (`next_k`): seconds from the decision until the player buys arm k later in the
   match (NaN if never). Used for labels only: share of the comparison group that buys X by
   match end is < 25% "choice", 25–50% "mixed", ≥ 50% "order".
7. **Covariates** (`covariate_columns`; about 425 columns at all ranks):
   - **State (18):** badge_level, rank_level, own_minus_team, team_rank_gap, in_calibration,
     win_streak, has_build, net_worth_at_buy, my_team_nw, enemy_team_nw, team_nw_diff_at_buy,
     n_prior_items, level, kills, deaths, assists, snapshot_age_s, game_time_s
     (= `visit_start`).
   - **Player history (6)** (`frame/history.py`; strictly earlier Ranked matches of the
     account, ordered by (start_time, match_id)): `prev_won`; `wr_2_6` and `wr_2_11` = win
     rate over matches 2–6 and 2–11 back (NaN if none); `n_hist` = number of earlier matches,
     capped at 11; `games_24h` = matches started in the previous 24 h; `prev_gap_h`.
   - **Lane (3):** one-hot of `assigned_lane` (1, 4, 6).
   - **Own inventory (about 150):** `own_<item>` for every item this hero buys in the window.
   - **Composition (about 85):** `ally_h<id>`, `enemy_h<id>` multi-hot for every hero in the
     window.
   - **Enemy inventory (about 150):** `enemy_<item>` = how many enemy players own each item.
   - **Objectives (8):** own and enemy t1/t2/base losses, own and enemy mid-boss claims before
     the decision.
   - **Not covariates:** `days_since_patch`, `duration_s`, every `nc_*`, `nc_next_won`, `won`.
8. **Dev split:** `dev` = splitmix64(match_id) mod 5 == 0 (`pipeline.is_dev`), 20% of matches.
   Raw `match_id % 5` is not uniform, which is why the hash exists.

---

## 4. Estimation: every calculation (`estimate/`, `analysis.py`, `phase3.py`, `summary.py`)

Notation per frame: rows i, arms k = 1..K, chosen arm A_i, outcome Y_i (win 0/1), eligibility
E_ik, covariates X_i, match cluster g(i).

### 4.1 Nuisance models, cross-fitted (`estimate/aipw.py`)
- **Folds:** `GroupKFold(5)` by match. Every prediction for a row comes from models trained on
  the other 4 folds.
- **Propensity** `fit_propensity`: one multiclass classifier P(A = k | X) per fold, predicted
  out of fold. It is then set to 0 for ineligible arms and renormalized:
  e_ik = p̂_ik·E_ik / Σ_j p̂_ij·E_ij. It uses no outcome, so it is fitted once per (frame,
  learner, covariate set) on **all** rows (dev + confirm) and cached as
  `prop_<learner>_<price>_<tag>_<nrows>.npy`. Tags: `h_all` (production covariates), `h_nc`
  (without `win_streak` and `prev_won`, for §4.1), `h_noskill` (without the skill columns, for
  the OVB benchmark).
- **Outcome** `fit_multiarm`: one model per (fold, arm), trained on that fold's rows that chose
  k, predicting μ_k(X) = P(Y = 1 | X, A = k) for the held-out fold. If a training fold has no
  row for arm k, that (fold, arm) is unsupported (`fold_absent`). Constant-outcome arms get the
  constant.
- **LightGBM** (`make_learner`): `n_estimators` 400, `learning_rate` 0.05, `num_leaves` 31,
  `min_child_samples` 100, `subsample` 0.8 (every iteration), `colsample_bytree` 0.8,
  `reg_lambda` 1. NaN handled natively. **Outcome models stop early** (`fit_outcome`): 10% of
  the training fold's matches held out, patience 50, at most 1,000 trees, skipped below 2,000
  rows. Propensities keep a fixed 400 trees.
- **Logistic** (cross-check): median imputation with missing indicators, StandardScaler,
  `LogisticRegression(C=1, max_iter=2000)` (multinomial lbfgs); `Ridge(alpha=1)` for continuous
  outcomes.
- **Parallelism:** fits run in a 4-worker thread pool (`FIT_WORKERS`, env `DL_FIT_WORKERS`),
  with LightGBM `n_jobs` = cores / 4. **Logistic gets no thread limit** (issue F1).

### 4.2 Support (trimming)
supported_ik = E_ik and e_ik ≥ **0.01** and μ_k(X_i) defined. Unsupported rows are dropped
from that arm's contrasts, never clipped. `trim_share_k` = 1 − supported / eligible. Each arm's
estimate therefore covers its own sub-population (issue B3).

### 4.3 Doubly robust scores
ψ_ik = μ_k(X_i) + 1{A_i = k}·(Y_i − μ_k(X_i)) / e_ik, on supported (i, k); NaN elsewhere.

### 4.4 Headline: "X instead of what they actually bought" (`vs_actual`)
On rows where k is supported, let d_i = ψ_ik − Y_i and o_i = 1{A_i ≠ k}:
- θ_k = mean(d) / mean(o) = E[Y(k) − Y | A ≠ k] over supported rows.
- **SE:** match-clustered, from the ratio's influence function (d_i − θ_k·o_i) / mean(o). Per
  cluster, sum the influence values; SE = sqrt(Σ_g S_g²) / n.
- Also reported: `n`, `n_chose`, `n_other`, `trim_share`, crossover shares within 600 s and
  by match end, and the crossover label.

### 4.5 Pairwise (`pairwise`)
θ_kj = mean(ψ_k − ψ_j) on rows where both are supported; match-clustered SE. Used for the
"X or Y?" matrix.

### 4.6 Effect over game time (`estimate/heterogeneity.py`, `dr_curve`)
Regress (d, o) on a cubic B-spline in game time (6 knots at quantiles); the curve is
numerator / denominator on a grid of 300 to 2,700 s every 150 s. Band: delta method from a
match-clustered sandwich over both regressions. Bands are pointwise.

### 4.7 Shrinkage and ranking (`empirical_bayes`)
Normal-normal empirical Bayes per (hero, window, price), across all arms in the frame (both
tiers), on calibrated effects and SEs. τ² is DerSimonian-Laird: max(0, (Q − (K−1)) / c), with
w = 1/se², Q = Σw(y − ȳ_w)², c = Σw − Σw²/Σw. The posterior mean is m + B(y − m), with
B = τ² / (τ² + se²). If τ² = 0, every arm shrinks to the same value (Haze 3200 and 6400 in
09-29).

### 4.8 Multiple testing (`estimate/fdr.py`, `phase3.main`)
p = two-sided normal p of the **calibrated** effect. Benjamini-Hochberg q runs **per tier** over
every arm at every price of one (hero, window). "Significant" = q < 0.05.

### 4.9 Calibration for residual confounding (`analysis.calibrate`, `pooled_calibration`)
Systematic error is modelled as N(μ, τ²) (Schuemie et al. 2018):
- **Source:** the next-match control (§5.2) at **all ranks on core arms**, pooled over both
  windows. μ and τ are DerSimonian-Laird over every arm and price; se(μ) is a match-clustered
  Poisson bootstrap with nuisances held fixed. Stored in
  `results/hero_<id>/calibration_oracle_plus.json`, built by `--stage calibrate`.
- **Values:** Haze μ −0.238pp, τ 0.261pp, se 0.072pp (80 arms). Billy μ −0.099pp, τ 0,
  se 0.089pp (87 arms).
- **Headline and curves:** effect_cal = θ − μ; se_cal = sqrt(se² + τ² + se(μ)²).
- **Pairwise:** not shifted (a common bias cancels); se_cal = sqrt(se² + 2τ²).
- Raw and calibrated values are both written.

### 4.10 Reporting flags (`summary.py`)
- `p_better` = Φ(effect_cal / se_cal). A flat-prior one-sided normal probability, not a
  posterior from the shrinkage model.
- `promising` = not significant, and `p_better` ≥ 0.9 or ≤ 0.1.
- `replicates` = the item ran in both windows with the same calibrated sign.
- `agree` / `counts` come from `confirm_agreement.csv` when logistic ran: same sign, and both
  significant or both not, at q < 0.05.

### 4.11 Sensitivity bounds (`analysis.ovb`, `ovb_benchmark`)
Chernozhukov, Cinelli, Newey, Sharma & Syrgkanis (2022), in DoubleML's parametrization: bias
≤ S·|ρ|·sqrt(cf_y·cf_d / (1 − cf_d)), with S² = σ²·ν², σ² = mean residual² and ν² from the
Riesz representer (eligibility-aware).
- `rv` = the confounder strength (cf_y = cf_d) that moves the estimate to 0; `rva` does the
  same for the edge of the 95% interval.
- Benchmark: refit without the skill columns (`h_noskill`) to get the cf_y and cf_d "as strong
  as rank", and the bias such a confounder would cause.
- cf_d = 1 now gives an unbounded bias (it used to divide by zero, fixed in `d17556f`).

### 4.12 Stages (`phase3.py`)
- `falsify`: §5 checks on all rows; writes negative-control tables, per-(arm, match) score sums
  (`*_scores.parquet`), split-half z and summaries, then the window-level pooled JSON.
- `confirm`: effects, pairwise and curves on the 80% non-dev rows. Propensities are reused from
  all rows; outcome models are fitted on confirm rows only.
- `shuffle` and `synth`: estimator nulls (§5.6).
- `calibrate`, `pool`, `compare`, `dev` as described above.

---

## 5. Falsification and diagnostics (PLAN §4; `analysis.py`)

Thresholds are frozen in `analysis.THRESHOLDS`. Since 2026-10-14, §4.1 and §4.4 are judged
once per (hero, window), pooled over prices.

### 5.1 §4.1 previous-match negative control (`nc_player`)
- **Outcome:** `nc_prev_won`. **Covariates:** production minus `win_streak` and `prev_won`
  (the history window starts two matches back, so the outcome is never a covariate).
  Propensities use tag `h_nc`.
- Headline effect per arm; then DerSimonian-Laird over every arm and price gives μ and τ.
- **Pass:** |μ| ≤ 0.2pp and τ ≤ 0.3pp.
- Reported beside it: `null_fail_rate` (how often the rule fails at these SEs with zero bias,
  assuming independent arms) and the bootstrap `chance_fail` (dependence included).

### 5.2 Next-match diagnostic (`nc_next`)
The same, with outcome `nc_next_won` (the account's next Ranked match) and the full production
covariates. It is not a gate, but it is the calibration source (§4.9).

### 5.3 §4.2 in-game momentum (`nc_ingame`)
- **Outcomes:** `nc_prior_souls` and `nc_prior_deaths`, each divided by its SD; LightGBM
  regressors for the outcome models.
- **Pass, per frame:** |effect| ≤ 0.1 SD for at least 90% of arms and mean |effect| ≤ 0.05 SD.

### 5.4 §4.3 SE calibration (`se_calibration`, dev rows)
- The arm with the most buyers is split at random into two fake arms. The fake arms'
  propensity is e_k / 2 and their outcome model is μ_k.
- 200 repetitions. **Pass:** 92% to 98% of 95% intervals cover zero.

### 5.5 §4.4 split-half stability (`split_half`)
- Matches are split at random into two halves. Each half fits its own propensities and
  outcome models.
- z = (θ_A − θ_B) / sqrt(se_A² + se_B²) per arm. **Pass, pooled per window:** at most 10% of
  arms have |z| > 1.96. The seed is the price, so every run uses the same halves (issue C5).

### 5.6 Estimator nulls
- **Shuffle** (`nc_player_shuffled`): the previous result is permuted across player-matches,
  one value per player-match, so the true effect is 0 and nothing is confounded. It cannot
  catch bias from the nuisance models fitting a real signal.
- **Semi-synthetic** (`nc_player_semisynthetic`): a pooled covariates-only model predicts the
  outcome out of fold, predictions are averaged per player-match, and Y* ~ Bernoulli(p). The
  true effect is 0, but the covariate signal is realistic. Run on Haze all ranks (previous −0.06
  and −0.03pp) and Oracle+ core-only (previous −0.01 and −0.02pp; next +0.34 and −0.22pp, within
  1.5 SEs). Not yet run on the two-tier design.

### 5.7 Pooled bootstrap (`bootstrap_nc`)
Per (arm, match) the sums of d and o are saved. A Poisson(1) weight per cluster (match or
account), 1,000 reps, all prices jointly, gives θ_b = Σw·d / Σw·o per arm. Reported: the SD of
the pooled μ, the ratio of bootstrap to nominal SE, and `chance_fail` (rule failures on draws
centred at the estimates). **Nuisance models are held fixed** (issue C2).

### 5.8 What the checks show (latest)
- **Haze all ranks (old core-only design):** §4.1 −0.38 and −0.21pp ± 0.10, fail. Next match
  −0.22 and −0.26pp. Shuffle +0.07 and +0.13. Synth −0.06 and −0.03. §4.2, §4.3 and §4.4 pass.
- **Haze Oracle+ (two tiers):** §4.1 −0.49 ± 0.23 (09-16) and +0.00 ± 0.21 (09-29); core-only
  −0.42 and −0.15. Next match +0.07 and −0.28. §4.3 coverage 0.905 to 0.95. §4.4 4.3% and 1.4%.
- **Billy Oracle+:** §4.1 −0.30 ± 0.34 and −0.13 ± 0.37; next match +0.05 and −0.32.
  §4.4 4.3% and 6.2%.
- **§4.2 fails in 13 of 16 Oracle+ frames, only because of rare arms.** Core arms: 99.5%
  within 0.1 SD, mean 0.022. Rare arms: 24% outside, but their z SD is 1.05, which is pure
  noise.
- **Signal vs noise** (z SD of real effects vs fake-outcome effects, Oracle+): Haze core 1.59
  vs 1.07/1.10; Haze rare 1.05 vs 0.97/0.94; Billy core 0.92 vs 0.97/1.03; Billy rare 0.68 vs
  0.69/0.93.
- **Replication across windows** (calibrated effects): Haze core r 0.63 (30 items), Haze rare
  −0.10 (37), Billy core 0.55 (19), Billy rare −0.16 (22).
- **LightGBM vs logistic** (Haze 09-16, 800 and 1600): core r 0.80, same sign 17/19; rare
  r 0.45, same sign 9/16. Logistic SEs are 1.09× (core) and 0.73× (rare) LightGBM's.

---

## 6. Issues log

Each entry: what, evidence, why it matters, what to try. Severity: **H** can change
conclusions, **M** weakens or biases them, **L** hygiene.

### A. Identification and confounding

**A1 (H). Residual confounding of about −0.25pp remains.**
- **Evidence:** Haze all ranks, previous match −0.30pp pooled (SE 0.07) and next match −0.24pp
  (SE 0.07), while shuffle and synth are ≈ 0.
- **What didn't fix it:** history covariates; adding within-window Haze experience, uncapped
  history count and account-id rank left the next-match control unchanged (−0.57 → −0.60pp,
  −0.46 → −0.49pp on two frames).
- **Pattern:** at 09-16 the most negative previous-match arms are the mainstream picks (Burst
  Fire, Ricochet, Active Reload, Bullet Lifesteal); the previous- and next-match per-arm
  effects correlate at r 0.06.
- **Why it matters:** it is the same size as many item effects. It is handled by calibration
  (§4.9), not removed.
- **Unmeasured candidates:** party/stack (`party` is always NULL), session form, time of day,
  region/server, smurf or account-sharing, hero experience (history has no hero_id),
  `days_since_patch` (A6).

**A2 (H). Calibration rests on transfer assumptions that are not tested.** The bias measured on
the next-match outcome, at all ranks, on core arms, pooled over windows, is applied to:
- the current-match outcome,
- Oracle+ only (Oracle+ own estimate −0.18 ± 0.16, consistent but noisy),
- rare arms (no evidence either way),
- each hero separately (Haze −0.24, Billy −0.10, so it differs by hero; a roster rule is
  undecided).

τ is added to every arm's SE, so a hero with τ = 0 (Billy) gets no widening for per-arm
spread. The memo of 2026-10-14 warned against calibrating away a failed check; Will chose it
knowingly.

**A3 (M). The next-match control has small built-in effects.**
- **Post-treatment path:** item → this match's result → next match (momentum, or MMR making
  the next match harder). Estimated negligible: a 1pp effect times the dependence of
  consecutive results.
- **Selection:** 8% of rows have no next match (the player stopped, or the window ended). If
  quitting after a loss relates to item choice, the control is biased. Untested.

**A4 (M). The headline compares against "what they actually bought".**
- This mix differs per arm and per row, so θ_k is not "X vs a fixed alternative". Two arms'
  headlines are not directly comparable; `pairwise` is the like-for-like contrast.
- Adding the extended tier changed the comparison group for core items, so core estimates
  moved between the core-only and two-tier runs.

**A5 (M). Crossover.** 17 core arm-rows are "order" (≥ 50% of the comparison group buys the
item later anyway), so their effect is a timing effect ("now vs later"). 28 are "mixed". Rare
items are almost all "choice".

**A6 (M). `days_since_patch` is not a covariate.** The meta moves over a 13-day window, and a
hero's win rate and item popularity can drift together. It is a cheap candidate covariate,
possibly related to A1. Untested.

**A7 (M). State is stale.**
- Economy, level and K/D/A come from the last snapshot, up to 299 s before the purchase
  (`snapshot_age_s` is a covariate, the values in between are not).
- A fight between the snapshot and the shop visit is invisible. §4.2 tests the interval that
  ends at the snapshot, not the gap after it, so this staleness is not tested.

**A8 (L). Unobservable in the data:** sell and re-buy (one row per player-item), party, region.
Item arrays are not time-sorted (handled).

### B. How the model interacts with this dataset

**B1 (H). Rare (extended) arms are data-starved for per-arm outcome models.**
- With 500 to 2,499 buys per window, a training fold has roughly 400 to 2,000 rows of that arm,
  against about 425 covariates. LightGBM with `min_child_samples` 100 and early stopping gives
  near-constant μ_k, so AIPW leans on inverse weighting (high variance).
- Evidence: rare-arm SEs about 4.6pp vs about 1.2pp core (Haze); LightGBM vs logistic on rare
  items r 0.45 and Rusted Barrel +10.0 vs +1.2pp; rare effects do not replicate across windows
  (r ≈ −0.1).
- **To try:** one pooled outcome model with arm as a feature (S-learner, or arm
  interactions), so rare arms borrow structure; fewer covariates for rare arms; cross-hero
  pooling (T6).

**B2 (H). "Promising" fires about 20% of the time on pure noise.** p_better ≥ 0.9 or ≤ 0.1 has
a 20% chance per item under the null. Observed rates:

| | Promising rate | Read |
|---|---|---|
| Haze core | 47% | real |
| Haze rare | 24% | ≈ chance |
| Billy core | 14% | below chance |
| Billy rare | 6% | below chance |

- "Promising in both windows, same direction" has about 2% null rate; Haze core 9 of 30 (30%),
  Haze rare 3 of 37 (8%), Billy 0. So most rare "promising" flags are noise.
- **To fix:** report a posterior from a proper hierarchical model with false-sign-rate
  control, or require replication and learner agreement for rare items.

**B3 (H). Trimming at a fixed propensity of 1% removes most rows for rare arms.**
- Median trim share: core 0.27 (90th percentile 0.72); rare 0.80 (90th percentile 0.94).
  Extra Charge at 800 is about 0.85 trimmed.
- Each rare-arm effect describes the narrow fifth of situations where the item is a plausible
  buy. That may be the right question, but it differs per arm, and the label on the page does
  not say so.
- **To try:** an arm-relative threshold (e.g. e ≥ 0.1 × the arm's base rate), overlap weights,
  or report the supported sub-population explicitly.

**B4 (M). Many sparse covariates for the sample size.** About 425 columns (own inventory about
150, enemy inventory about 150, composition about 85), mostly 0/1 or small counts, at Oracle+
sizes (Haze frames 57k to 125k rows, Billy smaller). `colsample_bytree` 0.8 helps. Collapsing
enemy inventories to category counts or using embeddings is untested.

**B5 (M). LightGBM hyperparameters were never tuned.** num_leaves 31, min_child 100, lr 0.05
are defaults chosen once. Only early stopping was tuned (on the dev set, logged).
- Evidence: with a fixed 400 trees the outcome models did worse than a constant on held-out
  negative-control outcomes (0.715 vs 0.693 log-loss). Propensities stop around 300 trees and
  were left at 400.
- **To try:** propensity calibration (reliability) checks and a small dev-set grid.

**B6 (M). Separate outcome model per arm.** This is a T-learner; nothing is shared across arms.
It is fine for core arms and poor for rare ones (B1).

**B7 (L). Propensity weight tails.** The top 1% of buyers carry 5% to 20% of the 1/e weight per
arm (Haze 09-29 800). They are not monitored routinely.

### C. Inference and statistics

**C1 (M). The frozen thresholds are absolute and do not scale with sample size.**
- §4.1 (|μ| ≤ 0.2pp, τ ≤ 0.3pp) fails by chance 17% to 25% (Haze) and 55% to 65% (Oracle+
  per hero).
- §4.2 (0.1 SD) fails noisy rare arms on noise alone.
- Pass/fail is therefore weak evidence either way. Use the bootstrap `chance_fail` and the
  z-based views beside it.

**C2 (M). The bootstrap holds nuisance models fixed.** It measures second-stage variance only.
Earlier shuffle reruns suggested the pooled mean varies more than its nominal FE SE (an SD of
0.25 across shuffles vs nominal 0.18 to 0.27), and one frame cancelled to 1e-8 on a shuffled
outcome (the arms' vs-actual estimates are negatively correlated). A full refit bootstrap is
expensive.

**C3 (M). Per-frame split-half is very noisy.** One 7-arm frame's z SD ranged 0.72 to 1.61 over
four seeds. Use the pooled rule.

**C4 (M). Empirical Bayes mixes tiers and can collapse.** EB runs across core and rare arms
together per price, on the flat normal model. With τ² = 0 every arm gets the same shrunk value
(Haze 09-29 at 3200 and 6400), which removes ranking information.

**C5 (L). Split-half uses seed = price.** Every run splits the same halves, so a flag repeats
across runs without being independent evidence (Extra Spirit z −4.0 and −3.5 were one split;
fresh seeds gave −0.13 to +1.39).

**C6 (L). `p_better` ignores shrinkage.** It is computed from the unshrunk calibrated effect.

### D. Protocol and process

**D1 (H). The dev/confirm rule has been bent.** PLAN §3.1 says win-outcome tables on the 80%
confirm rows are computed once, after the design is frozen. In practice:
- an all-ranks confirm ran 3 LightGBM frames (stopped; one frame's table was shown in chat);
- the Oracle+ core-only confirm ran and was reported;
- the population (Oracle+) and the item tiers were then changed, and confirm ran again.

Each change was logged in PLAN §8 before its run, and the tiers were chosen from purchase
counts, not outcomes. But design decisions were made after seeing confirm results, so this is
a garden-of-forking-paths risk. Any write-up must say so. A clean confirmation needs data the
design has not seen, e.g. the 09-29 days after 10-12, or the next patch.

**D2 (M). The plan's strongest validity test (§4.5 patch diff-in-diff) has never run.** It needs
two patches and patch-note directions coded in advance. With latest-patch-only production
(T1), keep it as a periodic check on a few heroes.

**D3 (M). Time estimates were unreliable.** Logistic was never timed before being scheduled;
it ran 15 to 25 times over estimate. Time one frame before queueing a stage at scale.

### E. Data quirks to remember (see also HANDOFF §9)
- Lake file URLs go stale within about 2 hours: re-read the manifest.
- Team badges are 0 since 2026-07-30; only Ranked has skill. Filter `game_mode = 'Normal'`.
- The rank and badge columns are UInt32: cast before subtracting. Item ids arrive as uint64.
- New heroes 84, 85 and 88 have no 09-16 data.
- History covers about 2 weeks before each window, so `n_hist` and `games_24h` are truncated
  for early-window rows of new or returning accounts.

### F. Compute and engineering

**F1 (H for cost). Logistic is extremely slow and probably thread-oversubscribed.**
- Haze Oracle+ two-tier: 79 min (800), 59 min (1600), and 3200 still running after 3.5 h when
  stopped.
- Likely cause: 4 parallel fits, each letting BLAS use all 16 cores.
- **Fix to try:** `threadpoolctl.threadpool_limits(FIT_THREADS)` around each logistic fit,
  cap `max_iter` at 500, skip the OVB refit, pairwise and curves, and fit outcome models only
  for flagged arms (proposal T3).

**F2 (M). The propensity cache key is (learner, price, tag, nrows) only.** If the covariates or
the arm list change but the row count does not, a stale cache is silently reused. Tags were
bumped by hand (`h_` prefix) when covariates changed. Fix: hash the covariate list, the arms and
the population into the key.

**F3 (L). Per-frame costs (Oracle+, LightGBM).**
- Haze falsify two-tier: about 17 to 20 min per window. Haze confirm: about 3 to 5 min per window.
- Billy is about 40% of that. All-ranks Billy falsify (core): about 17 min per window.
- CPU runs at 100% during fits (by design).

**F4 (L). Output layout.**
- `results/hero_<id>/patch_<patch>/`: all ranks, core arms (calibration source).
- `.../pop_oracle_plus/`: Oracle+, both tiers (current).
- `.../pop_oracle_plus/core_only_2026-10-15/` and `.../pre_history_2026-10-14/`: superseded.
- Logs: `results/overnight_2026-10-16.log` and others.

---

## 7. To-do, in priority order

Decisions for Will are marked **(Will)**.

**T1. Latest patch only (Will agreed in discussion; log it in PLAN §8 first).**
- Re-extract 09-29 through the latest available day (about 4 more days, about 30% more data):
  `python -m dl_item_effects.ingest.extract --patch 09-29-2026 --end <yesterday>`, then
  objectives. The windows now overlap the old one, so write to a new folder or overwrite after
  archiving.
- Point the runner at one patch.
- The new days also give a clean hold-out for D1: they were never seen by the design.

**T2. Fix logistic speed (F1)** and measure one frame before relying on any estimate.

**T3. Targeted agreement check.** Logistic on significant and promising items only.
- **Proposed rule (Will to confirm):** same calibrated sign, and for significant items
  logistic's 95% interval excludes 0; for promising items logistic `p_better` ≥ 0.8 (or ≤ 0.2).
- Gate rare-item "promising" on agreement.

**T4. Fix the rare-item weaknesses before the roster (B1 to B3).**
- An arm-relative trim or overlap weights.
- A pooled outcome model, or an S-learner.
- A proper posterior / false-sign-rate summary to replace `p_better`.
- **(Will):** whether §4.2 judges rare arms on a z basis.

**T5. Roster.** About 41 heroes at Oracle+, latest patch only. About 4 h for falsify plus
LightGBM confirm, plus the agreement check. Choose the calibration rule for the roster
**(Will)**:
- per hero, from all ranks (needs an all-ranks falsify per hero, about 50% more compute);
- a roster-wide pooled value;
- per hero at Oracle+ (noisy for small heroes).

**T6. Cross-hero pooling** for item effects, a hierarchical model per item across heroes. This
is the main lever for recognizing underused items.

**T7. Investigate A1.** Try `days_since_patch` and time-of-day covariates (time of day can come
from `start_time`). Re-extract history with hero_id for hero experience. Test next-match
missingness (A3).

**T8. Engineering.** Fix the cache key (F2). Phase 4 `validate/` module: a per-hero
falsification report and the WPA baseline (PLAN §4.7).

**T9. Phase 6 pages** (hero pages from cached tables) and **Phase 7** (patch diff-in-diff as a
periodic check, D2).

**T10.** Merge `plan-revision-and-phase-1` into `main` or open a PR when Will wants.

---

## 8. Commands (repo root, `PYTHONPATH=src`, one heavy job at a time)

```bash
python -m pytest -q
python -m dl_item_effects.phase3 --hero 72 --patch 09-29-2026 --stage falsify --learner lgbm                      # Oracle+, both tiers (defaults)
python -m dl_item_effects.phase3 --hero 72 --patch 09-29-2026 --stage falsify --learner lgbm --population all     # all ranks, core arms (calibration source)
python -m dl_item_effects.phase3 --hero 72 --patch 09-16-2026 --stage calibrate --population oracle_plus --cal-source all
python -m dl_item_effects.phase3 --hero 72 --patch 09-29-2026 --stage confirm --learner lgbm
python -m dl_item_effects.phase3 --hero 72 --patch 09-29-2026 --stage synth --learner lgbm
python -m dl_item_effects.phase3 --hero 72 --patch 09-29-2026 --stage pool --learner lgbm
python -m dl_item_effects.summary --heroes 13 72 --name haze_billy_oracle_plus
```

The `calibrate` stage needs `--patch` only because the CLI requires it; `--cal-patches`
chooses the windows to pool.
