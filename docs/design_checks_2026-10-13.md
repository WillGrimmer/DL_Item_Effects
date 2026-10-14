# Design checks before Phase 1

Run 2026-10-13 against the live data lake and the Phase 0 code. Nothing in `src/` was changed.
These checks are why `PLAN.md` changed on the same day. Data facts are in
[`data_dumps.md`](data_dumps.md); this file holds the design evidence.

## Bottom line

The Phase 0 estimator does what it says. Three things around it did not hold:

1. **The data it was written for no longer exists in that form.** New host, new layout, and the
   skill covariate (team badge) has been zero since 2026-07-30. Only Ranked matches carry skill
   data now, and there it is per player.
2. **"Same cost" is not "same price".** 30.5% of shop purchases consume a held component and
   pay less than list price.
3. **"X now instead of Y now" is often a question about order.** For the most-bought items, most
   players who pick Y buy X later in the same match. The estimator then correctly reports an
   effect near zero, however much owning X is worth.

## 1. Crossover: the comparison group buys X later

Mirage, 29,403 player-matches, 2026-10-10 to 10-13, Ranked and Unranked pooled (the two modes
agree; pair-level correlation 0.97). For each ordered pair (X, Y) among the six most-bought
items at a cost: of the players who bought Y while not owning X, the share who then bought X.

| Cost | within 30 s | within 3 min | within 10 min | later in the match | range of "later" across pairs |
|---|---|---|---|---|---|
| 800 | 0.9% | 3.6% | 10.4% | 18.5% | 3% to 86% |
| 1600 | 2.8% | 10.4% | 32.1% | 46.3% | 6% to 90% |
| 3200 | 0.8% | 2.8% | 8.7% | 15.0% | 3% to 88% |
| 6400 | 1.2% | 4.0% | 23.0% | 29.6% | 16% to 69% |

Values are medians across the 30 pairs at each cost. 65 of 120 pairs are at 25% or more, and 24
are at 50% or more.

- Mystic Regeneration after Headshot Booster: 72% within 3 minutes, 86% in the match.
- Toxic Bullets after Headhunter: 88% in the match.
- Ricochet and Escalating Exposure: 68% in each direction.
- Headshot Booster after Extra Stamina: 3%.

Per item: the share of an item's comparison rows that come from players who later buy it is 25%
or more for 14 of 39 on-support items, and 50% or more for six (Mystic Vulnerability 74%, Toxic
Bullets 73%, Escalating Exposure 56%, Ricochet 54%, Mystic Regeneration 52%, Healbane 51%).
Those six are each owned by 67% to 86% of Mirage players by the end of the match.

**Shop visits.** 36% of purchases sit in a visit (gaps of 30 s or less) with another purchase,
and 17% share a visit with another purchase at the same cost (26% at 1600). Inside a visit the
order is click order.

## 2. What the estimator returns under crossover (simulation)

Synthetic data with no unmeasured confounding, run through the repo's `decision_frame` and
`fit_multiarm`. Truth: ending the match owning X is worth +3.0pp. Estimates in percentage points.

**Crossover.** A share `p` of first-pickers buy the other item later. 600k player-matches.

| p | "X now vs Y now" truth | `pairwise(X, Y)` | 95% CI |
|---|---|---|---|
| 0.0 | 3.00 | 2.77 | 2.44 to 3.11 |
| 0.3 | 2.10 | 1.88 | 1.55 to 2.22 |
| 0.6 | 1.20 | 0.98 | 0.65 to 1.31 |
| 0.9 | 0.30 | 0.09 | -0.25 to 0.42 |

The estimator tracks the order effect, 3.0 × (1 − p). It is not biased for its own estimand. It
does not measure the value of owning X.

**Same-visit pairs.** Half the players buy X and Y in one visit, 1 to 3 s apart, random order.
The plain frame gives 0.91 when the order effect among the other players is 3.0 and the
population average is 1.5. Collapsing purchases within 30 s into one visit and dropping visits
that contain both arms gives 2.78 (2.31 to 3.25).

**An "ever owned X" estimand.** AIPW of win on "bought X at some point" recovers 3.0 when the
window covers every purchase. When match length depends on an unobserved factor that also
drives winning, it returns 1.3 and 0.2. Ownership-by-time needs censoring weights and
time-varying covariates. It is not a quick add-on.

**Headline baseline.** Four arms with shares 55 / 25 / 15 / 5%. The current headline ("vs a
uniformly random alternative") and "vs what players actually bought" are both unbiased with
honest SEs. For the 55% arm the SE is 0.40 against 0.15 (0.33 once rescaled to players who
bought something else); for the 5% arm 0.84 against 0.80. The choice between them is about
meaning, not precision.

**Open:** across these runs the estimate sits about 0.2pp under truth while coverage stays at
95 to 97%. Not yet explained. It may be the simulation's definition of truth.

## 3. Power in Ranked

Half-width of a 95% interval, approximated as `1.96 × sqrt(0.25/n_arm + 0.25/n_rest)`. This
ignores clustering and nuisance estimation, so it is optimistic.

- Mirage in Ranked: 10.9k player-matches a week. One week gives a median half-width of 1.87pp
  across 36 on-support items (1.10 to 3.31). A 1.0pp half-width is reached by 9 items in two
  weeks, 20 in four, 34 in eight.
- Scaled to one 13-day patch window by player-match count, assuming Mirage's item shares:

| Hero | Player-matches per window | Median half-width | Detectable at 80% power |
|---|---|---|---|
| Mirage (least played) | 23k | 1.3pp | 1.8pp |
| Median hero | 57k | 0.8pp | 1.2pp |
| Infernus, Lash | 96k | 0.6pp | 0.9pp |
| Haze | 129k | 0.5pp | 0.8pp |

- The old arm threshold of 500 buys gives a half-width of 4.4pp at best. A 2pp half-width needs
  about 2,400 buys.

## 4. Player-level signal available in Ranked

Ranked rows in the 09-30 to 10-12 window (2.47M):

- 97% link to the same player's previous Ranked match (searching back to 09-01). Median 5
  Ranked games per account in the window.
- Win rate after a win is 51.0%, after a loss 49.1%.
- Win rate by `player_rank_initial_win_streak`: 49.0% at 0, 50.0% at 1, 50.9% at 2, 51.9% at 3,
  55.1% at 6 or more.

So there is player-level win propensity beyond rank, and both a covariate for it (win streak)
and a same-scale negative-control outcome (previous match result) are in the table.

## 5. Tooling: DoubleML 0.11.4

- `DoubleMLAPOS` exists (since 0.9.0) with `causal_contrast` and `sensitivity_analysis`.
- Sensitivity on contrasts runs but is undocumented and untested upstream.
- `DoubleMLAPOS` with cluster data fails at construction. Single-level `DoubleMLAPO` clusters
  fine.
- No per-row restriction on which levels are available.
- Installs and runs on Python 3.14, which it does not officially support.

It can cross-check our estimator on a synthetic case without eligibility or clustering. It
cannot be the sensitivity engine.

## 6. Defects found in the Phase 0 code

`ingest/extract.py`
- Points at a host that no longer resolves and joins a table that no longer exists. With the
  new URL and a local shop-item table the SQL runs unchanged (3,000 matches, 514k rows).
- Filters `match_mode` only, so Street Brawl is included.
- `require_badge` tests `IS NOT NULL`; badges are 0, so it filters nothing.
- Purchases before the first snapshot (7.4%) get NULL level and K/D/A, and their net worth is
  backfilled from the first snapshot *after* the purchase.
- Same-click component rows are emitted as purchases.

`frame/risk_set.py`
- `owned_matrix` uses `sold > t`, so a component consumed by the upgrade is missing from the
  inventory at that decision.
- Same-second purchases each see the other as eligible.
- The re-buy path and the duplicate-drop never fire: the data has one row per (player, item).

`estimate/aipw.py`
- `learner='lgbm'` crashes when a training fold has exactly two arms
  (`objective='multiclass'`), which includes every cost with two on-support arms.
- The logistic path raises on NaN covariates.
- An arm missing from a training fold becomes unsupported for that whole fold, and
  `trim_share` does not count it.
- `seed` does nothing on the logistic path.

`assets/decoder.py`
- The docstring says shop items do not combine from components. 65 do.

## How this was produced

Four sub-agents ran in parallel (data semantics, crossover and volume, citations and tooling,
simulation); the data-source, skill-column, volume and previous-match checks were run directly.
All scripts were scratch files outside the repo and are not committed. Not verified: the price
actually paid (inferred from consumed components), whether the last day's lower volume is real,
and what happens to a component when its upgrade is sold within 15 s (restored in about half of
cases).
