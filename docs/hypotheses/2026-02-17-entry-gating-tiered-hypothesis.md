# Hypothesis: 2026-02-17 - Tiered Entry Gating With Disciplined Exits

## Status
- Evaluated: `2026-02-17`
- State: `Closed`
- Result summary source: `docs/milestones/2026-02-17-entry-tier-experiment-results.md`

## Verdict
`Partially supported` (and adopted as a working candidate):
- Tiered entry (`tiered_candidate_25_early_50`) improved entry timing/recall in focal windows.
- Risk profile stayed close to baseline on tested symbols, unlike full relaxation.
- Full weekly-only relaxation increased drawdown on TSLA and was treated as too aggressive.

## Evidence Snapshot
From `out/entry_tier_experiment/summary.csv` (window `2023-01-03` to `2026-02-13`, symbols `APO`, `TSLA`, `MSFT`):
- APO: baseline `+67.91%` vs tiered `+70.68%` (max DD both `12.37%`)
- TSLA: baseline `+36.70%` vs tiered `+38.13%` (max DD `21.45%` vs `21.44%`)
- MSFT: baseline `+18.54%` vs tiered `+20.83%` (max DD `16.88%` in both cases)

## Follow-up Decision
- Keep disciplined exits unchanged.
- Use `tiered_candidate_25_early_50` as the next iteration candidate.
- Keep `relaxed_weekly_full` as an upper-bound capture reference, not default execution mode.

## Context
Current v2 behavior shows disciplined exits and generally stable risk control, but entry timing can miss early portions of strong trend legs.

Recent analysis suggests:
- strict monthly+weekly+daily confirmation improves average quality,
- but can be late on some high-momentum reversals,
- and fully relaxing monthly confirmation can degrade expectancy.

## Observation
- The strategy appears stronger at exits than entries.
- Missing some early trend segments is likely a recall problem, not an exit discipline problem.
- The user workflow favors seeing more candidate opportunities, then applying discretion, while still relying on system-driven exits.

## Hypothesis
If we split entries into tiers (candidate vs confirmed) while keeping the current disciplined exit engine unchanged, then:
- we will improve opportunity capture (entry recall),
- without materially degrading risk-adjusted outcomes.

## Null Hypothesis
Tiered entry gating does not improve opportunity capture enough to justify added noise, and/or it worsens drawdown and expectancy versus the current strict baseline.

## Proposed Rule Change (Independent Variable)
Introduce entry tiers:

1. `CONFIRMED` (current strict rule):
- `MonthlyRiskOn=1`
- `WeeklyRiskOn=1`
- daily trigger active

2. `CANDIDATE` (broader alert rule):
- `WeeklyRiskOn=1`
- daily trigger active
- monthly confirmation not required

3. Optional `EARLY_CONFIRM` bridge:
- `WeeklyRiskOn=1`
- strong daily breakout/reclaim
- monthly improving (for example: close above monthly fast EMA and fast EMA rising), even if strict monthly is not yet fully on

Exit engine remains unchanged for all tiers.

## Experimental Design
Use the same symbols and windows already tracked (`APO`, `TSLA`, `MSFT`) and evaluate:
- baseline strict mode (`CONFIRMED` only),
- tiered mode with candidate alerts,
- optional tiered mode with early-confirm logic.

Keep fixed during experiments:
- exit logic and parameters,
- data source and preprocessing,
- execution assumptions.

## Primary Metrics (Dependent Variables)
1. Opportunity capture:
- number of valid opportunities surfaced,
- participation in top trend windows (time-in-trend and return capture).

2. Signal quality:
- `P(return > 0)` at +5/+10/+20/+40 bars,
- `P(return >= +5%)` at the same horizons,
- mean and median forward returns.

3. Risk impact:
- max drawdown,
- mean adverse excursion,
- churn (trade count and average hold duration).

## Acceptance Criteria
Treat the hypothesis as supported if tiered entry (vs strict baseline) shows:
1. Higher opportunity capture,
2. No material degradation in drawdown profile,
3. No material degradation in 20-bar expectancy and hit-rate quality.

## Failure Criteria
Treat the hypothesis as not supported if tiered entry causes:
1. Meaningful drawdown/churn increase,
2. Lower 20-bar expectancy with no compensating capture gain,
3. Lower practical usability due to excessive low-quality noise.

## Practical Implementation Note
Initial rollout should be decision-support first:
- `CANDIDATE` as alert-only,
- `CONFIRMED` as execution-grade signal,
- optional tier-based sizing later if evidence is positive.

## Next Step
Implement tier labels in output rows, then run the same probability/backtest framework per tier and compare against current v2 baseline.
