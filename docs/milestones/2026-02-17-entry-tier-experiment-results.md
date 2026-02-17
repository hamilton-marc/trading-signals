# Milestone: 2026-02-17 - Entry Tier Experiment Results (v2 Exit Discipline Unchanged)

## Objective
Test whether loosening entry gating (while keeping the current disciplined exits) improves opportunity capture without unacceptable risk degradation.

## Experiment Setup
- Symbols: `APO`, `TSLA`, `MSFT`
- Window: `2023-01-03` to `2026-02-13`
- Exit logic: unchanged from v2 baseline (`ATR_HARD_STOP` + weekly fail)
- Policies tested:
  1. `baseline_confirmed` (strict monthly+weekly+daily)
  2. `tiered_candidate_25_early_50` (candidate 25%, early-confirm 50%, confirmed 100%)
  3. `relaxed_weekly_full` (weekly+daily trigger at full size)

## Topline Results
Source: `out/entry_tier_experiment/summary.csv`

| Symbol | Mode | Return % | Max DD % | Round Trips |
|---|---|---:|---:|---:|
| APO | baseline_confirmed | 67.9114 | 12.3706 | 8 |
| APO | tiered_candidate_25_early_50 | 70.6766 | 12.3706 | 8 |
| APO | relaxed_weekly_full | 77.8850 | 12.3706 | 8 |
| TSLA | baseline_confirmed | 36.6973 | 21.4457 | 7 |
| TSLA | tiered_candidate_25_early_50 | 38.1308 | 21.4419 | 13 |
| TSLA | relaxed_weekly_full | 55.4554 | 27.5986 | 13 |
| MSFT | baseline_confirmed | 18.5397 | 16.8755 | 15 |
| MSFT | tiered_candidate_25_early_50 | 20.8276 | 16.8810 | 15 |
| MSFT | relaxed_weekly_full | 27.8325 | 16.8837 | 15 |

## Entry-Timing Observations in Focal Windows
- TSLA (`2024-09-01` to `2025-01-31`):
  - baseline first buy: `2024-11-07`
  - tiered/relaxed first buy: `2024-09-06`
- MSFT (`2025-04-15` to `2025-07-31`):
  - baseline first buy: `2025-06-03`
  - tiered/relaxed first buy: `2025-05-06`

Interpretation:
- Looser gating enters earlier on major trend transitions.
- Full relaxation materially increased TSLA drawdown.
- Tiered sizing captured earlier entries while keeping drawdown near baseline.

## Recommendation (For Next Iteration)
Use `tiered_candidate_25_early_50` as the next working candidate:
- preserves current exit discipline,
- improves entry recall and timing,
- keeps risk profile closer to baseline than full relaxation.

`relaxed_weekly_full` remains a useful upper-bound reference for capture potential but appears too aggressive for risk consistency on TSLA.

## Decision Summary (Capital Preservation Focus)
Follow-up discussion clarified practical priorities:
- Primary objective is capital preservation and psychological sustainability, not pure return maximization.
- Drawdowns are expected, but we prefer to fail fast on new trades when timing is wrong.
- Tiered entries are useful because they can provide multiple buy opportunities during strong moves, enabling adds to a winning position.

Operational framing for next chapter:
- Keep exits disciplined and rules-based.
- Prefer more candidate opportunities over overly sparse entries.
- Use tiered sizing to probe early, cut quickly if invalidated, and add when the move confirms.

## Visual Artifacts
- Notebook (entries, exits, equity by mode):
  - `notebooks/entry_tier_experiment_lab.ipynb`
- PNG charts:
  - `out/entry_tier_experiment/plots/APO_modes.png`
  - `out/entry_tier_experiment/plots/TSLA_modes.png`
  - `out/entry_tier_experiment/plots/MSFT_modes.png`

## Output Artifacts
- Per-mode per-symbol time series:
  - `out/entry_tier_experiment/baseline_confirmed/*.csv`
  - `out/entry_tier_experiment/tiered_candidate_25_early_50/*.csv`
  - `out/entry_tier_experiment/relaxed_weekly_full/*.csv`
- Aggregate summary:
  - `out/entry_tier_experiment/summary.csv`
