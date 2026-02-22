# Milestone: 2026-02-17 - MTF Entry/Exit v1 Retrospective (APO)

## Summary
This milestone documented why the new entry/exit pattern improved behavior on APO.

Core pattern that worked:
- Multi-timeframe entry confirmation (monthly + weekly + daily context).
- Asymmetric risk management:
  - hard to get in (multiple reasons),
  - easy to get out (any one valid risk trigger).
- Exit stack shifted from an over-sensitive equity kill-switch to price-structure exits (`EMA50_FAIL` and `ATR_TRAIL`), which improved trend capture while still limiting large givebacks.

This directly aligned with the intended behavior:
- participate in strong trends,
- avoid giving back massive gains during later chop/range.

## Problem We Observed
Early v1 rules were too restrictive and too reactive:

1. Entry timing lock (missed trend):
- Buy required same-bar alignment of:
  - `MonthlyUp=1`
  - `WeeklyUp=1`
  - `DailyCrossAboveEMA50=1`
- APO example:
  - `2024-08-27`: daily cross happened, but weekly was not up.
  - `2024-08-30`: monthly + weekly were both up, but cross flag was already off.
- Result: no buy near a major trend leg.

2. Exit aggressiveness (death by micro-exits):
- Equity kill-switch used `equity_below_ema OR drawdown_threshold`.
- This triggered very quickly after entries, causing repeated 1-bar holds.

## What Changed

### Entry changes
1. Momentum-positive requirement became optional.
- Default entry no longer requires momentum > 0.
- Momentum can still be enabled as an extra gate via `--require-momentum-positive-entry`.

2. Daily trigger changed from same-bar cross only to cross-window logic.
- New trigger:
  - cross today
  - OR close > EMA50 and a cross happened within `N` bars (`--entry-cross-lookback-bars`, default `15`).
- This preserved discipline while removing timing brittleness.

### Exit changes
1. Equity kill-switch mode changed to default `both`.
- New option: `--equity-kill-mode any|both`.
- Default: `both` (requires `equity_below_ema AND drawdown_threshold`).
- This prevents immediate exits from minor equity wobble.

2. Trend-failure confirmation increased.
- `--trend-fail-bars` default moved from `1` to `2`.
- Reduced churn from one-bar noise around EMA50.

## Why This Worked (Mechanics)
The combination improved behavior for APO because:

1. Entry became context-aware rather than instant-only.
- The recent-cross window let the system enter once higher-timeframe trend confirmation arrived, even if the exact cross bar had passed.

2. Exit hierarchy became less brittle.
- Equity kill-switch no longer dominated exits.
- Exit responsibility shifted to:
  - `EMA50_FAIL` for trend structure breaks,
  - `ATR_TRAIL` for volatility-adjusted protection.

3. Fewer premature exits increased time in valid trends.
- Average hold duration increased materially versus the aggressive-kill setup.
- Strong legs were captured more fully.

## APO Before vs After (Observed During Iteration)

### Earlier restrictive/aggressive phase
- Example state:
  - very low entry count (at one point only 1 executed buy),
  - exits dominated by `EQUITY_KILL`,
  - missed major 2024-09 to 2025-01 run-up window.

### Current default configuration (this milestone)
- Summary (APO):
  - `TradeExecutions`: 18
  - `RoundTrips`: 9
  - `WinRatePct`: 33.33
  - `EndingEquity`: 166668.89
  - `TotalReturnPct`: 66.6689
  - `MaxDrawdownPct`: 16.4177
- Exit mix:
  - `EMA50_FAIL`: 5
  - `ATR_TRAIL`: 4
  - `EQUITY_KILL`: 0

These values come from `out/mtf_entry_exit_v1_summary.csv` and `out/mtf_entry_exit_v1/APO.csv` after the latest defaults.

## Current Default Model (Important)
Entry:
- MonthlyUp
- WeeklyUp
- Daily trigger:
  - same-bar EMA50 cross up
  - or recent-cross window above EMA50 (`N=15` by default)
- Optional momentum positive gate (off by default)

Exit (any one):
- ATR trailing stop
- EMA50 trend-failure confirmation (`2` bars)
- equity kill-switch (default `both` mode)

## Reproduce
Run:
```bash
python3 -m scripts.strategies.mtf_entry_exit_v1
```

Inspect:
- `out/mtf_entry_exit_v1/APO.csv`
- `out/mtf_entry_exit_v1_summary.csv`
- `notebooks/mtf_entry_exit_v1_lab.ipynb`

## Key Takeaway
For APO, the successful pattern was not "more indicators."  
It was:
- **timing-robust entry confirmation** + **less brittle exits**.

That produced the desired shape:
- capture strongest trend legs,
- avoid catastrophic giveback in later range/chop.

## Commit Anchors
- `8c66491` - initial MTF v1 strategy engine
- `4638433` - momentum gate optional
- `5242e8c` - recent-cross entry window
- `34d743f` - relaxed exits (`equity-kill-mode=both`, trend-fail default 2)
