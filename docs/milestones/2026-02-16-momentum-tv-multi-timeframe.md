# Milestone: 2026-02-16 - Momentum TV + Multi-Timeframe Analysis

## Summary
This milestone established a stable baseline for multi-timeframe technical analysis and TradingView-style momentum signal replication.

Key outcomes:
- Added monthly data fetch support from Stooq and monthly trend analysis notebook tooling.
- Added and validated a TradingView-style momentum strategy implementation.
- Added daily/weekly/monthly timeframe options for momentum analysis.
- Added notebook visualizations for signal and momentum inspection.
- Captured and pushed all related changes to `main`.

## What Was Built

### Data Fetching
- `scripts.data.fetch_stooq_ohlc`
  - Added interval support (`daily` and `monthly`) via Stooq `i=d|m`.
  - Monthly outputs now supported in `out/monthly/`.

### Trend + Signal Workflow
- `scripts.signals.signal_engine`
  - Added monthly regime integration and later simplified gating behavior.
  - Added EMA-cross long trigger support (Close crossing above configurable EMA, default `EMA_50`).

### TradingView-Style Momentum
- New script: `scripts.indicators.momentum_strategy_tv`
  - Replicates Pine logic:
    - `MOM0 = Close - Close[length]`
    - `MOM1 = MOM0 - MOM0[1]`
    - Long stop while `MOM0 > 0 and MOM1 > 0`
    - Short stop while `MOM0 < 0 and MOM1 < 0`
  - Default `length=24`.
  - Supports `--timeframe daily|weekly|monthly`.
  - Output defaults:
    - Daily: `out/momentum_tv/`
    - Weekly: `out/momentum_tv_weekly/`
    - Monthly: `out/momentum_tv_monthly/`

### Notebooks
- `notebooks/monthly_trend_lab.ipynb`
  - Monthly trend analysis with full-history orientation.
- `notebooks/momentum_tv_lab.ipynb`
  - Visualizes TradingView-style momentum outputs.
  - Timeframe switch in notebook (`daily|weekly|monthly`).

## Key Decisions
- Keep momentum TV implementation separate (`scripts.indicators.momentum_strategy_tv`) from existing momentum engine (`scripts.indicators.momentum_strategy`) to avoid regression risk.
- Keep timeframe analysis as output-level exploration before deeper strategy engine integration.
- Treat monthly/weekly as first-class analysis options while preserving daily compatibility.

## Current Known-Good Commands

### Fetch data
```bash
python3 -m scripts.data.fetch_stooq_ohlc
python3 -m scripts.data.fetch_stooq_ohlc --interval m
```

### TradingView-style momentum
```bash
python3 -m scripts.indicators.momentum_strategy_tv --timeframe daily
python3 -m scripts.indicators.momentum_strategy_tv --timeframe weekly
python3 -m scripts.indicators.momentum_strategy_tv --timeframe monthly
```

### Existing momentum/signal pipeline (legacy + current)
```bash
python3 -m scripts.indicators.momentum_strategy --length 24
python3 -m scripts.signals.signal_engine --min-hold-bars 5
```

## Validation Results (at milestone)
- Daily momentum TV signals validated against TradingView **daily** chart behavior and reported as closely matching.
- Weekly and monthly momentum TV runs execute successfully and produce lower signal frequency than daily:
  - Daily: APO `38`, TSLA `40` events
  - Weekly: APO `12`, TSLA `5` events
  - Monthly: APO `3`, TSLA `9` events

## Outputs / Artifacts
- `out/daily/`
- `out/monthly/`
- `out/momentum_tv/`
- `out/momentum_tv_weekly/`
- `out/momentum_tv_monthly/`
- `out/signals/`
- Notebooks in `notebooks/`

## Known Gaps / Not Implemented Yet
- No strict 1:1 TradingView broker emulator parity (data feed + execution assumptions can still differ).
- No single unified strategy engine combining all timeframes and momentum modes yet.
- No automated tests yet for cross-timeframe aggregation and event parity.

## Next Steps (Suggested)
1. Add a compact parity-check script to compare TradingView-exported OHLC vs local OHLC signal diffs.
2. Add a timeframe-aware signal orchestration layer (e.g., weekly regime + daily execution).
3. Add lightweight tests for weekly/monthly aggregation correctness and momentum event generation.

## Commit Anchors
- `446a85c` - Add monthly data flow and simplify signal gating
- `8476681` - Add TradingView-style momentum strategy lab
- `21183d6` - Add multi-timeframe options for TV momentum analysis

## How To Use This Milestone
When starting a new session, reference this file first:
- `docs/milestones/2026-02-16-momentum-tv-multi-timeframe.md`

Use it as the canonical context snapshot for this project phase.
