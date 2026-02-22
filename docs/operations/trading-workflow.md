# Trading Workflow (Human-in-the-Loop)

## Purpose
Define an operational workflow to use the current system day-to-day for:
1. Finding new long opportunities.
2. Managing existing trades (especially exits).

This is intentionally **not** a fully automated trading system.

## Operating Model
- Research and signal generation are automated.
- Final trade decisions are human-reviewed.
- Workflow runs after market close (EOD data).

## Inputs
- Watchlist: `watchlist.txt`
- (Planned) current holdings list: `data/positions/current_positions.csv`

## Daily Workflow (EOD)

### Step 1: Refresh data
```bash
python3 -m scripts.data.fetch_stooq_ohlc --interval all --delay-seconds 0.4
```

### Step 2: Refresh indicators and signals
```bash
python3 -m scripts.indicators.compute_ema --periods 50,200
python3 -m scripts.indicators.trend_analyzer --buffer-pct 0.5 --confirm-bars 3
python3 -m scripts.indicators.momentum_strategy_tv_match --timeframe daily --length 24 --min-tick 0.01
python3 -m scripts.signals.signal_engine --min-hold-bars 5
python3 -m scripts.reports.recent_momentum_report
```

### Step 3: Review new long opportunities
Primary artifacts:
- `out/reports/recent_momentum_buys_5d.md`
- `out/reports/recent_momentum_buys_5d.csv`
- `notebooks/recent_signal_lab.ipynb`

Decision focus:
- Rank/score from report.
- Trend quality context from chart (EMA/trend alignment).
- Avoid low-quality setups in obvious downtrends unless explicitly trading reversals.

### Step 4: Review existing positions (exit management)
Per-symbol artifacts:
- `out/signals/<SYMBOL>.csv`
- `out/trend/<SYMBOL>.csv`

Current practical exit triggers (any one):
1. `SignalEvent = LONG_TO_SHORT`
2. Trend changes to `DOWNTREND`
3. (Optional tighter risk rule) close below `EMA_50`

## Weekly Workflow (Review + Calibration)
1. Review what signals were taken vs ignored.
2. Review avoidable losses and late exits.
3. Adjust one parameter at a time (if needed), then re-check outputs/backtests.
4. Log hypothesis/result links in `docs/hypotheses/` and `docs/milestones/`.

## Decision Checklist (Daily)
- [ ] Data updated successfully for watchlist.
- [ ] Recent-buy report generated.
- [ ] New long candidates reviewed (rank + chart context).
- [ ] Existing positions checked for exit triggers.
- [ ] Decisions logged (what was taken, skipped, and why).

## What Is Not Implemented Yet
- Automatic position-aware daily brief from holdings file.
- Broker integration and order execution.
- Portfolio-level risk engine.

## Next Operational Upgrade (when ready)
Create a single daily brief generator that merges:
1. New long candidates (watchlist scan).
2. Hold/exit alerts (current positions).
