# Trading Workflow (Human-in-the-Loop)

## Purpose
Define an operational workflow to use the current system day-to-day for:
1. Finding new long opportunities.
2. Managing existing trades (especially exits).

This is intentionally **not** a fully automated trading system.

Related policy:
- `docs/operations/trading-policy.md` (decision rules and signal hierarchy)

## Operating Model
- Research and signal generation are automated.
- Final trade decisions are human-reviewed.
- Workflow runs after market close (EOD data).

## Inputs
- Watchlist: `watchlist.txt`
- (Planned) current holdings list: `positions/current_positions.csv`

## Daily Workflow (EOD)

### Optional: Single-command runner
```bash
python3 -m scripts.operations.daily_run --label daily --fetch-interval all
```

### Step 1: Refresh data
```bash
python3 -m scripts.data.fetch_stooq_ohlc --interval all --delay-seconds 2.0 --delay-jitter-seconds 3.0
```
This fetches `daily` from Stooq and derives `weekly`/`monthly` locally from daily data.

### Step 2: Refresh indicators and signals
```bash
python3 -m scripts.indicators.compute_ema --periods 50,200
python3 -m scripts.indicators.trend_analyzer --buffer-pct 0.5 --confirm-bars 3
python3 -m scripts.indicators.momentum_strategy_tv_match --timeframe daily --length 24 --min-tick 0.01
python3 -m scripts.indicators.momentum_strategy_tv_match --timeframe weekly --length 24 --min-tick 0.01
python3 -m scripts.signals.signal_engine --min-hold-bars 5
python3 -m scripts.reports.recent_momentum_report
python3 -m scripts.reports.recent_momentum_report \
  --input-dir out/indicators/momentum_tv_match/weekly \
  --window-bars 2 \
  --out-csv out/reports/momentum/recent_momentum_buys_weekly_10d.csv \
  --out-md out/reports/momentum/recent_momentum_buys_weekly_10d.md
python3 -m scripts.reports.weekly_trend_watchlist_report \
  --window-bars 2 \
  --out-csv out/reports/momentum/weekly_trend_no_recent_momle_10d.csv \
  --out-md out/reports/momentum/weekly_trend_no_recent_momle_10d.md
```

### Optional: Tidy output metadata
```bash
python3 -m scripts.maintenance.tidy_out
python3 -m scripts.maintenance.verify_out_layout
```

### Step 3: Review new long opportunities
Primary artifacts:
- `out/reports/momentum/recent_momentum_buys_5d.md`
- `out/reports/momentum/recent_momentum_buys_5d.csv`
- `out/reports/momentum/recent_momentum_buys_weekly_10d.md`
- `out/reports/momentum/recent_momentum_buys_weekly_10d.csv`
- `out/reports/momentum/weekly_trend_no_recent_momle_10d.md`
- `out/reports/momentum/weekly_trend_no_recent_momle_10d.csv`
- `notebooks/recent_signal_lab.ipynb`
- `notebooks/recent_signal_lab_weekly.ipynb`
- `notebooks/weekly_trend_watchlist_lab.ipynb`

Decision focus:
- Rank/score from report.
- Trend quality context from chart (EMA/trend alignment).
- Continuation candidates where trend is strong but no fresh weekly `MomLE` appeared in last ~10 trading days.
- Avoid low-quality setups in obvious downtrends unless explicitly trading reversals.

## Signal Hierarchy (Current Default)
Use timeframes with clear priority to reduce daily noise.

1. Regime filter (direction): weekly and monthly context first.
2. Opportunity screen (candidates): weekly `MomLE` in last 2 weekly bars (~10 trading days).
3. Execution timing (optional refinement): daily chart only after symbol passes weekly screen.
4. Risk/exit: capital-preservation exits always override new-entry logic.

Practical interpretation:
- If weekly/monthly context is weak, do not treat daily `MomLE` as an execution-grade long signal.
- Use daily mostly for timing and risk management after higher timeframe alignment exists.
- Ignore isolated daily opposite momentum flips when higher timeframe trend remains intact.

### Step 4: Review existing positions (exit management)
Per-symbol artifacts:
- `out/signals/engine/<SYMBOL>.csv`
- `out/indicators/trend/<SYMBOL>.csv`

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

For planned output-layout migration, see: `docs/operations/out-structure-v2-plan.md`.
