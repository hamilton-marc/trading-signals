# trading-signals

End-of-day (EOD) trading signal research project.

The current focus is:
- fetch OHLC data from Stooq
- compute indicators and trend/momentum states
- scan for recent candidate signals
- run simple strategy/backtest experiments

## Requirements
- Python `3.10+`

Optional dependencies:
- `pandas` (needed by some strategy/experiment scripts)
- `matplotlib` and `jupyter` (for notebooks)

## Project Structure

Implementation code is organized under `scripts/`:
- `scripts/data/`: data fetching
- `scripts/indicators/`: EMA/trend/momentum calculations
- `scripts/signals/`: signal engine
- `scripts/strategies/`: backtests + MTF systems
- `scripts/reports/`: shortlist/ranking reports

Run commands via modules: `python3 -m scripts.<group>.<script>`
Root-level script wrappers were removed in the latest reorganization.

## Quick Workflow (Recommended)

1. Refresh market data

```bash
python3 -m scripts.data.fetch_stooq_ohlc --interval all --delay-seconds 1.2
```

2. Compute strict TradingView-style momentum events

```bash
python3 -m scripts.indicators.momentum_strategy_tv_match --timeframe daily --length 24 --min-tick 0.01
```

3. Build ranked recent-buy shortlist (last 5 bars)

```bash
python3 -m scripts.reports.recent_momentum_report
```

4. Open notebook for chart review of all recent symbols

- `notebooks/recent_signal_lab.ipynb`

Generated shortlist artifacts:
- `out/reports/momentum/recent_momentum_buys_5d.csv`
- `out/reports/momentum/recent_momentum_buys_5d.md`

## Notebook Usage

Start Jupyter from the repo root:

```bash
jupyter notebook
```

Recommended notebook sequence:
1. `notebooks/ema_lab.ipynb`
2. `notebooks/tv_momentum_match_lab.ipynb`
3. `notebooks/recent_signal_lab.ipynb`
4. `notebooks/backtest_lab.ipynb`

Notebook assumptions:
- Most notebooks read pre-generated CSV artifacts from `out/`
- Run the relevant CLI steps first (at minimum: fetch data + compute indicators/signals)
- Use the same Python environment where dependencies are installed (`matplotlib`, `pandas` when needed)

## Command Discovery

For any script/module options, run:

```bash
python3 -m scripts.<group>.<script> --help
```

## Out Folder Housekeeping

To declutter root-level files in `out/` after experiments:

```bash
python3 -m scripts.maintenance.tidy_out
```

This moves top-level metadata files into:
- `out/_meta/errors/`
- `out/_meta/latest/`
- `out/_meta/summaries/`
- `out/_meta/watchlists/`

## Core Commands

### Data Fetch (Stooq)

See also: `docs/data-fetching.md` for a focused fetch reference.

Default daily fetch:

```bash
python3 -m scripts.data.fetch_stooq_ohlc
```

Useful variants:

```bash
python3 -m scripts.data.fetch_stooq_ohlc --interval w
python3 -m scripts.data.fetch_stooq_ohlc --interval m
python3 -m scripts.data.fetch_stooq_ohlc --interval all
python3 -m scripts.data.fetch_stooq_ohlc --start-date 2025-01-01
python3 -m scripts.data.fetch_stooq_ohlc --dry-run
```

Writes:
- `out/data/daily/<SYMBOL>.csv`
- `out/data/weekly/<SYMBOL>.csv`
- `out/data/monthly/<SYMBOL>.csv`
- error files in `out/_meta/errors/` (by timeframe)

### Indicator Pipeline

EMA:

```bash
python3 -m scripts.indicators.compute_ema --periods 50,200
```

Trend analyzer:

```bash
python3 -m scripts.indicators.trend_analyzer
python3 -m scripts.indicators.trend_analyzer --buffer-pct 0.5 --confirm-bars 3
```

Momentum (state-machine version):

```bash
python3 -m scripts.indicators.momentum_strategy --length 24
```

### TradingView Momentum Variants

Flexible TV-style model:

```bash
python3 -m scripts.indicators.momentum_strategy_tv --timeframe daily
python3 -m scripts.indicators.momentum_strategy_tv --timeframe weekly
python3 -m scripts.indicators.momentum_strategy_tv --timeframe monthly
```

Strict Pine-match model:

```bash
python3 -m scripts.indicators.momentum_strategy_tv_match --timeframe daily --length 24 --min-tick 0.01
```

Strict model output:
- per-symbol: `out/indicators/momentum_tv_match/<timeframe>/<SYMBOL>.csv`
- latest summary: `out/_meta/latest/momentum_tv_match_<timeframe>_latest.csv`
- errors: `out/_meta/errors/momentum_tv_match_<timeframe>_errors.csv`

### Ranked Recent-Signal Report

```bash
python3 -m scripts.reports.recent_momentum_report
```

Default behavior:
- filters symbols with `MomLE` in last `5` bars
- ranks continuation quality using trend alignment, EMA slopes, ATR normalization, extension penalty, and freshness

Writes:
- `out/reports/momentum/recent_momentum_buys_5d.csv`
- `out/reports/momentum/recent_momentum_buys_5d.md`

### Signal Engine (v1)

```bash
python3 -m scripts.signals.signal_engine --min-hold-bars 5
python3 -m scripts.signals.signal_engine --min-hold-bars 5 --monthly-regime-filter
python3 -m scripts.signals.signal_engine --min-hold-bars 5 --allow-neutral-trend-entries
```

Writes:
- `out/signals/engine/<SYMBOL>.csv`
- `out/_meta/latest/signal_engine_latest.csv`

### Long-Only Backtest

```bash
python3 -m scripts.strategies.backtest_long --symbol APO --initial-capital 100000 --allocation-pct 5 --ema-stop-column EMA_50
python3 -m scripts.strategies.backtest_long --symbol APO --min-atr-pct 2.0 --atr-period 14
```

Writes:
- `out/backtests/long_only/<SYMBOL>_trades.csv`
- `out/backtests/long_only/<SYMBOL>_equity_curve.csv`
- `out/backtests/long_only/<SYMBOL>_summary.csv`

## Advanced Strategy/Research Tools

MTF entry/exit strategy:

```bash
python3 -m scripts.strategies.mtf_entry_exit_v1
python3 -m scripts.strategies.mtf_entry_exit_v2
```

Tiered-entry experiment:

```bash
python3 -m scripts.strategies.entry_tier_experiment
```

Hardening harness:

```bash
python3 -m scripts.strategies.mtf_v1_hardening
```

Signal quality study:

```bash
python3 -m scripts.strategies.signal_quality_study_v2
```

Note: some advanced scripts require `pandas`.

## Notebooks

Main notebooks currently used:
- `notebooks/ema_lab.ipynb`
- `notebooks/recent_signal_lab.ipynb`
- `notebooks/tv_momentum_match_lab.ipynb`
- `notebooks/backtest_lab.ipynb`

## Notes

- `watchlist.txt` is the primary symbol input.
- `out/` is for generated artifacts and is ignored by Git.
- Operational runbook: `docs/operations/trading-workflow.md`
- Files under `docs/milestones/` are historical snapshots and may include legacy command examples.
