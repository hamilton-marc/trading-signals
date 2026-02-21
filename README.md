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

Root-level `*.py` files are compatibility entrypoints, so commands like `python3 fetch_stooq_ohlc.py` still work.

## Quick Workflow (Recommended)

1. Refresh market data

```bash
python3 fetch_stooq_ohlc.py --interval all --delay-seconds 0.4
```

2. Compute strict TradingView-style momentum events

```bash
python3 momentum_strategy_tv_match.py --timeframe daily --length 24 --min-tick 0.01
```

3. Build ranked recent-buy shortlist (last 5 bars)

```bash
python3 recent_momentum_report.py
```

4. Open notebook for chart review of all recent symbols

- `notebooks/recent_signal_lab.ipynb`

Generated shortlist artifacts:
- `out/reports/recent_momentum_buys_5d.csv`
- `out/reports/recent_momentum_buys_5d.md`

## Core Commands

### Data Fetch (Stooq)

Default daily fetch:

```bash
python3 fetch_stooq_ohlc.py
```

Useful variants:

```bash
python3 fetch_stooq_ohlc.py --interval w
python3 fetch_stooq_ohlc.py --interval m
python3 fetch_stooq_ohlc.py --interval all
python3 fetch_stooq_ohlc.py --start-date 2025-01-01
python3 fetch_stooq_ohlc.py --dry-run
```

Writes:
- `out/daily/<SYMBOL>.csv`
- `out/weekly/<SYMBOL>.csv`
- `out/monthly/<SYMBOL>.csv`
- error files in `out/` (by timeframe)

### Indicator Pipeline

EMA:

```bash
python3 compute_ema.py --periods 50,200
```

Trend analyzer:

```bash
python3 trend_analyzer.py
python3 trend_analyzer.py --buffer-pct 0.5 --confirm-bars 3
```

Momentum (state-machine version):

```bash
python3 momentum_strategy.py --length 24
```

### TradingView Momentum Variants

Flexible TV-style model:

```bash
python3 momentum_strategy_tv.py --timeframe daily
python3 momentum_strategy_tv.py --timeframe weekly
python3 momentum_strategy_tv.py --timeframe monthly
```

Strict Pine-match model:

```bash
python3 momentum_strategy_tv_match.py --timeframe daily --length 24 --min-tick 0.01
```

Strict model output:
- per-symbol: `out/momentum_tv_match_<timeframe>/<SYMBOL>.csv`
- latest summary: `out/momentum_tv_match_<timeframe>_latest.csv`
- errors: `out/momentum_tv_match_<timeframe>_errors.csv`

### Ranked Recent-Signal Report

```bash
python3 recent_momentum_report.py
```

Default behavior:
- filters symbols with `MomLE` in last `5` bars
- ranks continuation quality using trend alignment, EMA slopes, ATR normalization, extension penalty, and freshness

Writes:
- `out/reports/recent_momentum_buys_5d.csv`
- `out/reports/recent_momentum_buys_5d.md`

### Signal Engine (v1)

```bash
python3 signal_engine.py --min-hold-bars 5
python3 signal_engine.py --min-hold-bars 5 --monthly-regime-filter
python3 signal_engine.py --min-hold-bars 5 --allow-neutral-trend-entries
```

Writes:
- `out/signals/<SYMBOL>.csv`
- `out/signal_latest.csv`

### Long-Only Backtest

```bash
python3 backtest_long.py --symbol APO --initial-capital 100000 --allocation-pct 5 --ema-stop-column EMA_50
python3 backtest_long.py --symbol APO --min-atr-pct 2.0 --atr-period 14
```

Writes:
- `out/backtests/<SYMBOL>_trades.csv`
- `out/backtests/<SYMBOL>_equity_curve.csv`
- `out/backtests/<SYMBOL>_summary.csv`

## Advanced Strategy/Research Tools

MTF entry/exit strategy:

```bash
python3 mtf_entry_exit_v1.py
python3 mtf_entry_exit_v2.py
```

Tiered-entry experiment:

```bash
python3 entry_tier_experiment.py
```

Hardening harness:

```bash
python3 mtf_v1_hardening.py
```

Signal quality study:

```bash
python3 signal_quality_study_v2.py
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
- `data/` currently stores archived/local reference datasets.
