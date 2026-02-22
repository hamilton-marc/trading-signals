# Out Folder v2 Migration Plan

## Goal
Define a cleaner `out/` structure that separates stable operational outputs from research/experiment artifacts, while avoiding breakage during migration.

This document is planning-only for v2. It does not assume all paths are already migrated.

## Target Structure (v2)

```text
out/
  data/
    daily/
    weekly/
    monthly/
  indicators/
    ema/
    trend/
    momentum/
    momentum_tv/
      daily/
      weekly/
      monthly/
    momentum_tv_match/
      daily/
      weekly/
      monthly/
  signals/
    engine/
    variants/
  reports/
    momentum/
  backtests/
    long_only/
    tv_match/
  strategies/
    mtf_entry_exit_v1/
    mtf_entry_exit_v2/
    entry_tier_experiment/
    signal_quality_v2/
    hardening/
  experiments/
    sweeps/
    comparisons/
    legacy/
  runs/
    YYYY-MM-DD_<label>/
      config/
      outputs/
      logs/
  _meta/
    errors/
    latest/
    summaries/
    watchlists/
```

## Mapping Rules

| Current path pattern | v2 target | Notes |
|---|---|---|
| `out/daily/*` | `out/data/daily/*` | Stooq daily OHLC |
| `out/weekly/*` | `out/data/weekly/*` | Stooq weekly OHLC |
| `out/monthly/*` | `out/data/monthly/*` | Stooq monthly OHLC |
| `out/indicators/*` | `out/indicators/ema/*` | EMA-enriched per-symbol CSVs |
| `out/trend/*` | `out/indicators/trend/*` | Trend analyzer outputs |
| `out/momentum/*` | `out/indicators/momentum/*` | Momentum outputs |
| `out/momentum_tv/*` | `out/indicators/momentum_tv/daily/*` | Flexible TV-style daily |
| `out/momentum_tv_weekly/*` | `out/indicators/momentum_tv/weekly/*` | Flexible TV-style weekly |
| `out/momentum_tv_monthly/*` | `out/indicators/momentum_tv/monthly/*` | Flexible TV-style monthly |
| `out/momentum_tv_match_daily/*` | `out/indicators/momentum_tv_match/daily/*` | Strict TV-match daily |
| `out/momentum_tv_match_weekly/*` | `out/indicators/momentum_tv_match/weekly/*` | Strict TV-match weekly |
| `out/signals/*` | `out/signals/engine/*` | Main signal engine per-symbol outputs |
| `out/signals_*/*` | `out/signals/variants/<name>/*` | Legacy signal variants |
| `out/reports/*` | `out/reports/momentum/*` | Report namespace by topic |
| `out/backtests/*` | `out/backtests/long_only/*` | Long-only backtest outputs |
| `out/backtests_tv_match/*` | `out/backtests/tv_match/*` | TV-match backtest outputs |
| `out/mtf_entry_exit_v1/*` | `out/strategies/mtf_entry_exit_v1/*` | Strategy outputs |
| `out/mtf_entry_exit_v2/*` | `out/strategies/mtf_entry_exit_v2/*` | Strategy outputs |
| `out/entry_tier_experiment*/*` | `out/strategies/entry_tier_experiment/<name>/*` | Preserve experiment flavor |
| `out/signal_quality_v2*/*` | `out/strategies/signal_quality_v2/<name>/*` | Preserve setup flavor |
| `out/hardening/*` | `out/strategies/hardening/*` | Hardening artifacts |
| `out/_sweep_v2/*` | `out/experiments/sweeps/_sweep_v2/*` | Legacy sweep tree |
| `out/_targeted_v2/*` | `out/experiments/comparisons/_targeted_v2/*` | Legacy targeted tree |
| `out/mom_sweep_*` | `out/experiments/sweeps/momentum/<name>/*` | High-volume sweep outputs |
| `out/momentum_tv_base*` | `out/experiments/comparisons/momentum_tv/<name>/*` | Baseline/guard/combo sets |
| `out/momentum_tv_cmp_*` | `out/experiments/comparisons/momentum_tv/<name>/*` | Comparison sets |
| `out/sig_cmp_*` | `out/experiments/comparisons/signals/<name>/*` | Signal comparison sets |
| `out/sig_tune_*` | `out/experiments/sweeps/signals/<name>/*` | Signal tuning sets |
| top-level `*_errors.csv` | `out/_meta/errors/*` | Already partially applied by `tidy_out` |
| top-level `*_latest.csv` | `out/_meta/latest/*` | Already partially applied by `tidy_out` |
| top-level `*_summary.csv` | `out/_meta/summaries/*` | Already partially applied by `tidy_out` |
| top-level watchlist txt files | `out/_meta/watchlists/*` | Already partially applied by `tidy_out` |

## Script Impact (Default path updates required)

### Data
- `scripts/data/fetch_stooq_ohlc.py`
  - `out/daily`, `out/weekly`, `out/monthly`
  - `out/stooq_errors.csv`, `out/stooq_weekly_errors.csv`, `out/stooq_monthly_errors.csv`

### Indicators
- `scripts/indicators/compute_ema.py`
  - `--input-dir` default `out/daily`
  - `--out-dir` default `out/indicators`
  - errors default `out/indicator_errors.csv`
- `scripts/indicators/trend_analyzer.py`
  - input default `out/indicators`
  - out default `out/trend`
  - latest/errors at top-level `out/`
- `scripts/indicators/momentum_strategy.py`
  - input default `out/daily`
  - out default `out/momentum`
  - latest/errors at top-level `out/`
- `scripts/indicators/momentum_strategy_tv.py`
  - data inputs default to `out/daily` and `out/monthly`
  - out dirs `out/momentum_tv*`
  - latest/errors at top-level `out/`
- `scripts/indicators/momentum_strategy_tv_match.py`
  - data inputs `out/daily`, `out/weekly`, `out/monthly`
  - out dirs `out/momentum_tv_match_<timeframe>`
  - latest/errors at top-level `out/`

### Signals
- `scripts/signals/signal_engine.py`
  - trend `out/trend`, momentum `out/momentum`, monthly `out/monthly`
  - out dir `out/signals`
  - latest/errors at top-level `out/`

### Reports
- `scripts/reports/recent_momentum_report.py`
  - input dir `out/momentum_tv_match_daily`
  - output files under `out/reports/`

### Strategies
- `scripts/strategies/backtest_long.py`
  - signals `out/signals`, trend `out/trend`, out `out/backtests`
- `scripts/strategies/mtf_entry_exit_v1.py`
  - daily/weekly/monthly input defaults in `out/`
  - outputs and latest/summary/errors under top-level `out/`
- `scripts/strategies/mtf_entry_exit_v2.py`
  - daily/weekly/monthly input defaults in `out/`
  - outputs and latest/summary/errors under top-level `out/`
- `scripts/strategies/entry_tier_experiment.py`
  - daily/weekly/monthly inputs + out dir under top-level `out/`
- `scripts/strategies/signal_quality_study_v2.py`
  - input `out/mtf_entry_exit_v2`, output `out/signal_quality_v2`
- `scripts/strategies/mtf_v1_hardening.py`
  - input file default `out/daily/APO.csv`
  - out dir default `out/hardening`

### Maintenance
- `scripts/maintenance/tidy_out.py`
  - should remain valid for top-level metadata files in any future layout

## Notebook Impact

Hard-coded or repeated references to old top-level `out/*` paths exist in notebooks, especially:
- `notebooks/recent_signal_lab.ipynb`
- `notebooks/backtest_lab.ipynb`
- `notebooks/signal_event_lab.ipynb`
- `notebooks/mtf_entry_exit_v1_lab.ipynb`
- `notebooks/mtf_entry_exit_v2_*.ipynb`
- `notebooks/tv_momentum_match_lab.ipynb`

Migration recommendation:
- introduce a single `OUT_ROOT = Path("out")` config cell in each notebook
- derive all paths from that root and new v2 subpaths
- avoid embedding absolute paths in notebook outputs

## No-Break Migration Sequence

1. Add a shared path module (`scripts/paths.py`) with v2 defaults and helper builders.
2. Update scripts to accept v2 defaults while still allowing CLI override to old paths.
3. Add compatibility symlinks or fallback logic for critical reads during transition.
4. Migrate data directories first: `daily/weekly/monthly`.
5. Migrate indicator outputs next (`ema`, `trend`, `momentum`, `momentum_tv*`).
6. Migrate signals/reports/backtests.
7. Migrate strategy and experiment trees.
8. Update notebooks to v2 path config cells.
9. Remove old path fallbacks only after at least one full EOD run succeeds.

## Acceptance Criteria

- One end-to-end EOD workflow completes with v2 defaults and no manual path edits.
- `README.md` and `docs/operations/trading-workflow.md` match the new structure.
- No script fails due to hard-coded legacy `out/*` paths.
- Notebook labs open and run with v2 path config cells.
