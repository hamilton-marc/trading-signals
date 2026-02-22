# Out Folder v2 Migration Plan

## Goal
Define a cleaner `out/` structure that separates stable operational outputs from research/experiment artifacts, while avoiding breakage during migration.

This document now serves as the v2 migration record and reference.

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

## Script Impact (Applied)

All primary scripts now default to v2 paths:

- Data fetch defaults:
  - `out/data/daily`, `out/data/weekly`, `out/data/monthly`
  - errors under `out/_meta/errors/`
- Indicator defaults:
  - EMA output in `out/indicators/ema`
  - trend output in `out/indicators/trend`
  - momentum output in `out/indicators/momentum`
  - TradingView momentum outputs in:
    - `out/indicators/momentum_tv/<timeframe>`
    - `out/indicators/momentum_tv_match/<timeframe>`
  - indicator latest/errors under `out/_meta/latest/` and `out/_meta/errors/`
- Signal engine defaults:
  - inputs from `out/indicators/*` and `out/data/monthly`
  - output in `out/signals/engine`
  - latest/errors in `out/_meta/`
- Report defaults:
  - recent momentum report input from `out/indicators/momentum_tv_match/daily`
  - outputs under `out/reports/momentum/`
- Strategy defaults:
  - long-only backtests in `out/backtests/long_only`
  - tv-match backtests in `out/backtests/tv_match`
  - MTF/entry/signal-quality/hardening outputs under `out/strategies/*`
- Maintenance:
  - `scripts/maintenance/tidy_out.py` remains valid for top-level metadata cleanup.

Compatibility symlinks were added for legacy high-traffic paths (`out/daily`, `out/monthly`, `out/trend`, etc.) during migration.

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

## Migration Sequence (Executed)

1. Updated script defaults to v2 path layout while retaining CLI overrides.
2. Added compatibility symlinks for legacy path continuity.
3. Migrated existing `out/` artifacts into v2 namespaces.
4. Updated docs and notebook path references to v2 defaults.
5. Ran smoke checks on core modules.

## Acceptance Criteria

- One end-to-end EOD workflow completes with v2 defaults and no manual path edits.
- `README.md` and `docs/operations/trading-workflow.md` match the new structure.
- No script fails due to hard-coded legacy `out/*` paths.
- Notebook labs open and run with v2 path config cells.
