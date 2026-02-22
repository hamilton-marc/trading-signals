# scripts/

Implementation modules are organized by responsibility:

- `data/`
  - data acquisition scripts (provider fetch)
- `indicators/`
  - technical indicator and trend/momentum transforms
- `signals/`
  - signal generation and orchestration
- `strategies/`
  - backtests, MTF systems, and validation harnesses
- `reports/`
  - ranked shortlist and report builders

Run modules directly, for example:

```bash
python3 -m scripts.data.fetch_stooq_ohlc --interval all
python3 -m scripts.indicators.momentum_strategy_tv_match --timeframe daily
python3 -m scripts.reports.recent_momentum_report
```

Maintenance helpers:

```bash
python3 -m scripts.maintenance.tidy_out
python3 -m scripts.maintenance.verify_out_layout
python3 -m scripts.maintenance.verify_out_layout --fail-on-legacy
```
