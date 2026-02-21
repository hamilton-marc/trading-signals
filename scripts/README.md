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

Root-level `*.py` files in the repository are compatibility wrappers that call into these modules.
