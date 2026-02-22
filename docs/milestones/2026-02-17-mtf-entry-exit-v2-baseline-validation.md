# Milestone: 2026-02-17 - MTF Entry/Exit v2 Baseline Validation (APO + TSLA)

## Summary
This milestone establishes `scripts.strategies.mtf_entry_exit_v2` as a new validated baseline.

APO still behaves well overall, and v2 materially improves robustness on TSLA versus v1.

Core v2 pattern:
- Regime-gated entries: monthly + weekly must both be risk-on.
- Daily triggers: breakout and pullback-reclaim.
- Asymmetric risk: one valid exit reason is enough.

## v2 Baseline Defaults
- `monthly_regime_mode=strict`
- `weekly_ema_period=13`
- `weekly_confirm_bars=2`
- `weekly_fail_bars=2`
- `daily_ema_period=50`
- `breakout_lookback=20`
- `momentum_length=24`
- `atr_period=14`
- `atr_trail_mult=2.5`
- `hard_stop_atr_mult=1.5`
- `enable_equity_kill=False` (off by default)

## Current Baseline Results
From `out/mtf_entry_exit_v2_summary.csv`:

| Symbol | Return % | Max DD % | Round Trips | Ending Equity |
|---|---:|---:|---:|---:|
| APO | 67.9114 | 12.3706 | 8 | 167911.43 |
| TSLA | 36.6973 | 21.4457 | 7 | 136697.27 |

## Comparison vs v1 Baseline
From `out/mtf_entry_exit_v1_summary.csv` and `out/mtf_entry_exit_v2_summary.csv`:

| Symbol | v1 Return % | v2 Return % | Delta Return % | v1 Max DD % | v2 Max DD % | Delta Max DD % |
|---|---:|---:|---:|---:|---:|---:|
| APO | 82.3636 | 67.9114 | -14.4522 | 13.3230 | 12.3706 | -0.9524 |
| TSLA | -26.5002 | 36.6973 | +63.1975 | 36.1166 | 21.4457 | -14.6709 |

Interpretation:
- APO remains strong with lower drawdown and fewer trades.
- TSLA is the major improvement: from negative return in v1 to positive return with substantially lower drawdown in v2.

## Event Behavior Snapshot
From `out/mtf_entry_exit_v2/APO.csv` and `out/mtf_entry_exit_v2/TSLA.csv`:
- APO exits: `ATR_HARD_STOP` only (8 exits).
- TSLA exits: `ATR_HARD_STOP` only (7 exits).
- APO first buy: `2023-05-31`.
- TSLA first buy: `2023-09-12`.

## Why APO Still Looks Good
- Regime gating reduces low-quality entries.
- Daily breakout/reclaim still allows participation in directional legs.
- ATR-based hard/trailing stop keeps exits objective and fast when trend quality deteriorates.
- Net result on APO: trend participation with controlled giveback.

## Known Limitation
- Strict monthly confirmation can delay entries on sharp reversals (example pattern observed on TSLA in 2024), which is a deliberate tradeoff for regime quality.

## Reproduce
Run strategy:
```bash
python3 -m scripts.strategies.mtf_entry_exit_v2
```

Visualize:
- `notebooks/mtf_entry_exit_v2_apo_lab.ipynb`
- `notebooks/mtf_entry_exit_v2_tsla_lab.ipynb`

## Milestone Outcome
v2 is now a documented baseline for further experiments.

Success criteria met:
- APO remains strong and stable.
- TSLA no longer breaks the strategy profile.
- Baseline metrics and visualization artifacts are checked in for objective regression testing.
