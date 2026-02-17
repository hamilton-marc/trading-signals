# trading-signals
A tool to provide trading signals based on a set of criteria.

## Requirements
- Python 3.10+

## Fetch OHLC From Stooq (Daily or Monthly)
Run:

```bash
python3 fetch_stooq_ohlc.py
```

Fetch monthly bars (longer history when available):

```bash
python3 fetch_stooq_ohlc.py --interval m
```

Optional start date filter (inclusive):

```bash
python3 fetch_stooq_ohlc.py --start-date 2025-01-01
```

Verify resolved Stooq URLs without fetching:

```bash
python3 fetch_stooq_ohlc.py --dry-run
```

Default behavior:
- Reads symbols from `watchlist.txt`
- Fetches historical OHLCV from Stooq
- Daily (`--interval d`, default):
  - Writes per-symbol CSV files to `out/daily/<SYMBOL>.csv`
  - Writes failures to `out/stooq_errors.csv`
- Monthly (`--interval m`):
  - Writes per-symbol CSV files to `out/monthly/<SYMBOL>.csv`
  - Writes failures to `out/stooq_monthly_errors.csv`

## Compute EMA
Run EMA-200 from downloaded daily data:

```bash
python3 compute_ema.py --period 200
```

Run multiple EMA lines together (example: EMA-50 and EMA-200):

```bash
python3 compute_ema.py --periods 50,200
```

Default behavior:
- Reads symbols from `watchlist.txt`
- Reads input files from `out/daily/<SYMBOL>.csv`
- Writes indicator files to `out/indicators/<SYMBOL>.csv`
- Adds EMA columns like `EMA_50`, `EMA_200` (blank until enough rows exist)

## Analyze Trend
Translate the TradingView trend logic using EMA (default: EMA-50 and EMA-200):

```bash
python3 trend_analyzer.py
```

Reduce sideways-market noise with buffer + confirmation:

```bash
python3 trend_analyzer.py --buffer-pct 0.5 --confirm-bars 3
```

Default behavior:
- Reads symbols from `watchlist.txt`
- Reads input files from `out/indicators/<SYMBOL>.csv`
- Classifies each row as `UPTREND`, `DOWNTREND`, or `NEUTRAL`
- Writes per-symbol output to `out/trend/<SYMBOL>.csv`
- Writes latest per-symbol trend to `out/trend_latest.csv`

## Momentum Strategy
Compute momentum long/short signals (default length: 24):

```bash
python3 momentum_strategy.py --length 24
```

Reduce momentum noise with confirmation and minimum-strength filters:

```bash
python3 momentum_strategy.py --length 24 --confirm-bars 3 --min-mom0-pct 1.0 --min-mom1-pct 0.2
```

Default behavior:
- Reads symbols from `watchlist.txt`
- Reads input files from `out/daily/<SYMBOL>.csv`
- Computes `MOM0` and `MOM1`
- Computes per-bar states:
  - `MomentumRawState` (unfiltered)
  - `MomentumCandidate` (strength-filtered)
  - `MomentumState` (confirmed, persistent)
- Emits transition events in `MomentumEvent` (`LONG_ENTRY`, `SHORT_ENTRY`, `LONG_TO_SHORT`, `SHORT_TO_LONG`)
- Writes per-symbol output to `out/momentum/<SYMBOL>.csv`
- Writes latest per-symbol momentum state to `out/momentum_latest.csv`

## Momentum Strategy (TradingView-Style Stops)
Replicate the provided TradingView momentum stop-entry behavior (`length=24` by default):

```bash
python3 momentum_strategy_tv.py
```

Run on weekly bars:

```bash
python3 momentum_strategy_tv.py --timeframe weekly
```

Run on monthly bars:

```bash
python3 momentum_strategy_tv.py --timeframe monthly
```

Use a different momentum length and tick size:

```bash
python3 momentum_strategy_tv.py --timeframe weekly --length 24 --min-tick 0.01
```

Reduce churn in sideways markets with trend-efficiency + EMA-spread guards:

```bash
python3 momentum_strategy_tv.py --timeframe daily --sideways-filter --er-lookback 20 --min-er 0.35 --ema-fast 10 --ema-slow 30 --min-ema-spread-pct 1.0
```

Reduce rapid reversal churn with a minimum hold period:

```bash
python3 momentum_strategy_tv.py --timeframe daily --min-hold-bars 5
```

Default behavior:
- Reads symbols from `watchlist.txt`
- Supports timeframe selection with `--timeframe daily|weekly|monthly` (default: `daily`)
- Input defaults by timeframe:
  - daily: reads `out/daily/<SYMBOL>.csv`
  - weekly: reads `out/daily/<SYMBOL>.csv` and aggregates into weekly bars
  - monthly: reads `out/monthly/<SYMBOL>.csv` (or aggregates monthly if daily data is passed)
- Computes:
  - `MOM0 = Close - Close[length]`
  - `MOM1 = MOM0 - MOM0[1]`
- Optional sideways guard (`--sideways-filter`):
  - requires `TrendEfficiency >= min_er`
  - requires `EMA spread % >= min_ema_spread_pct`
  - guard columns in output:
    - `TrendEfficiency`
    - `EmaSpreadPct`
    - `SidewaysFilterPass`
- Optional reversal hold guard:
  - `--min-hold-bars N` delays reversal entries until current state is held for at least `N` bars
  - use `0` (default) to disable hold guard
- While `MOM0 > 0` and `MOM1 > 0`, places long stop at `High + min_tick`; otherwise cancels
- While `MOM0 < 0` and `MOM1 < 0`, places short stop at `Low - min_tick`; otherwise cancels
- Output defaults by timeframe:
  - daily: `out/momentum_tv/<SYMBOL>.csv`, latest `out/momentum_tv_latest.csv`
  - weekly: `out/momentum_tv_weekly/<SYMBOL>.csv`, latest `out/momentum_tv_weekly_latest.csv`
  - monthly: `out/momentum_tv_monthly/<SYMBOL>.csv`, latest `out/momentum_tv_monthly_latest.csv`

## Signal Engine (v1)
Build cleaner final entries by combining trend regime + momentum transitions + breakout confirmation:

```bash
python3 signal_engine.py --min-hold-bars 5
```

Add a higher-timeframe monthly gate for buy-side entries:

```bash
python3 signal_engine.py --min-hold-bars 5 --monthly-regime-filter
```

Keep trend filter on, but allow entries during `NEUTRAL` trend:

```bash
python3 signal_engine.py --min-hold-bars 5 --allow-neutral-trend-entries
```

Default behavior:
- Reads trend files from `out/trend/<SYMBOL>.csv`
- Reads momentum files from `out/momentum/<SYMBOL>.csv`
- Long setup triggers:
  - momentum transition into long (`LONG_ENTRY` / `SHORT_TO_LONG`)
  - OR `Close` crossing above `EMA_50` (configurable via `--ema-cross-long-column`)
  - disable EMA-cross trigger with `--disable-ema-cross-long-trigger`
- Optional monthly buy-side gate:
  - enable with `--monthly-regime-filter`
  - reads monthly OHLC from `out/monthly/<SYMBOL>.csv`
  - requires running `python3 fetch_stooq_ohlc.py --interval m` first
  - when enabled, `LONG_ENTRY` and `SHORT_TO_LONG` require monthly `UPTREND`
  - short-side setups still follow daily logic
  - monthly trend defaults: `EMA_10/EMA_20`, `buffer=0.5%`, `confirm=2`
  - tune with:
    - `--monthly-fast-period`
    - `--monthly-slow-period`
    - `--monthly-buffer-pct`
    - `--monthly-confirm-bars`
- Optional breakout confirmation:
  - set `--breakout-lookback N` (e.g. `20`) to require breakouts
  - use `0` (default) to disable breakout requirement
- Long setup requires:
  - momentum transition into long
  - trend is `UPTREND` (unless `--disable-trend-filter`)
  - or trend is `NEUTRAL` if `--allow-neutral-trend-entries`
  - optional break above prior `N`-bar high
- Short setup requires:
  - momentum transition into short
  - trend is `DOWNTREND` (unless `--disable-trend-filter`)
  - or trend is `NEUTRAL` if `--allow-neutral-trend-entries`
  - optional break below prior `N`-bar low
- Enforces minimum hold bars before reversal (`--min-hold-bars`)
- Writes per-symbol signal output to `out/signals/<SYMBOL>.csv`
- Writes latest per-symbol signal state to `out/signal_latest.csv`

## Long-Only Backtest
Run a cash-only backtest that buys from trend/signal triggers and exits on bearish/stop conditions:

```bash
python3 backtest_long.py --symbol APO --initial-capital 100000 --allocation-pct 5 --ema-stop-column EMA_50
```

Add an ATR entry filter (example: require ATR(14) >= 2.0% of close):

```bash
python3 backtest_long.py --symbol APO --min-atr-pct 2.0 --atr-period 14
```

Default behavior:
- Reads signals from `out/signals/<SYMBOL>.csv`
- Reads trend/EMA data from `out/trend/<SYMBOL>.csv`
- Executes orders at next-day open (signals are generated after market close)
- Long-only:
  - buy trigger: transition into `UPTREND` or signal event `LONG_ENTRY` / `SHORT_TO_LONG`
  - exit trigger: transition into `DOWNTREND`, signal event `LONG_TO_SHORT`, or close below EMA stop
- Optional volatility gate for entries:
  - set `--min-atr-pct X` to require `ATR(atr_period) / Close * 100 >= X`
  - use `0` (default) to disable ATR gating
- If buy and exit triggers happen on the same day, exit has priority
- Uses one trade per day and allocates `allocation_pct` of current equity per buy
- No margin; buys are limited by available cash
- Writes:
  - `out/backtests/<SYMBOL>_trades.csv`
  - `out/backtests/<SYMBOL>_equity_curve.csv`
  - `out/backtests/<SYMBOL>_summary.csv`

## Multi-Timeframe Entry/Exit (v1)
Run a long-only strategy with strict multi-timeframe entry confluence and asymmetric exits:

```bash
python3 mtf_entry_exit_v1.py
```

Tune core risk controls (example):

```bash
python3 mtf_entry_exit_v1.py --atr-mult 2.0 --trend-fail-bars 2 --kill-max-drawdown-pct 15 --kill-cooldown-bars 8 --entry-cross-lookback-bars 15
```

Default behavior:
- Reads symbols from `watchlist.txt`
- Reads daily OHLC data from `out/daily/<SYMBOL>.csv`
- Entry requires all of:
  - monthly close above monthly EMA (default period `10`)
  - weekly close above weekly EMA (default period `20`)
  - daily trigger (default period `50`) where either:
    - close crosses above daily EMA on the current bar
    - or close is above daily EMA and a cross-above occurred within `--entry-cross-lookback-bars` bars (default `15`)
- Set `--entry-cross-lookback-bars 0` to require same-bar cross only
- Optional additional entry gate:
  - `--require-momentum-positive-entry` also requires daily momentum > 0
  - momentum uses `Close - Close[momentum_length]` (default `24`)
- Exit triggers on any one condition:
  - ATR trailing stop breach (default `ATR(14) * 2.5`)
  - trend-failure bars below daily EMA (default `2`)
  - equity kill-switch trigger (default mode `both`: requires both equity below strategy equity EMA and drawdown threshold breach)
- Uses next-bar open execution for both entries and exits
- Applies entry cooldown after kill-switch events (default `10` bars)
- Writes:
  - per-symbol diagnostics: `out/mtf_entry_exit_v1/<SYMBOL>.csv`
  - latest per-symbol status: `out/mtf_entry_exit_v1_latest.csv`
  - per-symbol summary: `out/mtf_entry_exit_v1_summary.csv`
  - symbol-level failures: `out/mtf_entry_exit_v1_errors.csv`

Visualization notebook:
- `notebooks/mtf_entry_exit_v1_lab.ipynb` plots price/actions, equity/drawdown, and confluence/cooldown state.
