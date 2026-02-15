# trading-signals
A tool to provide trading signals based on a set of criteria.

## Requirements
- Python 3.10+

## Fetch Daily OHLC From Stooq
Run:

```bash
python3 fetch_stooq_ohlc.py
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
- Fetches daily historical OHLCV from Stooq
- Writes per-symbol CSV files to `out/daily/<SYMBOL>.csv`
- Writes per-symbol failures to `out/stooq_errors.csv`

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
