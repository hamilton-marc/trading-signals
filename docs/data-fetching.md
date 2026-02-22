# Data Fetching (Stooq)

This project fetches end-of-day OHLCV data from `stooq.com` using:

```bash
python3 -m scripts.data.fetch_stooq_ohlc
```

## Inputs
- Symbols: `watchlist.txt` (one symbol per line, US ticker form)

## Timeframes
- Daily: `--interval d` (default)
- Weekly: `--interval w`
- Monthly: `--interval m`
- All in one run: `--interval all`

Examples:

```bash
python3 -m scripts.data.fetch_stooq_ohlc --interval all --delay-seconds 0.4
python3 -m scripts.data.fetch_stooq_ohlc --interval d --start-date 2024-01-01
python3 -m scripts.data.fetch_stooq_ohlc --dry-run
```

## Output Paths
- Daily: `out/daily/<SYMBOL>.csv`
- Weekly: `out/weekly/<SYMBOL>.csv`
- Monthly: `out/monthly/<SYMBOL>.csv`

Errors are written to `out/` with timeframe-specific filenames.

## Provider Request Shape
The fetcher resolves Stooq download URLs in this form:

`https://stooq.com/q/d/l/?s=<symbol>.us&i=<interval>`

Where interval maps to:
- `d` = daily
- `w` = weekly
- `m` = monthly

## Notes
- The runner is fault-tolerant per symbol (one failure does not abort all symbols).
- Weekly/monthly can also be derived later from daily in some downstream scripts, but fetching directly is supported.
