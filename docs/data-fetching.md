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
python3 -m scripts.data.fetch_stooq_ohlc --interval all --delay-seconds 2.0 --delay-jitter-seconds 3.0
python3 -m scripts.data.fetch_stooq_ohlc --interval d --start-date 2024-01-01
python3 -m scripts.data.fetch_stooq_ohlc --interval d --start-date 2026-02-20 --end-date 2026-02-27
python3 -m scripts.data.fetch_stooq_ohlc --dry-run
```

## Output Paths
- Daily: `out/data/daily/<SYMBOL>.csv`
- Weekly: `out/data/weekly/<SYMBOL>.csv`
- Monthly: `out/data/monthly/<SYMBOL>.csv`

Errors are written to `out/_meta/errors/` with timeframe-specific filenames.

## Provider Request Shape
The fetcher resolves Stooq download URLs in this form:

`https://stooq.com/q/d/l/?s=<symbol>.us&i=<interval>`

When date filters are available, it uses:

`https://stooq.com/q/d/l/?s=<symbol>.us&i=<interval>&f=<YYYYMMDD>&t=<YYYYMMDD>`

Where interval maps to:
- `d` = daily
- `w` = weekly
- `m` = monthly

## Notes
- The runner is fault-tolerant per symbol (one failure does not abort all symbols).
- Weekly/monthly can also be derived later from daily in some downstream scripts, but fetching directly is supported.
- Incremental mode is enabled by default (`--incremental`): for each symbol, the fetcher computes the next missing date from existing CSV data and requests only that date range from Stooq (`f`/`t`), then merges new rows.
- You can disable incremental behavior with `--no-incremental` to replace output files from the response.
- Request pacing defaults to randomized delay per symbol:
  - `--delay-seconds 2.0`
  - `--delay-jitter-seconds 3.0`
  - Actual sleep per symbol is uniformly sampled from `[2.0, 5.0]` seconds by default.
