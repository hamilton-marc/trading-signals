# Data Fetching (Stooq)

This project fetches end-of-day OHLCV data from `stooq.com` using:

```bash
python3 -m scripts.data.fetch_stooq_ohlc
```

## Inputs
- Symbols: `watchlists/watchlist.md` by default. Markdown headings and other decorative lines are ignored; symbol lines are parsed from the remaining content.
- In the split workspace, prefer an explicit sibling private path for live runs, for example `../trading-signals-private/watchlists/watchlist.md`.

## Timeframes
- Daily fetch from Stooq: `--interval d` (default)
- Weekly derived from local daily CSVs: `--interval w`
- Monthly derived from local daily CSVs: `--interval m`
- All in one run: `--interval all` (fetch daily, then derive weekly + monthly)

Examples:

```bash
export STOOQ_APIKEY=your_key_here
python3 -m scripts.data.fetch_stooq_ohlc --interval all --delay-seconds 2.0 --delay-jitter-seconds 3.0
python3 -m scripts.data.fetch_stooq_ohlc --interval d --start-date 2024-01-01
python3 -m scripts.data.fetch_stooq_ohlc --interval d --start-date 2026-02-20 --end-date 2026-02-27
python3 -m scripts.data.fetch_stooq_ohlc --interval w
python3 -m scripts.data.fetch_stooq_ohlc --dry-run
python3 -m scripts.data.fetch_stooq_ohlc --watchlist ../trading-signals-private/watchlists/watchlist.md
```

## Output Paths
- Daily: `out/data/daily/<SYMBOL>.csv`
- Weekly: `out/data/weekly/<SYMBOL>.csv`
- Monthly: `out/data/monthly/<SYMBOL>.csv`

Errors are written to `out/_meta/errors/` with timeframe-specific filenames.

## Provider Request Shape
The fetcher resolves Stooq download URLs in this form:

`https://stooq.com/q/d/l/?s=<symbol>.us&i=d`

When date filters are available, it uses:

`https://stooq.com/q/d/l/?s=<symbol>.us&i=d&f=<YYYYMMDD>&t=<YYYYMMDD>`

If you configure an API key, the fetcher appends:

`&apikey=<YOUR_KEY>`

## API Key Placement
- Preferred: store `export STOOQ_APIKEY=...` in `~/.bashrc` so shells and local agents consistently inherit it.
- Repo-local fallback: copy `.env.example` to `.env` in the repo root and place `STOOQ_APIKEY=...` there.
- The fetcher now loads repo-root `.env` automatically when present.
- `.env` is already ignored by Git and must not be committed.
- Install the repo shell hook with `bash scripts/maintenance/install_git_hooks.sh`.
- The hook blocks tracked `.env` files and likely `STOOQ_APIKEY` assignments, and runs `gitleaks` when installed.

## Notes
- The runner is fault-tolerant per symbol (one failure does not abort all symbols).
- Weekly/monthly are derived locally from `out/data/daily/<SYMBOL>.csv` to reduce provider requests.
- `--start-date` / `--end-date` apply to daily fetches; they are ignored when running `--interval w` or `--interval m`.
- Incremental mode is enabled by default (`--incremental`): for each symbol, the fetcher computes the next missing date from existing CSV data and requests only that date range from Stooq (`f`/`t`), then merges new rows.
- You can disable incremental behavior with `--no-incremental` to replace output files from the response.
- Request pacing defaults to randomized delay per symbol:
  - `--delay-seconds 2.0`
  - `--delay-jitter-seconds 3.0`
  - Actual sleep per symbol is uniformly sampled from `[2.0, 5.0]` seconds by default.
- Stooq now requires an API key for CSV downloads. Supply it with `--api-key` or by setting `STOOQ_APIKEY` before running fetch commands.
