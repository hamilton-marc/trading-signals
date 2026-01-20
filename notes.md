# Session Recap

## What we did
- Created a Marketstack fetch script (`fetch_marketstack.sh`) to pull daily EOD data.
- JSON responses are stored as `data/daily/{SYMBOL}.json` for now (no transformation).
- `data/weekly/` and `data/monthly/` exist for later derived data.
- Added `data/` to `.gitignore`.
- Removed `.python-version` and documented Python requirement in `README.md` (Python 3.10+).
- Cleared `data/` contents after hitting Alpha Vantage limits (folders remain).

## Current status
- Using Marketstack with `MARKETSTACK_API_KEY` and default base URL `http://api.marketstack.com/v1`.
- Watchlist reduced to 2 symbols to avoid rate limits.
- Observed rate-limit headers: 5 requests/second and `x-quota-limit: 100`.

## Next steps (when ready)
- Run `bash fetch_marketstack.sh` to refresh data.

## Sessions
- Most recent Codex session

```bash
codex resume 019bd803-89b7-7433-a548-9d7d33ca8c28
```
