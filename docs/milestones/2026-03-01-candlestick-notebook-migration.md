# Milestone: 2026-03-01 - Incremental Fetch Hardening + Candlestick Notebook Migration

## Summary
This milestone delivered two major improvements:
1. Data refresh reliability against Stooq limits (incremental daily-only fetch + local weekly/monthly derivation).
2. Notebook chart UX upgrade to candlesticks with one-symbol-per-row review layouts.

## Current Branch State
- `main` is at `537e9b9` and includes the Stooq fetch hardening work from `feature/ops-next`.
- Active feature branch: `feature/candlestick-notebooks` at `0382ade` (pushed to origin).
- Candlestick notebook work is **not yet merged into `main`**.

## What Was Delivered

### A) Data Fetching Improvements
Relevant commits:
- `3611186` Add incremental date-range fetch with randomized 2-5s pacing
- `5427b5c` Derive weekly/monthly from daily in Stooq fetch workflow
- `276d2db` Fix incremental Stooq fetch by setting default end date

Key behavior:
- Fetches only missing daily bars per symbol using `f/t` date bounds.
- Stops quickly on provider rate-limit responses.
- Derives `weekly` and `monthly` from local `daily` CSVs (no provider calls for higher timeframes).
- Request pacing uses randomized delay in `[2s, 5s]` by default.

### B) Candlestick Notebook Migration
Relevant commits (branch `feature/candlestick-notebooks`):
- `f54e196` Add reusable matplotlib candlestick plotting helpers
- `1d4ce26` Convert ema_lab to candlesticks with volume and style toggle
- `a3ed055` Migrate core signal notebooks to candle charts with volume panels
- `bcf35f3` Document candlestick notebook defaults and chart-style toggle
- `0382ade` Switch notebooks to 1-up layout with volume overlay and MACD panel

New shared module:
- `scripts/plotting/candles.py`
- `scripts/plotting/__init__.py`

Converted notebooks:
- `notebooks/ema_lab.ipynb`
- `notebooks/recent_signal_lab.ipynb`
- `notebooks/recent_signal_lab_weekly.ipynb`
- `notebooks/weekly_trend_watchlist_lab.ipynb`

Final chart layout in these notebooks:
- `CHART_STYLE = "candles"` default, `"line"` fallback.
- One symbol per row in multi-symbol labs.
- Price panel: candles + EMA overlays + event markers + volume overlay.
- Lower panel: MACD histogram + MACD line + signal line.

## Validation Performed
- All 4 converted notebooks executed in `.venv` with:
  - `CHART_STYLE="candles"`
  - `CHART_STYLE="line"`
- Matplotlib used with `Agg` backend for non-interactive validation.
- No runtime exceptions in validation runs.

## Data State Snapshot (as of 2026-03-01 session)
- Daily/weekly/monthly watchlist coverage was refreshed to latest trading date available (`2026-02-27`) during this session.
- Weekly and monthly were regenerated from daily after refresh.

## Recommended Next Step
Merge `feature/candlestick-notebooks` into `main` with a merge commit:

```bash
git checkout main
git pull --ff-only origin main
git merge --no-ff feature/candlestick-notebooks -m "Merge branch 'feature/candlestick-notebooks' into main"
git push origin main
```

## How To Use This Milestone
When starting a new session, reference:
- `docs/milestones/2026-03-01-candlestick-notebook-migration.md`

