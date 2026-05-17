# Private Overlay

This repo is the public engine for `trading-signals`.

When working in the split workspace, keep personal inputs in the sibling
private repo:

```text
trading-signals-workspace/
  trading-signals/
  trading-signals-private/
```

Typical private inputs:

- live watchlists
- current holdings
- private config overlays
- portfolio-specific notes

Example sibling paths:

```text
../trading-signals-private/watchlists/watchlist.md
../trading-signals-private/holdings/holdings.md
../trading-signals-private/configs/
```

Current CLI defaults in the public repo still point at in-repo files for
backward compatibility. Prefer passing explicit private paths for live work.

Examples:

```bash
python3 -m scripts.data.fetch_stooq_ohlc \
  --watchlist ../trading-signals-private/watchlists/watchlist.md

python3 -m scripts.operations.daily_run \
  --watchlist ../trading-signals-private/watchlists/watchlist.md \
  --label daily
```
