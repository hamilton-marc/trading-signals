#!/usr/bin/env bash

API_KEY="${ALPHA_VANTAGE_API_KEY:-}"
WATCHLIST_FILE="watchlist.txt"
DATA_DIR="data"
SLEEP_SECONDS="12"

if [ -z "$API_KEY" ]; then
  echo "ALPHA_VANTAGE_API_KEY is not set" >&2
  exit 1
fi

if [ ! -f "$WATCHLIST_FILE" ]; then
  echo "Missing $WATCHLIST_FILE" >&2
  exit 1
fi

mkdir -p "$DATA_DIR"

while IFS= read -r symbol || [ -n "$symbol" ]; do
  symbol="${symbol%$'\r'}"
  [ -z "$symbol" ] && continue
  case "$symbol" in
    \#*) continue ;;
  esac

  echo "Fetching $symbol"
  if ! curl -sS -f \
    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=${symbol}&outputsize=full&datatype=csv&apikey=${API_KEY}" \
    -o "${DATA_DIR}/${symbol}.csv"; then
    echo "Failed to fetch $symbol" >&2
  fi

  sleep "$SLEEP_SECONDS"
done < "$WATCHLIST_FILE"

echo "Done."
