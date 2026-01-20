#!/usr/bin/env bash

API_KEY="${MARKETSTACK_API_KEY:-}"
BASE_URL="${MARKETSTACK_BASE_URL:-https://api.marketstack.com/v2}"
WATCHLIST_FILE="watchlist.txt"
DATA_DIR="data"
DAILY_DIR="${DATA_DIR}/daily"
WEEKLY_DIR="${DATA_DIR}/weekly"
MONTHLY_DIR="${DATA_DIR}/monthly"
FETCH_LIMIT="${FETCH_LIMIT:-1000}"   # API page size per symbol.
SLEEP_SECONDS="${SLEEP_SECONDS:-1}" # Throttle between requests.
MAX_SYMBOLS="${MAX_SYMBOLS:-0}"     # 0 = no limit; set >0 to cap requests.
OUTPUT_FORMAT="json"
PRETTY_JSON="0"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --pretty)
      PRETTY_JSON="1"
      ;;
    --format)
      shift
      OUTPUT_FORMAT="${1:-}"
      ;;
    --format=*)
      OUTPUT_FORMAT="${1#*=}"
      ;;
    *)
      echo "Usage: $0 [--format json|csv] [--pretty]" >&2
      exit 1
      ;;
  esac
  shift
done

if [ -z "$API_KEY" ]; then
  echo "MARKETSTACK_API_KEY is not set" >&2
  exit 1
fi

case "$OUTPUT_FORMAT" in
  json|csv|both) ;;
  *)
    echo "Unsupported format: $OUTPUT_FORMAT (use json or csv)" >&2
    exit 1
    ;;
esac

if { [ "$PRETTY_JSON" -eq 1 ] && [ "$OUTPUT_FORMAT" = "json" ]; } || [ "$OUTPUT_FORMAT" = "csv" ]; then
  if ! command -v jq >/dev/null 2>&1; then
    echo "jq is required for CSV or pretty JSON output" >&2
    exit 1
  fi
fi

if [ ! -f "$WATCHLIST_FILE" ]; then
  echo "Missing $WATCHLIST_FILE" >&2
  exit 1
fi

# Marketstack provides daily EOD data; weekly/monthly can be derived later.
mkdir -p "$DAILY_DIR" "$WEEKLY_DIR" "$MONTHLY_DIR"

processed=0
while IFS= read -r symbol || [ -n "$symbol" ]; do
  # Trim Windows carriage returns if the watchlist has CRLF line endings.
  symbol="${symbol%$'\r'}"
  [ -z "$symbol" ] && continue
  case "$symbol" in
    \#*) continue ;;
  esac

  # Limit symbols per run to avoid exceeding request quotas.
  if [ "$MAX_SYMBOLS" -gt 0 ] && [ "$processed" -ge "$MAX_SYMBOLS" ]; then
    echo "Reached MAX_SYMBOLS=${MAX_SYMBOLS}; stopping."
    break
  fi

  processed=$((processed + 1))

  echo "Fetching $symbol"
  tmp_json="$(mktemp)"
  # Marketstack returns JSON; optionally convert to CSV.
  # Check your plan limits in the Marketstack dashboard if requests fail or data is truncated.
  if ! curl -sS -f \
    "${BASE_URL}/eod?access_key=${API_KEY}&symbols=${symbol}&limit=${FETCH_LIMIT}" \
    -o "$tmp_json"; then
    echo "Failed to fetch daily for $symbol" >&2
    rm -f "$tmp_json"
    continue
  fi

  case "$OUTPUT_FORMAT" in
    json)
      if [ "$PRETTY_JSON" -eq 1 ]; then
        if ! jq '.' "$tmp_json" > "${DAILY_DIR}/${symbol}.json"; then
          echo "Failed to write JSON for $symbol" >&2
          rm -f "$tmp_json"
        else
          rm -f "$tmp_json"
        fi
      else
        if ! mv "$tmp_json" "${DAILY_DIR}/${symbol}.json"; then
          echo "Failed to write JSON for $symbol" >&2
          rm -f "$tmp_json"
        fi
      fi
      ;;
    csv|both)
      if [ "$OUTPUT_FORMAT" = "both" ]; then
        if [ "$PRETTY_JSON" -eq 1 ]; then
          if ! jq '.' "$tmp_json" > "${DAILY_DIR}/${symbol}.json"; then
            echo "Failed to write JSON for $symbol" >&2
          fi
        else
          if ! cp "$tmp_json" "${DAILY_DIR}/${symbol}.json"; then
            echo "Failed to write JSON for $symbol" >&2
          fi
        fi
      fi

      if ! jq -r '
        def rows:
          if type == "object" and (.data? | type) == "array" then .data
          elif type == "array" then .
          else [] end;
        (["Date","Open","High","Low","Close","Volume"] | @csv),
        (rows | sort_by(.date)[] | [
          (.date // "" | tostring | .[0:10]),
          (.open // ""),
          (.high // ""),
          (.low // ""),
          (.close // ""),
          (.volume // "")
        ] | @csv)
      ' "$tmp_json" > "${DAILY_DIR}/${symbol}.csv"; then
        echo "Failed to write CSV for $symbol" >&2
      fi
      rm -f "$tmp_json"
      ;;
  esac

  sleep "$SLEEP_SECONDS"
done < "$WATCHLIST_FILE"

echo "Done."
