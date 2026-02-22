#!/usr/bin/env python3
"""Strict replica of TradingView Momentum Strategy stop-entry behavior."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from scripts.paths import (
    DATA_DAILY_DIR,
    DATA_MONTHLY_DIR,
    DATA_WEEKLY_DIR,
    momentum_tv_match_errors_file,
    momentum_tv_match_latest_file,
    momentum_tv_match_output_dir,
)


@dataclass
class SymbolResult:
    symbol: str
    rows: int
    output_path: Path


@dataclass
class SymbolError:
    symbol: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", default="watchlist.txt", help="Path to watchlist file")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol override")
    parser.add_argument(
        "--timeframe",
        default="weekly",
        choices=["d", "w", "m", "daily", "weekly", "monthly"],
        help="Analysis timeframe (default: weekly)",
    )
    parser.add_argument("--input-dir", default=None, help="Input OHLC directory (default depends on timeframe)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default depends on timeframe)")
    parser.add_argument("--latest-file", default=None, help="Latest summary CSV path (default depends on timeframe)")
    parser.add_argument("--errors-file", default=None, help="Errors CSV path (default depends on timeframe)")
    parser.add_argument("--length", type=int, default=24, help="Momentum length (default: 24)")
    parser.add_argument("--min-tick", type=float, default=0.01, help="Tick size for stop offsets (default: 0.01)")
    parser.add_argument(
        "--allow-pyramiding",
        action="store_true",
        help="Allow same-direction re-entries while already in position (default: disabled)",
    )
    return parser.parse_args()


def normalize_timeframe(value: str) -> str:
    lowered = value.strip().lower()
    mapping = {
        "d": "daily",
        "daily": "daily",
        "w": "weekly",
        "weekly": "weekly",
        "m": "monthly",
        "monthly": "monthly",
    }
    timeframe = mapping.get(lowered)
    if timeframe is None:
        raise ValueError(f"Unsupported timeframe: {value!r}")
    return timeframe


def resolve_paths(
    timeframe: str,
    input_dir_arg: str | None,
    out_dir_arg: str | None,
    latest_file_arg: str | None,
    errors_file_arg: str | None,
) -> tuple[Path, Path, Path, Path]:
    if input_dir_arg:
        input_dir = Path(input_dir_arg)
    else:
        default_inputs = {
            "daily": DATA_DAILY_DIR,
            "weekly": DATA_WEEKLY_DIR,
            "monthly": DATA_MONTHLY_DIR,
        }
        input_dir = default_inputs[timeframe]

    if out_dir_arg:
        out_dir = Path(out_dir_arg)
    else:
        out_dir = momentum_tv_match_output_dir(timeframe)

    if latest_file_arg:
        latest_file = Path(latest_file_arg)
    else:
        latest_file = momentum_tv_match_latest_file(timeframe)

    if errors_file_arg:
        errors_file = Path(errors_file_arg)
    else:
        errors_file = momentum_tv_match_errors_file(timeframe)

    return input_dir, out_dir, latest_file, errors_file


def read_watchlist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")

    symbols: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(line.upper())
    return symbols


def parse_symbols(value: str, watchlist_path: Path) -> list[str]:
    if value.strip():
        return [part.strip().upper() for part in value.split(",") if part.strip()]
    return read_watchlist(watchlist_path)


def parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return float(raw)


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"Input file has no rows: {path}")
    required = {"Date", "High", "Low", "Close"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Missing columns in {path}: {', '.join(sorted(missing))}")
    rows.sort(key=lambda row: row["Date"])
    return rows


def aggregate_rows(rows: list[dict[str, str]], timeframe: str) -> list[dict[str, str]]:
    if timeframe == "daily":
        return rows

    buckets: list[dict[str, object]] = []
    current_key: tuple[int, int] | None = None
    bucket: dict[str, object] | None = None

    for row in rows:
        row_date = date.fromisoformat(row["Date"])
        if timeframe == "weekly":
            iso_year, iso_week, _ = row_date.isocalendar()
            key = (iso_year, iso_week)
        else:
            key = (row_date.year, row_date.month)

        open_value = parse_float(row.get("Open"))
        high_value = parse_float(row.get("High"))
        low_value = parse_float(row.get("Low"))
        close_value = parse_float(row.get("Close"))
        if close_value is None:
            continue

        if key != current_key:
            if bucket is not None:
                buckets.append(bucket)
            bucket = {
                "Date": row["Date"],
                "Open": close_value if open_value is None else open_value,
                "High": close_value if high_value is None else high_value,
                "Low": close_value if low_value is None else low_value,
                "Close": close_value,
                "Volume": int(float((row.get("Volume") or "0").strip() or "0")),
            }
            current_key = key
            continue

        if bucket is None:
            continue

        bucket["Date"] = row["Date"]
        current_high = bucket.get("High")
        current_low = bucket.get("Low")
        if high_value is not None and (current_high is None or high_value > current_high):
            bucket["High"] = high_value
        if low_value is not None and (current_low is None or low_value < current_low):
            bucket["Low"] = low_value
        bucket["Close"] = close_value
        bucket["Volume"] = int(bucket.get("Volume", 0)) + int(float((row.get("Volume") or "0").strip() or "0"))

    if bucket is not None:
        buckets.append(bucket)

    out: list[dict[str, str]] = []
    for item in buckets:
        out.append(
            {
                "Date": str(item["Date"]),
                "Open": format_float(item.get("Open") if isinstance(item.get("Open"), float) else None),
                "High": format_float(item.get("High") if isinstance(item.get("High"), float) else None),
                "Low": format_float(item.get("Low") if isinstance(item.get("Low"), float) else None),
                "Close": format_float(item.get("Close") if isinstance(item.get("Close"), float) else None),
                "Volume": str(int(item.get("Volume", 0))),
            }
        )
    return out


def compute_momentum(values: list[float], length: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    for idx in range(length, len(values)):
        output[idx] = values[idx] - values[idx - length]
    return output


def enrich_rows(rows: list[dict[str, str]], length: int, min_tick: float, allow_pyramiding: bool) -> list[dict[str, str]]:
    closes: list[float] = []
    highs: list[float | None] = []
    lows: list[float | None] = []
    for row in rows:
        close_value = parse_float(row.get("Close"))
        if close_value is None:
            raise ValueError(f"Missing Close value on {row.get('Date', '')}")
        closes.append(close_value)
        highs.append(parse_float(row.get("High")))
        lows.append(parse_float(row.get("Low")))

    mom0 = compute_momentum(closes, length=length)
    mom1: list[float | None] = [None] * len(rows)
    for idx in range(1, len(rows)):
        prev = mom0[idx - 1]
        curr = mom0[idx]
        if prev is None or curr is None:
            continue
        mom1[idx] = curr - prev

    pending_long_stop: float | None = None
    pending_short_stop: float | None = None
    position = 0  # -1 short, 0 flat, 1 long

    for idx, row in enumerate(rows):
        high = highs[idx]
        low = lows[idx]
        event = ""
        action = ""
        trade_delta = 0
        fill_price: float | None = None

        # Fill prior-bar pending orders.
        if pending_long_stop is not None and high is not None and high >= pending_long_stop:
            can_fill_long = allow_pyramiding or position <= 0
            if can_fill_long:
                fill_price = pending_long_stop
                event = "MomLE"
                if position < 0:
                    trade_delta = 2
                    action = "REVERSE_TO_LONG"
                elif position == 0:
                    trade_delta = 1
                    action = "BUY"
                position = 1
                pending_short_stop = None
                if not allow_pyramiding:
                    pending_long_stop = None
        elif pending_short_stop is not None and low is not None and low <= pending_short_stop:
            can_fill_short = allow_pyramiding or position >= 0
            if can_fill_short:
                fill_price = pending_short_stop
                event = "MomSE"
                if position > 0:
                    trade_delta = -2
                    action = "REVERSE_TO_SHORT"
                elif position == 0:
                    trade_delta = -1
                    action = "SELL_SHORT"
                position = -1
                pending_long_stop = None
                if not allow_pyramiding:
                    pending_short_stop = None

        current_mom0 = mom0[idx]
        current_mom1 = mom1[idx]
        long_condition = current_mom0 is not None and current_mom1 is not None and current_mom0 > 0 and current_mom1 > 0
        short_condition = current_mom0 is not None and current_mom1 is not None and current_mom0 < 0 and current_mom1 < 0

        # Pine behavior: place/replace stop entries while condition is true, cancel otherwise.
        pending_long_stop = high + min_tick if long_condition and high is not None else None
        pending_short_stop = low - min_tick if short_condition and low is not None else None

        row["MOM0"] = format_float(current_mom0)
        row["MOM1"] = format_float(current_mom1)
        row["LongCondition"] = "1" if long_condition else "0"
        row["ShortCondition"] = "1" if short_condition else "0"
        row["LongStop"] = format_float(pending_long_stop)
        row["ShortStop"] = format_float(pending_short_stop)
        row["FillPrice"] = format_float(fill_price)
        row["Event"] = event
        row["Action"] = action
        row["TradeDelta"] = str(trade_delta)
        row["Position"] = "LONG" if position > 0 else "SHORT" if position < 0 else "FLAT"

    return rows


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "symbol",
                "Date",
                "Close",
                "MOM0",
                "MOM1",
                "LongCondition",
                "ShortCondition",
                "LongStop",
                "ShortStop",
                "FillPrice",
                "Event",
                "Action",
                "TradeDelta",
                "Position",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_errors(path: Path, errors: list[SymbolError]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "error"])
        writer.writeheader()
        for item in errors:
            writer.writerow({"symbol": item.symbol, "error": item.message})


def main() -> int:
    args = parse_args()
    try:
        timeframe = normalize_timeframe(args.timeframe)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    length = args.length
    min_tick = args.min_tick
    allow_pyramiding = args.allow_pyramiding
    if length <= 0:
        print("[error] --length must be > 0")
        return 1
    if min_tick <= 0:
        print("[error] --min-tick must be > 0")
        return 1

    try:
        symbols = parse_symbols(args.symbols, Path(args.watchlist))
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    if not symbols:
        print("[error] no symbols to process")
        return 1

    input_dir, out_dir, latest_file, errors_file = resolve_paths(
        timeframe=timeframe,
        input_dir_arg=args.input_dir,
        out_dir_arg=args.out_dir,
        latest_file_arg=args.latest_file,
        errors_file_arg=args.errors_file,
    )

    successes: list[SymbolResult] = []
    errors: list[SymbolError] = []
    latest_rows: list[dict[str, str]] = []

    for symbol in symbols:
        try:
            input_path = input_dir / f"{symbol}.csv"
            if not input_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {input_path}")

            source_rows = read_rows(input_path)
            rows = aggregate_rows(source_rows, timeframe=timeframe)
            enriched = enrich_rows(rows, length=length, min_tick=min_tick, allow_pyramiding=allow_pyramiding)

            output_path = out_dir / f"{symbol}.csv"
            write_rows(output_path, enriched)

            latest = enriched[-1]
            latest_rows.append(
                {
                    "symbol": symbol,
                    "Date": latest.get("Date", ""),
                    "Close": latest.get("Close", ""),
                    "MOM0": latest.get("MOM0", ""),
                    "MOM1": latest.get("MOM1", ""),
                    "LongCondition": latest.get("LongCondition", ""),
                    "ShortCondition": latest.get("ShortCondition", ""),
                    "LongStop": latest.get("LongStop", ""),
                    "ShortStop": latest.get("ShortStop", ""),
                    "FillPrice": latest.get("FillPrice", ""),
                    "Event": latest.get("Event", ""),
                    "Action": latest.get("Action", ""),
                    "TradeDelta": latest.get("TradeDelta", ""),
                    "Position": latest.get("Position", ""),
                }
            )
            successes.append(SymbolResult(symbol=symbol, rows=len(enriched), output_path=output_path))
            print(f"[ok] {symbol} -> {output_path} ({len(enriched)} rows)")
        except Exception as exc:
            errors.append(SymbolError(symbol=symbol, message=str(exc)))
            print(f"[fail] {symbol} -> {exc}")

    latest_rows.sort(key=lambda row: row["symbol"])
    write_latest(latest_file, latest_rows)
    write_errors(errors_file, errors)

    print("\nSummary")
    print(f"  timeframe: {timeframe}")
    print(f"  length: {length}")
    print(f"  min_tick: {min_tick}")
    print(f"  allow_pyramiding: {'on' if allow_pyramiding else 'off'}")
    print(f"  symbols: {len(symbols)}")
    print(f"  success: {len(successes)}")
    print(f"  failed:  {len(errors)}")
    print(f"  output dir: {out_dir}")
    print(f"  latest file: {latest_file}")
    print(f"  errors file: {errors_file}")

    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
