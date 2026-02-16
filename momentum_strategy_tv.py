#!/usr/bin/env python3
"""Replicate TradingView Momentum Strategy stop-entry behavior across multiple timeframes."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


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
    parser.add_argument(
        "--timeframe",
        default="daily",
        choices=["d", "w", "m", "daily", "weekly", "monthly"],
        help="Analysis timeframe: daily, weekly, or monthly (default: daily)",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with source OHLC CSV files (default depends on timeframe)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory for momentum CSV output (default depends on timeframe)",
    )
    parser.add_argument(
        "--latest-file",
        default=None,
        help="CSV file path for latest momentum signal per symbol (default depends on timeframe)",
    )
    parser.add_argument(
        "--errors-file",
        default=None,
        help="CSV file path for symbol-level errors (default depends on timeframe)",
    )
    parser.add_argument("--length", type=int, default=24, help="Momentum length (default: 24)")
    parser.add_argument(
        "--min-tick",
        type=float,
        default=0.01,
        help="Tick size used for stop offsets (default: 0.01)",
    )
    return parser.parse_args()


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


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("Input file has no data rows")
    required = {"Date", "High", "Low", "Close"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Input file missing required columns: {', '.join(sorted(missing))}")
    return rows


def parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return float(raw)


def parse_int(value: str | None) -> int:
    raw = (value or "").strip()
    if not raw:
        return 0
    return int(float(raw))


def format_float(value: float | None) -> str:
    if value is None:
        return ""
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


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
    normalized = mapping.get(lowered)
    if normalized is None:
        raise ValueError(f"Unsupported timeframe: {value!r}")
    return normalized


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
        input_dir = Path("out/monthly" if timeframe == "monthly" else "out/daily")

    if out_dir_arg:
        out_dir = Path(out_dir_arg)
    else:
        if timeframe == "daily":
            out_dir = Path("out/momentum_tv")
        elif timeframe == "weekly":
            out_dir = Path("out/momentum_tv_weekly")
        else:
            out_dir = Path("out/momentum_tv_monthly")

    if latest_file_arg:
        latest_file = Path(latest_file_arg)
    else:
        if timeframe == "daily":
            latest_file = Path("out/momentum_tv_latest.csv")
        elif timeframe == "weekly":
            latest_file = Path("out/momentum_tv_weekly_latest.csv")
        else:
            latest_file = Path("out/momentum_tv_monthly_latest.csv")

    if errors_file_arg:
        errors_file = Path(errors_file_arg)
    else:
        if timeframe == "daily":
            errors_file = Path("out/momentum_tv_errors.csv")
        elif timeframe == "weekly":
            errors_file = Path("out/momentum_tv_weekly_errors.csv")
        else:
            errors_file = Path("out/momentum_tv_monthly_errors.csv")

    return input_dir, out_dir, latest_file, errors_file


def aggregate_rows(rows: list[dict[str, str]], timeframe: str) -> list[dict[str, str]]:
    if timeframe == "daily":
        return sorted(rows, key=lambda row: row["Date"])

    ordered = sorted(rows, key=lambda row: row["Date"])
    buckets: list[dict[str, object]] = []
    current_key: tuple[int, int] | None = None
    bucket: dict[str, object] | None = None

    for row in ordered:
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
                "Open": open_value if open_value is not None else close_value,
                "High": high_value if high_value is not None else close_value,
                "Low": low_value if low_value is not None else close_value,
                "Close": close_value,
                "Volume": parse_int(row.get("Volume")),
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
        bucket["Volume"] = int(bucket.get("Volume", 0)) + parse_int(row.get("Volume"))

    if bucket is not None:
        buckets.append(bucket)

    output: list[dict[str, str]] = []
    for item in buckets:
        output.append(
            {
                "Date": str(item["Date"]),
                "Open": format_float(item.get("Open") if isinstance(item.get("Open"), float) else None),
                "High": format_float(item.get("High") if isinstance(item.get("High"), float) else None),
                "Low": format_float(item.get("Low") if isinstance(item.get("Low"), float) else None),
                "Close": format_float(item.get("Close") if isinstance(item.get("Close"), float) else None),
                "Volume": str(int(item.get("Volume", 0))),
            }
        )
    return output


def compute_momentum(values: list[float], length: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    for idx in range(length, len(values)):
        output[idx] = values[idx] - values[idx - length]
    return output


def momentum_raw_state(mom0: float | None, mom1: float | None) -> str:
    if mom0 is None or mom1 is None:
        return "NEUTRAL"
    if mom0 > 0 and mom1 > 0:
        return "LONG"
    if mom0 < 0 and mom1 < 0:
        return "SHORT"
    return "NEUTRAL"


def enrich_rows(rows: list[dict[str, str]], length: int, min_tick: float) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: row["Date"])

    closes: list[float] = []
    highs: list[float | None] = []
    lows: list[float | None] = []
    for row in ordered:
        close = parse_float(row.get("Close"))
        if close is None:
            raise ValueError(f"Missing Close value on {row.get('Date', '')}")
        closes.append(close)
        highs.append(parse_float(row.get("High")))
        lows.append(parse_float(row.get("Low")))

    mom0 = compute_momentum(closes, length=length)
    mom1: list[float | None] = [None] * len(ordered)
    for idx in range(1, len(ordered)):
        if mom0[idx] is None or mom0[idx - 1] is None:
            continue
        mom1[idx] = mom0[idx] - mom0[idx - 1]

    state = "FLAT"
    pending_long_stop: float | None = None
    pending_short_stop: float | None = None

    for idx, row in enumerate(ordered):
        high = highs[idx]
        low = lows[idx]
        event = ""
        action = ""
        fill_price: float | None = None

        # Fill stop orders carried from the prior bar.
        if pending_long_stop is not None and high is not None and high >= pending_long_stop:
            if state == "FLAT":
                event = "LONG_ENTRY"
                action = "BUY"
            elif state == "SHORT":
                event = "SHORT_TO_LONG"
                action = "REVERSE_TO_LONG"
            state = "LONG"
            fill_price = pending_long_stop
            pending_long_stop = None
            pending_short_stop = None
        elif pending_short_stop is not None and low is not None and low <= pending_short_stop:
            if state == "FLAT":
                event = "SHORT_ENTRY"
                action = "SELL_SHORT"
            elif state == "LONG":
                event = "LONG_TO_SHORT"
                action = "REVERSE_TO_SHORT"
            state = "SHORT"
            fill_price = pending_short_stop
            pending_long_stop = None
            pending_short_stop = None

        raw_state = momentum_raw_state(mom0[idx], mom1[idx])
        long_condition = raw_state == "LONG"
        short_condition = raw_state == "SHORT"

        # TradingView logic: keep/replace stop entry while condition is true; cancel otherwise.
        pending_long_stop = high + min_tick if long_condition and high is not None else None
        pending_short_stop = low - min_tick if short_condition and low is not None else None

        row["MOM0"] = "" if mom0[idx] is None else f"{mom0[idx]:.6f}"
        row["MOM1"] = "" if mom1[idx] is None else f"{mom1[idx]:.6f}"
        row["MomentumRawState"] = raw_state
        row["MomLongCondition"] = "1" if long_condition else "0"
        row["MomShortCondition"] = "1" if short_condition else "0"
        row["LongStop"] = "" if pending_long_stop is None else f"{pending_long_stop:.6f}"
        row["ShortStop"] = "" if pending_short_stop is None else f"{pending_short_stop:.6f}"
        row["FillPrice"] = "" if fill_price is None else f"{fill_price:.6f}"
        row["MomentumState"] = state
        row["MomentumEvent"] = event
        row["MomentumSignal"] = state
        row["MomentumColor"] = "GREEN" if state == "LONG" else "RED" if state == "SHORT" else "ORANGE"
        row["SignalAction"] = action

    return ordered


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
                "MomentumRawState",
                "MomentumState",
                "MomentumEvent",
                "LongStop",
                "ShortStop",
                "FillPrice",
                "SignalAction",
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

    if length <= 0:
        print("[error] length must be > 0")
        return 1
    if min_tick <= 0:
        print("[error] min-tick must be > 0")
        return 1

    try:
        symbols = read_watchlist(Path(args.watchlist))
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    if not symbols:
        print("[error] watchlist is empty")
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
            if not rows:
                raise ValueError(f"No rows available for timeframe={timeframe} in {input_path}")
            enriched = enrich_rows(rows, length=length, min_tick=min_tick)

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
                    "MomentumRawState": latest.get("MomentumRawState", ""),
                    "MomentumState": latest.get("MomentumState", ""),
                    "MomentumEvent": latest.get("MomentumEvent", ""),
                    "LongStop": latest.get("LongStop", ""),
                    "ShortStop": latest.get("ShortStop", ""),
                    "FillPrice": latest.get("FillPrice", ""),
                    "SignalAction": latest.get("SignalAction", ""),
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
    print(f"  symbols: {len(symbols)}")
    print(f"  success: {len(successes)}")
    print(f"  failed:  {len(errors)}")
    print(f"  latest file: {latest_file}")
    print(f"  errors file: {errors_file}")

    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
