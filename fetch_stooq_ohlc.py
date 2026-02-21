#!/usr/bin/env python3
"""Fetch historical OHLC data from Stooq for symbols in a watchlist."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import urlopen

STOOQ_URL_TEMPLATE = "https://stooq.com/q/d/l/?s={symbol}&i={interval}"
CSV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class FetchResult:
    symbol: str
    rows_written: int
    output_path: Path


@dataclass
class FetchError:
    symbol: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", default="watchlist.txt", help="Path to watchlist file")
    parser.add_argument(
        "--interval",
        choices=["d", "w", "m", "all"],
        default="d",
        help="Stooq interval: d=daily, w=weekly, m=monthly, all=d+w+m (default: d)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory for per-symbol CSV output "
            "(default: out/daily for d, out/weekly for w, out/monthly for m)"
        ),
    )
    parser.add_argument(
        "--errors-file",
        default=None,
        help=(
            "CSV file path for symbol-level errors "
            "(default: out/stooq_errors.csv for d, out/stooq_weekly_errors.csv for w, out/stooq_monthly_errors.csv for m)"
        ),
    )
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay after each symbol fetch attempt (default: 0.0)",
    )
    parser.add_argument(
        "--start-date",
        help="Optional inclusive start date filter in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved Stooq URLs and skip network fetch/writes",
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


def to_stooq_symbol(symbol: str) -> str:
    lowered = symbol.lower()
    if "." in lowered:
        return lowered
    return f"{lowered}.us"


def build_stooq_url(stooq_symbol: str, interval: str) -> str:
    return STOOQ_URL_TEMPLATE.format(symbol=quote_plus(stooq_symbol), interval=interval)


def interval_label(interval: str) -> str:
    labels = {
        "d": "daily",
        "w": "weekly",
        "m": "monthly",
    }
    return labels.get(interval, interval)


def resolve_output_paths(
    interval: str,
    out_dir_arg: str | None,
    errors_file_arg: str | None,
) -> tuple[Path, Path]:
    if out_dir_arg:
        out_dir = Path(out_dir_arg)
    else:
        if interval == "d":
            out_dir = Path("out/daily")
        elif interval == "w":
            out_dir = Path("out/weekly")
        else:
            out_dir = Path("out/monthly")

    if errors_file_arg:
        errors_path = Path(errors_file_arg)
    else:
        if interval == "d":
            errors_path = Path("out/stooq_errors.csv")
        elif interval == "w":
            errors_path = Path("out/stooq_weekly_errors.csv")
        else:
            errors_path = Path("out/stooq_monthly_errors.csv")

    return out_dir, errors_path


def parse_iso_date(value: str, *, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}: {value!r}. Use YYYY-MM-DD.") from exc


def fetch_stooq_rows(url: str, timeout: int) -> list[dict[str, str]]:
    with urlopen(url, timeout=timeout) as response:
        content = response.read().decode("utf-8")

    reader = csv.DictReader(content.splitlines())
    if not reader.fieldnames:
        raise ValueError("Stooq response did not contain CSV headers")

    missing = [col for col in CSV_COLUMNS if col not in reader.fieldnames]
    if missing:
        raise ValueError(f"Stooq CSV missing expected columns: {', '.join(missing)}")

    rows = [
        {col: row.get(col, "") for col in CSV_COLUMNS}
        for row in reader
        if any((row.get(col) or "").strip() for col in CSV_COLUMNS)
    ]
    if not rows:
        raise ValueError("No historical rows returned")
    return rows


def filter_rows_by_start_date(rows: list[dict[str, str]], start_date: date | None) -> list[dict[str, str]]:
    if start_date is None:
        return rows

    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        row_date = parse_iso_date(row["Date"], field_name="row Date")
        if row_date >= start_date:
            filtered_rows.append(row)
    return filtered_rows


def write_symbol_csv(path: Path, rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            row_count += 1

    return row_count


def write_errors_csv(path: Path, errors: list[FetchError]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "error"])
        writer.writeheader()
        for item in errors:
            writer.writerow({"symbol": item.symbol, "error": item.message})


def main() -> int:
    args = parse_args()

    watchlist_path = Path(args.watchlist)

    intervals = ["d", "w", "m"] if args.interval == "all" else [args.interval]
    if len(intervals) > 1 and (args.out_dir or args.errors_file):
        print("[error] --out-dir and --errors-file are only supported with a single interval run")
        return 1
    try:
        start_date = parse_iso_date(args.start_date, field_name="--start-date") if args.start_date else None
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1

    try:
        symbols = read_watchlist(watchlist_path)
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    if not symbols:
        print("[error] watchlist is empty")
        return 1

    if args.delay_seconds < 0:
        print("[error] --delay-seconds must be >= 0")
        return 1

    total_successes = 0
    total_failures = 0

    mode = "dry-run" if args.dry_run else "fetch"
    print(f"Mode: {mode}")
    print(f"Intervals: {', '.join(intervals)}")
    print(f"Symbols: {len(symbols)}")
    print(f"Delay seconds: {args.delay_seconds}")

    for interval in intervals:
        out_dir, errors_path = resolve_output_paths(
            interval=interval,
            out_dir_arg=args.out_dir,
            errors_file_arg=args.errors_file,
        )

        print(f"\nInterval {interval_label(interval)} ({interval})")
        successes: list[FetchResult] = []
        errors: list[FetchError] = []

        for symbol in symbols:
            stooq_symbol = to_stooq_symbol(symbol)
            url = build_stooq_url(stooq_symbol, interval=interval)

            if args.dry_run:
                print(f"[dry-run] {symbol} ({stooq_symbol}) -> {url}")
                continue

            try:
                rows = fetch_stooq_rows(url=url, timeout=args.timeout)
                rows = filter_rows_by_start_date(rows, start_date=start_date)
                output_path = out_dir / f"{symbol}.csv"
                row_count = write_symbol_csv(output_path, rows)
                successes.append(FetchResult(symbol=symbol, rows_written=row_count, output_path=output_path))
                print(f"[ok] {symbol} ({stooq_symbol}) -> {output_path} ({row_count} rows)")
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                message = str(exc)
                errors.append(FetchError(symbol=symbol, message=message))
                print(f"[fail] {symbol} ({stooq_symbol}) -> {message}")
            except Exception as exc:
                message = f"Unexpected error: {exc}"
                errors.append(FetchError(symbol=symbol, message=message))
                print(f"[fail] {symbol} ({stooq_symbol}) -> {message}")
            finally:
                if args.delay_seconds > 0:
                    time.sleep(args.delay_seconds)

        if not args.dry_run:
            write_errors_csv(errors_path, errors)

        print("  Summary")
        print(f"    success: {len(successes)}")
        print(f"    failed:  {len(errors)}")
        print(f"    out dir: {out_dir}")
        print(f"    errors file: {errors_path}")

        total_successes += len(successes)
        total_failures += len(errors)

    print("\nOverall Summary")
    print(f"  mode:    {mode}")
    print(f"  symbols: {len(symbols)}")
    print(f"  success: {total_successes}")
    print(f"  failed:  {total_failures}")

    if args.dry_run:
        return 0

    return 0 if total_successes > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
