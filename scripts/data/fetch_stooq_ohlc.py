#!/usr/bin/env python3
"""Fetch historical OHLC data from Stooq for symbols in a watchlist."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import random
import time
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from scripts.paths import data_dir_for_interval, stooq_errors_file

STOOQ_URL_BASE = "https://stooq.com/q/d/l/"
CSV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class FetchResult:
    symbol: str
    rows_fetched: int
    rows_written: int
    rows_total: int
    action: str
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
            "(default: out/data/daily for d, out/data/weekly for w, out/data/monthly for m)"
        ),
    )
    parser.add_argument(
        "--errors-file",
        default=None,
        help=(
            "CSV file path for symbol-level errors "
            "(default: out/_meta/errors/stooq_daily_errors.csv for d, out/_meta/errors/stooq_weekly_errors.csv for w, out/_meta/errors/stooq_monthly_errors.csv for m)"
        ),
    )
    parser.add_argument("--timeout", type=int, default=15, help="HTTP timeout in seconds")
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=2.0,
        help="Base delay after each symbol fetch attempt (default: 2.0)",
    )
    parser.add_argument(
        "--delay-jitter-seconds",
        type=float,
        default=3.0,
        help=(
            "Optional random additional delay per symbol. Actual delay is sampled "
            "uniformly from [delay-seconds, delay-seconds + delay-jitter-seconds] "
            "(default: 3.0)"
        ),
    )
    parser.add_argument(
        "--start-date",
        help="Optional inclusive start date filter in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        help="Optional inclusive end date filter in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--incremental",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fetch only dates newer than existing per-symbol CSVs by using "
            "Stooq f/t date params and local merge (default: on)"
        ),
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
    if lowered.startswith("^"):
        return lowered
    if "." in lowered:
        return lowered
    return f"{lowered}.us"


def format_stooq_date(value: date) -> str:
    return value.strftime("%Y%m%d")


def build_stooq_url(
    stooq_symbol: str,
    interval: str,
    *,
    from_date: date | None = None,
    to_date: date | None = None,
) -> str:
    params = {
        "s": stooq_symbol,
        "i": interval,
    }
    if from_date is not None:
        params["f"] = format_stooq_date(from_date)
    if to_date is not None:
        params["t"] = format_stooq_date(to_date)
    # Keep ^, . for index/ticker syntax while escaping query separators safely.
    return f"{STOOQ_URL_BASE}?{urlencode(params, safe='^.')}"


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
        out_dir = data_dir_for_interval(interval)

    if errors_file_arg:
        errors_path = Path(errors_file_arg)
    else:
        errors_path = stooq_errors_file(interval)

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


def read_existing_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [col for col in CSV_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Existing CSV missing expected columns: {', '.join(missing)}")
        rows = [
            {col: row.get(col, "") for col in CSV_COLUMNS}
            for row in reader
            if any((row.get(col) or "").strip() for col in CSV_COLUMNS)
        ]
    rows.sort(key=lambda item: item["Date"])
    return rows


def latest_row_date(rows: list[dict[str, str]]) -> date | None:
    if not rows:
        return None
    return parse_iso_date(rows[-1]["Date"], field_name="existing row Date")


def filter_rows_by_start_date(rows: list[dict[str, str]], start_date: date | None) -> list[dict[str, str]]:
    if start_date is None:
        return rows

    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        row_date = parse_iso_date(row["Date"], field_name="row Date")
        if row_date >= start_date:
            filtered_rows.append(row)
    return filtered_rows


def filter_rows_by_end_date(rows: list[dict[str, str]], end_date: date | None) -> list[dict[str, str]]:
    if end_date is None:
        return rows

    filtered_rows: list[dict[str, str]] = []
    for row in rows:
        row_date = parse_iso_date(row["Date"], field_name="row Date")
        if row_date <= end_date:
            filtered_rows.append(row)
    return filtered_rows


def merge_rows(
    existing_rows: list[dict[str, str]],
    incoming_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    by_date: dict[str, dict[str, str]] = {row["Date"]: row for row in existing_rows}
    new_rows = 0
    for row in incoming_rows:
        if row["Date"] not in by_date:
            new_rows += 1
        by_date[row["Date"]] = row

    merged = [by_date[key] for key in sorted(by_date.keys())]
    return merged, new_rows


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
        end_date = parse_iso_date(args.end_date, field_name="--end-date") if args.end_date else None
    except ValueError as exc:
        print(f"[error] {exc}")
        return 1
    if start_date and end_date and end_date < start_date:
        print("[error] --end-date must be >= --start-date")
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
    if args.delay_jitter_seconds < 0:
        print("[error] --delay-jitter-seconds must be >= 0")
        return 1

    total_successes = 0
    total_failures = 0

    mode = "dry-run" if args.dry_run else "fetch"
    print(f"Mode: {mode}")
    print(f"Intervals: {', '.join(intervals)}")
    print(f"Symbols: {len(symbols)}")
    print(f"Delay seconds: {args.delay_seconds}")
    max_delay = args.delay_seconds + args.delay_jitter_seconds
    print(
        "Delay jitter seconds: "
        f"{args.delay_jitter_seconds} "
        f"(actual per symbol in [{args.delay_seconds:.3f}, {max_delay:.3f}])"
    )
    print(f"Incremental: {args.incremental}")
    if start_date:
        print(f"Start date: {start_date.isoformat()}")
    if end_date:
        print(f"End date: {end_date.isoformat()}")

    for interval in intervals:
        out_dir, errors_path = resolve_output_paths(
            interval=interval,
            out_dir_arg=args.out_dir,
            errors_file_arg=args.errors_file,
        )

        print(f"\nInterval {interval_label(interval)} ({interval})")
        successes: list[FetchResult] = []
        errors: list[FetchError] = []
        skipped = 0

        for symbol in symbols:
            stooq_symbol = to_stooq_symbol(symbol)
            output_path = out_dir / f"{symbol}.csv"

            try:
                existing_rows = read_existing_rows(output_path) if args.incremental else []
                existing_latest = latest_row_date(existing_rows) if args.incremental else None

                effective_start = start_date
                if args.incremental and existing_latest is not None:
                    next_missing = existing_latest + timedelta(days=1)
                    effective_start = max(next_missing, start_date) if start_date else next_missing
                effective_end = end_date
                if effective_start and effective_end and effective_end < effective_start:
                    # Nothing new is needed for this symbol.
                    skipped += 1
                    successes.append(
                        FetchResult(
                            symbol=symbol,
                            rows_fetched=0,
                            rows_written=0,
                            rows_total=len(existing_rows),
                            action="skip",
                            output_path=output_path,
                        )
                    )
                    print(
                        f"[skip] {symbol} ({stooq_symbol}) -> no missing dates "
                        f"(existing through {existing_latest.isoformat() if existing_latest else 'n/a'})"
                    )
                    continue

                url = build_stooq_url(
                    stooq_symbol,
                    interval=interval,
                    from_date=effective_start,
                    to_date=effective_end,
                )

                if args.dry_run:
                    print(f"[dry-run] {symbol} ({stooq_symbol}) -> {url}")
                    continue

                rows = fetch_stooq_rows(url=url, timeout=args.timeout)
                rows = filter_rows_by_start_date(rows, start_date=effective_start)
                rows = filter_rows_by_end_date(rows, end_date=effective_end)

                if args.incremental:
                    merged_rows, new_rows = merge_rows(existing_rows, rows)
                    if new_rows == 0 and output_path.exists():
                        skipped += 1
                        successes.append(
                            FetchResult(
                                symbol=symbol,
                                rows_fetched=len(rows),
                                rows_written=0,
                                rows_total=len(existing_rows),
                                action="skip",
                                output_path=output_path,
                            )
                        )
                        print(
                            f"[skip] {symbol} ({stooq_symbol}) -> {output_path} "
                            f"(fetched={len(rows)} new=0 total={len(existing_rows)})"
                        )
                    else:
                        row_count = write_symbol_csv(output_path, merged_rows)
                        successes.append(
                            FetchResult(
                                symbol=symbol,
                                rows_fetched=len(rows),
                                rows_written=new_rows,
                                rows_total=row_count,
                                action="merge",
                                output_path=output_path,
                            )
                        )
                        print(
                            f"[ok] {symbol} ({stooq_symbol}) -> {output_path} "
                            f"(fetched={len(rows)} new={new_rows} total={row_count})"
                        )
                else:
                    row_count = write_symbol_csv(output_path, rows)
                    successes.append(
                        FetchResult(
                            symbol=symbol,
                            rows_fetched=len(rows),
                            rows_written=row_count,
                            rows_total=row_count,
                            action="replace",
                            output_path=output_path,
                        )
                    )
                    print(
                        f"[ok] {symbol} ({stooq_symbol}) -> {output_path} "
                        f"(rows={row_count})"
                    )
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                message = str(exc)
                errors.append(FetchError(symbol=symbol, message=message))
                print(f"[fail] {symbol} ({stooq_symbol}) -> {message}")
            except Exception as exc:
                message = f"Unexpected error: {exc}"
                errors.append(FetchError(symbol=symbol, message=message))
                print(f"[fail] {symbol} ({stooq_symbol}) -> {message}")
            finally:
                if args.delay_seconds > 0 or args.delay_jitter_seconds > 0:
                    sleep_seconds = random.uniform(
                        args.delay_seconds,
                        args.delay_seconds + args.delay_jitter_seconds,
                    )
                    time.sleep(sleep_seconds)

        if not args.dry_run:
            write_errors_csv(errors_path, errors)

        print("  Summary")
        print(f"    success: {len(successes)}")
        print(f"    skipped: {skipped}")
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
