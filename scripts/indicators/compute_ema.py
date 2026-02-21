#!/usr/bin/env python3
"""Compute EMA indicators from downloaded daily OHLC CSV files."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
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
    parser.add_argument("--input-dir", default="out/daily", help="Directory with source OHLC CSV files")
    parser.add_argument("--out-dir", default="out/indicators", help="Directory for indicator CSV output")
    parser.add_argument(
        "--errors-file",
        default="out/indicator_errors.csv",
        help="CSV file path for symbol-level errors",
    )
    parser.add_argument("--period", type=int, default=200, help="Single EMA period (default behavior)")
    parser.add_argument(
        "--periods",
        help="Optional comma-separated EMA periods (example: 50,200)",
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


def parse_periods(value: str | None, fallback_period: int) -> list[int]:
    if value is None:
        periods = [fallback_period]
    else:
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            raise ValueError("No EMA periods provided. Use --periods like 50,200.")
        periods = [int(part) for part in parts]

    cleaned: list[int] = []
    seen: set[int] = set()
    for period in periods:
        if period <= 0:
            raise ValueError(f"Invalid EMA period: {period}. Period must be > 0.")
        if period not in seen:
            cleaned.append(period)
            seen.add(period)
    return cleaned


def compute_ema(values: list[float], period: int) -> list[float | None]:
    if period <= 0:
        raise ValueError("Period must be greater than zero")

    ema: list[float | None] = [None] * len(values)
    if len(values) < period:
        return ema

    smoothing = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period
    ema[period - 1] = seed

    prev = seed
    for index in range(period, len(values)):
        current = (values[index] - prev) * smoothing + prev
        ema[index] = current
        prev = current

    return ema


def read_price_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError("Input file has no data rows")
    if "Date" not in rows[0] or "Close" not in rows[0]:
        raise ValueError("Input file must include Date and Close columns")
    return rows


def write_rows(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
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
    watchlist_path = Path(args.watchlist)
    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    errors_file = Path(args.errors_file)
    try:
        periods = parse_periods(args.periods, fallback_period=args.period)
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

    successes: list[SymbolResult] = []
    errors: list[SymbolError] = []

    for symbol in symbols:
        try:
            input_path = input_dir / f"{symbol}.csv"
            if not input_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {input_path}")

            rows = read_price_rows(input_path)
            rows.sort(key=lambda row: row["Date"])

            closes: list[float] = []
            for row in rows:
                closes.append(float(row["Close"]))

            for period in periods:
                ema_values = compute_ema(closes, period=period)
                indicator_col = f"EMA_{period}"
                for row, ema_value in zip(rows, ema_values, strict=True):
                    row[indicator_col] = "" if ema_value is None else f"{ema_value:.6f}"

            fieldnames = list(rows[0].keys())
            output_path = out_dir / f"{symbol}.csv"
            write_rows(output_path, rows, fieldnames=fieldnames)

            successes.append(SymbolResult(symbol=symbol, rows=len(rows), output_path=output_path))
            print(f"[ok] {symbol} -> {output_path} ({len(rows)} rows)")
        except Exception as exc:
            errors.append(SymbolError(symbol=symbol, message=str(exc)))
            print(f"[fail] {symbol} -> {exc}")

    write_errors(errors_file, errors)

    print("\nSummary")
    period_text = ",".join(str(period) for period in periods)
    print(f"  periods: {period_text}")
    print(f"  symbols: {len(symbols)}")
    print(f"  success: {len(successes)}")
    print(f"  failed:  {len(errors)}")
    print(f"  errors file: {errors_file}")

    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
