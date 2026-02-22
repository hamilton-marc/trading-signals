#!/usr/bin/env python3
"""Translate TradingView trend logic into Python using EMA columns."""

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
    parser.add_argument(
        "--input-dir",
        default="out/indicators/ema",
        help="Directory with per-symbol indicator CSV files",
    )
    parser.add_argument(
        "--out-dir",
        default="out/indicators/trend",
        help="Directory for per-symbol trend analysis CSV files",
    )
    parser.add_argument(
        "--latest-file",
        default="out/_meta/latest/trend_latest.csv",
        help="CSV file path for latest trend per symbol",
    )
    parser.add_argument(
        "--errors-file",
        default="out/_meta/errors/trend_errors.csv",
        help="CSV file path for symbol-level errors",
    )
    parser.add_argument("--fast-period", type=int, default=50, help="Fast EMA period")
    parser.add_argument("--slow-period", type=int, default=200, help="Slow EMA period")
    parser.add_argument(
        "--buffer-pct",
        type=float,
        default=0.5,
        help="Percent buffer applied around EMA comparisons to reduce sideways noise (default: 0.5)",
    )
    parser.add_argument(
        "--confirm-bars",
        type=int,
        default=3,
        help="Consecutive bars required to confirm a trend change (default: 3)",
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
    return rows


def parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return float(raw)


def classify_candidate_trend(
    close: float | None,
    fast_ema: float | None,
    slow_ema: float | None,
    buffer_ratio: float,
) -> str:
    if close is None or fast_ema is None or slow_ema is None:
        return "NEUTRAL"

    uptrend_price = close > fast_ema * (1.0 + buffer_ratio)
    uptrend_stack = fast_ema > slow_ema * (1.0 + buffer_ratio)
    downtrend_price = close < fast_ema * (1.0 - buffer_ratio)
    downtrend_stack = fast_ema < slow_ema * (1.0 - buffer_ratio)

    if uptrend_price and uptrend_stack:
        return "UPTREND"
    if downtrend_price and downtrend_stack:
        return "DOWNTREND"
    return "NEUTRAL"


def trend_color(trend: str) -> str:
    if trend == "UPTREND":
        return "GREEN"
    if trend == "DOWNTREND":
        return "RED"
    return "ORANGE"


def enrich_rows_with_trend(
    rows: list[dict[str, str]],
    fast_col: str,
    slow_col: str,
    buffer_ratio: float,
    confirm_bars: int,
) -> list[dict[str, str]]:
    enriched: list[dict[str, str]] = []
    current_trend = "NEUTRAL"
    pending_trend: str | None = None
    pending_count = 0

    for row in sorted(rows, key=lambda item: item["Date"]):
        close = parse_float(row.get("Close"))
        fast_ema = parse_float(row.get(fast_col))
        slow_ema = parse_float(row.get(slow_col))
        candidate_trend = classify_candidate_trend(
            close=close,
            fast_ema=fast_ema,
            slow_ema=slow_ema,
            buffer_ratio=buffer_ratio,
        )

        if candidate_trend == current_trend:
            pending_trend = None
            pending_count = 0
        else:
            if candidate_trend == pending_trend:
                pending_count += 1
            else:
                pending_trend = candidate_trend
                pending_count = 1

            if pending_trend is not None and pending_count >= confirm_bars:
                current_trend = pending_trend
                pending_trend = None
                pending_count = 0

        new_row = dict(row)
        new_row["TrendCandidate"] = candidate_trend
        new_row["Trend"] = current_trend
        new_row["TrendColor"] = trend_color(current_trend)
        enriched.append(new_row)
    return enriched


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
                "EMA_fast",
                "EMA_slow",
                "Trend",
                "TrendColor",
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
    fast_period = args.fast_period
    slow_period = args.slow_period
    buffer_pct = args.buffer_pct
    confirm_bars = args.confirm_bars
    buffer_ratio = buffer_pct / 100.0
    fast_col = f"EMA_{fast_period}"
    slow_col = f"EMA_{slow_period}"

    if fast_period <= 0 or slow_period <= 0:
        print("[error] fast-period and slow-period must be > 0")
        return 1

    if fast_period >= slow_period:
        print("[error] fast-period should be smaller than slow-period")
        return 1

    if buffer_pct < 0:
        print("[error] buffer-pct must be >= 0")
        return 1

    if confirm_bars <= 0:
        print("[error] confirm-bars must be > 0")
        return 1

    try:
        symbols = read_watchlist(Path(args.watchlist))
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    if not symbols:
        print("[error] watchlist is empty")
        return 1

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    latest_file = Path(args.latest_file)
    errors_file = Path(args.errors_file)

    successes: list[SymbolResult] = []
    errors: list[SymbolError] = []
    latest_rows: list[dict[str, str]] = []

    for symbol in symbols:
        try:
            input_path = input_dir / f"{symbol}.csv"
            if not input_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {input_path}")

            rows = read_rows(input_path)
            if fast_col not in rows[0] or slow_col not in rows[0]:
                raise ValueError(
                    f"Missing required EMA columns: {fast_col}, {slow_col}. "
                    "Run compute_ema.py with --periods 50,200 (or matching periods)."
                )

            enriched_rows = enrich_rows_with_trend(
                rows,
                fast_col=fast_col,
                slow_col=slow_col,
                buffer_ratio=buffer_ratio,
                confirm_bars=confirm_bars,
            )
            output_path = out_dir / f"{symbol}.csv"
            write_rows(output_path, enriched_rows)

            latest = enriched_rows[-1]
            latest_rows.append(
                {
                    "symbol": symbol,
                    "Date": latest["Date"],
                    "Close": latest.get("Close", ""),
                    "EMA_fast": latest.get(fast_col, ""),
                    "EMA_slow": latest.get(slow_col, ""),
                    "Trend": latest["Trend"],
                    "TrendColor": latest["TrendColor"],
                }
            )

            successes.append(SymbolResult(symbol=symbol, rows=len(enriched_rows), output_path=output_path))
            print(f"[ok] {symbol} -> {output_path} ({len(enriched_rows)} rows)")
        except Exception as exc:
            errors.append(SymbolError(symbol=symbol, message=str(exc)))
            print(f"[fail] {symbol} -> {exc}")

    latest_rows.sort(key=lambda item: item["symbol"])
    write_latest(latest_file, latest_rows)
    write_errors(errors_file, errors)

    print("\nSummary")
    print(f"  fast_period: {fast_period}")
    print(f"  slow_period: {slow_period}")
    print(f"  buffer_pct: {buffer_pct}")
    print(f"  confirm_bars: {confirm_bars}")
    print(f"  symbols: {len(symbols)}")
    print(f"  success: {len(successes)}")
    print(f"  failed:  {len(errors)}")
    print(f"  latest file: {latest_file}")
    print(f"  errors file: {errors_file}")

    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
