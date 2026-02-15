#!/usr/bin/env python3
"""Build final trade signals from trend, momentum transitions, and breakout confirmation."""

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
    parser.add_argument("--trend-dir", default="out/trend", help="Directory with trend CSV files")
    parser.add_argument("--momentum-dir", default="out/momentum", help="Directory with momentum CSV files")
    parser.add_argument("--out-dir", default="out/signals", help="Directory for per-symbol signal CSV files")
    parser.add_argument(
        "--latest-file",
        default="out/signal_latest.csv",
        help="CSV file path for latest signal state per symbol",
    )
    parser.add_argument(
        "--errors-file",
        default="out/signal_errors.csv",
        help="CSV file path for symbol-level errors",
    )
    parser.add_argument(
        "--breakout-lookback",
        type=int,
        default=0,
        help="Bars to look back for breakout confirmation. Use 0 to disable (default: 0).",
    )
    parser.add_argument(
        "--min-hold-bars",
        type=int,
        default=5,
        help="Minimum bars to hold a position before reversal (default: 5)",
    )
    parser.add_argument(
        "--disable-trend-filter",
        action="store_true",
        help="Allow signals regardless of trend regime",
    )
    parser.add_argument(
        "--allow-neutral-trend-entries",
        action="store_true",
        help="When trend filter is on, allow both long and short setups during NEUTRAL trend",
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


def merge_rows(
    trend_rows: list[dict[str, str]],
    momentum_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    trend_by_date = {row["Date"]: row for row in trend_rows}
    momentum_by_date = {row["Date"]: row for row in momentum_rows}
    common_dates = sorted(set(trend_by_date).intersection(momentum_by_date))
    if not common_dates:
        raise ValueError("No overlapping dates between trend and momentum inputs")

    merged: list[dict[str, str]] = []
    for date_key in common_dates:
        trend = trend_by_date[date_key]
        mom = momentum_by_date[date_key]
        merged.append(
            {
                "Date": date_key,
                "Open": trend.get("Open", ""),
                "High": trend.get("High", ""),
                "Low": trend.get("Low", ""),
                "Close": trend.get("Close", ""),
                "Volume": trend.get("Volume", ""),
                "Trend": (trend.get("Trend") or "NEUTRAL").strip(),
                "MomentumState": (mom.get("MomentumState") or mom.get("MomentumSignal") or "NEUTRAL").strip(),
                "MomentumEvent": (mom.get("MomentumEvent") or "").strip(),
            }
        )
    return merged


def build_signals(
    rows: list[dict[str, str]],
    breakout_lookback: int,
    min_hold_bars: int,
    use_trend_filter: bool,
    allow_neutral_trend_entries: bool,
) -> list[dict[str, str]]:
    highs: list[float | None] = [parse_float(row.get("High")) for row in rows]
    lows: list[float | None] = [parse_float(row.get("Low")) for row in rows]
    closes: list[float | None] = [parse_float(row.get("Close")) for row in rows]

    long_triggers = {"LONG_ENTRY", "SHORT_TO_LONG"}
    short_triggers = {"SHORT_ENTRY", "LONG_TO_SHORT"}

    state = "FLAT"
    bars_in_state = 0

    for idx, row in enumerate(rows):
        trend = row["Trend"]
        momentum_event = row["MomentumEvent"]
        close = closes[idx]
        high = highs[idx]
        low = lows[idx]

        breakout_high: float | None = None
        breakout_low: float | None = None
        breakout_long = breakout_lookback == 0
        breakout_short = breakout_lookback == 0

        if breakout_lookback > 0 and idx >= breakout_lookback:
            prev_highs = [value for value in highs[idx - breakout_lookback : idx] if value is not None]
            prev_lows = [value for value in lows[idx - breakout_lookback : idx] if value is not None]
            if prev_highs:
                breakout_high = max(prev_highs)
            if prev_lows:
                breakout_low = min(prev_lows)

            # Breakout confirmation uses intraday extremes to avoid missing transitions
            # where close finishes back inside the range.
            if high is not None and breakout_high is not None:
                breakout_long = high > breakout_high
            elif close is not None and breakout_high is not None:
                breakout_long = close > breakout_high

            if low is not None and breakout_low is not None:
                breakout_short = low < breakout_low
            elif close is not None and breakout_low is not None:
                breakout_short = close < breakout_low

        if not use_trend_filter:
            regime_long_ok = True
            regime_short_ok = True
        elif allow_neutral_trend_entries:
            regime_long_ok = trend in {"UPTREND", "NEUTRAL"}
            regime_short_ok = trend in {"DOWNTREND", "NEUTRAL"}
        else:
            regime_long_ok = trend == "UPTREND"
            regime_short_ok = trend == "DOWNTREND"

        long_setup = momentum_event in long_triggers and regime_long_ok and breakout_long
        short_setup = momentum_event in short_triggers and regime_short_ok and breakout_short

        signal_event = ""
        signal_action = ""

        if state != "FLAT":
            bars_in_state += 1

        if state == "FLAT":
            if long_setup:
                state = "LONG"
                bars_in_state = 0
                signal_event = "LONG_ENTRY"
                signal_action = "BUY"
            elif short_setup:
                state = "SHORT"
                bars_in_state = 0
                signal_event = "SHORT_ENTRY"
                signal_action = "SELL_SHORT"
        elif state == "LONG":
            if bars_in_state >= min_hold_bars and short_setup:
                state = "SHORT"
                bars_in_state = 0
                signal_event = "LONG_TO_SHORT"
                signal_action = "REVERSE_TO_SHORT"
        elif state == "SHORT":
            if bars_in_state >= min_hold_bars and long_setup:
                state = "LONG"
                bars_in_state = 0
                signal_event = "SHORT_TO_LONG"
                signal_action = "REVERSE_TO_LONG"

        row["BreakoutHigh"] = "" if breakout_high is None else f"{breakout_high:.6f}"
        row["BreakoutLow"] = "" if breakout_low is None else f"{breakout_low:.6f}"
        row["LongSetup"] = "1" if long_setup else "0"
        row["ShortSetup"] = "1" if short_setup else "0"
        row["SignalState"] = state
        row["SignalEvent"] = signal_event
        row["SignalAction"] = signal_action

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
                "Trend",
                "MomentumState",
                "MomentumEvent",
                "SignalState",
                "SignalEvent",
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
    breakout_lookback = args.breakout_lookback
    min_hold_bars = args.min_hold_bars
    use_trend_filter = not args.disable_trend_filter
    allow_neutral_trend_entries = args.allow_neutral_trend_entries

    if breakout_lookback < 0:
        print("[error] breakout-lookback must be >= 0")
        return 1
    if min_hold_bars < 0:
        print("[error] min-hold-bars must be >= 0")
        return 1

    try:
        symbols = read_watchlist(Path(args.watchlist))
    except Exception as exc:
        print(f"[error] {exc}")
        return 1

    if not symbols:
        print("[error] watchlist is empty")
        return 1

    trend_dir = Path(args.trend_dir)
    momentum_dir = Path(args.momentum_dir)
    out_dir = Path(args.out_dir)
    latest_file = Path(args.latest_file)
    errors_file = Path(args.errors_file)

    successes: list[SymbolResult] = []
    errors: list[SymbolError] = []
    latest_rows: list[dict[str, str]] = []

    for symbol in symbols:
        try:
            trend_path = trend_dir / f"{symbol}.csv"
            momentum_path = momentum_dir / f"{symbol}.csv"
            if not trend_path.exists():
                raise FileNotFoundError(f"Trend CSV not found: {trend_path}")
            if not momentum_path.exists():
                raise FileNotFoundError(f"Momentum CSV not found: {momentum_path}")

            trend_rows = read_rows(trend_path)
            momentum_rows = read_rows(momentum_path)
            merged = merge_rows(trend_rows, momentum_rows)
            signaled = build_signals(
                merged,
                breakout_lookback=breakout_lookback,
                min_hold_bars=min_hold_bars,
                use_trend_filter=use_trend_filter,
                allow_neutral_trend_entries=allow_neutral_trend_entries,
            )

            output_path = out_dir / f"{symbol}.csv"
            write_rows(output_path, signaled)

            latest = signaled[-1]
            latest_rows.append(
                {
                    "symbol": symbol,
                    "Date": latest.get("Date", ""),
                    "Close": latest.get("Close", ""),
                    "Trend": latest.get("Trend", ""),
                    "MomentumState": latest.get("MomentumState", ""),
                    "MomentumEvent": latest.get("MomentumEvent", ""),
                    "SignalState": latest.get("SignalState", ""),
                    "SignalEvent": latest.get("SignalEvent", ""),
                    "SignalAction": latest.get("SignalAction", ""),
                }
            )

            successes.append(SymbolResult(symbol=symbol, rows=len(signaled), output_path=output_path))
            print(f"[ok] {symbol} -> {output_path} ({len(signaled)} rows)")
        except Exception as exc:
            errors.append(SymbolError(symbol=symbol, message=str(exc)))
            print(f"[fail] {symbol} -> {exc}")

    latest_rows.sort(key=lambda row: row["symbol"])
    write_latest(latest_file, latest_rows)
    write_errors(errors_file, errors)

    print("\nSummary")
    print(f"  breakout_lookback: {breakout_lookback}")
    print(f"  min_hold_bars: {min_hold_bars}")
    print(f"  trend_filter: {'on' if use_trend_filter else 'off'}")
    print(f"  allow_neutral_trend_entries: {'on' if allow_neutral_trend_entries else 'off'}")
    print(f"  symbols: {len(symbols)}")
    print(f"  success: {len(successes)}")
    print(f"  failed:  {len(errors)}")
    print(f"  latest file: {latest_file}")
    print(f"  errors file: {errors_file}")

    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
