#!/usr/bin/env python3
"""Replicate the TradingView Momentum Strategy stop-entry behavior on daily bars."""

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
    parser.add_argument("--out-dir", default="out/momentum_tv", help="Directory for momentum CSV output")
    parser.add_argument(
        "--latest-file",
        default="out/momentum_tv_latest.csv",
        help="CSV file path for latest momentum signal per symbol",
    )
    parser.add_argument(
        "--errors-file",
        default="out/momentum_tv_errors.csv",
        help="CSV file path for symbol-level errors",
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
