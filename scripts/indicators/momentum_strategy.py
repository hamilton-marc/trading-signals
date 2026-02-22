#!/usr/bin/env python3
"""Compute Momentum Strategy signals from daily price data."""

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
    parser.add_argument("--input-dir", default="out/data/daily", help="Directory with source OHLC CSV files")
    parser.add_argument("--out-dir", default="out/indicators/momentum", help="Directory for momentum CSV output")
    parser.add_argument(
        "--latest-file",
        default="out/_meta/latest/momentum_latest.csv",
        help="CSV file path for latest momentum signal per symbol",
    )
    parser.add_argument(
        "--errors-file",
        default="out/_meta/errors/momentum_errors.csv",
        help="CSV file path for symbol-level errors",
    )
    parser.add_argument("--length", type=int, default=24, help="Momentum length (default: 24)")
    parser.add_argument(
        "--confirm-bars",
        type=int,
        default=3,
        help="Consecutive bars required to confirm LONG/SHORT transitions (default: 3)",
    )
    parser.add_argument(
        "--min-mom0-pct",
        type=float,
        default=1.0,
        help="Minimum absolute MOM0 as percent of close for a valid directional signal (default: 1.0)",
    )
    parser.add_argument(
        "--min-mom1-pct",
        type=float,
        default=0.2,
        help="Minimum absolute MOM1 as percent of close for a valid directional signal (default: 0.2)",
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
    if "Date" not in rows[0] or "Close" not in rows[0]:
        raise ValueError("Input file must include Date and Close columns")
    return rows


def compute_momentum(values: list[float], length: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    for idx in range(length, len(values)):
        output[idx] = values[idx] - values[idx - length]
    return output


def classify_raw_state(mom0: float | None, mom1: float | None) -> str:
    if mom0 is None or mom1 is None:
        return "NEUTRAL"
    if mom0 > 0 and mom1 > 0:
        return "LONG"
    if mom0 < 0 and mom1 < 0:
        return "SHORT"
    return "NEUTRAL"


def classify_state(
    mom0: float | None,
    mom1: float | None,
    close: float,
    min_mom0_pct: float,
    min_mom1_pct: float,
) -> tuple[str, str]:
    raw_state = classify_raw_state(mom0, mom1)
    if raw_state == "NEUTRAL":
        return "NEUTRAL", "ORANGE"

    mom0_abs_pct = (abs(mom0) / close) * 100.0 if close != 0 else 0.0
    mom1_abs_pct = (abs(mom1) / close) * 100.0 if close != 0 else 0.0
    if mom0_abs_pct < min_mom0_pct or mom1_abs_pct < min_mom1_pct:
        return "NEUTRAL", "ORANGE"

    if raw_state == "LONG":
        return "LONG", "GREEN"
    return "SHORT", "RED"


def enrich_rows(
    rows: list[dict[str, str]],
    length: int,
    confirm_bars: int,
    min_mom0_pct: float,
    min_mom1_pct: float,
) -> list[dict[str, str]]:
    ordered = sorted(rows, key=lambda row: row["Date"])
    closes = [float(row["Close"]) for row in ordered]

    mom0 = compute_momentum(closes, length=length)

    # mom1 follows the built-in pattern: momentum of mom0 with length 1.
    mom1: list[float | None] = [None] * len(mom0)
    for idx in range(1, len(mom0)):
        if mom0[idx] is None or mom0[idx - 1] is None:
            continue
        mom1[idx] = mom0[idx] - mom0[idx - 1]

    raw_state_values: list[str] = []
    candidate_values: list[str] = []
    for m0, m1, close in zip(mom0, mom1, closes, strict=True):
        raw_state_values.append(classify_raw_state(m0, m1))
        candidate, _ = classify_state(
            m0,
            m1,
            close=close,
            min_mom0_pct=min_mom0_pct,
            min_mom1_pct=min_mom1_pct,
        )
        candidate_values.append(candidate)

    # Persistent event-driven state:
    # - Start NEUTRAL.
    # - Enter LONG/SHORT only after confirmation.
    # - Ignore NEUTRAL candidates while in position to reduce sideways churn.
    # - Reverse only after confirmation of opposite direction.
    state_values: list[str] = []
    event_values: list[str] = []
    state_colors: list[str] = []
    current_state = "NEUTRAL"
    pending_state = ""
    pending_count = 0

    for candidate in candidate_values:
        event = ""
        if current_state == "NEUTRAL":
            if candidate in {"LONG", "SHORT"}:
                if candidate == pending_state:
                    pending_count += 1
                else:
                    pending_state = candidate
                    pending_count = 1
                if pending_count >= confirm_bars:
                    current_state = candidate
                    event = f"{candidate}_ENTRY"
                    pending_state = ""
                    pending_count = 0
            else:
                pending_state = ""
                pending_count = 0
        else:
            opposite = "SHORT" if current_state == "LONG" else "LONG"
            if candidate == opposite:
                if candidate == pending_state:
                    pending_count += 1
                else:
                    pending_state = candidate
                    pending_count = 1
                if pending_count >= confirm_bars:
                    event = f"{current_state}_TO_{candidate}"
                    current_state = candidate
                    pending_state = ""
                    pending_count = 0
            else:
                pending_state = ""
                pending_count = 0

        state_values.append(current_state)
        event_values.append(event)
        state_colors.append("GREEN" if current_state == "LONG" else "RED" if current_state == "SHORT" else "ORANGE")

    for idx, row in enumerate(ordered):
        row["MOM0"] = "" if mom0[idx] is None else f"{mom0[idx]:.6f}"
        row["MOM1"] = "" if mom1[idx] is None else f"{mom1[idx]:.6f}"
        row["MomentumRawState"] = raw_state_values[idx]
        row["MomentumCandidate"] = candidate_values[idx]
        row["MomentumState"] = state_values[idx]
        row["MomentumEvent"] = event_values[idx]
        row["MomentumColor"] = state_colors[idx]
        # Backward compatibility with earlier notebook/code paths.
        row["MomentumSignal"] = state_values[idx]

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
                "MomentumCandidate",
                "MomentumState",
                "MomentumEvent",
                "MomentumSignal",
                "MomentumColor",
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
    confirm_bars = args.confirm_bars
    min_mom0_pct = args.min_mom0_pct
    min_mom1_pct = args.min_mom1_pct
    if length <= 0:
        print("[error] length must be > 0")
        return 1
    if confirm_bars <= 0:
        print("[error] confirm-bars must be > 0")
        return 1
    if min_mom0_pct < 0 or min_mom1_pct < 0:
        print("[error] min-mom0-pct and min-mom1-pct must be >= 0")
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
            enriched_rows = enrich_rows(
                rows,
                length=length,
                confirm_bars=confirm_bars,
                min_mom0_pct=min_mom0_pct,
                min_mom1_pct=min_mom1_pct,
            )

            output_path = out_dir / f"{symbol}.csv"
            write_rows(output_path, enriched_rows)

            latest = enriched_rows[-1]
            latest_rows.append(
                {
                    "symbol": symbol,
                    "Date": latest.get("Date", ""),
                    "Close": latest.get("Close", ""),
                    "MOM0": latest.get("MOM0", ""),
                    "MOM1": latest.get("MOM1", ""),
                    "MomentumState": latest.get("MomentumState", ""),
                    "MomentumEvent": latest.get("MomentumEvent", ""),
                    "MomentumSignal": latest.get("MomentumSignal", ""),
                    "MomentumColor": latest.get("MomentumColor", ""),
                }
            )

            successes.append(SymbolResult(symbol=symbol, rows=len(enriched_rows), output_path=output_path))
            print(f"[ok] {symbol} -> {output_path} ({len(enriched_rows)} rows)")
        except Exception as exc:
            errors.append(SymbolError(symbol=symbol, message=str(exc)))
            print(f"[fail] {symbol} -> {exc}")

    latest_rows.sort(key=lambda row: row["symbol"])
    write_latest(latest_file, latest_rows)
    write_errors(errors_file, errors)

    print("\nSummary")
    print(f"  length: {length}")
    print(f"  confirm_bars: {confirm_bars}")
    print(f"  min_mom0_pct: {min_mom0_pct}")
    print(f"  min_mom1_pct: {min_mom1_pct}")
    print(f"  symbols: {len(symbols)}")
    print(f"  success: {len(successes)}")
    print(f"  failed:  {len(errors)}")
    print(f"  latest file: {latest_file}")
    print(f"  errors file: {errors_file}")

    return 0 if successes else 1


if __name__ == "__main__":
    raise SystemExit(main())
