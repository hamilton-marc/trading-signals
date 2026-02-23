#!/usr/bin/env python3
"""Build a weekly trend watchlist where no fresh MomLE signal appeared recently."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from scripts.paths import INDICATORS_MOMENTUM_TV_MATCH_DIR, REPORTS_MOMENTUM_DIR


@dataclass
class TrendRow:
    symbol: str
    score: float
    latest_date: str
    close: float
    ema_fast: float
    ema_slow: float
    close_vs_ema_fast_pct: float
    close_vs_ema_slow_pct: float
    ema_fast_slope_pct: float
    ema_slow_slope_pct: float
    last_momle_date: str
    bars_since_last_momle: int
    recent_signals_window: int


@dataclass
class SymbolError:
    symbol: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(INDICATORS_MOMENTUM_TV_MATCH_DIR / "weekly"),
        help="Directory with weekly momentum tv-match CSV files",
    )
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol override")
    parser.add_argument("--event", default="MomLE", help="Event name treated as buy signal")
    parser.add_argument(
        "--window-bars",
        type=int,
        default=2,
        help="Lookback window (weekly bars) treated as recent (2 ~= 10 trading days)",
    )
    parser.add_argument("--ema-fast-period", type=int, default=50, help="Fast EMA period")
    parser.add_argument("--ema-slow-period", type=int, default=200, help="Slow EMA period")
    parser.add_argument("--slope-bars", type=int, default=8, help="Slope lookback bars")
    parser.add_argument(
        "--require-slow-slope-up",
        action="store_true",
        help="Require slow EMA slope to be positive",
    )
    parser.add_argument(
        "--out-csv",
        default=str(REPORTS_MOMENTUM_DIR / "weekly_trend_no_recent_momle_10d.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--out-md",
        default=str(REPORTS_MOMENTUM_DIR / "weekly_trend_no_recent_momle_10d.md"),
        help="Markdown output path",
    )
    return parser.parse_args()


def parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return float(raw)


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def compute_ema(values: list[float], period: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return output
    alpha = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period
    output[period - 1] = seed
    prev = seed
    for idx in range(period, len(values)):
        prev = (values[idx] - prev) * alpha + prev
        output[idx] = prev
    return output


def slope_pct(series: list[float | None], idx: int, lookback: int) -> float:
    if lookback <= 0 or idx - lookback < 0:
        return 0.0
    now = series[idx]
    prev = series[idx - lookback]
    if now is None or prev is None or prev == 0:
        return 0.0
    return ((now - prev) / prev) * 100.0


def pct_vs(reference: float, value: float) -> float:
    if reference == 0:
        return 0.0
    return ((value - reference) / reference) * 100.0


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_symbol_files(input_dir: Path, symbols_arg: str) -> list[Path]:
    if symbols_arg.strip():
        symbols = [part.strip().upper() for part in symbols_arg.split(",") if part.strip()]
        return [input_dir / f"{symbol}.csv" for symbol in symbols]
    return sorted(input_dir.glob("*.csv"))


def score_symbol(
    *,
    close_vs_fast: float,
    close_vs_slow: float,
    fast_slope_pct: float,
    slow_slope_pct: float,
    bars_since_last_momle: int,
) -> float:
    score = 0.0
    score += clamp(close_vs_fast, 0.0, 20.0) * 0.9
    score += clamp(close_vs_slow, 0.0, 35.0) * 1.2
    score += clamp(fast_slope_pct, 0.0, 8.0) * 5.0
    score += clamp(slow_slope_pct, 0.0, 6.0) * 3.0
    # Keep names with a not-too-stale prior MomLE near the top while still allowing older setups.
    score += max(0.0, 30.0 - (bars_since_last_momle * 0.8))
    return score


def build_rows(
    *,
    symbol_files: list[Path],
    event_name: str,
    window_bars: int,
    ema_fast_period: int,
    ema_slow_period: int,
    slope_bars: int,
    require_slow_slope_up: bool,
) -> tuple[list[TrendRow], list[SymbolError]]:
    rows: list[TrendRow] = []
    errors: list[SymbolError] = []

    for csv_path in symbol_files:
        symbol = csv_path.stem.upper()
        if not csv_path.exists():
            errors.append(SymbolError(symbol=symbol, message=f"Missing file: {csv_path}"))
            continue

        try:
            ohlc = load_rows(csv_path)
        except Exception as exc:
            errors.append(SymbolError(symbol=symbol, message=f"Read error: {exc}"))
            continue

        if len(ohlc) < max(window_bars, ema_slow_period, slope_bars + 1):
            continue

        closes: list[float] = []
        dates: list[str] = []
        events: list[str] = []
        valid = True
        for raw in ohlc:
            close = parse_float(raw.get("Close"))
            date_value = (raw.get("Date") or "").strip()
            if close is None or not date_value:
                valid = False
                break
            closes.append(close)
            dates.append(date_value)
            events.append((raw.get("Event") or "").strip())
        if not valid:
            errors.append(SymbolError(symbol=symbol, message="Missing Date/Close values"))
            continue

        recent_slice = events[-window_bars:]
        recent_buy_count = sum(1 for value in recent_slice if value == event_name)
        if recent_buy_count > 0:
            continue

        last_signal_idx: int | None = None
        for idx in range(len(events) - 1, -1, -1):
            if events[idx] == event_name:
                last_signal_idx = idx
                break
        if last_signal_idx is None:
            # This watchlist is for continuation setups that had a prior momentum long event.
            continue

        ema_fast_series = compute_ema(closes, ema_fast_period)
        ema_slow_series = compute_ema(closes, ema_slow_period)
        latest_idx = len(closes) - 1
        ema_fast = ema_fast_series[latest_idx]
        ema_slow = ema_slow_series[latest_idx]
        if ema_fast is None or ema_slow is None:
            continue

        close = closes[latest_idx]
        fast_slope = slope_pct(ema_fast_series, latest_idx, slope_bars)
        slow_slope = slope_pct(ema_slow_series, latest_idx, slope_bars)

        # Strong obvious trend filter.
        if close <= ema_fast:
            continue
        if ema_fast <= ema_slow:
            continue
        if fast_slope <= 0:
            continue
        if require_slow_slope_up and slow_slope <= 0:
            continue

        close_vs_fast = pct_vs(ema_fast, close)
        close_vs_slow = pct_vs(ema_slow, close)
        bars_since = latest_idx - last_signal_idx
        score = score_symbol(
            close_vs_fast=close_vs_fast,
            close_vs_slow=close_vs_slow,
            fast_slope_pct=fast_slope,
            slow_slope_pct=slow_slope,
            bars_since_last_momle=bars_since,
        )

        rows.append(
            TrendRow(
                symbol=symbol,
                score=score,
                latest_date=dates[latest_idx],
                close=close,
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                close_vs_ema_fast_pct=close_vs_fast,
                close_vs_ema_slow_pct=close_vs_slow,
                ema_fast_slope_pct=fast_slope,
                ema_slow_slope_pct=slow_slope,
                last_momle_date=dates[last_signal_idx],
                bars_since_last_momle=bars_since,
                recent_signals_window=recent_buy_count,
            )
        )

    rows.sort(key=lambda item: item.score, reverse=True)
    return rows, errors


def write_csv(path: Path, rows: list[TrendRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "Rank",
        "Symbol",
        "Score",
        "LatestDate",
        "Close",
        "EMAFast",
        "EMASlow",
        "CloseVsEMAFastPct",
        "CloseVsEMASlowPct",
        "EMAFastSlopePct",
        "EMASlowSlopePct",
        "LastMomLEDate",
        "BarsSinceLastMomLE",
        "RecentSignalsWindow",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "Rank": str(idx),
                    "Symbol": row.symbol,
                    "Score": f"{row.score:.4f}",
                    "LatestDate": row.latest_date,
                    "Close": f"{row.close:.6f}",
                    "EMAFast": f"{row.ema_fast:.6f}",
                    "EMASlow": f"{row.ema_slow:.6f}",
                    "CloseVsEMAFastPct": f"{row.close_vs_ema_fast_pct:.4f}",
                    "CloseVsEMASlowPct": f"{row.close_vs_ema_slow_pct:.4f}",
                    "EMAFastSlopePct": f"{row.ema_fast_slope_pct:.4f}",
                    "EMASlowSlopePct": f"{row.ema_slow_slope_pct:.4f}",
                    "LastMomLEDate": row.last_momle_date,
                    "BarsSinceLastMomLE": str(row.bars_since_last_momle),
                    "RecentSignalsWindow": str(row.recent_signals_window),
                }
            )


def write_markdown(
    *,
    path: Path,
    rows: list[TrendRow],
    input_dir: Path,
    event_name: str,
    window_bars: int,
) -> None:
    lines: list[str] = []
    lines.append("# Weekly Trend Watchlist (No Recent MomLE)")
    lines.append("")
    lines.append(f"- Source: `{input_dir}`")
    lines.append(f"- Filter: no `{event_name}` in last **{window_bars}** weekly bars")
    lines.append(
        "- Strong trend rule: close > EMA50, EMA50 > EMA200, EMA50 slope > 0"
    )
    lines.append(f"- Ranked symbols: **{len(rows)}**")
    lines.append("")
    lines.append(
        "| Rank | Symbol | Score | Last MomLE | Bars Since | Latest Week | Close vs EMA50 % | EMA50 Slope % | Close vs EMA200 % |"
    )
    lines.append("|---:|---|---:|---|---:|---|---:|---:|---:|")
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"| {idx} | {row.symbol} | {row.score:.2f} | {row.last_momle_date} | {row.bars_since_last_momle} | "
            f"{row.latest_date} | {row.close_vs_ema_fast_pct:.2f} | {row.ema_fast_slope_pct:.2f} | {row.close_vs_ema_slow_pct:.2f} |"
        )
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"[error] Input directory not found: {input_dir}")
        return 1
    if args.window_bars <= 0:
        print("[error] --window-bars must be > 0")
        return 1
    if args.ema_fast_period <= 0 or args.ema_slow_period <= 0:
        print("[error] --ema-fast-period and --ema-slow-period must be > 0")
        return 1
    if args.ema_fast_period >= args.ema_slow_period:
        print("[error] --ema-fast-period should be smaller than --ema-slow-period")
        return 1
    if args.slope_bars <= 0:
        print("[error] --slope-bars must be > 0")
        return 1

    symbol_files = resolve_symbol_files(input_dir, args.symbols)
    ranked, errors = build_rows(
        symbol_files=symbol_files,
        event_name=args.event.strip(),
        window_bars=args.window_bars,
        ema_fast_period=args.ema_fast_period,
        ema_slow_period=args.ema_slow_period,
        slope_bars=args.slope_bars,
        require_slow_slope_up=bool(args.require_slow_slope_up),
    )

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    write_csv(out_csv, ranked)
    write_markdown(
        path=out_md,
        rows=ranked,
        input_dir=input_dir,
        event_name=args.event.strip(),
        window_bars=args.window_bars,
    )

    print(f"[ok] CSV: {out_csv}")
    print(f"[ok] MD:  {out_md}")
    print(f"[summary] scanned={len(symbol_files)} ranked={len(ranked)} errors={len(errors)}")
    if ranked:
        top = ", ".join(f"{row.symbol} ({row.score:.2f})" for row in ranked[:5])
        print(f"[top] {top}")
    if errors:
        print("[errors]")
        for err in errors[:20]:
            print(f"  - {err.symbol}: {err.message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
