#!/usr/bin/env python3
"""Build a ranked recent-buy report from strict TV-match momentum outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from scripts.paths import INDICATORS_MOMENTUM_TV_MATCH_DIR, REPORTS_MOMENTUM_DIR


@dataclass
class RankedSymbol:
    symbol: str
    score: float
    signal_date: str
    bars_since_signal: int
    recent_signals: int
    close: float
    ema_fast: float | None
    ema_slow: float | None
    close_vs_ema_fast_pct: float | None
    close_vs_ema_slow_pct: float | None
    ema_fast_slope_pct: float
    ema_slow_slope_pct: float
    atr_pct: float | None
    mom0_signal: float | None
    mom1_signal: float | None
    range_pos_252: float | None
    trend_status: str


@dataclass
class SymbolError:
    symbol: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(INDICATORS_MOMENTUM_TV_MATCH_DIR / "daily"),
        help="Directory with momentum tv-match CSV files",
    )
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol override")
    parser.add_argument(
        "--event",
        default="MomLE",
        help="Event name treated as a buy signal (default: MomLE)",
    )
    parser.add_argument(
        "--window-bars",
        type=int,
        default=5,
        help="Lookback window (bars) used to detect recent buy signals",
    )
    parser.add_argument(
        "--ema-fast-period",
        type=int,
        default=50,
        help="Fast EMA period used in ranking context",
    )
    parser.add_argument(
        "--ema-slow-period",
        type=int,
        default=200,
        help="Slow EMA period used in ranking context",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR period used in volatility normalization",
    )
    parser.add_argument(
        "--slope-bars",
        type=int,
        default=20,
        help="Lookback bars used for EMA slope percentage",
    )
    parser.add_argument(
        "--out-csv",
        default=str(REPORTS_MOMENTUM_DIR / "recent_momentum_buys_5d.csv"),
        help="CSV output path",
    )
    parser.add_argument(
        "--out-md",
        default=str(REPORTS_MOMENTUM_DIR / "recent_momentum_buys_5d.md"),
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


def compute_atr_wilder(rows: list[dict[str, str]], period: int) -> list[float | None]:
    output: list[float | None] = [None] * len(rows)
    if period <= 0 or len(rows) < period:
        return output

    tr_values: list[float | None] = [None] * len(rows)
    prev_close: float | None = None
    for idx, row in enumerate(rows):
        high = parse_float(row.get("High"))
        low = parse_float(row.get("Low"))
        close = parse_float(row.get("Close"))
        if high is None or low is None or close is None:
            prev_close = close
            continue
        if prev_close is None:
            tr_values[idx] = high - low
        else:
            tr_values[idx] = max(high - low, abs(high - prev_close), abs(low - prev_close))
        prev_close = close

    seed_window = tr_values[:period]
    if any(value is None for value in seed_window):
        return output

    seed = sum(float(value) for value in seed_window) / period
    output[period - 1] = seed
    prev = seed
    for idx in range(period, len(rows)):
        tr = tr_values[idx]
        if tr is None:
            continue
        prev = ((prev * (period - 1)) + tr) / period
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


def pct_vs(reference: float | None, value: float) -> float | None:
    if reference is None or reference == 0:
        return None
    return ((value - reference) / reference) * 100.0


def trend_status(close: float, ema_fast: float | None, ema_slow: float | None) -> str:
    if ema_fast is None or ema_slow is None:
        return "UNKNOWN"
    if close > ema_slow and close > ema_fast and ema_fast > ema_slow:
        return "ALIGNED"
    if close > ema_slow and close > ema_fast:
        return "MIXED"
    return "WEAK"


def score_symbol(
    *,
    close: float,
    ema_fast: float | None,
    ema_slow: float | None,
    ema_fast_slope: float,
    ema_slow_slope: float,
    atr: float | None,
    mom0_signal: float | None,
    mom1_signal: float | None,
    range_pos_252: float | None,
    bars_since_signal: int,
    window_bars: int,
) -> float:
    score = 0.0

    if ema_slow is not None:
        score += 25.0 if close > ema_slow else -25.0
    if ema_fast is not None:
        score += 20.0 if close > ema_fast else -20.0
    if ema_fast is not None and ema_slow is not None:
        score += 15.0 if ema_fast > ema_slow else -15.0

    score += clamp(ema_fast_slope * 4.0, -10.0, 10.0)
    score += clamp(ema_slow_slope * 8.0, -10.0, 10.0)

    close_vs_fast = pct_vs(ema_fast, close)
    if close_vs_fast is not None and close_vs_fast > 20.0:
        score -= clamp((close_vs_fast - 20.0) * 0.7, 0.0, 15.0)

    atr_pct = None if atr is None or close == 0 else (atr / close) * 100.0
    if atr_pct is not None and atr_pct > 4.0:
        score -= clamp((atr_pct - 4.0) * 2.0, 0.0, 12.0)

    if atr is not None and atr > 0:
        if mom0_signal is not None:
            score += clamp((mom0_signal / atr) * 6.0, -8.0, 8.0)
        if mom1_signal is not None:
            score += clamp((mom1_signal / atr) * 4.0, -6.0, 6.0)

    if range_pos_252 is not None:
        score += (range_pos_252 - 0.5) * 20.0

    freshness_max = max(0, window_bars - 1)
    score += max(0, freshness_max - bars_since_signal) * 1.5

    return score


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_symbol_files(input_dir: Path, symbols_arg: str) -> list[Path]:
    if symbols_arg.strip():
        symbols = [part.strip().upper() for part in symbols_arg.split(",") if part.strip()]
        return [input_dir / f"{symbol}.csv" for symbol in symbols]
    return sorted(input_dir.glob("*.csv"))


def build_ranked_rows(
    *,
    symbol_files: list[Path],
    event_name: str,
    window_bars: int,
    ema_fast_period: int,
    ema_slow_period: int,
    atr_period: int,
    slope_bars: int,
) -> tuple[list[RankedSymbol], list[SymbolError]]:
    ranked: list[RankedSymbol] = []
    errors: list[SymbolError] = []

    for csv_path in symbol_files:
        symbol = csv_path.stem.upper()
        if not csv_path.exists():
            errors.append(SymbolError(symbol=symbol, message=f"missing file: {csv_path}"))
            continue

        rows = load_rows(csv_path)
        if len(rows) < window_bars:
            continue

        required = {"Date", "High", "Low", "Close", "Event"}
        missing = required.difference(rows[0].keys())
        if missing:
            errors.append(SymbolError(symbol=symbol, message=f"missing columns: {', '.join(sorted(missing))}"))
            continue

        closes: list[float] = []
        for row in rows:
            close = parse_float(row.get("Close"))
            if close is None:
                closes = []
                break
            closes.append(close)
        if not closes:
            errors.append(SymbolError(symbol=symbol, message="invalid close series"))
            continue

        start_idx = max(0, len(rows) - window_bars)
        recent_indices = list(range(start_idx, len(rows)))
        event_indices = [idx for idx in recent_indices if (rows[idx].get("Event") or "").strip() == event_name]
        if not event_indices:
            continue

        last_signal_idx = event_indices[-1]
        recent_count = len(event_indices)
        last_idx = len(rows) - 1

        ema_fast_series = compute_ema(closes, ema_fast_period)
        ema_slow_series = compute_ema(closes, ema_slow_period)
        atr_series = compute_atr_wilder(rows, atr_period)

        close_last = closes[last_idx]
        ema_fast_last = ema_fast_series[last_idx]
        ema_slow_last = ema_slow_series[last_idx]
        atr_last = atr_series[last_idx]

        ema_fast_slope = slope_pct(ema_fast_series, last_idx, slope_bars)
        ema_slow_slope = slope_pct(ema_slow_series, last_idx, slope_bars)
        close_vs_ema_fast_pct = pct_vs(ema_fast_last, close_last)
        close_vs_ema_slow_pct = pct_vs(ema_slow_last, close_last)
        atr_pct = None if atr_last is None or close_last == 0 else (atr_last / close_last) * 100.0

        lookback_window = closes[-252:] if len(closes) >= 252 else closes
        low_252 = min(lookback_window)
        high_252 = max(lookback_window)
        range_pos_252 = None
        if high_252 > low_252:
            range_pos_252 = (close_last - low_252) / (high_252 - low_252)

        mom0_signal = parse_float(rows[last_signal_idx].get("MOM0"))
        mom1_signal = parse_float(rows[last_signal_idx].get("MOM1"))
        bars_since_signal = last_idx - last_signal_idx

        score = score_symbol(
            close=close_last,
            ema_fast=ema_fast_last,
            ema_slow=ema_slow_last,
            ema_fast_slope=ema_fast_slope,
            ema_slow_slope=ema_slow_slope,
            atr=atr_last,
            mom0_signal=mom0_signal,
            mom1_signal=mom1_signal,
            range_pos_252=range_pos_252,
            bars_since_signal=bars_since_signal,
            window_bars=window_bars,
        )

        ranked.append(
            RankedSymbol(
                symbol=symbol,
                score=score,
                signal_date=rows[last_signal_idx]["Date"],
                bars_since_signal=bars_since_signal,
                recent_signals=recent_count,
                close=close_last,
                ema_fast=ema_fast_last,
                ema_slow=ema_slow_last,
                close_vs_ema_fast_pct=close_vs_ema_fast_pct,
                close_vs_ema_slow_pct=close_vs_ema_slow_pct,
                ema_fast_slope_pct=ema_fast_slope,
                ema_slow_slope_pct=ema_slow_slope,
                atr_pct=atr_pct,
                mom0_signal=mom0_signal,
                mom1_signal=mom1_signal,
                range_pos_252=range_pos_252,
                trend_status=trend_status(close_last, ema_fast_last, ema_slow_last),
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.symbol))
    return ranked, errors


def format_opt(value: float | None, places: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{places}f}"


def write_csv(path: Path, rows: list[RankedSymbol]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "Rank",
        "Symbol",
        "Score",
        "SignalDate",
        "BarsSinceSignal",
        "RecentSignalsWindow",
        "TrendStatus",
        "Close",
        "EMAFast",
        "EMASlow",
        "CloseVsEMAFastPct",
        "CloseVsEMASlowPct",
        "EMAFastSlopePct",
        "EMASlowSlopePct",
        "ATRPct",
        "MOM0AtSignal",
        "MOM1AtSignal",
        "RangePos252",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "Rank": str(rank),
                    "Symbol": row.symbol,
                    "Score": f"{row.score:.4f}",
                    "SignalDate": row.signal_date,
                    "BarsSinceSignal": str(row.bars_since_signal),
                    "RecentSignalsWindow": str(row.recent_signals),
                    "TrendStatus": row.trend_status,
                    "Close": f"{row.close:.6f}",
                    "EMAFast": format_opt(row.ema_fast, places=6),
                    "EMASlow": format_opt(row.ema_slow, places=6),
                    "CloseVsEMAFastPct": format_opt(row.close_vs_ema_fast_pct, places=4),
                    "CloseVsEMASlowPct": format_opt(row.close_vs_ema_slow_pct, places=4),
                    "EMAFastSlopePct": f"{row.ema_fast_slope_pct:.4f}",
                    "EMASlowSlopePct": f"{row.ema_slow_slope_pct:.4f}",
                    "ATRPct": format_opt(row.atr_pct, places=4),
                    "MOM0AtSignal": format_opt(row.mom0_signal, places=6),
                    "MOM1AtSignal": format_opt(row.mom1_signal, places=6),
                    "RangePos252": format_opt(row.range_pos_252, places=4),
                }
            )


def write_markdown(
    *,
    path: Path,
    rows: list[RankedSymbol],
    errors: list[SymbolError],
    input_dir: Path,
    event_name: str,
    window_bars: int,
    ema_fast_period: int,
    ema_slow_period: int,
    atr_period: int,
    slope_bars: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Recent Momentum Buy Signals (Ranked)")
    lines.append("")
    lines.append(f"- Source: `{input_dir}`")
    lines.append(f"- Recent signal rule: `{event_name}` in the last **{window_bars}** bars")
    lines.append(f"- Ranked symbols: **{len(rows)}**")
    lines.append("- Score intent: higher implies better continuation quality (trend + momentum + volatility/extension controls)")
    lines.append("")
    lines.append("Scoring components:")
    lines.append(f"- Trend alignment using `Close`, `EMA_{ema_fast_period}`, `EMA_{ema_slow_period}`")
    lines.append(f"- EMA slope strength over `{slope_bars}` bars")
    lines.append(f"- Volatility normalization and penalty via `ATR({atr_period})`")
    lines.append("- Penalty for extreme extension above fast EMA")
    lines.append("- Freshness bonus for very recent signals")
    lines.append("")

    if rows:
        lines.append(
            "| Rank | Symbol | Score | Signal Date | Bars Since | Recent Signals | Trend | ATR % | Close vs EMA200 % |"
        )
        lines.append("|---:|---|---:|---|---:|---:|---|---:|---:|")
        for rank, row in enumerate(rows, start=1):
            lines.append(
                "| "
                + f"{rank} | {row.symbol} | {row.score:.2f} | {row.signal_date} | "
                + f"{row.bars_since_signal} | {row.recent_signals} | {row.trend_status} | "
                + f"{format_opt(row.atr_pct, 2)} | {format_opt(row.close_vs_ema_slow_pct, 2)} |"
            )
    else:
        lines.append("No symbols matched the recent signal filter.")

    if errors:
        lines.append("")
        lines.append("Errors / skipped files:")
        for error in errors:
            lines.append(f"- `{error.symbol}`: {error.message}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)

    if args.window_bars <= 0:
        print("[error] --window-bars must be > 0")
        return 1
    if args.ema_fast_period <= 0 or args.ema_slow_period <= 0:
        print("[error] EMA periods must be > 0")
        return 1
    if args.atr_period <= 0:
        print("[error] --atr-period must be > 0")
        return 1
    if args.slope_bars <= 0:
        print("[error] --slope-bars must be > 0")
        return 1
    if not input_dir.exists():
        print(f"[error] input directory not found: {input_dir}")
        return 1

    symbol_files = resolve_symbol_files(input_dir=input_dir, symbols_arg=args.symbols)
    ranked_rows, errors = build_ranked_rows(
        symbol_files=symbol_files,
        event_name=args.event.strip(),
        window_bars=args.window_bars,
        ema_fast_period=args.ema_fast_period,
        ema_slow_period=args.ema_slow_period,
        atr_period=args.atr_period,
        slope_bars=args.slope_bars,
    )

    write_csv(path=out_csv, rows=ranked_rows)
    write_markdown(
        path=out_md,
        rows=ranked_rows,
        errors=errors,
        input_dir=input_dir,
        event_name=args.event.strip(),
        window_bars=args.window_bars,
        ema_fast_period=args.ema_fast_period,
        ema_slow_period=args.ema_slow_period,
        atr_period=args.atr_period,
        slope_bars=args.slope_bars,
    )

    print(f"[ok] CSV: {out_csv}")
    print(f"[ok] MD:  {out_md}")
    print(f"[summary] scanned={len(symbol_files)} ranked={len(ranked_rows)} errors={len(errors)}")
    if ranked_rows:
        print(
            "[top] "
            + ", ".join(
                f"{idx + 1}:{item.symbol} ({item.score:.2f})"
                for idx, item in enumerate(ranked_rows[:5])
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
