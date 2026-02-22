#!/usr/bin/env python3
"""Run automated APO-only hardening validation for mtf_entry_exit_v1."""

from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from scripts.paths import DATA_DAILY_DIR, STRATEGIES_HARDENING_DIR


@dataclass
class RunMetrics:
    label: str
    total_return_pct: float
    ending_equity: float
    max_drawdown_pct: float
    trade_executions: int
    round_trips: int
    win_rate_pct: float
    avg_hold_bars: float
    hold_count: int
    score: float


@dataclass
class Segment:
    name: str
    start: date
    end: date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="APO", help="Symbol to validate (default: APO)")
    parser.add_argument("--input-file", default=str(DATA_DAILY_DIR / "APO.csv"), help="Daily source CSV file")
    parser.add_argument(
        "--strategy-script",
        default="scripts/strategies/mtf_entry_exit_v1.py",
        help="Path to strategy script",
    )
    parser.add_argument("--out-dir", default=str(STRATEGIES_HARDENING_DIR), help="Directory for hardening artifacts")
    parser.add_argument(
        "--min-segment-bars",
        type=int,
        default=120,
        help="Minimum bars required for a walk-forward segment",
    )
    parser.add_argument(
        "--max-dd-worsen-pct",
        type=float,
        default=2.0,
        help="Max allowed full-sample drawdown worsening in percentage points vs baseline",
    )
    parser.add_argument(
        "--min-trade-ratio",
        type=float,
        default=0.50,
        help="Min allowed trade count ratio vs baseline",
    )
    parser.add_argument(
        "--max-trade-ratio",
        type=float,
        default=1.75,
        help="Max allowed trade count ratio vs baseline",
    )
    parser.add_argument(
        "--min-hold-ratio",
        type=float,
        default=0.80,
        help="Min allowed avg hold-bar ratio vs baseline",
    )
    return parser.parse_args()


def parse_float(value: str | None) -> float:
    raw = (value or "").strip()
    if not raw:
        return 0.0
    return float(raw)


def parse_int(value: str | None) -> int:
    raw = (value or "").strip()
    if not raw:
        return 0
    return int(float(raw))


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def filter_rows_by_date(rows: list[dict[str, str]], start: date, end: date) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        row_date = date.fromisoformat(row["Date"])
        if start <= row_date <= end:
            filtered.append(row)
    return filtered


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_hold_stats(strategy_rows: list[dict[str, str]]) -> tuple[float, int]:
    buy_indices: list[int] = []
    hold_bars: list[int] = []
    for idx, row in enumerate(strategy_rows):
        action = (row.get("ExecutedAction") or "").strip()
        if action == "BUY":
            buy_indices.append(idx)
        elif action == "SELL" and buy_indices:
            entry_idx = buy_indices.pop(0)
            hold_bars.append(idx - entry_idx)
    if not hold_bars:
        return 0.0, 0
    return sum(hold_bars) / len(hold_bars), len(hold_bars)


def metric_score(total_return_pct: float, max_drawdown_pct: float) -> float:
    return total_return_pct - (0.35 * max_drawdown_pct)


def run_variant(
    *,
    root: Path,
    strategy_script: Path,
    symbol: str,
    source_rows: list[dict[str, str]],
    segment: Segment,
    label: str,
    variant_args: list[str],
) -> RunMetrics:
    scoped_rows = filter_rows_by_date(source_rows, start=segment.start, end=segment.end)
    if len(scoped_rows) < 2:
        raise ValueError(f"{label}/{segment.name}: not enough rows in date range")

    with tempfile.TemporaryDirectory(prefix="mtf_v1_harden_") as temp_dir_raw:
        temp_dir = Path(temp_dir_raw)
        input_dir = temp_dir / "input"
        out_dir = temp_dir / "out"
        summary_file = temp_dir / "summary.csv"
        latest_file = temp_dir / "latest.csv"
        errors_file = temp_dir / "errors.csv"
        watchlist_file = temp_dir / "watchlist.txt"

        input_dir.mkdir(parents=True, exist_ok=True)
        write_rows(input_dir / f"{symbol}.csv", scoped_rows)
        watchlist_file.write_text(f"{symbol}\n", encoding="utf-8")

        cmd = [
            sys.executable,
            str(strategy_script),
            "--watchlist",
            str(watchlist_file),
            "--input-dir",
            str(input_dir),
            "--out-dir",
            str(out_dir),
            "--summary-file",
            str(summary_file),
            "--latest-file",
            str(latest_file),
            "--errors-file",
            str(errors_file),
        ] + variant_args

        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"{label}/{segment.name} failed: {proc.stderr or proc.stdout}")

        summary_rows = read_rows(summary_file)
        if not summary_rows:
            raise ValueError(f"{label}/{segment.name}: missing summary rows")
        row = summary_rows[0]

        strategy_rows = read_rows(out_dir / f"{symbol}.csv")
        avg_hold_bars, hold_count = compute_hold_stats(strategy_rows)

        total_return_pct = parse_float(row.get("TotalReturnPct"))
        max_drawdown_pct = parse_float(row.get("MaxDrawdownPct"))
        return RunMetrics(
            label=label,
            total_return_pct=total_return_pct,
            ending_equity=parse_float(row.get("EndingEquity")),
            max_drawdown_pct=max_drawdown_pct,
            trade_executions=parse_int(row.get("TradeExecutions")),
            round_trips=parse_int(row.get("RoundTrips")),
            win_rate_pct=parse_float(row.get("WinRatePct")),
            avg_hold_bars=avg_hold_bars,
            hold_count=hold_count,
            score=metric_score(total_return_pct, max_drawdown_pct),
        )


def build_segments(source_rows: list[dict[str, str]], min_segment_bars: int) -> list[Segment]:
    dates = [date.fromisoformat(row["Date"]) for row in source_rows]
    first_date = min(dates)
    last_date = max(dates)

    candidates = [
        Segment(name="seg_2023", start=max(first_date, date(2023, 1, 1)), end=min(last_date, date(2023, 12, 31))),
        Segment(name="seg_2024", start=max(first_date, date(2024, 1, 1)), end=min(last_date, date(2024, 12, 31))),
        Segment(name="seg_2025_plus", start=max(first_date, date(2025, 1, 1)), end=last_date),
    ]

    valid: list[Segment] = []
    for segment in candidates:
        if segment.start > segment.end:
            continue
        count = 0
        for row in source_rows:
            row_date = date.fromisoformat(row["Date"])
            if segment.start <= row_date <= segment.end:
                count += 1
        if count >= min_segment_bars:
            valid.append(segment)
    return valid


def variant_grid() -> list[tuple[str, list[str]]]:
    variants: list[tuple[str, list[str]]] = [("baseline", [])]

    cross_lookbacks = [10, 15, 20]
    trend_fails = [2, 3]
    atr_mults = [2.0, 2.5, 3.0]
    dd_thresholds = [15.0, 20.0, 25.0]

    for lookback, trend_fail, atr_mult, dd_thresh in itertools.product(
        cross_lookbacks,
        trend_fails,
        atr_mults,
        dd_thresholds,
    ):
        label = (
            f"lb{lookback}_tf{trend_fail}_atr{atr_mult:.1f}_dd{dd_thresh:.0f}"
        )
        args = [
            "--entry-cross-lookback-bars",
            str(lookback),
            "--trend-fail-bars",
            str(trend_fail),
            "--atr-mult",
            str(atr_mult),
            "--kill-max-drawdown-pct",
            str(dd_thresh),
            "--equity-kill-mode",
            "both",
        ]
        variants.append((label, args))
    return variants


def write_leaderboard(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("No leaderboard rows to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    symbol = args.symbol.upper()
    source_rows = read_rows(Path(args.input_file))

    strategy_script = Path(args.strategy_script)
    if not strategy_script.exists():
        print(f"[error] strategy script not found: {strategy_script}")
        return 1

    full_segment = Segment(
        name="full",
        start=date.fromisoformat(source_rows[0]["Date"]),
        end=date.fromisoformat(source_rows[-1]["Date"]),
    )
    wf_segments = build_segments(source_rows, min_segment_bars=args.min_segment_bars)
    if len(wf_segments) < 2:
        print("[error] Need at least 2 valid walk-forward segments")
        return 1

    variants = variant_grid()

    # Baseline metrics for full sample and each segment.
    baseline_label, baseline_args = variants[0]
    baseline_full = run_variant(
        root=root,
        strategy_script=strategy_script,
        symbol=symbol,
        source_rows=source_rows,
        segment=full_segment,
        label=baseline_label,
        variant_args=baseline_args,
    )
    baseline_wf = {
        segment.name: run_variant(
            root=root,
            strategy_script=strategy_script,
            symbol=symbol,
            source_rows=source_rows,
            segment=segment,
            label=baseline_label,
            variant_args=baseline_args,
        )
        for segment in wf_segments
    }

    leaderboard: list[dict[str, str]] = []
    for label, variant_args in variants:
        full = run_variant(
            root=root,
            strategy_script=strategy_script,
            symbol=symbol,
            source_rows=source_rows,
            segment=full_segment,
            label=label,
            variant_args=variant_args,
        )

        wf_wins = 0
        wf_scores: list[float] = []
        for segment in wf_segments:
            current = run_variant(
                root=root,
                strategy_script=strategy_script,
                symbol=symbol,
                source_rows=source_rows,
                segment=segment,
                label=label,
                variant_args=variant_args,
            )
            wf_scores.append(current.score)
            if current.score > baseline_wf[segment.name].score:
                wf_wins += 1

        dd_limit = baseline_full.max_drawdown_pct + args.max_dd_worsen_pct
        min_trade = baseline_full.trade_executions * args.min_trade_ratio
        max_trade = baseline_full.trade_executions * args.max_trade_ratio
        min_hold = baseline_full.avg_hold_bars * args.min_hold_ratio

        full_pass = (
            full.ending_equity > baseline_full.ending_equity
            and full.max_drawdown_pct <= dd_limit
            and min_trade <= full.trade_executions <= max_trade
            and full.avg_hold_bars >= min_hold
        )
        wf_pass = wf_wins >= max(2, len(wf_segments) - 1)
        pass_all = full_pass and wf_pass

        leaderboard.append(
            {
                "label": label,
                "pass_all": "1" if pass_all else "0",
                "full_pass": "1" if full_pass else "0",
                "wf_pass": "1" if wf_pass else "0",
                "wf_wins": str(wf_wins),
                "wf_segments": str(len(wf_segments)),
                "ending_equity": f"{full.ending_equity:.2f}",
                "baseline_equity": f"{baseline_full.ending_equity:.2f}",
                "total_return_pct": f"{full.total_return_pct:.4f}",
                "max_drawdown_pct": f"{full.max_drawdown_pct:.4f}",
                "baseline_max_drawdown_pct": f"{baseline_full.max_drawdown_pct:.4f}",
                "trade_executions": str(full.trade_executions),
                "baseline_trade_executions": str(baseline_full.trade_executions),
                "avg_hold_bars": f"{full.avg_hold_bars:.4f}",
                "baseline_avg_hold_bars": f"{baseline_full.avg_hold_bars:.4f}",
                "score": f"{full.score:.4f}",
                "baseline_score": f"{baseline_full.score:.4f}",
                "variant_args": " ".join(variant_args),
            }
        )

    leaderboard.sort(
        key=lambda row: (
            int(row["pass_all"]),
            float(row["ending_equity"]),
            float(row["score"]),
        ),
        reverse=True,
    )

    out_dir = Path(args.out_dir)
    write_leaderboard(out_dir / "mtf_v1_leaderboard.csv", leaderboard)

    passing = [row for row in leaderboard if row["pass_all"] == "1"]
    print("Hardening Validation Summary")
    print(f"  symbol: {symbol}")
    print(f"  variants tested: {len(leaderboard)}")
    print(f"  pass_all variants: {len(passing)}")
    print(f"  baseline ending equity: {baseline_full.ending_equity:.2f}")
    print(f"  baseline max drawdown %: {baseline_full.max_drawdown_pct:.4f}")
    print(f"  baseline trades: {baseline_full.trade_executions}")
    print(f"  baseline avg hold bars: {baseline_full.avg_hold_bars:.4f}")
    print(f"  leaderboard: {out_dir / 'mtf_v1_leaderboard.csv'}")

    print("\nTop 10 Variants")
    print("label | pass_all | wf_wins | ending_equity | max_drawdown_pct | trades | avg_hold_bars | score")
    for row in leaderboard[:10]:
        print(
            f"{row['label']} | {row['pass_all']} | {row['wf_wins']}/{row['wf_segments']} | "
            f"{row['ending_equity']} | {row['max_drawdown_pct']} | {row['trade_executions']} | "
            f"{row['avg_hold_bars']} | {row['score']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
