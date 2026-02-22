#!/usr/bin/env python3
"""Quantify v2 trigger quality across symbols with forward-return probabilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from scripts.paths import STRATEGIES_MTF_V2_DIR, STRATEGIES_SIGNAL_QUALITY_V2_DIR


@dataclass
class StudyConfig:
    symbols: list[str]
    input_dir: Path
    out_dir: Path
    event_mode: str
    horizons: list[int]
    align_window: bool
    min_samples: int
    date_from: str | None
    date_to: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", default="watchlist.txt", help="Path to watchlist")
    parser.add_argument(
        "--symbols",
        default="",
        help="Comma-separated symbols; if omitted, read from watchlist",
    )
    parser.add_argument("--input-dir", default=str(STRATEGIES_MTF_V2_DIR), help="Directory with per-symbol strategy CSVs")
    parser.add_argument("--out-dir", default=str(STRATEGIES_SIGNAL_QUALITY_V2_DIR), help="Output directory")
    parser.add_argument(
        "--event-mode",
        choices=["trigger_edge_flat", "trigger_edge_all", "entry_setup_edge"],
        default="trigger_edge_flat",
        help="Event extraction mode",
    )
    parser.add_argument(
        "--horizons",
        default="5,10,20,40",
        help="Forward horizons in trading bars, comma-separated",
    )
    parser.add_argument(
        "--align-window",
        action="store_true",
        default=True,
        help="Use common date intersection across symbols (default: true)",
    )
    parser.add_argument(
        "--no-align-window",
        dest="align_window",
        action="store_false",
        help="Disable common-window alignment",
    )
    parser.add_argument("--min-samples", type=int, default=15, help="Minimum samples for ranked context tables")
    parser.add_argument("--date-from", default="", help="Optional inclusive start date YYYY-MM-DD")
    parser.add_argument("--date-to", default="", help="Optional inclusive end date YYYY-MM-DD")
    return parser.parse_args()


def read_watchlist(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Watchlist not found: {path}")
    symbols: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        symbols.append(line.upper())
    return symbols


def parse_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols.strip():
        return [item.strip().upper() for item in args.symbols.split(",") if item.strip()]
    return read_watchlist(Path(args.watchlist))


def parse_horizons(raw: str) -> list[int]:
    values: list[int] = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(int(text))
    unique = sorted(set(values))
    if not unique or min(unique) <= 0:
        raise ValueError("Horizon values must be positive integers")
    return unique


def as_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)


def as_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_symbol_frame(path: Path, symbol: str) -> pd.DataFrame:
    csv_path = path / f"{symbol}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing strategy CSV for {symbol}: {csv_path}")
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    required = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "DailyBreakout",
        "PullbackReclaim",
        "EntryTrigger",
        "EntrySetup",
        "MonthlyRiskOn",
        "WeeklyRiskOn",
        "RegimeOn",
        "MomentumPositive",
        "DailyEMA",
        "ATR",
        "PositionState",
    ]
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"{symbol} missing required columns: {', '.join(missing)}")

    df = df.sort_values("Date").reset_index(drop=True)
    df["Symbol"] = symbol
    for col in [
        "DailyBreakout",
        "PullbackReclaim",
        "EntryTrigger",
        "EntrySetup",
        "MonthlyRiskOn",
        "WeeklyRiskOn",
        "RegimeOn",
        "MomentumPositive",
    ]:
        df[col] = as_int(df[col])
    for col in ["Open", "High", "Low", "Close", "DailyEMA", "ATR"]:
        df[col] = as_float(df[col])
    return df


def apply_date_filters(df: pd.DataFrame, date_from: str | None, date_to: str | None) -> pd.DataFrame:
    output = df
    if date_from:
        output = output[output["Date"] >= pd.Timestamp(date_from)]
    if date_to:
        output = output[output["Date"] <= pd.Timestamp(date_to)]
    return output.reset_index(drop=True)


def align_date_window(frames: dict[str, pd.DataFrame]) -> tuple[pd.Timestamp, pd.Timestamp]:
    min_dates = [frame["Date"].min() for frame in frames.values()]
    max_dates = [frame["Date"].max() for frame in frames.values()]
    start = max(min_dates)
    end = min(max_dates)
    if start > end:
        raise ValueError("No overlapping date window across symbols")
    return start, end


def compute_event_mask(df: pd.DataFrame, event_mode: str) -> pd.Series:
    trigger = df["EntryTrigger"] == 1
    setup = df["EntrySetup"] == 1
    trigger_edge = trigger & (~trigger.shift(1, fill_value=False))
    setup_edge = setup & (~setup.shift(1, fill_value=False))
    flat = df["PositionState"].astype(str) == "FLAT"

    if event_mode == "trigger_edge_flat":
        return trigger_edge & flat
    if event_mode == "trigger_edge_all":
        return trigger_edge
    if event_mode == "entry_setup_edge":
        return setup_edge & flat
    raise ValueError(f"Unsupported event mode: {event_mode}")


def trigger_type_from_row(row: pd.Series) -> str:
    breakout = int(row["DailyBreakout"]) == 1
    reclaim = int(row["PullbackReclaim"]) == 1
    if breakout and reclaim:
        return "BOTH"
    if breakout:
        return "BREAKOUT"
    if reclaim:
        return "RECLAIM"
    return "UNKNOWN"


def attach_forward_metrics(events: pd.DataFrame, frame: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    close = frame["Close"]
    high = frame["High"]
    low = frame["Low"]
    output = events.copy()

    for h in horizons:
        fwd_close = close.shift(-h)
        output[f"ret_{h}d"] = (fwd_close.loc[events.index].values / output["Close"].values) - 1.0

        mfe_vals: list[float | None] = []
        mae_vals: list[float | None] = []
        for idx, entry_close in zip(events.index.tolist(), output["Close"].tolist()):
            end_idx = idx + h
            if end_idx >= len(frame) or entry_close <= 0:
                mfe_vals.append(None)
                mae_vals.append(None)
                continue
            window_high = high.iloc[idx + 1 : end_idx + 1].max()
            window_low = low.iloc[idx + 1 : end_idx + 1].min()
            mfe_vals.append((window_high / entry_close) - 1.0)
            mae_vals.append((window_low / entry_close) - 1.0)

        output[f"mfe_{h}d"] = mfe_vals
        output[f"mae_{h}d"] = mae_vals
        output[f"pos_{h}d"] = output[f"ret_{h}d"] > 0.0
        output[f"ge5_{h}d"] = output[f"ret_{h}d"] >= 0.05
    return output


def summarise_groups(
    events: pd.DataFrame,
    group_cols: list[str],
    horizons: list[int],
    min_samples: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = events.groupby(group_cols, dropna=False)
    for key, subset in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row: dict[str, object] = {col: value for col, value in zip(group_cols, key)}
        row["events"] = len(subset)
        for h in horizons:
            ret_col = f"ret_{h}d"
            mfe_col = f"mfe_{h}d"
            mae_col = f"mae_{h}d"
            pos_col = f"pos_{h}d"
            ge5_col = f"ge5_{h}d"

            valid = subset[ret_col].dropna()
            if valid.empty:
                row[f"valid_{h}d"] = 0
                row[f"mean_ret_{h}d"] = None
                row[f"median_ret_{h}d"] = None
                row[f"p_pos_{h}d"] = None
                row[f"p_ge5_{h}d"] = None
                row[f"mean_mfe_{h}d"] = None
                row[f"mean_mae_{h}d"] = None
                continue

            row[f"valid_{h}d"] = len(valid)
            row[f"mean_ret_{h}d"] = float(valid.mean())
            row[f"median_ret_{h}d"] = float(valid.median())
            row[f"p_pos_{h}d"] = float(subset.loc[valid.index, pos_col].mean())
            row[f"p_ge5_{h}d"] = float(subset.loc[valid.index, ge5_col].mean())
            row[f"mean_mfe_{h}d"] = float(subset.loc[valid.index, mfe_col].mean())
            row[f"mean_mae_{h}d"] = float(subset.loc[valid.index, mae_col].mean())

        row["meets_min_samples"] = len(subset) >= min_samples
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values("events", ascending=False).reset_index(drop=True)


def build_lift_table(context_summary: pd.DataFrame, horizons: list[int], min_samples: int) -> pd.DataFrame:
    if context_summary.empty:
        return context_summary

    overall = context_summary[(context_summary["RegimeBucket"] == "ALL") & (context_summary["TriggerType"] == "ALL")]
    if overall.empty:
        return pd.DataFrame()
    base = overall.iloc[0]

    candidate = context_summary[
        (context_summary["RegimeBucket"] != "ALL")
        & (context_summary["TriggerType"] != "ALL")
        & (context_summary["events"] >= min_samples)
    ].copy()
    if candidate.empty:
        return candidate

    h = 20 if 20 in horizons else horizons[-1]
    candidate["lift_p_ge5"] = candidate[f"p_ge5_{h}d"] - float(base[f"p_ge5_{h}d"])
    candidate["lift_mean_ret"] = candidate[f"mean_ret_{h}d"] - float(base[f"mean_ret_{h}d"])
    candidate["score"] = (
        candidate["lift_p_ge5"].fillna(0.0) * 0.6
        + candidate["lift_mean_ret"].fillna(0.0) * 0.4
    )
    candidate = candidate.sort_values(["score", "events"], ascending=[False, False]).reset_index(drop=True)
    return candidate


def format_pct(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return f"{float(value) * 100.0:.2f}%"


def write_markdown_report(
    out_path: Path,
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    event_mode: str,
    horizons: list[int],
    events: pd.DataFrame,
    overall: pd.DataFrame,
    regime: pd.DataFrame,
    ranked: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Signal Quality Study (v2)")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- Symbols: {', '.join(symbols)}")
    lines.append(f"- Window: {start.date()} to {end.date()}")
    lines.append(f"- Event mode: `{event_mode}`")
    lines.append(f"- Horizons: {', '.join(str(h) for h in horizons)} trading bars")
    lines.append(f"- Total events analyzed: {len(events)}")
    lines.append("")

    if not overall.empty:
        row = overall.iloc[0]
        lines.append("## Baseline Event Probabilities")
        for h in horizons:
            lines.append(
                f"- `{h}d`: P(>0)={format_pct(row[f'p_pos_{h}d'])}, "
                f"P(>=+5%)={format_pct(row[f'p_ge5_{h}d'])}, "
                f"mean={format_pct(row[f'mean_ret_{h}d'])}, "
                f"median={format_pct(row[f'median_ret_{h}d'])}"
            )
        lines.append("")

    if not regime.empty:
        lines.append("## Regime Effect (All Trigger Types)")
        for _, row in regime.iterrows():
            rb = row["RegimeBucket"]
            ev = int(row["events"])
            h = 20 if 20 in horizons else horizons[-1]
            lines.append(
                f"- `{rb}` (n={ev}): P(>=+5% @ {h}d)={format_pct(row[f'p_ge5_{h}d'])}, "
                f"mean @ {h}d={format_pct(row[f'mean_ret_{h}d'])}"
            )
        lines.append("")

    if not ranked.empty:
        lines.append("## Highest-Lift Contexts")
        h = 20 if 20 in horizons else horizons[-1]
        for _, row in ranked.head(8).iterrows():
            lines.append(
                f"- `{row['RegimeBucket']} | {row['TriggerType']}` (n={int(row['events'])}): "
                f"P(>=+5% @ {h}d)={format_pct(row[f'p_ge5_{h}d'])}, "
                f"mean @ {h}d={format_pct(row[f'mean_ret_{h}d'])}, "
                f"lift={format_pct(row['lift_p_ge5'])}"
            )
        lines.append("")

    lines.append("## Output Files")
    lines.append("- `events.csv`: one row per event with forward-return/MFE/MAE metrics")
    lines.append("- `overall_summary.csv`: baseline quality summary")
    lines.append("- `regime_summary.csv`: monthly/weekly regime-conditioned probabilities")
    lines.append("- `context_summary.csv`: regime + trigger-type summary")
    lines.append("- `ranked_contexts.csv`: context lift ranking vs baseline")
    lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    symbols = parse_symbols(args)
    if not symbols:
        print("[error] no symbols provided")
        return 1
    horizons = parse_horizons(args.horizons)
    cfg = StudyConfig(
        symbols=symbols,
        input_dir=Path(args.input_dir),
        out_dir=Path(args.out_dir),
        event_mode=args.event_mode,
        horizons=horizons,
        align_window=args.align_window,
        min_samples=args.min_samples,
        date_from=args.date_from.strip() or None,
        date_to=args.date_to.strip() or None,
    )

    frames: dict[str, pd.DataFrame] = {}
    for symbol in cfg.symbols:
        frame = load_symbol_frame(cfg.input_dir, symbol)
        frame = apply_date_filters(frame, cfg.date_from, cfg.date_to)
        if frame.empty:
            raise ValueError(f"{symbol} has no rows after date filter")
        frames[symbol] = frame

    if cfg.align_window:
        common_start, common_end = align_date_window(frames)
        for symbol in cfg.symbols:
            frame = frames[symbol]
            frame = frame[(frame["Date"] >= common_start) & (frame["Date"] <= common_end)].reset_index(drop=True)
            frames[symbol] = frame
    else:
        all_dates = pd.concat([frame["Date"] for frame in frames.values()], ignore_index=True)
        common_start = all_dates.min()
        common_end = all_dates.max()

    event_frames: list[pd.DataFrame] = []
    for symbol, frame in frames.items():
        frame = frame.reset_index(drop=True)
        event_mask = compute_event_mask(frame, cfg.event_mode)
        events = frame[event_mask].copy()
        if events.empty:
            continue
        events["TriggerType"] = events.apply(trigger_type_from_row, axis=1)
        events["RegimeBucket"] = (
            "M"
            + events["MonthlyRiskOn"].astype(int).astype(str)
            + "W"
            + events["WeeklyRiskOn"].astype(int).astype(str)
        )
        events["AtrPct"] = events["ATR"] / events["Close"]
        events["CloseVsDailyEMA"] = (events["Close"] / events["DailyEMA"]) - 1.0
        events = attach_forward_metrics(events, frame, cfg.horizons)
        event_frames.append(events)

    if not event_frames:
        print("[error] no events found")
        return 1

    all_events = pd.concat(event_frames, ignore_index=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    events_out_cols = [
        "Symbol",
        "Date",
        "Close",
        "TriggerType",
        "DailyBreakout",
        "PullbackReclaim",
        "EntrySetup",
        "MonthlyRiskOn",
        "WeeklyRiskOn",
        "RegimeOn",
        "RegimeBucket",
        "MomentumPositive",
        "AtrPct",
        "CloseVsDailyEMA",
    ]
    for h in cfg.horizons:
        events_out_cols.extend([f"ret_{h}d", f"mfe_{h}d", f"mae_{h}d", f"pos_{h}d", f"ge5_{h}d"])
    all_events[events_out_cols].to_csv(cfg.out_dir / "events.csv", index=False)

    overall_events = all_events.copy()
    overall_events["RegimeBucket"] = "ALL"
    overall_events["TriggerType"] = "ALL"
    overall_summary = summarise_groups(overall_events, ["RegimeBucket", "TriggerType"], cfg.horizons, cfg.min_samples)
    overall_summary.to_csv(cfg.out_dir / "overall_summary.csv", index=False)

    regime_events = all_events.copy()
    regime_events["TriggerType"] = "ALL"
    regime_summary = summarise_groups(regime_events, ["RegimeBucket", "TriggerType"], cfg.horizons, cfg.min_samples)
    regime_summary.to_csv(cfg.out_dir / "regime_summary.csv", index=False)

    context_summary = summarise_groups(all_events, ["RegimeBucket", "TriggerType"], cfg.horizons, cfg.min_samples)
    baseline_row = overall_summary.iloc[[0]].copy()
    context_with_baseline = pd.concat([baseline_row, context_summary], ignore_index=True)
    context_with_baseline.to_csv(cfg.out_dir / "context_summary.csv", index=False)

    ranked = build_lift_table(context_with_baseline, cfg.horizons, cfg.min_samples)
    ranked.to_csv(cfg.out_dir / "ranked_contexts.csv", index=False)

    write_markdown_report(
        out_path=cfg.out_dir / "report.md",
        symbols=cfg.symbols,
        start=common_start,
        end=common_end,
        event_mode=cfg.event_mode,
        horizons=cfg.horizons,
        events=all_events,
        overall=overall_summary,
        regime=regime_summary[regime_summary["RegimeBucket"] != "ALL"],
        ranked=ranked,
    )

    print("Signal Quality Study Complete")
    print(f"  symbols: {', '.join(cfg.symbols)}")
    print(f"  window: {common_start.date()} -> {common_end.date()}")
    print(f"  events: {len(all_events)}")
    print(f"  out dir: {cfg.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
