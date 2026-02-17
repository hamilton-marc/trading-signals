#!/usr/bin/env python3
"""Run a multi-timeframe confluence long strategy with asymmetric exit controls."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass
class StrategyConfig:
    daily_ema_period: int
    weekly_ema_period: int
    monthly_ema_period: int
    momentum_length: int
    require_momentum_positive_entry: bool
    entry_cross_lookback_bars: int
    atr_period: int
    atr_mult: float
    trend_fail_bars: int
    equity_ema_period: int
    kill_max_drawdown_pct: float
    kill_cooldown_bars: int
    initial_capital: float


@dataclass
class SymbolError:
    symbol: str
    message: str


@dataclass
class SymbolSummary:
    symbol: str
    rows: int
    trade_executions: int
    round_trips: int
    win_rate_pct: float
    ending_equity: float
    total_return_pct: float
    realized_pnl: float
    max_drawdown_pct: float
    final_shares: int
    final_cash: float
    latest_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", default="watchlist.txt", help="Path to watchlist file")
    parser.add_argument("--input-dir", default="out/daily", help="Directory with daily OHLC CSV files")
    parser.add_argument("--out-dir", default="out/mtf_entry_exit_v1", help="Directory for per-symbol output CSVs")
    parser.add_argument(
        "--latest-file",
        default="out/mtf_entry_exit_v1_latest.csv",
        help="CSV path for latest strategy status per symbol",
    )
    parser.add_argument(
        "--summary-file",
        default="out/mtf_entry_exit_v1_summary.csv",
        help="CSV path for summary metrics per symbol",
    )
    parser.add_argument(
        "--errors-file",
        default="out/mtf_entry_exit_v1_errors.csv",
        help="CSV path for per-symbol failures",
    )
    parser.add_argument("--daily-ema-period", type=int, default=50, help="Daily EMA period for trigger and trend-failure exit")
    parser.add_argument("--weekly-ema-period", type=int, default=20, help="Weekly EMA period for higher timeframe filter")
    parser.add_argument("--monthly-ema-period", type=int, default=10, help="Monthly EMA period for higher timeframe filter")
    parser.add_argument("--momentum-length", type=int, default=24, help="Daily momentum lookback length")
    parser.add_argument(
        "--require-momentum-positive-entry",
        action="store_true",
        help="Require positive daily momentum in addition to multi-timeframe confluence for entry",
    )
    parser.add_argument(
        "--entry-cross-lookback-bars",
        type=int,
        default=15,
        help=(
            "Allow daily trigger when close is above daily EMA and a cross-above happened within this many bars "
            "(default: 15, use 0 to require same-bar cross only)"
        ),
    )
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period for trailing stop")
    parser.add_argument("--atr-mult", type=float, default=2.5, help="ATR multiplier for trailing stop")
    parser.add_argument("--trend-fail-bars", type=int, default=1, help="Consecutive bars below daily EMA required for exit")
    parser.add_argument("--equity-ema-period", type=int, default=50, help="Equity EMA period for kill-switch monitoring")
    parser.add_argument(
        "--kill-max-drawdown-pct",
        type=float,
        default=20.0,
        help="Max strategy drawdown percent trigger for equity kill switch",
    )
    parser.add_argument(
        "--kill-cooldown-bars",
        type=int,
        default=10,
        help="Bars to pause entries after equity kill-switch triggers",
    )
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Starting capital")
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


def parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return float(raw)


def compute_ema(values: list[float], period: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return output

    alpha = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period
    output[period - 1] = seed
    prev = seed
    for idx in range(period, len(values)):
        current = (values[idx] - prev) * alpha + prev
        output[idx] = current
        prev = current
    return output


def compute_momentum(values: list[float], length: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    if length <= 0:
        return output
    for idx in range(length, len(values)):
        output[idx] = values[idx] - values[idx - length]
    return output


def compute_atr_wilder(rows: list[dict[str, object]], period: int) -> list[float | None]:
    output: list[float | None] = [None] * len(rows)
    if period <= 0 or len(rows) < period:
        return output

    tr_values: list[float | None] = [None] * len(rows)
    prev_close: float | None = None
    for idx, row in enumerate(rows):
        high = row.get("High")
        low = row.get("Low")
        close = row.get("Close")
        if not isinstance(high, float) or not isinstance(low, float) or not isinstance(close, float):
            prev_close = close if isinstance(close, float) else prev_close
            continue
        if prev_close is None:
            tr_values[idx] = high - low
        else:
            tr_values[idx] = max(high - low, abs(high - prev_close), abs(low - prev_close))
        prev_close = close

    window = tr_values[:period]
    if any(value is None for value in window):
        return output

    seed = sum(float(value) for value in window) / period
    output[period - 1] = seed
    prev_atr = seed
    for idx in range(period, len(rows)):
        tr = tr_values[idx]
        if tr is None:
            continue
        atr = ((prev_atr * (period - 1)) + tr) / period
        output[idx] = atr
        prev_atr = atr
    return output


def aggregate_closes(rows: list[dict[str, object]], timeframe: str) -> tuple[list[date], list[float]]:
    if timeframe not in {"weekly", "monthly"}:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    buckets: list[tuple[date, float]] = []
    current_key: tuple[int, int] | None = None
    current_date: date | None = None
    current_close: float | None = None

    for row in rows:
        row_date = row.get("DateObj")
        close = row.get("Close")
        if not isinstance(row_date, date) or not isinstance(close, float):
            continue

        if timeframe == "weekly":
            iso_year, iso_week, _ = row_date.isocalendar()
            key = (iso_year, iso_week)
        else:
            key = (row_date.year, row_date.month)

        if key != current_key:
            if current_key is not None and current_date is not None and current_close is not None:
                buckets.append((current_date, current_close))
            current_key = key
            current_date = row_date
            current_close = close
        else:
            current_date = row_date
            current_close = close

    if current_key is not None and current_date is not None and current_close is not None:
        buckets.append((current_date, current_close))

    end_dates = [item[0] for item in buckets]
    closes = [item[1] for item in buckets]
    return end_dates, closes


def map_completed_state_to_daily(
    daily_dates: list[date],
    period_end_dates: list[date],
    period_states: list[bool | None],
) -> list[bool | None]:
    mapped: list[bool | None] = [None] * len(daily_dates)
    period_idx = -1
    for idx, day in enumerate(daily_dates):
        while period_idx + 1 < len(period_end_dates) and period_end_dates[period_idx + 1] <= day:
            period_idx += 1
        mapped[idx] = period_states[period_idx] if period_idx >= 0 else None
    return mapped


def read_daily_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for raw in reader:
            row_date = date.fromisoformat(raw["Date"])
            open_price = parse_float(raw.get("Open"))
            high_price = parse_float(raw.get("High"))
            low_price = parse_float(raw.get("Low"))
            close_price = parse_float(raw.get("Close"))
            if close_price is None:
                continue
            rows.append(
                {
                    "Date": raw["Date"],
                    "DateObj": row_date,
                    "Open": close_price if open_price is None else open_price,
                    "High": close_price if high_price is None else high_price,
                    "Low": close_price if low_price is None else low_price,
                    "Close": close_price,
                    "Volume": raw.get("Volume", ""),
                }
            )
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    rows.sort(key=lambda row: row["Date"])
    return rows


def run_strategy(symbol: str, rows: list[dict[str, object]], cfg: StrategyConfig) -> tuple[list[dict[str, str]], SymbolSummary]:
    dates = [row["DateObj"] for row in rows]
    closes = [float(row["Close"]) for row in rows]
    daily_ema = compute_ema(closes, cfg.daily_ema_period)
    momentum = compute_momentum(closes, cfg.momentum_length)
    atr_values = compute_atr_wilder(rows, cfg.atr_period)

    weekly_dates, weekly_closes = aggregate_closes(rows, timeframe="weekly")
    weekly_ema = compute_ema(weekly_closes, cfg.weekly_ema_period)
    weekly_up_state: list[bool | None] = []
    for idx, close in enumerate(weekly_closes):
        ema = weekly_ema[idx]
        weekly_up_state.append(None if ema is None else close > ema)
    weekly_up = map_completed_state_to_daily(dates, weekly_dates, weekly_up_state)

    monthly_dates, monthly_closes = aggregate_closes(rows, timeframe="monthly")
    monthly_ema = compute_ema(monthly_closes, cfg.monthly_ema_period)
    monthly_up_state: list[bool | None] = []
    for idx, close in enumerate(monthly_closes):
        ema = monthly_ema[idx]
        monthly_up_state.append(None if ema is None else close > ema)
    monthly_up = map_completed_state_to_daily(dates, monthly_dates, monthly_up_state)

    cash = cfg.initial_capital
    shares = 0
    entry_cost = 0.0

    pending_action = ""
    pending_reason = ""
    pending_source_date = ""

    trade_executions = 0
    round_trips = 0
    winning_round_trips = 0
    realized_pnl_total = 0.0

    equity_peak = cfg.initial_capital
    max_drawdown_pct = 0.0

    equity_ema: float | None = None
    equity_ema_ready = 0
    equity_ema_alpha = 2.0 / (cfg.equity_ema_period + 1.0)

    trail_stop: float | None = None
    trend_fail_count = 0
    cooldown_remaining = 0

    output_rows: list[dict[str, str]] = []
    last_cross_above_idx: int | None = None

    for idx, row in enumerate(rows):
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])
        date_text = str(row["Date"])

        executed_action = ""
        executed_reason = ""
        executed_source_date = ""
        realized_trade_pnl = 0.0

        if pending_action == "BUY" and shares == 0 and open_price > 0:
            buy_shares = int(cash // open_price)
            if buy_shares > 0:
                cost = buy_shares * open_price
                shares = buy_shares
                cash -= cost
                entry_cost = cost
                trail_stop = None
                trend_fail_count = 0
                trade_executions += 1
                executed_action = "BUY"
                executed_reason = pending_reason
                executed_source_date = pending_source_date
        elif pending_action == "EXIT" and shares > 0:
            proceeds = shares * open_price
            realized_trade_pnl = proceeds - entry_cost
            realized_pnl_total += realized_trade_pnl
            round_trips += 1
            if realized_trade_pnl > 0:
                winning_round_trips += 1
            cash += proceeds
            shares = 0
            entry_cost = 0.0
            trail_stop = None
            trend_fail_count = 0
            trade_executions += 1
            executed_action = "SELL"
            executed_reason = pending_reason
            executed_source_date = pending_source_date

        pending_action = ""
        pending_reason = ""
        pending_source_date = ""

        equity = cash + shares * close_price
        if equity > equity_peak:
            equity_peak = equity
        drawdown_pct = 0.0 if equity_peak == 0 else ((equity_peak - equity) / equity_peak) * 100.0
        if drawdown_pct > max_drawdown_pct:
            max_drawdown_pct = drawdown_pct

        if equity_ema is None:
            equity_ema = equity
            equity_ema_ready = 1
        else:
            equity_ema = (equity - equity_ema) * equity_ema_alpha + equity_ema
            equity_ema_ready += 1

        equity_ema_is_ready = equity_ema_ready >= cfg.equity_ema_period
        equity_below_ema = equity_ema_is_ready and equity_ema is not None and equity < equity_ema

        d_ema = daily_ema[idx]
        mom = momentum[idx]
        atr = atr_values[idx]
        m_up = monthly_up[idx] is True
        w_up = weekly_up[idx] is True

        prev_close = closes[idx - 1] if idx > 0 else None
        prev_ema = daily_ema[idx - 1] if idx > 0 else None
        daily_cross_up = (
            prev_close is not None
            and prev_ema is not None
            and d_ema is not None
            and prev_close <= prev_ema
            and close_price > d_ema
        )
        if daily_cross_up:
            last_cross_above_idx = idx
        mom_positive = mom is not None and mom > 0

        recent_cross_window_ok = (
            cfg.entry_cross_lookback_bars > 0
            and d_ema is not None
            and close_price > d_ema
            and last_cross_above_idx is not None
            and (idx - last_cross_above_idx) <= cfg.entry_cross_lookback_bars
        )
        daily_trigger = daily_cross_up or recent_cross_window_ok

        entry_setup = m_up and w_up and daily_trigger
        if cfg.require_momentum_positive_entry:
            entry_setup = entry_setup and mom_positive

        atr_trail_candidate: float | None = None
        if shares > 0 and atr is not None:
            atr_trail_candidate = close_price - (cfg.atr_mult * atr)
            trail_stop = atr_trail_candidate if trail_stop is None else max(trail_stop, atr_trail_candidate)

        atr_trail_broken = shares > 0 and trail_stop is not None and close_price < trail_stop

        if shares > 0 and d_ema is not None and close_price < d_ema:
            trend_fail_count += 1
        elif shares > 0:
            trend_fail_count = 0
        else:
            trend_fail_count = 0
        trend_fail_exit = shares > 0 and trend_fail_count >= cfg.trend_fail_bars

        equity_kill = shares > 0 and (equity_below_ema or drawdown_pct >= cfg.kill_max_drawdown_pct)
        if equity_kill and cfg.kill_cooldown_bars > 0 and cooldown_remaining < cfg.kill_cooldown_bars:
            cooldown_remaining = cfg.kill_cooldown_bars

        exit_reason = ""
        if equity_kill:
            exit_reason = "EQUITY_KILL"
        elif atr_trail_broken:
            exit_reason = "ATR_TRAIL"
        elif trend_fail_exit:
            exit_reason = "EMA50_FAIL"

        next_pending_action = ""
        next_pending_reason = ""
        if idx < len(rows) - 1:
            if shares > 0 and exit_reason:
                next_pending_action = "EXIT"
                next_pending_reason = exit_reason
            elif shares == 0 and cooldown_remaining == 0 and entry_setup:
                next_pending_action = "BUY"
                next_pending_reason = "MTF_CONFLUENCE"

        pending_action = next_pending_action
        pending_reason = next_pending_reason
        pending_source_date = date_text if pending_action else ""

        output_rows.append(
            {
                "Date": date_text,
                "Open": f"{open_price:.6f}",
                "High": f"{high_price:.6f}",
                "Low": f"{low_price:.6f}",
                "Close": f"{close_price:.6f}",
                "MonthlyUp": "1" if m_up else "0",
                "WeeklyUp": "1" if w_up else "0",
                "DailyEMA": "" if d_ema is None else f"{d_ema:.6f}",
                "DailyCrossAboveEMA": "1" if daily_cross_up else "0",
                "DailyTriggerRecentCross": "1" if recent_cross_window_ok else "0",
                "DailyTrigger": "1" if daily_trigger else "0",
                "Momentum": "" if mom is None else f"{mom:.6f}",
                "MomentumPositive": "1" if mom_positive else "0",
                "EntrySetup": "1" if entry_setup else "0",
                "EntryMomentumConfirm": "1" if mom_positive else "0",
                "ATR": "" if atr is None else f"{atr:.6f}",
                "ATRTrailCandidate": "" if atr_trail_candidate is None else f"{atr_trail_candidate:.6f}",
                "ATRTrailStop": "" if trail_stop is None else f"{trail_stop:.6f}",
                "ATRTrailBroken": "1" if atr_trail_broken else "0",
                "TrendFailCount": str(trend_fail_count),
                "TrendFailExit": "1" if trend_fail_exit else "0",
                "Equity": f"{equity:.6f}",
                "EquityPeak": f"{equity_peak:.6f}",
                "DrawdownPct": f"{drawdown_pct:.6f}",
                "EquityEMA": "" if equity_ema is None else f"{equity_ema:.6f}",
                "EquityBelowEMA": "1" if equity_below_ema else "0",
                "EquityKill": "1" if equity_kill else "0",
                "CooldownRemaining": str(cooldown_remaining),
                "PositionState": "LONG" if shares > 0 else "FLAT",
                "Shares": str(shares),
                "Cash": f"{cash:.6f}",
                "PendingAction": pending_action,
                "PendingReason": pending_reason,
                "ExitReason": exit_reason,
                "ExecutedAction": executed_action,
                "ExecutedReason": executed_reason,
                "ExecutedSourceDate": executed_source_date,
                "RealizedTradePnL": "" if executed_action != "SELL" else f"{realized_trade_pnl:.6f}",
            }
        )

        if cooldown_remaining > 0:
            cooldown_remaining -= 1

    final_equity = cash + shares * closes[-1]
    total_return_pct = 0.0 if cfg.initial_capital == 0 else ((final_equity - cfg.initial_capital) / cfg.initial_capital) * 100.0
    win_rate_pct = 0.0 if round_trips == 0 else (winning_round_trips / round_trips) * 100.0

    summary = SymbolSummary(
        symbol=symbol,
        rows=len(output_rows),
        trade_executions=trade_executions,
        round_trips=round_trips,
        win_rate_pct=win_rate_pct,
        ending_equity=final_equity,
        total_return_pct=total_return_pct,
        realized_pnl=realized_pnl_total,
        max_drawdown_pct=max_drawdown_pct,
        final_shares=shares,
        final_cash=cash,
        latest_date=str(rows[-1]["Date"]),
    )
    return output_rows, summary


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "symbol",
        "Date",
        "Close",
        "MonthlyUp",
        "WeeklyUp",
        "DailyCrossAboveEMA",
        "DailyTriggerRecentCross",
        "DailyTrigger",
        "MomentumPositive",
        "EntrySetup",
        "ATRTrailStop",
        "TrendFailCount",
        "Equity",
        "DrawdownPct",
        "EquityKill",
        "CooldownRemaining",
        "PositionState",
        "PendingAction",
        "PendingReason",
        "ExecutedAction",
        "ExecutedReason",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[SymbolSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Symbol",
                "Rows",
                "TradeExecutions",
                "RoundTrips",
                "WinRatePct",
                "EndingEquity",
                "TotalReturnPct",
                "RealizedPnL",
                "MaxDrawdownPct",
                "FinalShares",
                "FinalCash",
                "LatestDate",
            ],
        )
        writer.writeheader()
        for item in rows:
            writer.writerow(
                {
                    "Symbol": item.symbol,
                    "Rows": str(item.rows),
                    "TradeExecutions": str(item.trade_executions),
                    "RoundTrips": str(item.round_trips),
                    "WinRatePct": f"{item.win_rate_pct:.2f}",
                    "EndingEquity": f"{item.ending_equity:.2f}",
                    "TotalReturnPct": f"{item.total_return_pct:.4f}",
                    "RealizedPnL": f"{item.realized_pnl:.2f}",
                    "MaxDrawdownPct": f"{item.max_drawdown_pct:.4f}",
                    "FinalShares": str(item.final_shares),
                    "FinalCash": f"{item.final_cash:.2f}",
                    "LatestDate": item.latest_date,
                }
            )


def write_errors(path: Path, errors: list[SymbolError]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["symbol", "error"])
        writer.writeheader()
        for error in errors:
            writer.writerow({"symbol": error.symbol, "error": error.message})


def validate_args(args: argparse.Namespace) -> str | None:
    if args.daily_ema_period <= 0:
        return "--daily-ema-period must be > 0"
    if args.weekly_ema_period <= 0:
        return "--weekly-ema-period must be > 0"
    if args.monthly_ema_period <= 0:
        return "--monthly-ema-period must be > 0"
    if args.momentum_length <= 0:
        return "--momentum-length must be > 0"
    if args.atr_period <= 0:
        return "--atr-period must be > 0"
    if args.atr_mult <= 0:
        return "--atr-mult must be > 0"
    if args.trend_fail_bars <= 0:
        return "--trend-fail-bars must be > 0"
    if args.equity_ema_period <= 0:
        return "--equity-ema-period must be > 0"
    if args.kill_max_drawdown_pct < 0:
        return "--kill-max-drawdown-pct must be >= 0"
    if args.kill_cooldown_bars < 0:
        return "--kill-cooldown-bars must be >= 0"
    if args.entry_cross_lookback_bars < 0:
        return "--entry-cross-lookback-bars must be >= 0"
    if args.initial_capital <= 0:
        return "--initial-capital must be > 0"
    return None


def main() -> int:
    args = parse_args()
    validation_error = validate_args(args)
    if validation_error is not None:
        print(f"[error] {validation_error}")
        return 1

    try:
        symbols = read_watchlist(Path(args.watchlist))
    except Exception as exc:
        print(f"[error] {exc}")
        return 1
    if not symbols:
        print("[error] watchlist is empty")
        return 1

    cfg = StrategyConfig(
        daily_ema_period=args.daily_ema_period,
        weekly_ema_period=args.weekly_ema_period,
        monthly_ema_period=args.monthly_ema_period,
        momentum_length=args.momentum_length,
        require_momentum_positive_entry=args.require_momentum_positive_entry,
        entry_cross_lookback_bars=args.entry_cross_lookback_bars,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        trend_fail_bars=args.trend_fail_bars,
        equity_ema_period=args.equity_ema_period,
        kill_max_drawdown_pct=args.kill_max_drawdown_pct,
        kill_cooldown_bars=args.kill_cooldown_bars,
        initial_capital=args.initial_capital,
    )

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    latest_file = Path(args.latest_file)
    summary_file = Path(args.summary_file)
    errors_file = Path(args.errors_file)

    latest_rows: list[dict[str, str]] = []
    summaries: list[SymbolSummary] = []
    errors: list[SymbolError] = []

    for symbol in symbols:
        try:
            input_path = input_dir / f"{symbol}.csv"
            if not input_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {input_path}")
            daily_rows = read_daily_rows(input_path)
            strategy_rows, summary = run_strategy(symbol, daily_rows, cfg)

            output_path = out_dir / f"{symbol}.csv"
            write_rows(output_path, strategy_rows)

            latest = strategy_rows[-1]
            latest_rows.append(
                {
                    "symbol": symbol,
                    "Date": latest.get("Date", ""),
                    "Close": latest.get("Close", ""),
                    "MonthlyUp": latest.get("MonthlyUp", ""),
                    "WeeklyUp": latest.get("WeeklyUp", ""),
                    "DailyCrossAboveEMA": latest.get("DailyCrossAboveEMA", ""),
                    "DailyTriggerRecentCross": latest.get("DailyTriggerRecentCross", ""),
                    "DailyTrigger": latest.get("DailyTrigger", ""),
                    "MomentumPositive": latest.get("MomentumPositive", ""),
                    "EntrySetup": latest.get("EntrySetup", ""),
                    "ATRTrailStop": latest.get("ATRTrailStop", ""),
                    "TrendFailCount": latest.get("TrendFailCount", ""),
                    "Equity": latest.get("Equity", ""),
                    "DrawdownPct": latest.get("DrawdownPct", ""),
                    "EquityKill": latest.get("EquityKill", ""),
                    "CooldownRemaining": latest.get("CooldownRemaining", ""),
                    "PositionState": latest.get("PositionState", ""),
                    "PendingAction": latest.get("PendingAction", ""),
                    "PendingReason": latest.get("PendingReason", ""),
                    "ExecutedAction": latest.get("ExecutedAction", ""),
                    "ExecutedReason": latest.get("ExecutedReason", ""),
                }
            )
            summaries.append(summary)
            print(f"[ok] {symbol} -> {output_path} ({summary.rows} rows)")
        except Exception as exc:
            errors.append(SymbolError(symbol=symbol, message=str(exc)))
            print(f"[fail] {symbol} -> {exc}")

    latest_rows.sort(key=lambda row: row["symbol"])
    summaries.sort(key=lambda item: item.symbol)

    write_latest(latest_file, latest_rows)
    write_summary(summary_file, summaries)
    write_errors(errors_file, errors)

    print("\nSummary")
    print(f"  symbols: {len(symbols)}")
    print(f"  success: {len(summaries)}")
    print(f"  failed:  {len(errors)}")
    print(f"  out dir: {out_dir}")
    print(f"  latest file:  {latest_file}")
    print(f"  summary file: {summary_file}")
    print(f"  errors file:  {errors_file}")

    return 0 if summaries else 1


if __name__ == "__main__":
    raise SystemExit(main())
