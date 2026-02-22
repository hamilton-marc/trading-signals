#!/usr/bin/env python3
"""Run a simple long-only backtest from trend + signal outputs."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Trade:
    date: str
    action: str
    reason: str
    source_date: str
    price: float
    shares: int
    cash_after: float
    position_shares_after: int
    avg_cost_after: float
    realized_pnl: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="APO", help="Symbol to backtest (default: APO)")
    parser.add_argument("--signals-dir", default="out/signals/engine", help="Directory with signal CSV files")
    parser.add_argument("--trend-dir", default="out/indicators/trend", help="Directory with trend CSV files")
    parser.add_argument("--out-dir", default="out/backtests/long_only", help="Directory for backtest outputs")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Starting cash")
    parser.add_argument(
        "--allocation-pct",
        type=float,
        default=5.0,
        help="Percent of current equity allocated on each buy signal",
    )
    parser.add_argument(
        "--ema-stop-column",
        default="EMA_50",
        help="EMA column used as trailing stop reference (default: EMA_50)",
    )
    parser.add_argument(
        "--hard-stop-pct",
        type=float,
        default=50.0,
        help="Percent below average cost for catastrophic stop floor (default: 50)",
    )
    parser.add_argument(
        "--atr-period",
        type=int,
        default=14,
        help="ATR lookback period used for optional entry filter (default: 14)",
    )
    parser.add_argument(
        "--min-atr-pct",
        type=float,
        default=0.0,
        help="Minimum ATR as percent of close required for new entries (0 disables filter)",
    )
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows


def parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return float(raw)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_atr_wilder(rows: list[dict[str, str]], period: int) -> list[float | None]:
    atr: list[float | None] = [None] * len(rows)
    if period <= 0 or len(rows) < period:
        return atr

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
        return atr

    seed = sum(float(value) for value in seed_window) / period
    atr[period - 1] = seed
    prev_atr = seed

    for idx in range(period, len(rows)):
        tr = tr_values[idx]
        if tr is None:
            continue
        current = ((prev_atr * (period - 1)) + tr) / period
        atr[idx] = current
        prev_atr = current

    return atr


def run_backtest(
    symbol: str,
    signal_rows: list[dict[str, str]],
    trend_rows: list[dict[str, str]],
    initial_capital: float,
    allocation_pct: float,
    ema_stop_column: str,
    hard_stop_pct: float,
    atr_period: int,
    min_atr_pct: float,
) -> tuple[list[Trade], list[dict[str, str]], dict[str, str]]:
    signal_by_date = {row["Date"]: row for row in signal_rows}
    merged_rows: list[dict[str, str]] = []
    for row in sorted(trend_rows, key=lambda item: item["Date"]):
        signal_row = signal_by_date.get(row["Date"])
        if signal_row is None:
            continue
        merged_rows.append(
            {
                "Date": row["Date"],
                "Open": row.get("Open", ""),
                "High": row.get("High", ""),
                "Low": row.get("Low", ""),
                "Close": row.get("Close", ""),
                "Trend": row.get("Trend", signal_row.get("Trend", "")).strip() or "NEUTRAL",
                "SignalEvent": (signal_row.get("SignalEvent") or "").strip(),
                "EMA_STOP": row.get(ema_stop_column, ""),
            }
        )

    if len(merged_rows) < 3:
        raise ValueError("Need at least 3 overlapping rows to backtest")

    atr_values = compute_atr_wilder(merged_rows, period=atr_period)

    cash = initial_capital
    shares = 0
    avg_cost = 0.0
    allocation_ratio = allocation_pct / 100.0
    hard_stop_ratio = hard_stop_pct / 100.0

    pending_action: dict[str, str] | None = None
    trades: list[Trade] = []
    equity_rows: list[dict[str, str]] = []
    prev_trend = "NEUTRAL"

    max_equity = initial_capital
    max_drawdown = 0.0
    realized_pnl_total = 0.0
    round_trip_count = 0
    winning_round_trips = 0

    for idx, row in enumerate(merged_rows):
        trade_date = row["Date"]
        open_price = parse_float(row.get("Open"))
        close_price = parse_float(row.get("Close"))
        trend = row.get("Trend", "NEUTRAL")
        signal_event = row.get("SignalEvent", "")
        ema_stop = parse_float(row.get("EMA_STOP"))
        atr_value = atr_values[idx]
        atr_pct = None if atr_value is None or close_price is None or close_price == 0 else (atr_value / close_price) * 100.0

        if open_price is None or close_price is None:
            prev_trend = trend
            continue

        executed_action = ""
        executed_reason = ""
        source_date = ""
        realized_pnl = 0.0
        shares_delta = 0

        if pending_action is not None:
            action = pending_action["type"]
            reason = pending_action["reason"]
            source_date = pending_action["source_date"]
            executed_reason = reason

            if action == "EXIT" and shares > 0:
                proceeds = shares * open_price
                realized_pnl = (open_price - avg_cost) * shares
                cash += proceeds
                realized_pnl_total += realized_pnl
                round_trip_count += 1
                if realized_pnl > 0:
                    winning_round_trips += 1
                shares_delta = -shares
                shares = 0
                avg_cost = 0.0
                executed_action = "SELL"
            elif action == "BUY":
                equity_at_open = cash + shares * open_price
                target_value = equity_at_open * allocation_ratio
                buy_value = min(target_value, cash)
                buy_shares = int(math.floor(buy_value / open_price))
                if buy_shares > 0:
                    cost = buy_shares * open_price
                    total_cost_basis = avg_cost * shares + cost
                    shares += buy_shares
                    cash -= cost
                    avg_cost = total_cost_basis / shares
                    shares_delta = buy_shares
                    executed_action = "BUY"

            if executed_action:
                trades.append(
                    Trade(
                        date=trade_date,
                        action=executed_action,
                        reason=executed_reason,
                        source_date=source_date,
                        price=open_price,
                        shares=abs(shares_delta),
                        cash_after=cash,
                        position_shares_after=shares,
                        avg_cost_after=avg_cost,
                        realized_pnl=realized_pnl,
                    )
                )
            pending_action = None

        active_stop: float | None = None
        if shares > 0:
            stop_candidates: list[float] = []
            if ema_stop is not None:
                stop_candidates.append(ema_stop)
            if avg_cost > 0:
                stop_candidates.append(avg_cost * (1.0 - hard_stop_ratio))
            if stop_candidates:
                active_stop = max(stop_candidates)

        exit_reason = ""
        if shares > 0 and trend == "DOWNTREND" and prev_trend != "DOWNTREND":
            exit_reason = "DOWNTREND"
        if shares > 0 and signal_event == "LONG_TO_SHORT":
            exit_reason = "LONG_TO_SHORT"
        if shares > 0 and active_stop is not None and close_price < active_stop:
            exit_reason = "EMA_STOP"

        buy_from_trend = trend == "UPTREND" and prev_trend != "UPTREND"
        buy_from_signal = signal_event in {"LONG_ENTRY", "SHORT_TO_LONG"}
        atr_filter_ok = min_atr_pct <= 0.0 or (atr_pct is not None and atr_pct >= min_atr_pct)
        buy_reason = ""
        if buy_from_trend and buy_from_signal:
            buy_reason = "UP+BUY_SIGNAL"
        elif buy_from_trend:
            buy_reason = "UPTREND"
        elif buy_from_signal:
            buy_reason = "BUY_SIGNAL"

        if idx < len(merged_rows) - 1:
            if exit_reason:
                pending_action = {
                    "type": "EXIT",
                    "reason": exit_reason,
                    "source_date": trade_date,
                }
            elif buy_reason and atr_filter_ok:
                pending_action = {
                    "type": "BUY",
                    "reason": buy_reason,
                    "source_date": trade_date,
                }

        equity = cash + shares * close_price
        max_equity = max(max_equity, equity)
        drawdown = 0.0 if max_equity == 0 else (max_equity - equity) / max_equity
        max_drawdown = max(max_drawdown, drawdown)

        equity_rows.append(
            {
                "Date": trade_date,
                "Open": f"{open_price:.6f}",
                "Close": f"{close_price:.6f}",
                "Trend": trend,
                "SignalEvent": signal_event,
                "EMA_Stop": "" if active_stop is None else f"{active_stop:.6f}",
                "ATR": "" if atr_value is None else f"{atr_value:.6f}",
                "ATR_Pct": "" if atr_pct is None else f"{atr_pct:.6f}",
                "ATRFilterPass": "1" if atr_filter_ok else "0",
                "PendingAction": "" if pending_action is None else pending_action["type"],
                "PendingReason": "" if pending_action is None else pending_action["reason"],
                "Cash": f"{cash:.6f}",
                "Shares": str(shares),
                "AvgCost": "" if shares == 0 else f"{avg_cost:.6f}",
                "Equity": f"{equity:.6f}",
                "DrawdownPct": f"{drawdown * 100.0:.6f}",
                "ExecutedAction": executed_action,
                "ExecutedReason": executed_reason,
                "ExecutedSourceDate": source_date,
            }
        )

        prev_trend = trend

    ending_equity = float(equity_rows[-1]["Equity"])
    total_return_pct = 0.0 if initial_capital == 0 else (ending_equity - initial_capital) / initial_capital * 100.0
    win_rate_pct = 0.0 if round_trip_count == 0 else winning_round_trips / round_trip_count * 100.0

    summary = {
        "Symbol": symbol,
        "InitialCapital": f"{initial_capital:.2f}",
        "EndingEquity": f"{ending_equity:.2f}",
        "TotalReturnPct": f"{total_return_pct:.4f}",
        "RealizedPnL": f"{realized_pnl_total:.2f}",
        "TradeExecutions": str(len(trades)),
        "RoundTrips": str(round_trip_count),
        "WinRatePct": f"{win_rate_pct:.2f}",
        "MaxDrawdownPct": f"{max_drawdown * 100.0:.4f}",
        "FinalShares": str(shares),
        "FinalCash": f"{cash:.2f}",
        "LastDate": equity_rows[-1]["Date"],
        "ATRPeriod": str(atr_period),
        "MinATRPercent": f"{min_atr_pct:.4f}",
    }

    return trades, equity_rows, summary


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper()
    signals_path = Path(args.signals_dir) / f"{symbol}.csv"
    trend_path = Path(args.trend_dir) / f"{symbol}.csv"
    out_dir = Path(args.out_dir)

    if args.initial_capital <= 0:
        print("[error] --initial-capital must be > 0")
        return 1
    if args.allocation_pct <= 0 or args.allocation_pct > 100:
        print("[error] --allocation-pct must be > 0 and <= 100")
        return 1
    if args.hard_stop_pct < 0 or args.hard_stop_pct > 100:
        print("[error] --hard-stop-pct must be between 0 and 100")
        return 1
    if args.atr_period <= 0:
        print("[error] --atr-period must be > 0")
        return 1
    if args.min_atr_pct < 0:
        print("[error] --min-atr-pct must be >= 0")
        return 1

    if not signals_path.exists():
        print(f"[error] Missing signals file: {signals_path}")
        return 1
    if not trend_path.exists():
        print(f"[error] Missing trend file: {trend_path}")
        return 1

    signal_rows = read_rows(signals_path)
    trend_rows = read_rows(trend_path)
    if args.ema_stop_column not in trend_rows[0]:
        print(
            f"[error] Trend file {trend_path} is missing {args.ema_stop_column}. "
            "Regenerate trend data using indicator files with that EMA column."
        )
        return 1

    trades, equity_rows, summary = run_backtest(
        symbol=symbol,
        signal_rows=signal_rows,
        trend_rows=trend_rows,
        initial_capital=args.initial_capital,
        allocation_pct=args.allocation_pct,
        ema_stop_column=args.ema_stop_column,
        hard_stop_pct=args.hard_stop_pct,
        atr_period=args.atr_period,
        min_atr_pct=args.min_atr_pct,
    )

    trades_path = out_dir / f"{symbol}_trades.csv"
    equity_path = out_dir / f"{symbol}_equity_curve.csv"
    summary_path = out_dir / f"{symbol}_summary.csv"

    trade_dict_rows = [
        {
            "Date": trade.date,
            "Action": trade.action,
            "Reason": trade.reason,
            "SourceDate": trade.source_date,
            "Price": f"{trade.price:.6f}",
            "Shares": str(trade.shares),
            "CashAfter": f"{trade.cash_after:.6f}",
            "PositionSharesAfter": str(trade.position_shares_after),
            "AvgCostAfter": "" if trade.position_shares_after == 0 else f"{trade.avg_cost_after:.6f}",
            "RealizedPnL": f"{trade.realized_pnl:.6f}",
        }
        for trade in trades
    ]

    write_csv(
        trades_path,
        [
            "Date",
            "Action",
            "Reason",
            "SourceDate",
            "Price",
            "Shares",
            "CashAfter",
            "PositionSharesAfter",
            "AvgCostAfter",
            "RealizedPnL",
        ],
        trade_dict_rows,
    )
    write_csv(
        equity_path,
        [
            "Date",
            "Open",
            "Close",
            "Trend",
            "SignalEvent",
            "EMA_Stop",
            "ATR",
            "ATR_Pct",
            "ATRFilterPass",
            "PendingAction",
            "PendingReason",
            "Cash",
            "Shares",
            "AvgCost",
            "Equity",
            "DrawdownPct",
            "ExecutedAction",
            "ExecutedReason",
            "ExecutedSourceDate",
        ],
        equity_rows,
    )
    write_csv(summary_path, list(summary.keys()), [summary])

    print(f"[ok] Trades:  {trades_path}")
    print(f"[ok] Equity:  {equity_path}")
    print(f"[ok] Summary: {summary_path}")
    print("\nBacktest Summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
