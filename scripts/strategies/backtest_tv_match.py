#!/usr/bin/env python3
"""Backtest strict TradingView-style momentum match outputs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from scripts.paths import BACKTESTS_TV_MATCH_DIR, momentum_tv_match_output_dir
from scripts.watchlists import DEFAULT_WATCHLIST_PATH, read_watchlist


@dataclass
class Trade:
    date: str
    action: str
    price: float
    shares_delta: float
    cash_after: float
    shares_after: float
    equity_after: float
    realized_pnl: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", default=DEFAULT_WATCHLIST_PATH, help="Path to watchlist file")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbol override")
    parser.add_argument(
        "--timeframe",
        default="daily",
        choices=["d", "w", "m", "daily", "weekly", "monthly"],
        help="Momentum timeframe to backtest (default: daily)",
    )
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with strict momentum match CSV files (default depends on timeframe)",
    )
    parser.add_argument("--out-dir", default=str(BACKTESTS_TV_MATCH_DIR), help="Output directory for backtest CSV/MD files")
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Starting capital per symbol")
    parser.add_argument(
        "--position-sizing",
        choices=["full_equity"],
        default="full_equity",
        help="Capital allocation rule for entries (default: full_equity)",
    )
    parser.add_argument(
        "--mode",
        choices=["long_only", "long_short", "both"],
        default="long_only",
        help="Backtest mode (default: long_only)",
    )
    return parser.parse_args()


def normalize_timeframe(value: str) -> str:
    lowered = value.strip().lower()
    mapping = {
        "d": "daily",
        "daily": "daily",
        "w": "weekly",
        "weekly": "weekly",
        "m": "monthly",
        "monthly": "monthly",
    }
    timeframe = mapping.get(lowered)
    if timeframe is None:
        raise ValueError(f"Unsupported timeframe: {value!r}")
    return timeframe


def parse_symbols(value: str, watchlist_path: Path) -> list[str]:
    if value.strip():
        return [part.strip().upper() for part in value.split(",") if part.strip()]
    return read_watchlist(watchlist_path)


def parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return float(raw)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    required = {"Date", "Close", "Event", "FillPrice"}
    missing = required.difference(rows[0].keys())
    if missing:
        raise ValueError(f"Missing required columns in {path}: {', '.join(sorted(missing))}")
    rows.sort(key=lambda row: row["Date"])
    return rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def comparison_stem(symbols: list[str], timeframe: str, mode: str) -> str:
    if not symbols:
        suffix = "none"
    else:
        suffix = "_".join(symbols)
    return f"comparison_{timeframe}_{mode}_{suffix}"


def trade_label(direction_before: int, direction_after: int) -> str:
    if direction_before == 0 and direction_after > 0:
        return "BUY_LONG"
    if direction_before > 0 and direction_after == 0:
        return "SELL_LONG"
    if direction_before == 0 and direction_after < 0:
        return "SELL_SHORT"
    if direction_before < 0 and direction_after == 0:
        return "BUY_TO_COVER"
    if direction_before < 0 and direction_after > 0:
        return "REVERSE_TO_LONG"
    if direction_before > 0 and direction_after < 0:
        return "REVERSE_TO_SHORT"
    raise ValueError(f"Unsupported transition: {direction_before} -> {direction_after}")


def should_enter_long(mode: str, event: str, shares: float) -> bool:
    if event != "MomLE":
        return False
    if mode == "long_only":
        return shares == 0.0
    return shares <= 0.0


def should_enter_short(mode: str, event: str, shares: float) -> bool:
    if mode != "long_short" or event != "MomSE":
        return False
    return shares >= 0.0


def mark_equity(cash: float, shares: float, close_price: float) -> float:
    return cash + (shares * close_price)


def run_backtest(
    *,
    mode: str,
    rows: list[dict[str, str]],
    initial_capital: float,
) -> tuple[list[Trade], list[dict[str, str]], dict[str, str]]:
    if mode not in {"long_only", "long_short"}:
        raise ValueError(f"Unsupported mode: {mode}")

    cash = initial_capital
    shares = 0.0
    entry_price: float | None = None
    max_equity = initial_capital
    max_drawdown_pct = 0.0
    trades: list[Trade] = []
    equity_rows: list[dict[str, str]] = []
    closed_trades = 0
    winning_trades = 0

    for row in rows:
        close_price = parse_float(row.get("Close"))
        if close_price is None:
            continue

        event = (row.get("Event") or "").strip()
        fill_price = parse_float(row.get("FillPrice"))
        executed_action = ""
        realized_pnl = 0.0

        if fill_price is not None and should_enter_long(mode, event, shares):
            prev_direction = -1 if shares < 0 else 0 if shares == 0 else 1
            old_shares = shares
            if shares < 0:
                realized_pnl = (-shares) * ((entry_price or fill_price) - fill_price)
                cash += shares * fill_price
                shares = 0.0
                closed_trades += 1
                if realized_pnl > 0:
                    winning_trades += 1
            next_shares = 0.0 if fill_price == 0 else cash / fill_price
            cash -= next_shares * fill_price
            shares += next_shares
            entry_price = fill_price
            executed_action = trade_label(prev_direction, 1)
            trades.append(
                Trade(
                    date=row["Date"],
                    action=executed_action,
                    price=fill_price,
                    shares_delta=shares - old_shares,
                    cash_after=cash,
                    shares_after=shares,
                    equity_after=mark_equity(cash, shares, close_price),
                    realized_pnl=realized_pnl,
                )
            )
        elif fill_price is not None and should_enter_short(mode, event, shares):
            prev_direction = 1 if shares > 0 else 0 if shares == 0 else -1
            old_shares = shares
            if shares > 0:
                realized_pnl = shares * (fill_price - (entry_price or fill_price))
                cash += shares * fill_price
                shares = 0.0
                closed_trades += 1
                if realized_pnl > 0:
                    winning_trades += 1
            short_shares = 0.0 if fill_price == 0 else cash / fill_price
            cash += short_shares * fill_price
            shares -= short_shares
            entry_price = fill_price
            executed_action = trade_label(prev_direction, -1)
            trades.append(
                Trade(
                    date=row["Date"],
                    action=executed_action,
                    price=fill_price,
                    shares_delta=shares - old_shares,
                    cash_after=cash,
                    shares_after=shares,
                    equity_after=mark_equity(cash, shares, close_price),
                    realized_pnl=realized_pnl,
                )
            )
        elif fill_price is not None and mode == "long_only" and event == "MomSE" and shares > 0:
            old_shares = shares
            realized_pnl = shares * (fill_price - (entry_price or fill_price))
            cash += shares * fill_price
            shares = 0.0
            entry_price = None
            closed_trades += 1
            if realized_pnl > 0:
                winning_trades += 1
            executed_action = "SELL_LONG"
            trades.append(
                Trade(
                    date=row["Date"],
                    action=executed_action,
                    price=fill_price,
                    shares_delta=shares - old_shares,
                    cash_after=cash,
                    shares_after=shares,
                    equity_after=mark_equity(cash, shares, close_price),
                    realized_pnl=realized_pnl,
                )
            )

        equity = mark_equity(cash, shares, close_price)
        max_equity = max(max_equity, equity)
        drawdown_pct = 0.0 if max_equity <= 0 else ((max_equity - equity) / max_equity) * 100.0
        max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

        equity_rows.append(
            {
                "Date": row["Date"],
                "Close": f"{close_price:.6f}",
                "Event": event,
                "FillPrice": "" if fill_price is None else f"{fill_price:.6f}",
                "Action": executed_action,
                "Cash": f"{cash:.6f}",
                "Shares": f"{shares:.6f}",
                "Equity": f"{equity:.6f}",
                "DrawdownPct": f"{drawdown_pct:.6f}",
                "Position": "LONG" if shares > 0 else "SHORT" if shares < 0 else "FLAT",
            }
        )

    ending_equity = parse_float(equity_rows[-1]["Equity"]) if equity_rows else initial_capital
    total_return_pct = 0.0 if initial_capital == 0 else ((ending_equity - initial_capital) / initial_capital) * 100.0
    win_rate_pct = 0.0 if closed_trades == 0 else (winning_trades / closed_trades) * 100.0
    summary = {
        "Mode": mode,
        "InitialCapital": f"{initial_capital:.2f}",
        "EndingEquity": f"{ending_equity:.2f}",
        "TotalReturnPct": f"{total_return_pct:.4f}",
        "Trades": str(len(trades)),
        "ClosedTrades": str(closed_trades),
        "WinRatePct": f"{win_rate_pct:.2f}",
        "MaxDrawdownPct": f"{max_drawdown_pct:.4f}",
        "LastDate": equity_rows[-1]["Date"] if equity_rows else "",
    }
    return trades, equity_rows, summary


def write_markdown_summary(
    *,
    path: Path,
    timeframe: str,
    mode: str,
    initial_capital: float,
    input_dir: Path,
    summary_rows: list[dict[str, str]],
) -> None:
    lines = [
        f"# TV-Match Backtest Comparison ({timeframe.title()}, {mode.replace('_', ' ')})",
        "",
        f"- Source: `{input_dir}`",
        "- Fill model: execute on `FillPrice` of the event bar (`MomLE` / `MomSE`)",
        "- Costs: none (fees/slippage not modeled)",
        f"- Starting capital per symbol: ${initial_capital:,.0f}",
        "",
        "| Symbol | Mode | Ending Equity | Return % | Trades | Closed Trades | Win Rate % | Max DD % | Last Date |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['Symbol']} | {row['Mode']} | {row['EndingEquity']} | {row['TotalReturnPct']} | "
            f"{row['Trades']} | {row['ClosedTrades']} | {row['WinRatePct']} | {row['MaxDrawdownPct']} | {row['LastDate']} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    watchlist_path = Path(args.watchlist)
    timeframe = normalize_timeframe(args.timeframe)
    symbols = parse_symbols(args.symbols, watchlist_path)
    input_dir = Path(args.input_dir) if args.input_dir else momentum_tv_match_output_dir(timeframe)
    out_dir = Path(args.out_dir)

    if args.initial_capital <= 0:
        print("[error] --initial-capital must be > 0")
        return 1
    if not symbols:
        print("[error] No symbols selected")
        return 1

    modes = ["long_only", "long_short"] if args.mode == "both" else [args.mode]
    comparison_rows: list[dict[str, str]] = []

    for symbol in symbols:
        input_path = input_dir / f"{symbol}.csv"
        if not input_path.exists():
            print(f"[error] Missing momentum match file: {input_path}")
            return 1
        rows = read_rows(input_path)
        symbol_summary_rows: list[dict[str, str]] = []

        for mode in modes:
            trades, equity_rows, summary = run_backtest(mode=mode, rows=rows, initial_capital=args.initial_capital)
            summary_with_symbol = {"Symbol": symbol, **summary}
            symbol_summary_rows.append(summary_with_symbol)
            comparison_rows.append(summary_with_symbol)

            trades_path = out_dir / f"{symbol}_{timeframe}_{mode}_trades.csv"
            equity_path = out_dir / f"{symbol}_{timeframe}_{mode}_equity.csv"
            write_csv(
                trades_path,
                [
                    "Date",
                    "Action",
                    "Price",
                    "SharesDelta",
                    "CashAfter",
                    "SharesAfter",
                    "EquityAfter",
                    "RealizedPnL",
                ],
                [
                    {
                        "Date": trade.date,
                        "Action": trade.action,
                        "Price": f"{trade.price:.6f}",
                        "SharesDelta": f"{trade.shares_delta:.6f}",
                        "CashAfter": f"{trade.cash_after:.6f}",
                        "SharesAfter": f"{trade.shares_after:.6f}",
                        "EquityAfter": f"{trade.equity_after:.6f}",
                        "RealizedPnL": f"{trade.realized_pnl:.6f}",
                    }
                    for trade in trades
                ],
            )
            write_csv(
                equity_path,
                ["Date", "Close", "Event", "FillPrice", "Action", "Cash", "Shares", "Equity", "DrawdownPct", "Position"],
                equity_rows,
            )
            print(f"[ok] {symbol} {mode} trades:  {trades_path}")
            print(f"[ok] {symbol} {mode} equity:  {equity_path}")

        summary_path = out_dir / f"{symbol}_tv_match_{timeframe}_summary.csv"
        write_csv(
            summary_path,
            ["Symbol", "Mode", "InitialCapital", "EndingEquity", "TotalReturnPct", "Trades", "ClosedTrades", "WinRatePct", "MaxDrawdownPct", "LastDate"],
            symbol_summary_rows,
        )
        print(f"[ok] {symbol} summary: {summary_path}")

    stem = comparison_stem(symbols, timeframe, args.mode)
    comparison_csv = out_dir / f"{stem}.csv"
    comparison_md = out_dir / f"{stem}.md"
    write_csv(
        comparison_csv,
        ["Symbol", "Mode", "InitialCapital", "EndingEquity", "TotalReturnPct", "Trades", "ClosedTrades", "WinRatePct", "MaxDrawdownPct", "LastDate"],
        comparison_rows,
    )
    write_markdown_summary(
        path=comparison_md,
        timeframe=timeframe,
        mode=args.mode,
        initial_capital=args.initial_capital,
        input_dir=input_dir,
        summary_rows=comparison_rows,
    )
    print(f"[ok] Comparison CSV: {comparison_csv}")
    print(f"[ok] Comparison MD:  {comparison_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
