#!/usr/bin/env python3
"""Run tiered-entry experiments with unchanged exit discipline and save visuals."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.paths import DATA_DAILY_DIR, DATA_MONTHLY_DIR, DATA_WEEKLY_DIR, STRATEGIES_ENTRY_TIER_DIR


@dataclass
class Policy:
    name: str
    candidate_alloc: float
    early_alloc: float
    confirmed_alloc: float


@dataclass
class Config:
    symbols: list[str]
    daily_input_dir: Path
    weekly_input_dir: Path
    monthly_input_dir: Path
    allow_derived_fallback: bool
    out_dir: Path
    date_from: date | None
    date_to: date | None
    initial_capital: float
    daily_ema_period: int
    weekly_ema_period: int
    weekly_confirm_bars: int
    weekly_fail_bars: int
    monthly_fast_ema_period: int
    monthly_slow_ema_period: int
    breakout_lookback: int
    momentum_length: int
    atr_period: int
    atr_trail_mult: float
    hard_stop_atr_mult: float
    make_plots: bool


@dataclass
class Summary:
    mode: str
    symbol: str
    rows: int
    trade_executions: int
    round_trips: int
    win_rate_pct: float
    ending_equity: float
    total_return_pct: float
    realized_pnl: float
    max_drawdown_pct: float
    latest_date: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchlist", default="watchlist.txt", help="Path to watchlist")
    parser.add_argument("--symbols", default="", help="Optional comma-separated symbols override")
    parser.add_argument("--daily-input-dir", default=str(DATA_DAILY_DIR))
    parser.add_argument("--weekly-input-dir", default=str(DATA_WEEKLY_DIR))
    parser.add_argument("--monthly-input-dir", default=str(DATA_MONTHLY_DIR))
    parser.add_argument("--allow-derived-fallback", action="store_true")
    parser.add_argument("--out-dir", default=str(STRATEGIES_ENTRY_TIER_DIR))
    parser.add_argument("--date-from", default="2023-01-03", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--date-to", default="", help="Inclusive YYYY-MM-DD")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--daily-ema-period", type=int, default=50)
    parser.add_argument("--weekly-ema-period", type=int, default=13)
    parser.add_argument("--weekly-confirm-bars", type=int, default=2)
    parser.add_argument("--weekly-fail-bars", type=int, default=2)
    parser.add_argument("--monthly-fast-ema-period", type=int, default=10)
    parser.add_argument("--monthly-slow-ema-period", type=int, default=20)
    parser.add_argument("--breakout-lookback", type=int, default=20)
    parser.add_argument("--momentum-length", type=int, default=24)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--atr-trail-mult", type=float, default=2.5)
    parser.add_argument("--hard-stop-atr-mult", type=float, default=1.5)
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG chart generation")
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


def parse_date_or_none(value: str) -> date | None:
    text = value.strip()
    if not text:
        return None
    return date.fromisoformat(text)


def parse_symbols(args: argparse.Namespace) -> list[str]:
    if args.symbols.strip():
        return [part.strip().upper() for part in args.symbols.split(",") if part.strip()]
    return read_watchlist(Path(args.watchlist))


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


def compute_prev_rolling_max(values: list[float], lookback: int) -> list[float | None]:
    output: list[float | None] = [None] * len(values)
    if lookback <= 0:
        return output
    for idx in range(lookback, len(values)):
        output[idx] = max(values[idx - lookback : idx])
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

    seed_window = tr_values[:period]
    if any(v is None for v in seed_window):
        return output
    seed = sum(float(v) for v in seed_window) / period
    output[period - 1] = seed
    prev = seed
    for idx in range(period, len(rows)):
        tr = tr_values[idx]
        if tr is None:
            continue
        current = ((prev * (period - 1)) + tr) / period
        output[idx] = current
        prev = current
    return output


def read_ohlc_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for raw in reader:
            try:
                row_date = date.fromisoformat((raw.get("Date") or "").strip())
            except ValueError:
                continue
            def f(name: str) -> float | None:
                text = (raw.get(name) or "").strip()
                return None if not text else float(text)
            close = f("Close")
            if close is None:
                continue
            open_price = f("Open")
            high_price = f("High")
            low_price = f("Low")
            rows.append(
                {
                    "DateObj": row_date,
                    "Date": row_date.isoformat(),
                    "Open": close if open_price is None else open_price,
                    "High": close if high_price is None else high_price,
                    "Low": close if low_price is None else low_price,
                    "Close": close,
                }
            )
    rows.sort(key=lambda r: str(r["Date"]))
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    return rows


def read_close_rows(path: Path) -> list[tuple[date, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[tuple[date, float]] = []
        for raw in reader:
            text = (raw.get("Close") or "").strip()
            if not text:
                continue
            rows.append((date.fromisoformat(raw["Date"]), float(text)))
    rows.sort(key=lambda x: x[0])
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    return rows


def aggregate_closes(rows: list[dict[str, object]], timeframe: str) -> list[tuple[date, float]]:
    out: list[tuple[date, float]] = []
    current_key: tuple[int, int] | None = None
    current_date: date | None = None
    current_close: float | None = None
    for row in rows:
        row_date = row["DateObj"]
        close = row["Close"]
        assert isinstance(row_date, date)
        assert isinstance(close, float)
        if timeframe == "weekly":
            y, w, _ = row_date.isocalendar()
            key = (y, w)
        elif timeframe == "monthly":
            key = (row_date.year, row_date.month)
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if key != current_key:
            if current_date is not None and current_close is not None:
                out.append((current_date, current_close))
            current_key = key
            current_date = row_date
            current_close = close
        else:
            current_date = row_date
            current_close = close
    if current_date is not None and current_close is not None:
        out.append((current_date, current_close))
    return out


def map_to_daily(daily_dates: list[date], period_dates: list[date], period_values: list[object]) -> list[object | None]:
    mapped: list[object | None] = [None] * len(daily_dates)
    idx = -1
    for i, day in enumerate(daily_dates):
        while idx + 1 < len(period_dates) and period_dates[idx + 1] <= day:
            idx += 1
        mapped[i] = period_values[idx] if idx >= 0 else None
    return mapped


def build_weekly_regime(
    weekly_closes: list[float],
    weekly_ema: list[float | None],
    confirm_bars: int,
) -> tuple[list[bool], list[int]]:
    risk_on: list[bool] = [False] * len(weekly_closes)
    below_count: list[int] = [0] * len(weekly_closes)
    above_streak = 0
    below_streak = 0
    state = False
    for idx, close_price in enumerate(weekly_closes):
        ema = weekly_ema[idx]
        prev_ema = weekly_ema[idx - 1] if idx > 0 else None
        if ema is None:
            above_streak = 0
            below_streak = 0
            state = False
            continue
        close_above = close_price > ema
        close_below = close_price < ema
        ema_rising = prev_ema is not None and ema > prev_ema
        if close_above:
            above_streak += 1
        else:
            above_streak = 0
        if close_below:
            below_streak += 1
        else:
            below_streak = 0
        below_count[idx] = below_streak
        base_up = close_above and ema_rising
        if not base_up:
            state = False
        elif not state and above_streak >= confirm_bars:
            state = True
        risk_on[idx] = state
    return risk_on, below_count


def load_symbol_inputs(cfg: Config, symbol: str) -> tuple[list[dict[str, object]], list[tuple[date, float]], list[tuple[date, float]], str, str]:
    daily_path = cfg.daily_input_dir / f"{symbol}.csv"
    daily_rows = read_ohlc_rows(daily_path)
    weekly_path = cfg.weekly_input_dir / f"{symbol}.csv"
    monthly_path = cfg.monthly_input_dir / f"{symbol}.csv"
    if weekly_path.exists():
        weekly_rows = read_close_rows(weekly_path)
        weekly_source = "EXTERNAL"
    elif cfg.allow_derived_fallback:
        weekly_rows = aggregate_closes(daily_rows, "weekly")
        weekly_source = "DERIVED_DAILY"
    else:
        raise FileNotFoundError(f"Weekly input missing: {weekly_path}")
    if monthly_path.exists():
        monthly_rows = read_close_rows(monthly_path)
        monthly_source = "EXTERNAL"
    elif cfg.allow_derived_fallback:
        monthly_rows = aggregate_closes(daily_rows, "monthly")
        monthly_source = "DERIVED_DAILY"
    else:
        raise FileNotFoundError(f"Monthly input missing: {monthly_path}")
    return daily_rows, weekly_rows, monthly_rows, weekly_source, monthly_source


def filter_daily_rows(rows: list[dict[str, object]], start: date | None, end: date | None) -> list[dict[str, object]]:
    out = []
    for row in rows:
        day = row["DateObj"]
        assert isinstance(day, date)
        if start is not None and day < start:
            continue
        if end is not None and day > end:
            continue
        out.append(row)
    if not out:
        raise ValueError("No daily rows after date filter")
    return out


def run_policy(
    symbol: str,
    policy: Policy,
    cfg: Config,
    daily_rows: list[dict[str, object]],
    weekly_rows: list[tuple[date, float]],
    monthly_rows: list[tuple[date, float]],
    weekly_source: str,
    monthly_source: str,
) -> tuple[pd.DataFrame, Summary]:
    rows = filter_daily_rows(daily_rows, cfg.date_from, cfg.date_to)
    dates = [r["DateObj"] for r in rows]
    closes = [float(r["Close"]) for r in rows]
    daily_ema = compute_ema(closes, cfg.daily_ema_period)
    momentum = compute_momentum(closes, cfg.momentum_length)
    atr_values = compute_atr_wilder(rows, cfg.atr_period)
    breakout_prev_high = compute_prev_rolling_max(closes, cfg.breakout_lookback)

    weekly_dates = [d for d, _ in weekly_rows]
    weekly_closes = [c for _, c in weekly_rows]
    weekly_ema = compute_ema(weekly_closes, cfg.weekly_ema_period)
    weekly_risk_on, weekly_below = build_weekly_regime(weekly_closes, weekly_ema, cfg.weekly_confirm_bars)
    weekly_risk_on_daily = map_to_daily(dates, weekly_dates, weekly_risk_on)
    weekly_below_daily = map_to_daily(dates, weekly_dates, weekly_below)

    monthly_dates = [d for d, _ in monthly_rows]
    monthly_closes = [c for _, c in monthly_rows]
    monthly_fast = compute_ema(monthly_closes, cfg.monthly_fast_ema_period)
    monthly_slow = compute_ema(monthly_closes, cfg.monthly_slow_ema_period)
    monthly_strict: list[bool] = [False] * len(monthly_closes)
    monthly_early: list[bool] = [False] * len(monthly_closes)
    for idx, close_price in enumerate(monthly_closes):
        fast = monthly_fast[idx]
        slow = monthly_slow[idx]
        prev_fast = monthly_fast[idx - 1] if idx > 0 else None
        fast_rising = bool(fast is not None and prev_fast is not None and fast > prev_fast)
        monthly_early[idx] = bool(fast is not None and close_price > fast and fast_rising)
        monthly_strict[idx] = bool(fast is not None and slow is not None and close_price > fast and fast > slow)

    monthly_strict_daily = map_to_daily(dates, monthly_dates, monthly_strict)
    monthly_early_daily = map_to_daily(dates, monthly_dates, monthly_early)

    cash = cfg.initial_capital
    shares = 0
    avg_entry_price = 0.0
    position_alloc_target = 0.0
    hard_stop: float | None = None
    trail_stop: float | None = None

    pending_target_alloc: float | None = None
    pending_reason = ""
    pending_source_date = ""

    trade_executions = 0
    round_trips = 0
    wins = 0
    realized_total = 0.0
    equity_peak = cfg.initial_capital
    max_drawdown = 0.0

    output_rows: list[dict[str, object]] = []

    for idx, row in enumerate(rows):
        open_price = float(row["Open"])
        close_price = float(row["Close"])
        date_text = str(row["Date"])

        executed_action = ""
        executed_reason = ""
        executed_source_date = ""
        realized_trade_pnl = 0.0

        if pending_target_alloc is not None:
            target_alloc = pending_target_alloc
            pending_target_alloc = None
            if target_alloc <= 0 and shares > 0:
                proceeds = shares * open_price
                realized_trade_pnl = (open_price - avg_entry_price) * shares
                realized_total += realized_trade_pnl
                round_trips += 1
                if realized_trade_pnl > 0:
                    wins += 1
                cash += proceeds
                shares = 0
                avg_entry_price = 0.0
                position_alloc_target = 0.0
                hard_stop = None
                trail_stop = None
                trade_executions += 1
                executed_action = "SELL"
                executed_reason = pending_reason
                executed_source_date = pending_source_date
            elif target_alloc > 0:
                equity_open = cash + shares * open_price
                target_value = target_alloc * equity_open
                target_shares = int(target_value // open_price) if open_price > 0 else shares
                if target_shares > shares:
                    delta = target_shares - shares
                    cost = delta * open_price
                    if cost <= cash and delta > 0:
                        new_total_shares = shares + delta
                        avg_entry_price = (
                            ((avg_entry_price * shares) + (open_price * delta)) / new_total_shares if shares > 0 else open_price
                        )
                        cash -= cost
                        shares = new_total_shares
                        position_alloc_target = target_alloc
                        atr_now = atr_values[idx]
                        if atr_now is not None:
                            candidate_hard = open_price - (cfg.hard_stop_atr_mult * atr_now)
                            hard_stop = candidate_hard if hard_stop is None else max(hard_stop, candidate_hard)
                        trade_executions += 1
                        executed_action = "BUY" if shares == delta else "SCALE_IN"
                        executed_reason = pending_reason
                        executed_source_date = pending_source_date

        pending_reason = ""
        pending_source_date = ""

        equity = cash + shares * close_price
        equity_peak = max(equity_peak, equity)
        drawdown_pct = 0.0 if equity_peak <= 0 else ((equity_peak - equity) / equity_peak) * 100.0
        max_drawdown = max(max_drawdown, drawdown_pct)

        d_ema = daily_ema[idx]
        mom = momentum[idx]
        atr = atr_values[idx]
        prev_close = closes[idx - 1] if idx > 0 else None
        prev_ema = daily_ema[idx - 1] if idx > 0 else None
        prev_high = breakout_prev_high[idx]
        weekly_on = weekly_risk_on_daily[idx] is True
        monthly_on = monthly_strict_daily[idx] is True
        monthly_early_on = monthly_early_daily[idx] is True
        daily_breakout = bool(d_ema is not None and prev_high is not None and close_price > d_ema and close_price > prev_high)
        momentum_positive = bool(mom is not None and mom > 0)
        pullback_reclaim = bool(
            d_ema is not None
            and prev_ema is not None
            and prev_close is not None
            and prev_close <= prev_ema
            and close_price > d_ema
            and momentum_positive
        )
        trigger = daily_breakout or pullback_reclaim
        candidate = weekly_on and trigger
        confirmed = monthly_on and weekly_on and trigger
        early_confirm = weekly_on and monthly_early_on and trigger and not monthly_on

        tier_signal = "NONE"
        desired_alloc = 0.0
        if confirmed:
            tier_signal = "CONFIRMED"
            desired_alloc = policy.confirmed_alloc
        elif early_confirm:
            tier_signal = "EARLY_CONFIRM"
            desired_alloc = policy.early_alloc
        elif candidate:
            tier_signal = "CANDIDATE"
            desired_alloc = policy.candidate_alloc

        atr_trail_candidate: float | None = None
        if shares > 0 and atr is not None:
            atr_trail_candidate = close_price - (cfg.atr_trail_mult * atr)
            trail_stop = atr_trail_candidate if trail_stop is None else max(trail_stop, atr_trail_candidate)

        active_stop: float | None = None
        if shares > 0:
            stop_candidates: list[float] = []
            if trail_stop is not None:
                stop_candidates.append(trail_stop)
            if hard_stop is not None:
                stop_candidates.append(hard_stop)
            if stop_candidates:
                active_stop = max(stop_candidates)
        stop_exit = bool(shares > 0 and active_stop is not None and close_price < active_stop)
        wb = weekly_below_daily[idx]
        weekly_fail_exit = bool(shares > 0 and isinstance(wb, int) and wb >= cfg.weekly_fail_bars)
        exit_reason = ""
        if stop_exit:
            exit_reason = "ATR_HARD_STOP"
        elif weekly_fail_exit:
            exit_reason = "WEEKLY_FAIL"

        if idx < len(rows) - 1:
            if shares > 0 and exit_reason:
                pending_target_alloc = 0.0
                pending_reason = exit_reason
                pending_source_date = date_text
            elif shares == 0 and desired_alloc > 0:
                pending_target_alloc = desired_alloc
                pending_reason = tier_signal
                pending_source_date = date_text
            elif shares > 0 and desired_alloc > position_alloc_target + 1e-9:
                pending_target_alloc = desired_alloc
                pending_reason = f"UPGRADE_{tier_signal}"
                pending_source_date = date_text

        output_rows.append(
            {
                "Date": date_text,
                "Open": open_price,
                "Close": close_price,
                "WeeklyTrendSource": weekly_source,
                "MonthlyTrendSource": monthly_source,
                "Policy": policy.name,
                "DailyEMA": d_ema,
                "ATR": atr,
                "MonthlyRiskOn": 1 if monthly_on else 0,
                "MonthlyEarlyOn": 1 if monthly_early_on else 0,
                "WeeklyRiskOn": 1 if weekly_on else 0,
                "DailyBreakout": 1 if daily_breakout else 0,
                "PullbackReclaim": 1 if pullback_reclaim else 0,
                "EntryTrigger": 1 if trigger else 0,
                "CandidateEntry": 1 if candidate else 0,
                "ConfirmedEntry": 1 if confirmed else 0,
                "EarlyConfirmEntry": 1 if early_confirm else 0,
                "TierSignal": tier_signal,
                "DesiredAlloc": desired_alloc,
                "PositionAllocTarget": position_alloc_target,
                "ATRTrailStop": trail_stop,
                "HardStop": hard_stop,
                "ActiveStop": active_stop,
                "StopExit": 1 if stop_exit else 0,
                "WeeklyFailExit": 1 if weekly_fail_exit else 0,
                "Equity": equity,
                "EquityPeak": equity_peak,
                "DrawdownPct": drawdown_pct,
                "PositionState": "LONG" if shares > 0 else "FLAT",
                "Shares": shares,
                "Cash": cash,
                "PendingTargetAlloc": pending_target_alloc,
                "PendingReason": pending_reason,
                "ExecutedAction": executed_action,
                "ExecutedReason": executed_reason,
                "ExecutedSourceDate": executed_source_date,
                "RealizedTradePnL": realized_trade_pnl if executed_action == "SELL" else None,
                "ExitReason": exit_reason,
            }
        )

    df = pd.DataFrame(output_rows)
    final_equity = float(df.iloc[-1]["Equity"])
    total_return_pct = ((final_equity - cfg.initial_capital) / cfg.initial_capital) * 100.0
    win_rate_pct = (wins / round_trips) * 100.0 if round_trips > 0 else 0.0
    summary = Summary(
        mode=policy.name,
        symbol=symbol,
        rows=len(df),
        trade_executions=trade_executions,
        round_trips=round_trips,
        win_rate_pct=win_rate_pct,
        ending_equity=final_equity,
        total_return_pct=total_return_pct,
        realized_pnl=realized_total,
        max_drawdown_pct=max_drawdown,
        latest_date=str(df.iloc[-1]["Date"]),
    )
    return df, summary


def make_symbol_plot(symbol: str, dfs: dict[str, pd.DataFrame], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    modes = list(dfs.keys())
    fig, axes = plt.subplots(len(modes), 2, figsize=(15, 4.4 * len(modes)), sharex=False)
    if len(modes) == 1:
        axes = [axes]  # type: ignore[assignment]

    for row_idx, mode in enumerate(modes):
        df = dfs[mode]
        dates = pd.to_datetime(df["Date"])
        ax_p = axes[row_idx][0]
        ax_e = axes[row_idx][1]

        ax_p.plot(dates, df["Close"], color="#1f77b4", linewidth=1.3, label="Close")
        if "DailyEMA" in df.columns:
            ax_p.plot(dates, pd.to_numeric(df["DailyEMA"], errors="coerce"), color="#ff7f0e", linewidth=1.0, label="Daily EMA")

        buys = df[df["ExecutedAction"].isin(["BUY", "SCALE_IN"])]
        sells = df[df["ExecutedAction"] == "SELL"]
        if not buys.empty:
            ax_p.scatter(pd.to_datetime(buys["Date"]), buys["Close"], marker="^", s=40, color="#2ca02c", label="Buy/Scale In")
        if not sells.empty:
            ax_p.scatter(pd.to_datetime(sells["Date"]), sells["Close"], marker="v", s=40, color="#d62728", label="Sell")
        ax_p.set_title(f"{symbol} | {mode} | Entries/Exits")
        ax_p.grid(alpha=0.25)
        ax_p.legend(loc="best", fontsize=8)

        ax_e.plot(dates, df["Equity"], color="#1f77b4", linewidth=1.5, label="Equity")
        ax_dd = ax_e.twinx()
        ax_dd.fill_between(dates, 0.0, pd.to_numeric(df["DrawdownPct"], errors="coerce"), color="#d62728", alpha=0.15, label="Drawdown %")
        ax_e.set_title(f"{symbol} | {mode} | Equity Curve")
        ax_e.grid(alpha=0.25)
        l1, t1 = ax_e.get_legend_handles_labels()
        l2, t2 = ax_dd.get_legend_handles_labels()
        ax_e.legend(l1 + l2, t1 + t2, loc="best", fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def write_mode_summary(path: Path, summaries: list[Summary]) -> None:
    rows = [
        {
            "Mode": item.mode,
            "Symbol": item.symbol,
            "Rows": item.rows,
            "TradeExecutions": item.trade_executions,
            "RoundTrips": item.round_trips,
            "WinRatePct": f"{item.win_rate_pct:.2f}",
            "EndingEquity": f"{item.ending_equity:.2f}",
            "TotalReturnPct": f"{item.total_return_pct:.4f}",
            "RealizedPnL": f"{item.realized_pnl:.2f}",
            "MaxDrawdownPct": f"{item.max_drawdown_pct:.4f}",
            "LatestDate": item.latest_date,
        }
        for item in summaries
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> int:
    args = parse_args()
    symbols = parse_symbols(args)
    if not symbols:
        print("[error] no symbols found")
        return 1
    cfg = Config(
        symbols=symbols,
        daily_input_dir=Path(args.daily_input_dir),
        weekly_input_dir=Path(args.weekly_input_dir),
        monthly_input_dir=Path(args.monthly_input_dir),
        allow_derived_fallback=args.allow_derived_fallback,
        out_dir=Path(args.out_dir),
        date_from=parse_date_or_none(args.date_from),
        date_to=parse_date_or_none(args.date_to),
        initial_capital=args.initial_capital,
        daily_ema_period=args.daily_ema_period,
        weekly_ema_period=args.weekly_ema_period,
        weekly_confirm_bars=args.weekly_confirm_bars,
        weekly_fail_bars=args.weekly_fail_bars,
        monthly_fast_ema_period=args.monthly_fast_ema_period,
        monthly_slow_ema_period=args.monthly_slow_ema_period,
        breakout_lookback=args.breakout_lookback,
        momentum_length=args.momentum_length,
        atr_period=args.atr_period,
        atr_trail_mult=args.atr_trail_mult,
        hard_stop_atr_mult=args.hard_stop_atr_mult,
        make_plots=not args.no_plots,
    )
    policies = [
        Policy("baseline_confirmed", candidate_alloc=0.0, early_alloc=0.0, confirmed_alloc=1.0),
        Policy("tiered_candidate_25_early_50", candidate_alloc=0.25, early_alloc=0.5, confirmed_alloc=1.0),
        Policy("relaxed_weekly_full", candidate_alloc=1.0, early_alloc=1.0, confirmed_alloc=1.0),
    ]

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: list[Summary] = []
    per_symbol_plots: dict[str, dict[str, pd.DataFrame]] = {}

    for symbol in cfg.symbols:
        try:
            daily_rows, weekly_rows, monthly_rows, w_source, m_source = load_symbol_inputs(cfg, symbol)
            per_symbol_plots[symbol] = {}
            for policy in policies:
                df, summary = run_policy(
                    symbol=symbol,
                    policy=policy,
                    cfg=cfg,
                    daily_rows=daily_rows,
                    weekly_rows=weekly_rows,
                    monthly_rows=monthly_rows,
                    weekly_source=w_source,
                    monthly_source=m_source,
                )
                mode_dir = cfg.out_dir / policy.name
                mode_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(mode_dir / f"{symbol}.csv", index=False)
                per_symbol_plots[symbol][policy.name] = df
                all_summaries.append(summary)
                print(f"[ok] {policy.name} | {symbol} -> {mode_dir / f'{symbol}.csv'} ({len(df)} rows)")
        except Exception as exc:
            print(f"[fail] {symbol} -> {exc}")

    write_mode_summary(cfg.out_dir / "summary.csv", all_summaries)

    if cfg.make_plots:
        for symbol, mode_frames in per_symbol_plots.items():
            if mode_frames:
                out_path = cfg.out_dir / "plots" / f"{symbol}_modes.png"
                make_symbol_plot(symbol, mode_frames, out_path)

    print("\nExperiment Complete")
    print(f"  symbols: {len(cfg.symbols)}")
    print(f"  policies: {len(policies)}")
    print(f"  out dir: {cfg.out_dir}")
    print(f"  summary: {cfg.out_dir / 'summary.csv'}")
    if cfg.make_plots:
        print(f"  plots: {cfg.out_dir / 'plots'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
