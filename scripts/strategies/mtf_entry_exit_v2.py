#!/usr/bin/env python3
"""Run a regime-aware MTF long strategy (v2) with breakout/reclaim entries."""

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
    weekly_confirm_bars: int
    weekly_fail_bars: int
    monthly_fast_ema_period: int
    monthly_slow_ema_period: int
    monthly_regime_mode: str
    breakout_lookback: int
    momentum_length: int
    atr_period: int
    atr_trail_mult: float
    hard_stop_atr_mult: float
    initial_capital: float
    enable_equity_kill: bool
    equity_ema_period: int
    equity_kill_mode: str
    kill_max_drawdown_pct: float
    kill_cooldown_bars: int


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
    parser.add_argument("--daily-input-dir", default="out/daily", help="Directory with daily OHLC CSV files")
    parser.add_argument("--weekly-input-dir", default="out/weekly", help="Directory with weekly OHLC CSV files")
    parser.add_argument("--monthly-input-dir", default="out/monthly", help="Directory with monthly OHLC CSV files")
    parser.add_argument(
        "--allow-derived-fallback",
        action="store_true",
        help="Allow deriving weekly/monthly bars from daily if external files are missing",
    )
    parser.add_argument("--out-dir", default="out/mtf_entry_exit_v2", help="Directory for per-symbol output CSVs")
    parser.add_argument(
        "--latest-file",
        default="out/mtf_entry_exit_v2_latest.csv",
        help="CSV path for latest strategy status per symbol",
    )
    parser.add_argument(
        "--summary-file",
        default="out/mtf_entry_exit_v2_summary.csv",
        help="CSV path for summary metrics per symbol",
    )
    parser.add_argument(
        "--errors-file",
        default="out/mtf_entry_exit_v2_errors.csv",
        help="CSV path for per-symbol failures",
    )
    parser.add_argument("--daily-ema-period", type=int, default=50, help="Daily EMA period for entry filters")
    parser.add_argument("--weekly-ema-period", type=int, default=13, help="Weekly EMA period for regime filter")
    parser.add_argument(
        "--weekly-confirm-bars",
        type=int,
        default=2,
        help="Consecutive weekly closes above weekly EMA required to turn regime ON",
    )
    parser.add_argument(
        "--weekly-fail-bars",
        type=int,
        default=2,
        help="Consecutive weekly closes below weekly EMA required for trend-failure exit",
    )
    parser.add_argument("--monthly-fast-ema-period", type=int, default=10, help="Monthly fast EMA period")
    parser.add_argument("--monthly-slow-ema-period", type=int, default=20, help="Monthly slow EMA period")
    parser.add_argument(
        "--monthly-regime-mode",
        choices=["strict", "early"],
        default="strict",
        help="strict: close>fast and fast>slow, early: close>fast and fast rising",
    )
    parser.add_argument(
        "--breakout-lookback",
        type=int,
        default=20,
        help="Daily breakout lookback bars (close > prior N-bar close high)",
    )
    parser.add_argument("--momentum-length", type=int, default=24, help="Daily momentum lookback length")
    parser.add_argument("--atr-period", type=int, default=14, help="ATR period for stop calculations")
    parser.add_argument("--atr-trail-mult", type=float, default=2.5, help="ATR multiplier for trailing stop")
    parser.add_argument(
        "--hard-stop-atr-mult",
        type=float,
        default=1.5,
        help="ATR multiplier for fixed hard stop from entry",
    )
    parser.add_argument("--initial-capital", type=float, default=100000.0, help="Starting capital")
    parser.add_argument(
        "--enable-equity-kill",
        action="store_true",
        help="Enable equity kill-switch as a safety net",
    )
    parser.add_argument("--equity-ema-period", type=int, default=50, help="Equity EMA period for kill-switch")
    parser.add_argument(
        "--equity-kill-mode",
        choices=["any", "both"],
        default="both",
        help="Kill-switch logic: any = below EMA OR drawdown trip, both = below EMA AND drawdown trip",
    )
    parser.add_argument(
        "--kill-max-drawdown-pct",
        type=float,
        default=20.0,
        help="Drawdown threshold for kill-switch",
    )
    parser.add_argument(
        "--kill-cooldown-bars",
        type=int,
        default=10,
        help="Bars to block new entries after kill-switch exit",
    )
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


def parse_float(value: str | None) -> float | None:
    raw = (value or "").strip()
    if not raw:
        return None
    return float(raw)


def parse_int(value: str | None) -> int:
    raw = (value or "").strip()
    if not raw:
        return 0
    return int(float(raw))


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

    window = tr_values[:period]
    if any(v is None for v in window):
        return output

    seed = sum(float(v) for v in window) / period
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


def read_daily_rows(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, object]] = []
        for raw in reader:
            close_price = parse_float(raw.get("Close"))
            if close_price is None:
                continue
            row_date = date.fromisoformat(raw["Date"])
            open_price = parse_float(raw.get("Open"))
            high_price = parse_float(raw.get("High"))
            low_price = parse_float(raw.get("Low"))
            rows.append(
                {
                    "Date": raw["Date"],
                    "DateObj": row_date,
                    "Open": close_price if open_price is None else open_price,
                    "High": close_price if high_price is None else high_price,
                    "Low": close_price if low_price is None else low_price,
                    "Close": close_price,
                }
            )
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    rows.sort(key=lambda r: r["Date"])
    return rows


def read_close_rows(path: Path) -> list[tuple[date, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: list[tuple[date, float]] = []
        for raw in reader:
            close_price = parse_float(raw.get("Close"))
            if close_price is None:
                continue
            rows.append((date.fromisoformat(raw["Date"]), close_price))
    if not rows:
        raise ValueError(f"No usable rows in {path}")
    rows.sort(key=lambda item: item[0])
    return rows


def aggregate_closes(rows: list[dict[str, object]], timeframe: str) -> list[tuple[date, float]]:
    buckets: list[tuple[date, float]] = []
    current_key: tuple[int, int] | None = None
    current_date: date | None = None
    current_close: float | None = None

    for row in rows:
        row_date = row.get("DateObj")
        close_price = row.get("Close")
        if not isinstance(row_date, date) or not isinstance(close_price, float):
            continue
        if timeframe == "weekly":
            iso_year, iso_week, _ = row_date.isocalendar()
            key = (iso_year, iso_week)
        elif timeframe == "monthly":
            key = (row_date.year, row_date.month)
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        if key != current_key:
            if current_date is not None and current_close is not None:
                buckets.append((current_date, current_close))
            current_key = key
            current_date = row_date
            current_close = close_price
        else:
            current_date = row_date
            current_close = close_price

    if current_date is not None and current_close is not None:
        buckets.append((current_date, current_close))
    return buckets


def map_to_daily(daily_dates: list[date], period_dates: list[date], period_values: list[object]) -> list[object | None]:
    mapped: list[object | None] = [None] * len(daily_dates)
    period_idx = -1
    for idx, day in enumerate(daily_dates):
        while period_idx + 1 < len(period_dates) and period_dates[period_idx + 1] <= day:
            period_idx += 1
        mapped[idx] = period_values[period_idx] if period_idx >= 0 else None
    return mapped


def build_weekly_regime(
    weekly_closes: list[float],
    weekly_ema: list[float | None],
    confirm_bars: int,
) -> tuple[list[bool], list[int], list[bool], list[bool]]:
    risk_on: list[bool] = [False] * len(weekly_closes)
    below_count: list[int] = [0] * len(weekly_closes)
    close_above_flags: list[bool] = [False] * len(weekly_closes)
    ema_rising_flags: list[bool] = [False] * len(weekly_closes)

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

        close_above_flags[idx] = close_above
        ema_rising_flags[idx] = ema_rising

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

    return risk_on, below_count, close_above_flags, ema_rising_flags


def run_strategy(
    symbol: str,
    daily_rows: list[dict[str, object]],
    weekly_rows: list[tuple[date, float]],
    monthly_rows: list[tuple[date, float]],
    cfg: StrategyConfig,
    weekly_source: str,
    monthly_source: str,
) -> tuple[list[dict[str, str]], SymbolSummary]:
    daily_dates = [row["DateObj"] for row in daily_rows]
    closes = [float(row["Close"]) for row in daily_rows]
    daily_ema = compute_ema(closes, cfg.daily_ema_period)
    momentum = compute_momentum(closes, cfg.momentum_length)
    atr_values = compute_atr_wilder(daily_rows, cfg.atr_period)
    breakout_prev_high = compute_prev_rolling_max(closes, cfg.breakout_lookback)

    weekly_dates = [item[0] for item in weekly_rows]
    weekly_closes = [item[1] for item in weekly_rows]
    weekly_ema = compute_ema(weekly_closes, cfg.weekly_ema_period)
    weekly_risk_on, weekly_below_count, weekly_close_above, weekly_ema_rising = build_weekly_regime(
        weekly_closes=weekly_closes,
        weekly_ema=weekly_ema,
        confirm_bars=cfg.weekly_confirm_bars,
    )
    weekly_risk_on_daily = map_to_daily(daily_dates, weekly_dates, weekly_risk_on)
    weekly_below_count_daily = map_to_daily(daily_dates, weekly_dates, weekly_below_count)
    weekly_ema_daily = map_to_daily(daily_dates, weekly_dates, weekly_ema)
    weekly_close_above_daily = map_to_daily(daily_dates, weekly_dates, weekly_close_above)
    weekly_ema_rising_daily = map_to_daily(daily_dates, weekly_dates, weekly_ema_rising)

    monthly_dates = [item[0] for item in monthly_rows]
    monthly_closes = [item[1] for item in monthly_rows]
    monthly_fast_ema = compute_ema(monthly_closes, cfg.monthly_fast_ema_period)
    monthly_slow_ema = compute_ema(monthly_closes, cfg.monthly_slow_ema_period)
    monthly_risk_on_flags: list[bool] = [False] * len(monthly_closes)
    monthly_fast_rising_flags: list[bool] = [False] * len(monthly_closes)
    for idx, close_price in enumerate(monthly_closes):
        fast = monthly_fast_ema[idx]
        slow = monthly_slow_ema[idx]
        prev_fast = monthly_fast_ema[idx - 1] if idx > 0 else None
        fast_rising = bool(fast is not None and prev_fast is not None and fast > prev_fast)
        monthly_fast_rising_flags[idx] = fast_rising
        if cfg.monthly_regime_mode == "early":
            monthly_risk_on_flags[idx] = bool(fast is not None and close_price > fast and fast_rising)
        else:
            monthly_risk_on_flags[idx] = bool(fast is not None and slow is not None and close_price > fast and fast > slow)

    monthly_risk_on_daily = map_to_daily(daily_dates, monthly_dates, monthly_risk_on_flags)
    monthly_fast_daily = map_to_daily(daily_dates, monthly_dates, monthly_fast_ema)
    monthly_slow_daily = map_to_daily(daily_dates, monthly_dates, monthly_slow_ema)
    monthly_fast_rising_daily = map_to_daily(daily_dates, monthly_dates, monthly_fast_rising_flags)

    cash = cfg.initial_capital
    shares = 0
    entry_cost = 0.0
    hard_stop: float | None = None
    trail_stop: float | None = None
    cooldown_remaining = 0

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
    equity_ema_count = 0
    equity_alpha = 2.0 / (cfg.equity_ema_period + 1.0)

    output_rows: list[dict[str, str]] = []

    for idx, row in enumerate(daily_rows):
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])
        date_text = str(row["Date"])

        executed_action = ""
        executed_reason = ""
        executed_source_date = ""
        realized_trade_pnl = 0.0

        # Execute prior-day signal at today's open.
        if pending_action == "BUY" and shares == 0 and open_price > 0:
            buy_shares = int(cash // open_price)
            if buy_shares > 0:
                cost = buy_shares * open_price
                shares = buy_shares
                cash -= cost
                entry_cost = cost
                trail_stop = None
                atr_at_entry = atr_values[idx]
                if atr_at_entry is not None:
                    hard_stop = open_price - (cfg.hard_stop_atr_mult * atr_at_entry)
                else:
                    hard_stop = None
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
            hard_stop = None
            trail_stop = None
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
            equity_ema_count = 1
        else:
            equity_ema = (equity - equity_ema) * equity_alpha + equity_ema
            equity_ema_count += 1
        equity_ema_ready = equity_ema_count >= cfg.equity_ema_period
        equity_below_ema = bool(equity_ema_ready and equity_ema is not None and equity < equity_ema)
        equity_drawdown_trip = drawdown_pct >= cfg.kill_max_drawdown_pct
        equity_kill = False
        if cfg.enable_equity_kill and shares > 0:
            if cfg.equity_kill_mode == "both":
                equity_kill = equity_below_ema and equity_drawdown_trip
            else:
                equity_kill = equity_below_ema or equity_drawdown_trip
            if equity_kill and cfg.kill_cooldown_bars > 0 and cooldown_remaining < cfg.kill_cooldown_bars:
                cooldown_remaining = cfg.kill_cooldown_bars

        d_ema = daily_ema[idx]
        mom = momentum[idx]
        atr = atr_values[idx]
        prev_close = closes[idx - 1] if idx > 0 else None
        prev_ema = daily_ema[idx - 1] if idx > 0 else None
        prev_high = breakout_prev_high[idx]

        monthly_risk_on = monthly_risk_on_daily[idx] is True
        weekly_risk_on_now = weekly_risk_on_daily[idx] is True
        regime_on = monthly_risk_on and weekly_risk_on_now

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
        entry_trigger = daily_breakout or pullback_reclaim
        entry_setup = regime_on and entry_trigger

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
        weekly_below = weekly_below_count_daily[idx]
        weekly_fail_exit = bool(shares > 0 and isinstance(weekly_below, int) and weekly_below >= cfg.weekly_fail_bars)

        exit_reason = ""
        if stop_exit:
            exit_reason = "ATR_HARD_STOP"
        elif weekly_fail_exit:
            exit_reason = "WEEKLY_FAIL"
        elif equity_kill:
            exit_reason = "EQUITY_KILL"

        entry_reason = ""
        if daily_breakout and pullback_reclaim:
            entry_reason = "BREAKOUT_AND_RECLAIM"
        elif daily_breakout:
            entry_reason = "BREAKOUT"
        elif pullback_reclaim:
            entry_reason = "RECLAIM"

        if idx < len(daily_rows) - 1:
            if shares > 0 and exit_reason:
                pending_action = "EXIT"
                pending_reason = exit_reason
                pending_source_date = date_text
            elif shares == 0 and cooldown_remaining == 0 and entry_setup:
                pending_action = "BUY"
                pending_reason = entry_reason or "ENTRY"
                pending_source_date = date_text

        output_rows.append(
            {
                "Date": date_text,
                "Open": f"{open_price:.6f}",
                "High": f"{high_price:.6f}",
                "Low": f"{low_price:.6f}",
                "Close": f"{close_price:.6f}",
                "WeeklyTrendSource": weekly_source,
                "MonthlyTrendSource": monthly_source,
                "MonthlyRegimeMode": cfg.monthly_regime_mode,
                "DailyEMA": "" if d_ema is None else f"{d_ema:.6f}",
                "Momentum": "" if mom is None else f"{mom:.6f}",
                "MomentumPositive": "1" if momentum_positive else "0",
                "BreakoutPrevHigh": "" if prev_high is None else f"{prev_high:.6f}",
                "DailyBreakout": "1" if daily_breakout else "0",
                "PullbackReclaim": "1" if pullback_reclaim else "0",
                "EntryTrigger": "1" if entry_trigger else "0",
                "MonthlyRiskOn": "1" if monthly_risk_on else "0",
                "MonthlyFastEMA": "" if monthly_fast_daily[idx] is None else f"{float(monthly_fast_daily[idx]):.6f}",
                "MonthlySlowEMA": "" if monthly_slow_daily[idx] is None else f"{float(monthly_slow_daily[idx]):.6f}",
                "MonthlyFastRising": "1" if monthly_fast_rising_daily[idx] is True else "0",
                "WeeklyRiskOn": "1" if weekly_risk_on_now else "0",
                "WeeklyEMA": "" if weekly_ema_daily[idx] is None else f"{float(weekly_ema_daily[idx]):.6f}",
                "WeeklyCloseAboveEMA": "1" if weekly_close_above_daily[idx] is True else "0",
                "WeeklyEMARising": "1" if weekly_ema_rising_daily[idx] is True else "0",
                "WeeklyBelowCount": str(weekly_below) if isinstance(weekly_below, int) else "",
                "RegimeOn": "1" if regime_on else "0",
                "EntrySetup": "1" if entry_setup else "0",
                "EntryReason": entry_reason,
                "ATR": "" if atr is None else f"{atr:.6f}",
                "ATRTrailCandidate": "" if atr_trail_candidate is None else f"{atr_trail_candidate:.6f}",
                "ATRTrailStop": "" if trail_stop is None else f"{trail_stop:.6f}",
                "HardStop": "" if hard_stop is None else f"{hard_stop:.6f}",
                "ActiveStop": "" if active_stop is None else f"{active_stop:.6f}",
                "StopExit": "1" if stop_exit else "0",
                "WeeklyFailExit": "1" if weekly_fail_exit else "0",
                "Equity": f"{equity:.6f}",
                "EquityPeak": f"{equity_peak:.6f}",
                "DrawdownPct": f"{drawdown_pct:.6f}",
                "EquityEMA": "" if equity_ema is None else f"{equity_ema:.6f}",
                "EquityBelowEMA": "1" if equity_below_ema else "0",
                "EquityDrawdownTrip": "1" if equity_drawdown_trip else "0",
                "EquityKill": "1" if equity_kill else "0",
                "CooldownRemaining": str(cooldown_remaining),
                "PositionState": "LONG" if shares > 0 else "FLAT",
                "Shares": str(shares),
                "Cash": f"{cash:.6f}",
                "ExitReason": exit_reason,
                "PendingAction": pending_action,
                "PendingReason": pending_reason,
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
        latest_date=str(daily_rows[-1]["Date"]),
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
        "WeeklyTrendSource",
        "MonthlyTrendSource",
        "MonthlyRegimeMode",
        "MonthlyRiskOn",
        "WeeklyRiskOn",
        "RegimeOn",
        "DailyBreakout",
        "PullbackReclaim",
        "EntrySetup",
        "ATRTrailStop",
        "HardStop",
        "ActiveStop",
        "WeeklyBelowCount",
        "DrawdownPct",
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


def write_summary(path: Path, summaries: list[SymbolSummary]) -> None:
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
        for item in summaries:
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
        for item in errors:
            writer.writerow({"symbol": item.symbol, "error": item.message})


def validate_args(args: argparse.Namespace) -> str | None:
    if args.daily_ema_period <= 0:
        return "--daily-ema-period must be > 0"
    if args.weekly_ema_period <= 0:
        return "--weekly-ema-period must be > 0"
    if args.weekly_confirm_bars <= 0:
        return "--weekly-confirm-bars must be > 0"
    if args.weekly_fail_bars <= 0:
        return "--weekly-fail-bars must be > 0"
    if args.monthly_fast_ema_period <= 0:
        return "--monthly-fast-ema-period must be > 0"
    if args.monthly_slow_ema_period <= 0:
        return "--monthly-slow-ema-period must be > 0"
    if args.monthly_fast_ema_period >= args.monthly_slow_ema_period:
        return "--monthly-fast-ema-period should be smaller than --monthly-slow-ema-period"
    if args.breakout_lookback <= 0:
        return "--breakout-lookback must be > 0"
    if args.momentum_length <= 0:
        return "--momentum-length must be > 0"
    if args.atr_period <= 0:
        return "--atr-period must be > 0"
    if args.atr_trail_mult <= 0:
        return "--atr-trail-mult must be > 0"
    if args.hard_stop_atr_mult <= 0:
        return "--hard-stop-atr-mult must be > 0"
    if args.initial_capital <= 0:
        return "--initial-capital must be > 0"
    if args.equity_ema_period <= 0:
        return "--equity-ema-period must be > 0"
    if args.kill_max_drawdown_pct < 0:
        return "--kill-max-drawdown-pct must be >= 0"
    if args.kill_cooldown_bars < 0:
        return "--kill-cooldown-bars must be >= 0"
    if args.equity_kill_mode not in {"any", "both"}:
        return "--equity-kill-mode must be one of: any, both"
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
        weekly_confirm_bars=args.weekly_confirm_bars,
        weekly_fail_bars=args.weekly_fail_bars,
        monthly_fast_ema_period=args.monthly_fast_ema_period,
        monthly_slow_ema_period=args.monthly_slow_ema_period,
        monthly_regime_mode=args.monthly_regime_mode,
        breakout_lookback=args.breakout_lookback,
        momentum_length=args.momentum_length,
        atr_period=args.atr_period,
        atr_trail_mult=args.atr_trail_mult,
        hard_stop_atr_mult=args.hard_stop_atr_mult,
        initial_capital=args.initial_capital,
        enable_equity_kill=args.enable_equity_kill,
        equity_ema_period=args.equity_ema_period,
        equity_kill_mode=args.equity_kill_mode,
        kill_max_drawdown_pct=args.kill_max_drawdown_pct,
        kill_cooldown_bars=args.kill_cooldown_bars,
    )

    daily_input_dir = Path(args.daily_input_dir)
    weekly_input_dir = Path(args.weekly_input_dir)
    monthly_input_dir = Path(args.monthly_input_dir)
    allow_derived_fallback = args.allow_derived_fallback

    out_dir = Path(args.out_dir)
    latest_file = Path(args.latest_file)
    summary_file = Path(args.summary_file)
    errors_file = Path(args.errors_file)

    latest_rows: list[dict[str, str]] = []
    summaries: list[SymbolSummary] = []
    errors: list[SymbolError] = []

    for symbol in symbols:
        try:
            daily_path = daily_input_dir / f"{symbol}.csv"
            if not daily_path.exists():
                raise FileNotFoundError(f"Daily input CSV not found: {daily_path}")
            daily_rows = read_daily_rows(daily_path)

            weekly_path = weekly_input_dir / f"{symbol}.csv"
            monthly_path = monthly_input_dir / f"{symbol}.csv"

            if weekly_path.exists():
                weekly_rows = read_close_rows(weekly_path)
                weekly_source = "EXTERNAL"
            elif allow_derived_fallback:
                weekly_rows = aggregate_closes(daily_rows, timeframe="weekly")
                weekly_source = "DERIVED_DAILY"
            else:
                raise FileNotFoundError(f"Weekly input CSV not found: {weekly_path}")

            if monthly_path.exists():
                monthly_rows = read_close_rows(monthly_path)
                monthly_source = "EXTERNAL"
            elif allow_derived_fallback:
                monthly_rows = aggregate_closes(daily_rows, timeframe="monthly")
                monthly_source = "DERIVED_DAILY"
            else:
                raise FileNotFoundError(f"Monthly input CSV not found: {monthly_path}")

            strategy_rows, summary = run_strategy(
                symbol=symbol,
                daily_rows=daily_rows,
                weekly_rows=weekly_rows,
                monthly_rows=monthly_rows,
                cfg=cfg,
                weekly_source=weekly_source,
                monthly_source=monthly_source,
            )

            output_path = out_dir / f"{symbol}.csv"
            write_rows(output_path, strategy_rows)

            latest = strategy_rows[-1]
            latest_rows.append(
                {
                    "symbol": symbol,
                    "Date": latest.get("Date", ""),
                    "Close": latest.get("Close", ""),
                    "WeeklyTrendSource": latest.get("WeeklyTrendSource", ""),
                    "MonthlyTrendSource": latest.get("MonthlyTrendSource", ""),
                    "MonthlyRegimeMode": latest.get("MonthlyRegimeMode", ""),
                    "MonthlyRiskOn": latest.get("MonthlyRiskOn", ""),
                    "WeeklyRiskOn": latest.get("WeeklyRiskOn", ""),
                    "RegimeOn": latest.get("RegimeOn", ""),
                    "DailyBreakout": latest.get("DailyBreakout", ""),
                    "PullbackReclaim": latest.get("PullbackReclaim", ""),
                    "EntrySetup": latest.get("EntrySetup", ""),
                    "ATRTrailStop": latest.get("ATRTrailStop", ""),
                    "HardStop": latest.get("HardStop", ""),
                    "ActiveStop": latest.get("ActiveStop", ""),
                    "WeeklyBelowCount": latest.get("WeeklyBelowCount", ""),
                    "DrawdownPct": latest.get("DrawdownPct", ""),
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
