"""Canonical output path definitions for v2 layout."""

from __future__ import annotations

from pathlib import Path

OUT_ROOT = Path("out")

DATA_DIR = OUT_ROOT / "data"
DATA_DAILY_DIR = DATA_DIR / "daily"
DATA_WEEKLY_DIR = DATA_DIR / "weekly"
DATA_MONTHLY_DIR = DATA_DIR / "monthly"

INDICATORS_DIR = OUT_ROOT / "indicators"
INDICATORS_EMA_DIR = INDICATORS_DIR / "ema"
INDICATORS_TREND_DIR = INDICATORS_DIR / "trend"
INDICATORS_MOMENTUM_DIR = INDICATORS_DIR / "momentum"
INDICATORS_MOMENTUM_TV_DIR = INDICATORS_DIR / "momentum_tv"
INDICATORS_MOMENTUM_TV_MATCH_DIR = INDICATORS_DIR / "momentum_tv_match"

SIGNALS_DIR = OUT_ROOT / "signals"
SIGNALS_ENGINE_DIR = SIGNALS_DIR / "engine"
SIGNALS_VARIANTS_DIR = SIGNALS_DIR / "variants"

REPORTS_DIR = OUT_ROOT / "reports"
REPORTS_MOMENTUM_DIR = REPORTS_DIR / "momentum"

BACKTESTS_DIR = OUT_ROOT / "backtests"
BACKTESTS_LONG_ONLY_DIR = BACKTESTS_DIR / "long_only"
BACKTESTS_TV_MATCH_DIR = BACKTESTS_DIR / "tv_match"

STRATEGIES_DIR = OUT_ROOT / "strategies"
STRATEGIES_MTF_V1_DIR = STRATEGIES_DIR / "mtf_entry_exit_v1"
STRATEGIES_MTF_V2_DIR = STRATEGIES_DIR / "mtf_entry_exit_v2"
STRATEGIES_ENTRY_TIER_DIR = STRATEGIES_DIR / "entry_tier_experiment"
STRATEGIES_SIGNAL_QUALITY_V2_DIR = STRATEGIES_DIR / "signal_quality_v2"
STRATEGIES_HARDENING_DIR = STRATEGIES_DIR / "hardening"

META_DIR = OUT_ROOT / "_meta"
META_ERRORS_DIR = META_DIR / "errors"
META_LATEST_DIR = META_DIR / "latest"
META_SUMMARIES_DIR = META_DIR / "summaries"
META_WATCHLISTS_DIR = META_DIR / "watchlists"


def stooq_errors_file(interval: str) -> Path:
    if interval == "d":
        return META_ERRORS_DIR / "stooq_daily_errors.csv"
    if interval == "w":
        return META_ERRORS_DIR / "stooq_weekly_errors.csv"
    return META_ERRORS_DIR / "stooq_monthly_errors.csv"


def data_dir_for_interval(interval: str) -> Path:
    if interval == "d":
        return DATA_DAILY_DIR
    if interval == "w":
        return DATA_WEEKLY_DIR
    return DATA_MONTHLY_DIR


def momentum_tv_output_dir(timeframe: str) -> Path:
    return INDICATORS_MOMENTUM_TV_DIR / timeframe


def momentum_tv_latest_file(timeframe: str) -> Path:
    return META_LATEST_DIR / f"momentum_tv_{timeframe}_latest.csv"


def momentum_tv_errors_file(timeframe: str) -> Path:
    return META_ERRORS_DIR / f"momentum_tv_{timeframe}_errors.csv"


def momentum_tv_match_output_dir(timeframe: str) -> Path:
    return INDICATORS_MOMENTUM_TV_MATCH_DIR / timeframe


def momentum_tv_match_latest_file(timeframe: str) -> Path:
    return META_LATEST_DIR / f"momentum_tv_match_{timeframe}_latest.csv"


def momentum_tv_match_errors_file(timeframe: str) -> Path:
    return META_ERRORS_DIR / f"momentum_tv_match_{timeframe}_errors.csv"
