#!/usr/bin/env python3
"""Verify the v2 out/ layout and optionally fail on legacy paths."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_DIRS = [
    Path("out/data/daily"),
    Path("out/data/weekly"),
    Path("out/data/monthly"),
    Path("out/indicators/ema"),
    Path("out/indicators/trend"),
    Path("out/indicators/momentum"),
    Path("out/indicators/momentum_tv/daily"),
    Path("out/indicators/momentum_tv/weekly"),
    Path("out/indicators/momentum_tv/monthly"),
    Path("out/indicators/momentum_tv_match/daily"),
    Path("out/indicators/momentum_tv_match/weekly"),
    Path("out/indicators/momentum_tv_match/monthly"),
    Path("out/signals/engine"),
    Path("out/signals/variants"),
    Path("out/reports/momentum"),
    Path("out/backtests/long_only"),
    Path("out/backtests/tv_match"),
    Path("out/strategies/mtf_entry_exit_v1"),
    Path("out/strategies/mtf_entry_exit_v2"),
    Path("out/strategies/entry_tier_experiment"),
    Path("out/strategies/signal_quality_v2"),
    Path("out/strategies/hardening"),
    Path("out/experiments/sweeps"),
    Path("out/experiments/comparisons"),
    Path("out/experiments/legacy"),
    Path("out/runs"),
    Path("out/_meta/errors"),
    Path("out/_meta/latest"),
    Path("out/_meta/summaries"),
    Path("out/_meta/watchlists"),
]

LEGACY_PATHS = [
    Path("out/daily"),
    Path("out/weekly"),
    Path("out/monthly"),
    Path("out/trend"),
    Path("out/momentum"),
    Path("out/momentum_tv"),
    Path("out/momentum_tv_weekly"),
    Path("out/momentum_tv_monthly"),
    Path("out/momentum_tv_match_daily"),
    Path("out/momentum_tv_match_weekly"),
    Path("out/momentum_tv_match_monthly"),
    Path("out/mtf_entry_exit_v1"),
    Path("out/mtf_entry_exit_v2"),
    Path("out/entry_tier_experiment"),
    Path("out/signal_quality_v2"),
    Path("out/hardening"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fail-on-legacy",
        action="store_true",
        help="Fail if any legacy out/ compatibility path exists.",
    )
    return parser.parse_args()


def describe_kind(path: Path) -> str:
    if path.is_symlink():
        return "symlink"
    if path.is_dir():
        return "dir"
    if path.is_file():
        return "file"
    return "missing"


def main() -> int:
    args = parse_args()
    missing_required = [path for path in REQUIRED_DIRS if not path.exists()]
    existing_legacy = [path for path in LEGACY_PATHS if path.exists() or path.is_symlink()]

    print("V2 out/ layout check")
    print(f"  required directories: {len(REQUIRED_DIRS)}")
    print(f"  missing required:     {len(missing_required)}")
    print(f"  legacy paths found:   {len(existing_legacy)}")
    print(f"  fail-on-legacy:       {'on' if args.fail_on_legacy else 'off'}")

    if missing_required:
        print("\nMissing required directories:")
        for path in missing_required:
            print(f"  - {path}")

    if existing_legacy:
        print("\nLegacy paths present:")
        for path in existing_legacy:
            kind = describe_kind(path)
            if path.is_symlink():
                print(f"  - {path} ({kind} -> {path.resolve()})")
            else:
                print(f"  - {path} ({kind})")

    if missing_required:
        return 1
    if args.fail_on_legacy and existing_legacy:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
