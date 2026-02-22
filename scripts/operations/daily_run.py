#!/usr/bin/env python3
"""Run the daily v2 workflow and persist run artifacts under out/runs/."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from scripts.paths import META_LATEST_DIR, OUT_ROOT, REPORTS_MOMENTUM_DIR


@dataclass
class StepResult:
    name: str
    command: list[str]
    log_file: str
    return_code: int
    duration_seconds: float
    started_at: str
    ended_at: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="daily", help="Run label used in out/runs/<date>_<label>")
    parser.add_argument("--watchlist", default="watchlist.txt", help="Watchlist file path")
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol override (creates a temporary effective watchlist)",
    )
    parser.add_argument(
        "--fetch-interval",
        choices=["d", "w", "m", "all"],
        default="d",
        help="Stooq interval used in the fetch step (default: d)",
    )
    parser.add_argument(
        "--fetch-delay-seconds",
        type=float,
        default=0.4,
        help="Delay between Stooq symbol requests (default: 0.4)",
    )
    parser.add_argument("--start-date", default="", help="Optional start date for fetch step (YYYY-MM-DD)")
    parser.add_argument("--ema-periods", default="50,200", help="EMA periods for compute_ema")
    parser.add_argument("--trend-buffer-pct", type=float, default=0.5, help="Trend analyzer buffer percent")
    parser.add_argument("--trend-confirm-bars", type=int, default=3, help="Trend analyzer confirm bars")
    parser.add_argument("--mom-length", type=int, default=24, help="Momentum length for momentum scripts")
    parser.add_argument("--mom-min-tick", type=float, default=0.01, help="Min tick for TV-match momentum")
    parser.add_argument("--signal-min-hold-bars", type=int, default=5, help="Signal engine min hold bars")
    parser.add_argument("--backtest-symbol", default="", help="Optional symbol for backtest_long")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip fetch step")
    parser.add_argument("--skip-report", action="store_true", help="Skip recent report step")
    parser.add_argument("--skip-layout-check", action="store_true", help="Skip strict layout check step")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue pipeline after a failed step")
    parser.add_argument("--run-dir", default="", help="Optional explicit run directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def normalize_label(raw: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", raw.strip()).strip("-")
    return value or "daily"


def parse_symbols(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def build_run_dir(label: str, explicit: str) -> Path:
    if explicit.strip():
        path = Path(explicit.strip())
        path.mkdir(parents=True, exist_ok=True)
        return path

    timestamp = datetime.now()
    base = OUT_ROOT / "runs" / f"{timestamp:%Y-%m-%d}_{normalize_label(label)}"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base

    with_time = OUT_ROOT / "runs" / f"{timestamp:%Y-%m-%d}_{normalize_label(label)}_{timestamp:%H%M%S}"
    with_time.mkdir(parents=True, exist_ok=True)
    return with_time


def ensure_effective_watchlist(
    *,
    source_watchlist: Path,
    symbols_override: list[str],
    config_dir: Path,
) -> Path:
    target = config_dir / "watchlist_effective.txt"
    if symbols_override:
        target.write_text("\n".join(symbols_override) + "\n", encoding="utf-8")
        return target

    if not source_watchlist.exists():
        raise FileNotFoundError(f"Watchlist not found: {source_watchlist}")
    target.write_text(source_watchlist.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def run_step(
    *,
    name: str,
    command: list[str],
    log_file: Path,
    dry_run: bool,
) -> StepResult:
    started = datetime.now()
    start_monotonic = time.perf_counter()
    if dry_run:
        log_file.write_text("[dry-run] " + " ".join(command) + "\n", encoding="utf-8")
        return StepResult(
            name=name,
            command=command,
            log_file=str(log_file),
            return_code=0,
            duration_seconds=0.0,
            started_at=started.isoformat(timespec="seconds"),
            ended_at=started.isoformat(timespec="seconds"),
        )

    proc = subprocess.run(command, capture_output=True, text=True)
    ended = datetime.now()
    duration = time.perf_counter() - start_monotonic
    combined = ""
    if proc.stdout:
        combined += proc.stdout
    if proc.stderr:
        if combined and not combined.endswith("\n"):
            combined += "\n"
        combined += proc.stderr
    log_file.write_text(combined, encoding="utf-8")
    return StepResult(
        name=name,
        command=command,
        log_file=str(log_file),
        return_code=proc.returncode,
        duration_seconds=round(duration, 3),
        started_at=started.isoformat(timespec="seconds"),
        ended_at=ended.isoformat(timespec="seconds"),
    )


def write_summary(
    *,
    run_dir: Path,
    args: argparse.Namespace,
    step_results: list[StepResult],
    started_at: datetime,
    ended_at: datetime,
    copied_artifacts: list[str],
) -> None:
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    overall_success = all(item.return_code == 0 for item in step_results)
    payload = {
        "run_dir": str(run_dir),
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "duration_seconds": round((ended_at - started_at).total_seconds(), 3),
        "overall_success": overall_success,
        "args": vars(args),
        "steps": [asdict(item) for item in step_results],
        "copied_artifacts": copied_artifacts,
    }
    summary_json = outputs_dir / "summary.json"
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Daily Run Summary",
        "",
        f"- run dir: `{run_dir}`",
        f"- started: `{payload['started_at']}`",
        f"- ended: `{payload['ended_at']}`",
        f"- duration seconds: `{payload['duration_seconds']}`",
        f"- overall success: `{payload['overall_success']}`",
        "",
        "## Steps",
        "",
        "| Step | Status | Duration(s) | Log |",
        "|---|---:|---:|---|",
    ]
    for item in step_results:
        status = "ok" if item.return_code == 0 else f"fail ({item.return_code})"
        lines.append(f"| `{item.name}` | {status} | {item.duration_seconds} | `{item.log_file}` |")
    if copied_artifacts:
        lines.extend(["", "## Copied Artifacts", ""])
        for artifact in copied_artifacts:
            lines.append(f"- `{artifact}`")
    (outputs_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_outputs(run_dir: Path) -> list[str]:
    outputs_dir = run_dir / "outputs"
    target_root = outputs_dir / "artifacts"
    target_root.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    candidates = [
        REPORTS_MOMENTUM_DIR / "recent_momentum_buys_5d.csv",
        REPORTS_MOMENTUM_DIR / "recent_momentum_buys_5d.md",
        META_LATEST_DIR / "signal_engine_latest.csv",
        META_LATEST_DIR / "trend_latest.csv",
        META_LATEST_DIR / "momentum_latest.csv",
        META_LATEST_DIR / "momentum_tv_match_daily_latest.csv",
    ]
    for source in candidates:
        if not source.exists():
            continue
        rel = source.relative_to(OUT_ROOT)
        destination = target_root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(str(destination))
    return copied


def print_step_result(result: StepResult) -> None:
    status = "ok" if result.return_code == 0 else f"fail ({result.return_code})"
    print(f"[{status}] {result.name} ({result.duration_seconds}s)")
    print(f"  log: {result.log_file}")


def main() -> int:
    args = parse_args()
    symbols_override = parse_symbols(args.symbols)
    source_watchlist = Path(args.watchlist)
    run_dir = build_run_dir(args.label, args.run_dir)
    config_dir = run_dir / "config"
    logs_dir = run_dir / "logs"
    config_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    effective_watchlist = ensure_effective_watchlist(
        source_watchlist=source_watchlist,
        symbols_override=symbols_override,
        config_dir=config_dir,
    )

    (config_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    (config_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "effective_watchlist": str(effective_watchlist),
                "symbols_override": symbols_override,
                "python": sys.executable,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    py = sys.executable
    step_commands: list[tuple[str, list[str], bool]] = []
    if not args.skip_fetch:
        fetch_cmd = [
            py,
            "-m",
            "scripts.data.fetch_stooq_ohlc",
            "--watchlist",
            str(effective_watchlist),
            "--interval",
            args.fetch_interval,
            "--delay-seconds",
            str(args.fetch_delay_seconds),
        ]
        if args.start_date.strip():
            fetch_cmd.extend(["--start-date", args.start_date.strip()])
        step_commands.append(("01_fetch_stooq", fetch_cmd, True))

    step_commands.extend(
        [
            (
                "02_compute_ema",
                [
                    py,
                    "-m",
                    "scripts.indicators.compute_ema",
                    "--watchlist",
                    str(effective_watchlist),
                    "--periods",
                    args.ema_periods,
                ],
                True,
            ),
            (
                "03_trend_analyzer",
                [
                    py,
                    "-m",
                    "scripts.indicators.trend_analyzer",
                    "--watchlist",
                    str(effective_watchlist),
                    "--buffer-pct",
                    str(args.trend_buffer_pct),
                    "--confirm-bars",
                    str(args.trend_confirm_bars),
                ],
                True,
            ),
            (
                "04_momentum_state",
                [
                    py,
                    "-m",
                    "scripts.indicators.momentum_strategy",
                    "--watchlist",
                    str(effective_watchlist),
                    "--length",
                    str(args.mom_length),
                ],
                True,
            ),
            (
                "05_momentum_tv_match_daily",
                [
                    py,
                    "-m",
                    "scripts.indicators.momentum_strategy_tv_match",
                    "--watchlist",
                    str(effective_watchlist),
                    "--timeframe",
                    "daily",
                    "--length",
                    str(args.mom_length),
                    "--min-tick",
                    str(args.mom_min_tick),
                ],
                True,
            ),
            (
                "06_signal_engine",
                [
                    py,
                    "-m",
                    "scripts.signals.signal_engine",
                    "--watchlist",
                    str(effective_watchlist),
                    "--min-hold-bars",
                    str(args.signal_min_hold_bars),
                ],
                True,
            ),
        ]
    )

    if not args.skip_report:
        report_cmd = [py, "-m", "scripts.reports.recent_momentum_report"]
        if symbols_override:
            report_cmd.extend(["--symbols", ",".join(symbols_override)])
        step_commands.append(("07_recent_report", report_cmd, True))

    if args.backtest_symbol.strip():
        step_commands.append(
            (
                "08_backtest",
                [
                    py,
                    "-m",
                    "scripts.strategies.backtest_long",
                    "--symbol",
                    args.backtest_symbol.strip().upper(),
                ],
                False,
            )
        )

    if not args.skip_layout_check:
        step_commands.append(
            (
                "09_verify_out_layout",
                [py, "-m", "scripts.maintenance.verify_out_layout", "--fail-on-legacy"],
                True,
            )
        )

    started_at = datetime.now()
    step_results: list[StepResult] = []
    print(f"Run dir: {run_dir}")
    print(f"Effective watchlist: {effective_watchlist}")
    if symbols_override:
        print(f"Symbols override: {','.join(symbols_override)}")

    for step_name, command, required in step_commands:
        print(f"\n[run] {step_name}")
        print("  cmd:", " ".join(command))
        log_file = logs_dir / f"{step_name}.log"
        result = run_step(name=step_name, command=command, log_file=log_file, dry_run=args.dry_run)
        step_results.append(result)
        print_step_result(result)
        if result.return_code != 0 and required and not args.continue_on_error:
            print(f"[stop] required step failed: {step_name}")
            break

    copied_artifacts = copy_outputs(run_dir)
    ended_at = datetime.now()
    write_summary(
        run_dir=run_dir,
        args=args,
        step_results=step_results,
        started_at=started_at,
        ended_at=ended_at,
        copied_artifacts=copied_artifacts,
    )

    required_failed = any(
        result.return_code != 0
        for result, (_, _, required) in zip(step_results, step_commands, strict=False)
        if required
    )
    if required_failed:
        print("\n[summary] run completed with failures")
        print(f"  summary: {run_dir / 'outputs' / 'summary.md'}")
        return 1

    print("\n[summary] run completed successfully")
    print(f"  summary: {run_dir / 'outputs' / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

