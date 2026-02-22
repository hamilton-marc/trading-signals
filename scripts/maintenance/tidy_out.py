#!/usr/bin/env python3
"""Tidy top-level files in out/ into categorized metadata folders."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="out", help="Output root directory (default: out)")
    parser.add_argument("--dry-run", action="store_true", help="Print planned moves without applying them")
    return parser.parse_args()


def classify_file(path: Path) -> str:
    name = path.name
    if name.endswith(".csv") and "_errors" in name:
        return "errors"
    if name.endswith(".csv") and "_latest" in name:
        return "latest"
    if name.endswith("_summary.csv"):
        return "summaries"
    if name.startswith("watchlist") and name.endswith(".txt"):
        return "watchlists"
    if name.endswith("_watchlist.txt"):
        return "watchlists"
    return "misc"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.exists():
        print(f"[error] out directory not found: {out_dir}")
        return 1
    if not out_dir.is_dir():
        print(f"[error] out path is not a directory: {out_dir}")
        return 1

    meta_root = out_dir / "_meta"
    moved = 0
    skipped = 0

    for path in sorted(out_dir.iterdir()):
        if path.is_dir():
            continue
        category = classify_file(path)
        target_dir = meta_root / category
        target_path = target_dir / path.name

        if target_path.exists():
            skipped += 1
            print(f"[skip] {path} -> {target_path} (target exists)")
            continue

        if args.dry_run:
            print(f"[dry-run] {path} -> {target_path}")
            moved += 1
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        path.rename(target_path)
        moved += 1
        print(f"[move] {path} -> {target_path}")

    mode = "dry-run" if args.dry_run else "apply"
    print(f"\nSummary ({mode})")
    print(f"  moved:   {moved}")
    print(f"  skipped: {skipped}")
    print(f"  meta:    {meta_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
