# AGENTS.md — Instructions for Coding Agents (Codex)

This repo is an end-of-day (EOD) stock scanner. The goal is to iterate in small, verifiable steps.

## Operating Mode
- Prefer **micro-iterations** with small diffs.
- Implement **only what the current story/spec asks for**. Do not add “nice to have” features.
- If something is ambiguous, make a reasonable assumption and document it in the relevant story/spec. Do not block on questions.

## Scope Discipline (Anti-Drift)
- Do **not** redesign the architecture or refactor code that already works unless explicitly requested.
- Prefer **additive changes** over refactors.
- Do not rename files, modules, or CLI commands once introduced unless required for correctness.
- Do not introduce new configuration systems (YAML/TOML/env frameworks) unless explicitly requested.

## Project Goals (High Level)
- Run after US market close using **daily EOD data** (no real-time requirement).
- Scan symbols from `watchlist.txt`.
- Output a **CSV** artifact in `out/` that is easy to review.
- Later iterations will add momentum signals and notifications.

## Current MVP Philosophy
- Start with scaffolding and contracts first.
- Add one capability at a time:
    1) read watchlist
    2) fetch EOD data
    3) compute indicator
    4) detect signal
    5) notify (email/push) — last

## Tech Choices
- Language: **Python 3.x** (works in a venv).
- Dependencies: keep minimal.
    - Standard library preferred for early iterations.
    - Add third-party packages only when clearly justified (and keep the list short).
- Data source: must support EOD daily bars; prefer free/cheap and simple. Do not require API keys for the very first iterations unless the story explicitly says so.

## Code Style Preferences (Python)
- Prefer an **object-oriented** design where it improves clarity (small classes with clear responsibilities).
- Use **type hints** throughout (function signatures + class attributes where practical).
- Prefer `dataclasses` for simple data containers.
- Keep modules cohesive (avoid “god files” and overly large classes).
- When introducing interfaces/abstractions, keep them lightweight and aligned to the current story.

## Validation / Config
- Prefer **Pydantic** models for configuration and validating external inputs.
- Prefer `dataclasses` for internal immutable data records.

## Output Contracts (Stability Anchor)
- Output folder: `out/` (create if missing).
- CSV filenames should include the date (YYYY-MM-DD) as specified by the current story/spec.
- Preserve CSV column names once introduced; only add new columns (don’t rename) unless requested.

## Error Handling
- Do not crash the whole run because one symbol fails.
- Record per-symbol failures in output (e.g., an `error` column) and continue.

## Testing / Verification
- Every iteration must include a simple way to verify behavior:
    - Either a small unit test, or
    - A deterministic “golden” output/schema check, or
    - A CLI `--help` + example run documented in the story/spec.
- When you add code, ensure `python -m compileall .` succeeds.

## Unit Testing
- Use **pytest** for all new tests.
- Place tests under `tests/` and name files `test_*.py`.
- Prefer small, focused tests and “golden” checks (e.g., CSV schema/columns, watchlist parsing).
- Do not add additional test frameworks unless explicitly requested.

## Suggested Repo Layout (Keep Simple)
- Do not introduce a complex layout until the spec requires it.

## Communication Style (in code + PR notes)
- Leave brief comments explaining assumptions.
- Keep functions small and readable.
- When making changes, summarize:
    - what you changed
    - how to run/verify it
    - what you intentionally did NOT do yet
