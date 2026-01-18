# Trading Signals — SPEC (Iterative MVP)

## Purpose
Build a small end-of-day (EOD) stock scanning tool that reads a personal watchlist and produces a daily CSV shortlist.
This project is intentionally built in small iterations so the solution can evolve and remain a learning experience.

## Core Constraints
- EOD analysis only (run after market close).
- Keep each change set small and verifiable.
- Prefer additive changes over refactors.
- Preserve stable contracts (CLI + output schema) once introduced.
- Avoid introducing new dependencies unless clearly justified.

## Non-Goals (for now)
- No auto-trading / brokerage integration.
- No backtesting framework.
- No web UI.
- No scanning the entire market universe (watchlist only).
- No API integration in early iterations (start offline with manually downloaded data).

---

## Inputs

### Watchlist
- Source: exported watchlists from TradingView / FinViz (or manually curated)
- Canonical file: `watchlist.txt`
    - One symbol per line
    - Ignore blank lines
    - Ignore lines starting with `#`

### Market Data (Offline / Manual)
- Data is provided manually as CSV files in `data/`
- Canonical format (per symbol):
    - Path: `data/{SYMBOL}.csv` (e.g., `data/AAPL.csv`)
    - Columns required (case-insensitive accepted):
        - `Date`
        - `Open`
        - `High`
        - `Low`
        - `Close`
        - `Volume`
- Data granularity:
    - Daily bars
    - Monthly bars will be derived by resampling daily data (later iteration)

---

## Output Contract (Stable)
All iterations must produce a CSV file in `out/`.

### Output Location
- Directory: `out/` (create if missing)

### Filename Format
- `out/shortlist_YYYY-MM-DD.csv`

### CSV Columns (stable)
- `date` (YYYY-MM-DD)
- `symbol`
- `close`
- `trend_dir_monthly`  # up|down|flat|unknown
- `trend_dir_daily`    # up|down|flat|unknown
- `reason`
- `error`

Notes:
- If a field is not available yet, leave it blank or use a placeholder.
- Do not rename columns once introduced. Only add new columns if needed later.

---

## Run Contract (Stable)
- Provide one simple command to run the scanner locally (venv assumed).
- The command should work on Linux.
- The program should exit `0` even if some symbols fail; failures are recorded per-symbol in the CSV.

---

## Current Iteration (ONLY implement this)
Agents: implement only the section below. Do not implement backlog items.

### Iteration 001 — Scaffolding + CSV Skeleton (NO data parsing yet)
**Goal:** Create a runnable skeleton and establish the CSV contract.

#### Behavior
- Read symbols from `watchlist.txt`
- Create `out/` if missing
- Write `out/shortlist_YYYY-MM-DD.csv`
- Include one row per symbol

#### CSV values for this iteration
- `date`: today’s date (YYYY-MM-DD)
- `symbol`: the symbol from watchlist
- `close`: blank
- `trend_dir_monthly`: `unknown`
- `trend_dir_daily`: `unknown`
- `reason`: `placeholder`
- `error`: blank unless something fails

#### Console output
Print a short summary, e.g.:
- number of symbols read
- output CSV path

#### Explicitly Out of Scope (do NOT implement)
- Do NOT read or parse `data/*.csv`
- Do NOT compute trend or momentum
- Do NOT send notifications
- Do NOT add complex configuration

---

## Backlog (uncommitted ideas — do NOT implement unless promoted)
- Iteration 002: Load `data/{SYMBOL}.csv` and populate `close` (latest close)
- Iteration 003: Compute `trend_dir_daily` from daily bars (rule TBD)
- Iteration 004: Resample daily bars into monthly bars; compute `trend_dir_monthly`
- Iteration 005: Compute a momentum indicator and detect directional signals:
    - `momentum_signal` in {up, down, none}
    - Treat `up`/`down` as candidates for entries and/or exits (rules TBD)
- Iteration 006: Notifications (email/push/etc.)
- Add unit tests (watchlist parsing + CSV schema “golden” check)

---

## Definition of Done (each iteration)
- Small diff
- Runs locally
- Produces the CSV output in `out/`
- Matches the output schema exactly
- No surprise features outside the “Current Iteration” scope
