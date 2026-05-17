# Trading Workspace Migration And Public/Private Split

## Status

Completed on May 17, 2026.

This document is retained as a record of the completed split work. The active
public-repo guidance now lives in `docs/private-overlay.md`.

## Outcome Summary

- `trading-signals/` remains the public engine repo
- `trading-signals-private/` is the private companion repo for personal inputs
- public default watchlist and holdings files are non-personal sample fixtures
- private watchlist and holdings were copied into the private repo
- public docs now describe the sibling private-overlay workflow
- the private watchlist path was validated end to end from the public repo
- large-watchlist comparison backtests were hardened to avoid overlong filenames

## Summary

Create a dedicated parent workspace at:

```text
/home/unionhills/Projects/GitHub/trading-signals-workspace/
  trading-signals/
  trading-signals-private/
```

Move the current public repo into that workspace, let the private repo live beside it, and refactor the public repo so private inputs are external rather than first-class tracked assets.

## Implementation Changes

### Phase 1: Workspace move

- Create `trading-signals-workspace/` under `~/Projects/GitHub/`.
- Move the existing `trading-signals/` repo into that folder without renaming the repo itself.
- Do not create the private repo from this repo or via automation.
- You create `trading-signals-private` on GitHub yourself and clone it into the sibling path.
- Keep Git history intact by moving the folders directly, not by copying files into a new repo.

### Phase 2: Immediate verification after move

- Open the public repo from its new path and verify:
  - `git status` is clean
  - the current branch and remotes are intact
  - the virtualenv or interpreter path still works, or is intentionally recreated
- Run a small smoke test from `trading-signals/`:
  - one fetch-related command in dry-run mode
  - one indicator command
  - one backtest command
- Open one representative notebook and verify repo-root assumptions still hold.

### Phase 3: Private repo bootstrap

- In `trading-signals-private/`, create initial folders:
  - `watchlists/`
  - `holdings/`
  - `configs/`
  - `docs/`
  - optional `out/` if you want private outputs separated later
- Move personal assets into the private repo first:
  - real watchlists
  - holdings
  - portfolio-specific notes
  - any future broker or automation settings
- Do not move reusable engine code, generic notebooks, or core docs into the private repo.

### Phase 4: Public repo input refactor

- Change the public repo’s posture from “my live workspace” to “engine + examples.”
- Keep only sample/default inputs in the public repo:
  - example watchlists
  - sample holdings template if needed
  - template config files
- Refactor commands and docs so the supported pattern is explicit external input, for example:
  - `--watchlist ../trading-signals-private/watchlists/core_etfs.md`
  - future `--config ../trading-signals-private/configs/daily.yaml`
- Audit CLI entrypoints and remove assumptions that personal repo-tracked files are the canonical default where practical.

### Phase 5: Documentation cleanup

- Update the public README to explain:
  - this repo is the engine
  - private portfolio inputs belong in a separate repo
  - example commands can point to either sample files or sibling private files
- Add a short “private overlay” doc in the public repo.
- Review `docs/product-vision.md` and `docs/operations/` for language that is too portfolio-specific and rewrite only the parts that should be product-facing.

### Phase 6: Workflow conventions

- Normal work:
  - run Codex from `trading-signals/` for engine work
  - run Codex from `trading-signals-private/` for private inputs/docs
- Cross-repo work:
  - run Codex from `trading-signals-workspace/` only when one session needs both repos
- Prefer relative sibling paths between the repos rather than absolute paths in docs and commands.

## Test Plan

- After the move, verify:
  - `git status` clean in both repos
  - remote URLs preserved
  - branch history preserved
- In `trading-signals/`, run:
  - one dry-run fetch command
  - one momentum generation command
  - one backtest command
- Verify at least one notebook opens successfully from the new path.
- Verify the public repo can consume a watchlist passed from `../trading-signals-private/...`.
- Verify ignored local files and secrets remain untracked in both repos.

## Assumptions

- The public repo remains open source.
- `trading-signals-private` will be private and hold real portfolio preferences.
- You will create the private GitHub repo yourself and clone it manually into the workspace.
- The repo move is acceptable now, even if some path-sensitive local setup needs minor repair.
- The first pass should optimize for clean separation and reversibility, not perfect config architecture.
