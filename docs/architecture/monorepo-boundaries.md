# Monorepo Boundaries: Research, Engine, Operations, App

## Purpose

Define the target subsystem boundaries for `trading-signals` as it evolves from
a research workspace into a product-oriented utility.

The main architectural constraint is:

- research may remain exploratory and LLM-assisted
- operations must remain deterministic and must not require an LLM at runtime

This document describes the intended boundary model for a single repo.

## North Star

The repo supports two connected but distinct modes of work:

1. Research
2. Operations

Research exists to discover, test, and refine strategies.

Operations exists to run approved strategies on a schedule, notify the user,
and eventually place broker actions for explicitly authorized instruments.

These modes may share code, but they must not share an ambiguous runtime.

## Boundary Rule

Allowed:

- research depends on engine code
- operations depends on engine code
- app/backend services depend on engine and operations code
- research outputs versioned configs and policy decisions that operations can consume

Not allowed:

- operations depending on notebooks
- operations depending on milestone docs or hypothesis docs
- operations requiring an LLM to generate, interpret, or approve signals
- research-only scripts being part of the production critical path

## Subsystems

### 1. Research

Purpose:

- develop hypotheses
- compare indicators and entry/exit rules
- backtest candidate strategies
- explore parameter changes
- review charts and reports

Typical artifacts:

- notebooks
- experiment scripts
- hypothesis docs
- milestone retrospectives
- ad hoc reports

Characteristics:

- human-driven
- exploratory
- may use LLM assistance
- allowed to change frequently

### 2. Engine

Purpose:

- deterministic market-data transforms
- indicator calculations
- signal generation
- reusable strategy logic

Typical artifacts:

- fetch logic
- indicator logic
- signal logic
- strategy logic
- reusable data and output models

Characteristics:

- deterministic
- testable
- versionable
- usable by both research and operations

### 3. Operations

Purpose:

- scheduled signal generation
- notification delivery
- permission-aware trade decisioning
- broker integration
- audit logging
- operational persistence and state management

Typical artifacts:

- daily runners
- scheduler jobs
- notification adapters
- broker adapters
- position-state logic
- execution policy logic

Characteristics:

- no LLM dependency on critical path
- stable inputs
- explicit configs
- explicit failure handling
- auditable outputs

### 4. App

Purpose:

- manage watchlists
- manage instrument-level auto-trade permissions
- manage capital deployment preferences
- display signal history and execution history
- present current state and alerts

Typical artifacts:

- API server
- web UI
- mobile app integration
- auth and account management

Characteristics:

- user-facing control plane
- consumes operational state
- does not own core signal logic

Preferred lightweight shape:

```text
app/
  api/
  web/
```

This naming is preferred over heavier terms like `backend/` and `frontend/`
because the intended product is still a relatively small utility, not an
enterprise platform.

## Promotion Flow

Research and operations should be connected by promotion, not by implicit reuse.

Target flow:

1. Research develops or modifies a strategy.
2. Research validates it with notebooks, backtests, and review.
3. A strategy version is approved for operational use.
4. That approved version is captured in a stable config or policy artifact.
5. Operations loads that versioned artifact and runs it deterministically.
6. Any later strategy change must be promoted again.

This keeps production behavior explainable and reproducible.

## Runtime Separation

### Research runtime

- notebooks
- manual CLI runs
- experiment-only scripts
- optional LLM-assisted review

### Operations runtime

- scheduled daily or intraday jobs
- persistent storage
- notifications
- broker execution
- monitoring and alerts

The operations runtime must be able to execute successfully even if no LLM is
available.

## Data And State Model

The long-term product should distinguish between shared market data and
user-specific state.

Shared or reusable:

- raw OHLCV history
- derived common indicators when strategy-independent
- reference benchmark data

User-specific or account-specific:

- watchlists
- holdings and positions
- execution permissions
- capital sizing preferences
- signal history by policy version
- execution history
- broker account mappings
- notification preferences

## Initial Monorepo Shape

This is a target structure, not an immediate migration requirement:

```text
trading-signals/
  docs/
  notebooks/
  research/
  engine/
  ops/
  app/
  configs/
```

When the product-facing surface is added, `app/` should generally take this
form:

```text
app/
  api/
  web/
```

Practical mapping from the current repo:

- `notebooks/` stays research-oriented
- parts of `scripts/indicators`, `scripts/signals`, and `scripts/strategies`
  likely migrate toward `engine/`
- `scripts/operations` likely migrates toward `ops/`
- future UI/backend code belongs under `app/`
- approved operational strategy definitions belong under `configs/`

No immediate file move is required if the boundaries are documented and upheld.

## Operational Product Scope

The first operational product slice should likely do the following:

1. Run a scheduled signal job.
2. Store generated signals in a persistent database.
3. Notify the user by one delivery channel.
4. Track instrument-level auto-trade permissions.
5. Support human-in-the-loop approval before live execution.
6. Add selective broker execution only after the above is stable.

This sequence keeps broker risk behind a narrower, testable boundary.

## Design Principles

- Prefer explicit config over hidden defaults.
- Keep runtime boundaries clearer than repo boundaries.
- Treat the engine as a reusable deterministic library or service.
- Keep user-specific preferences out of public sample fixtures.
- Record all production-relevant decisions with timestamps and version context.
- Separate signal generation from order execution.
- Make it possible to run paper trading before live trading.

## Immediate Next Decisions

The next architecture decisions to resolve are:

1. What an approved operational strategy artifact looks like.
2. Where persistent operational state will live.
3. What the first notification channel will be.
4. Whether the first execution mode is alert-only or paper trading.
5. How instrument-level auto-trade permission is represented.
6. Whether the first app surface is a simple web UI or API-first backend.

## Relation To Existing Docs

- Product intent: `docs/product-vision.md`
- Current human-in-the-loop workflow: `docs/operations/trading-workflow.md`
- Current private-input overlay: `docs/private-overlay.md`

This document should serve as the boundary reference when evolving the repo
structure and deciding whether a change belongs to research, engine, operations,
or app code.
