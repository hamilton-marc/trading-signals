# Initiative Framework - Trading Signals

## Purpose
This document captures implementation-oriented detail that supports the product vision:
- concrete capabilities,
- scope boundaries,
- success checks,
- and near-term roadmap planning.

Use this as the bridge between high-level vision and specific epics/initiatives.

## Target User
- Primary user: solo trader/learner (project owner).

## Vision Alignment Anchors
Each initiative in this file must reinforce at least one of the core vision outcomes:
- Define entries, exits, and risk before trade placement.
- Remove discretionary overrides and mid-trade negotiation.
- Improve discipline and repeatability over optimization complexity.
- Shift focus from individual trades to system refinement.

## Core Capabilities
1. Data ingestion and caching from external providers.
2. Indicator and strategy computation (trend, momentum, signal state).
3. Multi-timeframe analysis (daily, weekly, monthly).
4. Human-review artifacts (CSV + notebook visualizations).
5. Lightweight backtesting for sanity checks.

## Current Scope
- End-of-day US equities workflow.
- CSV-first outputs under `out/`.
- Rule-based indicators and strategies.
- Notebook-driven exploration and validation.

## Out of Scope (for now)
- Intraday execution and live brokerage integration.
- Portfolio optimization and risk engine.
- Heavy ML/AI prediction systems.
- Complex distributed infrastructure.

## Success Criteria
- Reliable daily run with per-symbol fault tolerance.
- Signal output is understandable and explainable per row (no hidden logic).
- Multi-timeframe outputs are available and visually verifiable.
- Backtest metrics are stable enough to compare rule revisions.
- Manual overrides trend down over time and are documented when used.
- Rule compliance (entries/exits/risk) is auditable from output artifacts.

## Product Surface (Artifacts)
- Scripts: deterministic CLI tools for fetch/compute/signal/backtest.
- Outputs: CSV files under `out/`.
- Research UI: notebooks under `notebooks/`.
- Context docs: milestone and decision snapshots under `docs/`.

## Near-Term Initiatives (Vision-Aligned)

### Initiative 1: Rule Definition and Enforcement
Objective:
Create explicit, machine-enforced rule contracts for entry, exit, and risk so decisions are not negotiated during volatility.

Supports vision outcomes:
- Define entries, exits, and risk before trade placement.
- Remove discretionary overrides and mid-trade negotiation.

Initial deliverables:
- Formal rule specification docs for active strategies.
- Output columns that show which rule fired and why.
- Explicit handling for invalid/ambiguous signal states.

### Initiative 2: Multi-Timeframe Discipline Layer
Objective:
Operationalize daily, weekly, and monthly context so the system promotes consistent, precommitted behavior instead of reactive overrides.

Supports vision outcomes:
- Improve discipline and repeatability over optimization complexity.
- Shift focus from individual trades to system refinement.

Initial deliverables:
- Stable timeframe aggregation and signal generation workflows.
- Documented policy for higher-timeframe context vs execution timeframe.
- Notebook parity checks for timeframe-based behavior.

### Initiative 3: Execution Auditability and Override Tracking
Objective:
Make every decision inspectable and track every override so discipline can be measured and improved.

Supports vision outcomes:
- Remove discretionary overrides and mid-trade negotiation.
- Shift focus from individual trades to system refinement.

Initial deliverables:
- Run metadata and parameter stamps in outputs.
- Event-level traceability from input to final action.
- Override logging convention (reason + timestamp + outcome).

### Initiative 4: Reliability and Regression Guardrails
Objective:
Increase trust in the system by preventing silent behavior drift in aggregation, indicator computation, and signal events.

Supports vision outcomes:
- Improve discipline and repeatability over optimization complexity.
- Shift focus from individual trades to system refinement.

Initial deliverables:
- Lightweight tests for key transformations and event logic.
- Baseline comparison snapshots for strategy revisions.
- Failure-tolerant per-symbol execution reporting.

### Initiative 5: Decision-Support Output Standardization
Objective:
Produce consistent daily shortlist artifacts that are clear enough to execute without ad-hoc reinterpretation.

Supports vision outcomes:
- Define entries, exits, and risk before trade placement.
- Improve discipline and repeatability over optimization complexity.

Initial deliverables:
- Standard shortlist schema and ranking fields.
- Clear per-symbol state summary (trend, momentum, risk context).
- Stable output locations and naming conventions.

## Long-Term Direction
Evolve from indicator playground to a consistent decision-support product:
- predictable daily operation,
- consistent ranking and shortlist generation,
- and documented strategy iterations with measurable outcomes.
