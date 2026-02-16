# Product Vision - Trading Signals

## Purpose
Build a practical end-of-day trading signal tool that helps a discretionary trader quickly identify higher-quality setups, with clear, testable logic and multi-timeframe context.

## Problem Statement
Manual chart review across many symbols and timeframes is slow and inconsistent. We need a lightweight system that:
- pulls price data reliably,
- computes transparent technical signals,
- and outputs actionable, reviewable artifacts.

## Target User
- Primary user: solo trader/learner (project owner).
- Secondary user (future): small group of collaborators reviewing signal outputs.

## Product Principles
- Keep it simple, inspectable, and reproducible.
- Prefer transparent rules over black-box models.
- Build in small, runnable increments.
- Optimize for learning speed and decision clarity.

## Vision Outcome
Given a watchlist, produce a daily shortlist of symbols with clearly explained signal states, grounded in daily/weekly/monthly technical context.

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
- Signal output is understandable and explainable per row.
- Multi-timeframe outputs are available and visually verifiable.
- Backtest metrics are stable enough to compare rule revisions.

## Product Surface (Artifacts)
- Scripts: deterministic CLI tools for fetch/compute/signal/backtest.
- Outputs: CSV files under `out/`.
- Research UI: notebooks under `notebooks/`.
- Context docs: milestone and decision snapshots under `docs/`.

## Near-Term Roadmap
1. Stabilize momentum/trend parity with TradingView behavior by timeframe.
2. Define a unified signal policy combining daily execution with weekly/monthly context.
3. Add lightweight regression tests for aggregation and event generation.
4. Standardize a daily shortlist schema and ranking fields.
5. Add basic run metadata (run date, params, source) to outputs.

## Long-Term Direction
Evolve from indicator playground to a consistent decision-support product:
- predictable daily operation,
- consistent ranking and shortlist generation,
- and documented strategy iterations with measurable outcomes.
