# AGENTS.md — Trading Signals (Flexible Mode)

This repository is a personal learning project focused on building an end-of-day (EOD) trading signal tool.

The goal is steady progress, clarity, and skill development — not rigid process compliance.

---

## Operating Philosophy

- Build in **small, runnable steps**.
- Keep the system working at all times.
- Prefer **simple over clever**.
- Refactor when it improves clarity.
- Avoid premature architecture decisions.

This project is allowed to evolve.

---

## Current Direction (High Level)

- Read symbols from `watchlist.txt`
- Fetch end-of-day price data
- Generate a daily CSV shortlist
- Gradually add trend / momentum logic

Details may change as we learn.

---

## Scope Guidelines (Lightweight)

- Do not add features unrelated to the current task.
- Refactoring is allowed if it simplifies or clarifies the design.
- Architecture can evolve — do not freeze contracts too early.
- Avoid introducing heavy frameworks unless clearly justified.

---

## Git Workflow

- Bias toward small, frequent commits at meaningful checkpoints.
- Commit when a unit of work is complete and verifiable, even if the overall initiative is still in progress.
- For major new features or initiatives:
  - Push the current feature branch.
  - Merge it into `main`.
  - Start a new branch from updated `main` before beginning the new initiative.

---

## Technology Preferences

- Python 3.10+
- Keep dependencies minimal (but don’t be dogmatic).
- Prefer:
  - Type hints
  - Small cohesive modules
  - `dataclasses` for simple data structures
- Use third-party libraries only when they meaningfully reduce complexity.

---

## Data Philosophy

- Data provider may change.
- JSON or CSV storage is acceptable.
- Weekly/monthly data may be derived from daily.
- Caching locally is preferred over repeated API calls.

Do not hard-wire assumptions that make provider changes difficult.

---

## Output Conventions

- Write artifacts to `out/`
- CSV is preferred for human review.
- Early schema changes are acceptable.
- Once the project stabilizes, contracts can harden.

---

## Security Hygiene

- Never commit credentials, tokens, API keys, private keys, `.env` values, or authentication cookies.
- Do not commit personally identifiable information (PII) unless explicitly required for a task.
- Avoid leaking local machine details in tracked artifacts (for example absolute paths in notebook outputs).
- Keep generated outputs and caches in ignored locations unless they are intentionally part of versioned results.
- Before commit/push, run configured secret scanning checks when available (`pre-commit`, CI workflows).
- If a potential leak is found, stop and remediate before continuing (remove from files, regenerate outputs, re-check).

---

## Error Handling

- One symbol failing should not crash the entire run.
- Log or record per-symbol failures.
- Prefer resilience over strict failure.

---

## Testing & Verification

Each iteration should have a simple verification path:
- A CLI run example
- A small unit test
- Or a clear printed summary

Keep tests focused and lightweight.

---

## Learning Mode Enabled

This is a growth project.

It is acceptable to:
- Ask why a command works
- Replace Bash with Python (or vice versa)
- Refactor for understanding
- Try different data providers
- Simplify earlier decisions

The objective is mastery and clarity — not process perfection.

---

## Communication Style (for coding agents)

When making changes:
- Summarize what changed
- Explain how to run/verify it
- Call out any assumptions
- Note what was intentionally not implemented yet

Keep diffs small. Keep explanations clear.
