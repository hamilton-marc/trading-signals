# Trading Policy (Weekly-First)

## Purpose
Define the default decision policy for discretionary execution using this project.  
This policy is intended to reduce signal noise, protect capital, and keep behavior consistent.

## Scope
- Time horizon: swing and position trading.
- Data cadence: end-of-day (EOD) only.
- Execution timing: evaluate after close, act on next trading day.
- Portfolio style: long bias unless a separate short policy is explicitly enabled.

## Decision Hierarchy
1. Higher-timeframe regime decides direction: monthly + weekly.
2. Weekly momentum decides candidate eligibility.
3. Daily chart is used for timing/refinement only.
4. Risk exits always override entry logic.

## Entry Policy (Long)
All of the following should be true before treating a setup as execution-grade:
1. Weekly `MomLE` occurred in the last 2 weekly bars (about 10 trading days).
2. Weekly regime is risk-on.
3. Monthly regime is risk-on.

Optional refinement:
- Use daily context to improve entry timing.
- Do not use daily signals alone to override weak weekly/monthly context.

## Exit Policy (Capital Preservation First)
Exit if any one trigger is active:
1. `SignalEvent = LONG_TO_SHORT`
2. Trend flips to `DOWNTREND`
3. Close breaks risk stop (for example, close below `EMA_50` when this rule is enabled)

If both entry and exit appear at once, prioritize exit.

## Noise-Control Rules
- Ignore isolated daily momentum flips when higher-timeframe regime remains intact.
- Avoid new long entries in obvious downtrends unless running a deliberate reversal experiment.
- Prefer fewer, higher-quality signals over frequent low-conviction signals.

## Operational Artifacts
- Weekly candidate report:
  - `out/reports/momentum/recent_momentum_buys_weekly_10d.md`
  - `out/reports/momentum/recent_momentum_buys_weekly_10d.csv`
- Daily candidate report (secondary):
  - `out/reports/momentum/recent_momentum_buys_5d.md`
  - `out/reports/momentum/recent_momentum_buys_5d.csv`
- Chart review notebooks:
  - `notebooks/recent_signal_lab_weekly.ipynb`
  - `notebooks/recent_signal_lab.ipynb`

## Change Control
- Default rule: change one decision parameter at a time.
- Keep a brief note of why a rule changed and what outcome is expected.
- Re-evaluate weekly before making additional policy changes.
