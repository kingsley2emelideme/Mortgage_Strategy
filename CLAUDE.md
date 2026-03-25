# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`mortgage_analysis.py` is a single-file institutional-grade mortgage risk engine for Canadian residential mortgage portfolios. It models fixed vs. variable rate trade-offs using stochastic interest rate simulation, vectorized amortization, and opportunity cost analytics.

## Running the Analysis

```bash
# Run the full pipeline (simulations + 3 matplotlib figures)
python mortgage_analysis.py
```

No build step. Dependencies: `numpy`, `pandas`, `matplotlib`. Install with:

```bash
pip install numpy pandas matplotlib
```

## Architecture

Everything lives in `mortgage_analysis.py`, organized into five layers:

1. **Data classes** (`MortgageParams`, `VasicekParams`, `LumpSumSpec`, `OpportunityCostParams`) — frozen dataclasses that serve as typed input contracts. Default values represent a real Calgary portfolio (~$360k, 20yr, 4.1% fixed / 3.35% variable).

2. **`VasicekRateEngine`** — Monte Carlo short-rate simulator using the Vasicek SDE (Euler-Maruyama). Generates `(n_months, n_paths)` rate matrices. Used to model variable-rate scenarios.

3. **`AmortizationEngine`** — Three amortization methods:
   - `amortize()` — single path → returns a `pd.DataFrame` schedule
   - `amortize_bulk()` — vectorized across all MC paths → returns terminal balances and total interest arrays
   - `amortize_bulk_full()` → returns the full `(term_months, n_paths)` balance trajectory

4. **`RiskAnalytics`** — Three analytics:
   - `invest_the_difference()` — models a brokerage account funded by the fixed/variable payment delta
   - `breakeven_inflation()` — Fisher equation, returns the inflation rate at which real debt cost = 0
   - `lump_sum_convexity()` — sweeps lump-sum sizes to quantify non-linear interest savings

5. **`ExecutiveDashboard`** — Composes all layers into matplotlib figures:
   - `render_deterministic_analysis()` — fixed vs. variable vs. hedged (deterministic scenarios)
   - `render()` — full 4-pane stochastic dashboard with fan charts and equity trajectories
   - Standalone `plot_convexity_report()` function for the convexity side-report

`main()` runs all three figures in sequence and prints key metrics to stdout.

## Key Design Decisions

- **Vectorized bulk operations**: `amortize_bulk` and `amortize_bulk_full` avoid per-path DataFrame overhead. The inner loop is over time (not paths), enabling NumPy broadcast across paths at each timestep.
- **Lump-sum timing**: Applied *after* the regular payment within the same month (see the comment "this is the fix!" in `amortize()`). Changing this ordering affects balance trajectories.
- **Frozen dataclasses**: All parameter objects are immutable. Mutation requires creating a new instance (e.g., `MortgageParams(principal=400_000)`).
- **RNG seed**: `ExecutiveDashboard` defaults to `seed=42`. Pass `seed=None` for non-reproducible runs.
