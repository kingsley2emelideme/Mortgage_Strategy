# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains two components:

1. **`mortgage_analysis.py`** — single-file institutional-grade mortgage risk engine for Canadian residential mortgage portfolios. Models fixed vs. variable rate trade-offs using stochastic interest rate simulation, vectorized amortization, and opportunity cost analytics.
2. **`app.py`** — Flask web application that exposes `mortgage_analysis.py` via REST API endpoints and serves an interactive frontend.

## Running the Analysis

```bash
# Run the full pipeline (simulations + 3 matplotlib figures)
python mortgage_analysis.py

# Run the web app locally
python app.py
# or with gunicorn (production)
gunicorn app:app
```

No build step. Dependencies for the analysis script:

```bash
pip install numpy pandas matplotlib
```

Full dependencies (including web app):

```bash
pip install -r requirements.txt
```

## Architecture

Everything lives in `mortgage_analysis.py`, organized into five layers:

1. **Data classes** (`MortgageParams`, `VasicekParams`, `LumpSumSpec`, `OpportunityCostParams`) — frozen dataclasses that serve as typed input contracts. Default values: `$750,000` principal, `25yr` amortization, `4.1%` fixed / `3.35%` variable. `MortgageParams` also carries a `payment_frequency` field (`"monthly"`, `"biweekly"`, `"weekly"`) which drives `periods_per_year`, `total_periods`, and `term_periods` derived properties.

2. **`VasicekRateEngine`** — Monte Carlo short-rate simulator using the Vasicek SDE (Euler-Maruyama). Generates `(n_periods, n_paths)` rate matrices. `dt` is set to `1 / periods_per_year` so the simulation scales correctly for monthly, bi-weekly, or weekly frequencies.

3. **`AmortizationEngine`** — Three amortization methods:
   - `amortize()` — single path → returns a `pd.DataFrame` schedule
   - `amortize_bulk()` — vectorized across all MC paths → returns terminal balances and total interest arrays
   - `amortize_bulk_full()` → returns the full `(term_periods, n_paths)` balance trajectory
   - `level_payment()` — static helper; accepts `periods_per_year` to compute the correct per-period payment for any frequency.

4. **`RiskAnalytics`** — Three analytics:
   - `invest_the_difference()` — models a brokerage account funded by the fixed/variable payment delta
   - `breakeven_inflation()` — Fisher equation, returns the inflation rate at which real debt cost = 0
   - `lump_sum_convexity()` — sweeps lump-sum sizes to quantify non-linear interest savings

5. **`ExecutiveDashboard`** — Composes all layers into matplotlib figures:
   - `render_deterministic_analysis()` — fixed vs. variable vs. hedged (deterministic scenarios)
   - `render()` — full 4-pane stochastic dashboard with fan charts and equity trajectories
   - Standalone `plot_convexity_report()` function for the convexity side-report

`main()` runs all three figures in sequence and prints key metrics to stdout.

## Web App (`app.py`)

- `GET /` — serves the interactive frontend (`templates/index.html`)
- `POST /api/calculate` — runs the full deterministic + stochastic analysis; accepts all `MortgageParams` fields including `payment_frequency`
- `POST /api/amortize` — returns a paginated amortization schedule for a single strategy
- `GET /methodology` — serves the methodology reference page (`templates/methodology.html`)

The `lump_sum_month` parameter is always specified in months in the API; `app.py` converts it to the correct period index for non-monthly frequencies.

## Key Design Decisions

- **Payment frequency**: `MortgageParams.payment_frequency` controls whether calculations run on monthly (12), bi-weekly (26), or weekly (52) periods. All amortization loops and Vasicek `dt` scale automatically via `periods_per_year`.
- **Vectorized bulk operations**: `amortize_bulk` and `amortize_bulk_full` avoid per-path DataFrame overhead. The inner loop is over time (not paths), enabling NumPy broadcast across paths at each timestep.
- **Lump-sum timing**: Applied *after* the regular payment within the same period (see the comment "this is the fix!" in `amortize()`). Changing this ordering affects balance trajectories.
- **Frozen dataclasses**: All parameter objects are immutable. Mutation requires creating a new instance (e.g., `MortgageParams(principal=750_000)`).
- **RNG seed**: `ExecutiveDashboard` defaults to `seed=42`. Pass `seed=None` for non-reproducible runs.
