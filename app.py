"""
Mortgage Rate Strategy — Flask Web Application
===============================================
Serves the frontend and exposes REST API endpoints backed by mortgage_analysis.py.
"""

from __future__ import annotations

import traceback
from typing import Any, Dict

import numpy as np
from flask import Flask, jsonify, render_template, request

from mortgage_analysis import (
    AmortizationEngine,
    LumpSumSpec,
    MortgageParams,
    OpportunityCostParams,
    RiskAnalytics,
    VasicekParams,
    VasicekRateEngine,
)

app = Flask(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_params(data: Dict[str, Any]) -> tuple[MortgageParams, VasicekParams, OpportunityCostParams]:
    """Extract and validate parameters from JSON request body."""
    mp = MortgageParams(
        principal=float(data.get("principal", 360_404.0)),
        amortization_years=int(data.get("amortization_years", 20)),
        fixed_rate=float(data.get("fixed_rate", 0.0410)),
        var_rate_start=float(data.get("var_rate_start", 0.0335)),
        term_months=int(data.get("term_months", 60)),
    )
    vp = VasicekParams(
        kappa=float(data.get("kappa", 0.35)),
        theta=float(data.get("theta", 0.035)),
        sigma=float(data.get("sigma", 0.012)),
        floor=float(data.get("floor", 0.0225)),
    )
    opp = OpportunityCostParams(
        equity_cagr=float(data.get("equity_cagr", 0.07)),
    )
    return mp, vp, opp


def _parse_lump_sum(data: Dict[str, Any]) -> LumpSumSpec | None:
    amount = float(data.get("lump_sum_amount", 0))
    month = int(data.get("lump_sum_month", 12))
    return LumpSumSpec(amount=amount, month=month) if amount > 0 else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/calculate", methods=["POST"])
def calculate():
    """
    Run the full deterministic + stochastic analysis.

    Returns JSON with:
      - amortization schedules for Fixed, Variable, Hedged (+ optional Stress)
      - Vasicek percentile fan chart data
      - opportunity-cost portfolio trajectory
      - key summary metrics
    """
    try:
        data = request.get_json(force=True) or {}
        mp, vp, opp = _parse_params(data)
        lump_sum = _parse_lump_sum(data)
        n_sims = int(data.get("n_sims", 2000))

        engine = AmortizationEngine(mp)
        risk = RiskAnalytics(mp, opp)
        rate_sim = VasicekRateEngine(vp, seed=42)

        fixed_pmt = AmortizationEngine.level_payment(mp.principal, mp.fixed_rate, mp.total_months)
        var_pmt = AmortizationEngine.level_payment(mp.principal, mp.var_rate_start, mp.total_months)

        n = mp.term_months
        flat_f = np.full(n, mp.fixed_rate)
        flat_v = np.full(n, mp.var_rate_start)

        # -- Deterministic schedules --
        df_fixed = engine.amortize(flat_f, fixed_pmt)
        df_var = engine.amortize(flat_v, var_pmt)
        df_hedged = engine.amortize(flat_v, fixed_pmt, lump_sum=lump_sum)

        # Stress: variable rate +2 %
        stress_rate = mp.var_rate_start + 0.02
        flat_stress = np.full(n, stress_rate)
        stress_pmt = AmortizationEngine.level_payment(mp.principal, stress_rate, mp.total_months)
        df_stress = engine.amortize(flat_stress, stress_pmt)

        months = list(range(1, n + 1))

        def schedule_to_dict(df):
            return {
                "months": months,
                "interest": df["Interest"].round(2).tolist(),
                "principal": df["Principal"].round(2).tolist(),
                "balance": df["Balance"].round(2).tolist(),
                "cumulative_interest": df["Interest"].cumsum().round(2).tolist(),
            }

        # -- Stochastic fan chart --
        rate_paths = rate_sim.simulate(n, n_sims, mp.var_rate_start)
        pcts = rate_sim.percentiles(rate_paths, quantiles=(0.10, 0.25, 0.50, 0.75, 0.90))

        # -- MC terminal interest distribution --
        _, mc_interests = engine.amortize_bulk(rate_paths, var_pmt, lump_sum=lump_sum)
        hist_counts, hist_edges = np.histogram(mc_interests, bins=40)
        hist_centers = ((hist_edges[:-1] + hist_edges[1:]) / 2).round(0).tolist()

        # -- Opportunity cost portfolio --
        portfolio = risk.invest_the_difference(fixed_pmt, var_pmt, lump_sum=lump_sum)

        # -- Convexity --
        ls_amounts = np.arange(0, 60_001, 5_000, dtype=np.float64)
        cvx_df = risk.lump_sum_convexity(mp.fixed_rate, fixed_pmt, ls_amounts)

        # -- Break-even inflation --
        be_fixed = risk.breakeven_inflation(mp.fixed_rate)
        be_var = risk.breakeven_inflation(mp.var_rate_start)

        return jsonify({
            "schedules": {
                "fixed": schedule_to_dict(df_fixed),
                "variable": schedule_to_dict(df_var),
                "hedged": schedule_to_dict(df_hedged),
                "stress": schedule_to_dict(df_stress),
            },
            "fan_chart": {
                "months": months,
                "p10": [round(x, 5) for x in pcts[0.10].tolist()],
                "p25": [round(x, 5) for x in pcts[0.25].tolist()],
                "p50": [round(x, 5) for x in pcts[0.50].tolist()],
                "p75": [round(x, 5) for x in pcts[0.75].tolist()],
                "p90": [round(x, 5) for x in pcts[0.90].tolist()],
            },
            "mc_histogram": {
                "centers": hist_centers,
                "counts": hist_counts.tolist(),
                "var_total": round(float(df_var["Interest"].sum()), 2),
                "fixed_total": round(float(df_fixed["Interest"].sum()), 2),
            },
            "opportunity_cost": {
                "months": months,
                "portfolio": portfolio.round(2).tolist(),
                "fixed_extra_paid": round(float((df_fixed["Interest"].sum() - df_var["Interest"].sum())), 2),
            },
            "convexity": {
                "amounts": cvx_df["LumpSum"].tolist(),
                "total_interest": cvx_df["TotalInterest"].round(2).tolist(),
                "marginal_saving": cvx_df["MarginalSaving"].round(2).tolist(),
            },
            "summary": {
                "fixed_payment": round(fixed_pmt, 2),
                "var_payment": round(var_pmt, 2),
                "payment_delta": round(fixed_pmt - var_pmt, 2),
                "fixed_total_interest": round(float(df_fixed["Interest"].sum()), 2),
                "var_total_interest": round(float(df_var["Interest"].sum()), 2),
                "hedged_total_interest": round(float(df_hedged["Interest"].sum()), 2),
                "breakeven_inflation_fixed": round(be_fixed * 100, 2),
                "breakeven_inflation_var": round(be_var * 100, 2),
                "fixed_terminal_balance": round(float(df_fixed["Balance"].iloc[-1]), 2),
                "var_terminal_balance": round(float(df_var["Balance"].iloc[-1]), 2),
                "hedged_terminal_balance": round(float(df_hedged["Balance"].iloc[-1]), 2),
                "portfolio_terminal": round(float(portfolio[-1]), 2),
                "n_sims": n_sims,
            },
        })

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/amortize", methods=["POST"])
def amortize():
    """
    Return a full paginated amortization schedule table for a single strategy.

    Expects: all mortgage params + strategy in ["fixed","variable","hedged","stress"]
    """
    try:
        data = request.get_json(force=True) or {}
        mp, _vp, _opp = _parse_params(data)
        lump_sum = _parse_lump_sum(data)
        strategy = data.get("strategy", "fixed")

        engine = AmortizationEngine(mp)
        n = mp.term_months

        if strategy == "fixed":
            rate = mp.fixed_rate
            pmt = AmortizationEngine.level_payment(mp.principal, rate, mp.total_months)
            rates = np.full(n, rate)
            df = engine.amortize(rates, pmt)
        elif strategy == "variable":
            rate = mp.var_rate_start
            pmt = AmortizationEngine.level_payment(mp.principal, rate, mp.total_months)
            rates = np.full(n, rate)
            df = engine.amortize(rates, pmt)
        elif strategy == "hedged":
            fixed_pmt = AmortizationEngine.level_payment(mp.principal, mp.fixed_rate, mp.total_months)
            rates = np.full(n, mp.var_rate_start)
            df = engine.amortize(rates, fixed_pmt, lump_sum=lump_sum)
        elif strategy == "stress":
            stress_rate = mp.var_rate_start + 0.02
            pmt = AmortizationEngine.level_payment(mp.principal, stress_rate, mp.total_months)
            rates = np.full(n, stress_rate)
            df = engine.amortize(rates, pmt)
        else:
            return jsonify({"error": f"Unknown strategy: {strategy}"}), 400

        df["CumulativeInterest"] = df["Interest"].cumsum()
        df["CumulativePrincipal"] = df["Principal"].cumsum()

        rows = df.round(2).to_dict(orient="records")
        return jsonify({"rows": rows, "strategy": strategy})

    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
