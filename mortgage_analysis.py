"""
Institutional-Grade Mortgage Risk Engine
=========================================
Asset-Liability Management (ALM) tool for residential mortgage portfolios.

Implements:
    - Vasicek SDE stochastic rate simulation (vectorized)
    - Vectorized amortization engine (NumPy)
    - Opportunity-cost / equity-portfolio comparison
    - Break-even inflation via Fisher Equation
    - Convexity analysis for lump-sum prepayments
    - 4-pane executive dashboard (Matplotlib/GridSpec)

Author: Senior Quantitative Developer ó Financial Risk Management
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------------------------------------------------------------
# 1. DATA CLASSES ó Strict Input Contracts
# ------------------------------------------------------------------------------


@dataclass(frozen=True)
class MortgageParams:
    """Immutable contract for mortgage terms.

    Attributes:
        principal: Original loan balance in CAD.
        amortization_years: Full amortization horizon.
        fixed_rate: Annual fixed coupon rate (e.g. 0.0410).
        var_rate_start: Initial annual variable rate (e.g. 0.0335).
        term_months: Evaluation / renewal window (default 60 = 5 yr).
    """

    principal: float = 360_404.0
    amortization_years: int = 20
    fixed_rate: float = 0.0410
    var_rate_start: float = 0.0335
    term_months: int = 60

    @property
    def total_months(self) -> int:
        return self.amortization_years * 12


@dataclass(frozen=True)
class VasicekParams:
    """Vasicek SDE calibration set.

    dr_t = kappa * (theta - r_t) * dt + sigma * dW_t

    Attributes:
        kappa: Mean-reversion speed.
        theta: Long-run equilibrium rate.
        sigma: Instantaneous volatility.
        floor: Hard floor on simulated rates (regulatory minimum).
    """

    kappa: float = 0.35
    theta: float = 0.035
    sigma: float = 0.012
    floor: float = 0.0225


@dataclass(frozen=True)
class LumpSumSpec:
    """Lump-sum prepayment specification.

    Attributes:
        amount: Prepayment amount in CAD.
        month: 1-indexed month in which the prepayment occurs.
    """

    amount: float = 25_000.0
    month: int = 12


@dataclass(frozen=True)
class OpportunityCostParams:
    """Parameters for the equity-portfolio counterfactual.

    Attributes:
        equity_cagr: Assumed compound annual growth rate for equities.
    """

    equity_cagr: float = 0.07


# ------------------------------------------------------------------------------
# 2. STOCHASTIC RATE ENGINE
# ------------------------------------------------------------------------------


class VasicekRateEngine:
    """Vectorized Vasicek short-rate simulator.

    Generates Bank of Canada overnight-rate proxy paths via the
    EulerñMaruyama discretisation of the Vasicek SDE.

    Args:
        vasicek_params: Calibration parameters for the SDE.
        seed: Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        vasicek_params: VasicekParams = VasicekParams(),
        seed: Optional[int] = None,
    ) -> None:
        self.p = vasicek_params
        self._rng = np.random.default_rng(seed)

    def simulate(
        self,
        n_months: int,
        n_paths: int,
        r0: float,
    ) -> np.ndarray:
        """Generate rate paths.

        Args:
            n_months: Number of monthly time-steps.
            n_paths: Number of Monte-Carlo paths.
            r0: Initial short rate.

        Returns:
            np.ndarray of shape (n_months, n_paths) with annualised rates.
        """
        dt: float = 1.0 / 12.0
        paths = np.empty((n_months, n_paths), dtype=np.float64)
        paths[0, :] = r0

        # Pre-draw all shocks at once ó avoids Python-level loop overhead
        z = self._rng.standard_normal((n_months - 1, n_paths))

        sqrt_dt = np.sqrt(dt)
        for t in range(1, n_months):
            drift = self.p.kappa * (self.p.theta - paths[t - 1, :]) * dt
            diffusion = self.p.sigma * sqrt_dt * z[t - 1, :]
            paths[t, :] = np.maximum(self.p.floor, paths[t - 1, :] + drift + diffusion)

        return paths

    def percentiles(
        self,
        paths: np.ndarray,
        quantiles: Tuple[float, ...] = (0.10, 0.50, 0.90),
    ) -> Dict[float, np.ndarray]:
        """Compute cross-sectional percentiles of rate paths.

        Args:
            paths: Shape (n_months, n_paths).
            quantiles: Desired quantiles.

        Returns:
            Mapping from quantile ? 1-D array of length n_months.
        """
        return {q: np.quantile(paths, q, axis=1) for q in quantiles}


# ------------------------------------------------------------------------------
# 3. VECTORIZED AMORTIZATION ENGINE (CORRECTED)
# ------------------------------------------------------------------------------


class AmortizationEngine:
    """High-performance amortization calculator.

    All heavy paths are pure NumPy; the per-scenario loop is kept
    intentionally (balance is path-dependent) but operates on scalars
    via compiled ufuncs ó sufficient for 10 000+ MC iterations < 2 s.

    Args:
        mortgage_params: Core mortgage contract.
    """

    def __init__(self, mortgage_params: MortgageParams = MortgageParams()) -> None:
        self.mp = mortgage_params

    # -- static helpers ----------------------------------------------------

    @staticmethod
    def level_payment(principal: float, annual_rate: float, n_months: int) -> float:
        """Standard annuity payment formula.

        Args:
            principal: Loan balance.
            annual_rate: Annual nominal rate.
            n_months: Total amortization months.

        Returns:
            Monthly payment amount.
        """
        r = annual_rate / 12.0
        if r == 0.0:
            return principal / n_months
        return principal * (r * (1 + r) ** n_months) / ((1 + r) ** n_months - 1)

    # -- single-path vectorized amortization -------------------------------

    def amortize(
        self,
        annual_rates: np.ndarray,
        monthly_payment: float,
        lump_sum: Optional[LumpSumSpec] = None,
    ) -> pd.DataFrame:
        """Run amortization schedule for a single rate path.

        Args:
            annual_rates: 1-D array of length ``term_months``.
            monthly_payment: Constant monthly payment.
            lump_sum: Optional prepayment specification.

        Returns:
            DataFrame with columns Month, Rate, Interest, Principal, Balance.
        """
        n = self.mp.term_months
        balance = float(self.mp.principal)

        interest_arr = np.empty(n, dtype=np.float64)
        principal_arr = np.empty(n, dtype=np.float64)
        balance_arr = np.empty(n, dtype=np.float64)

        ls_idx = (lump_sum.month - 1) if lump_sum else -1
        ls_amt = lump_sum.amount if lump_sum else 0.0

        for i in range(n):
            # Calculate interest on current balance
            monthly_rate = annual_rates[i] / 12.0
            interest = balance * monthly_rate
            
            # Calculate principal payment from regular payment
            princ = monthly_payment - interest
            
            # Apply regular principal payment
            balance = max(0.0, balance - princ)
            
            # Apply lump sum payment AFTER regular payment (this is the fix!)
            lump_sum_this_month = 0.0
            if i == ls_idx:
                lump_sum_this_month = min(ls_amt, balance)  # Can't pay more than remaining balance
                balance = max(0.0, balance - lump_sum_this_month)
            
            # Record values
            interest_arr[i] = interest
            principal_arr[i] = princ + lump_sum_this_month  # Total principal includes lump sum
            balance_arr[i] = balance

        return pd.DataFrame(
            {
                "Month": np.arange(1, n + 1),
                "Rate": annual_rates[:n],
                "Interest": interest_arr,
                "Principal": principal_arr,
                "Balance": balance_arr,
            }
        )

    def amortize_bulk(
        self,
        rate_matrix: np.ndarray,
        monthly_payment: float,
        lump_sum: Optional[LumpSumSpec] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized bulk amortization across many rate paths.

        Optimised inner loop ó avoids DataFrame overhead per path.

        Args:
            rate_matrix: Shape (term_months, n_paths).
            monthly_payment: Constant monthly payment.
            lump_sum: Optional prepayment specification.

        Returns:
            Tuple of (terminal balances, total interests), both shape (n_paths,).
        """
        n = self.mp.term_months
        n_paths = rate_matrix.shape[1]
        balances = np.full(n_paths, self.mp.principal, dtype=np.float64)
        interests = np.zeros(n_paths, dtype=np.float64)

        ls_idx = (lump_sum.month - 1) if lump_sum else -1
        ls_amt = lump_sum.amount if lump_sum else 0.0

        for t in range(n):
            # Calculate interest
            monthly_rate = rate_matrix[t, :] / 12.0
            interest = balances * monthly_rate
            interests += interest
            
            # Apply regular principal payment
            princ = monthly_payment - interest
            balances = np.maximum(0.0, balances - princ)
            
            # Apply lump sum payment
            if t == ls_idx:
                lump_sum_payment = np.minimum(ls_amt, balances)
                balances = np.maximum(0.0, balances - lump_sum_payment)

        return balances, interests

    def amortize_bulk_full(
        self,
        rate_matrix: np.ndarray,
        monthly_payment: float,
        lump_sum: Optional[LumpSumSpec] = None,
    ) -> np.ndarray:
        """Full balance trajectory across all paths.

        Args:
            rate_matrix: Shape (term_months, n_paths).
            monthly_payment: Constant monthly payment.
            lump_sum: Optional prepayment specification.

        Returns:
            np.ndarray of shape (term_months, n_paths) ó balance at each step.
        """
        n = self.mp.term_months
        n_paths = rate_matrix.shape[1]
        balance_matrix = np.empty((n, n_paths), dtype=np.float64)
        balances = np.full(n_paths, self.mp.principal, dtype=np.float64)

        ls_idx = (lump_sum.month - 1) if lump_sum else -1
        ls_amt = lump_sum.amount if lump_sum else 0.0

        for t in range(n):
            # Calculate interest and apply regular payment
            monthly_rate = rate_matrix[t, :] / 12.0
            interest = balances * monthly_rate
            princ = monthly_payment - interest
            balances = np.maximum(0.0, balances - princ)
            
            # Apply lump sum payment
            if t == ls_idx:
                lump_sum_payment = np.minimum(ls_amt, balances)
                balances = np.maximum(0.0, balances - lump_sum_payment)
                
            balance_matrix[t, :] = balances

        return balance_matrix


# ------------------------------------------------------------------------------
# 4. RISK & ANALYTICS LAYER
# ------------------------------------------------------------------------------


class RiskAnalytics:
    """Quantitative overlays: opportunity cost, break-even inflation, convexity.

    Args:
        mortgage_params: Core mortgage contract.
        opp_params: Equity-portfolio assumptions.
    """

    def __init__(
        self,
        mortgage_params: MortgageParams = MortgageParams(),
        opp_params: OpportunityCostParams = OpportunityCostParams(),
    ) -> None:
        self.mp = mortgage_params
        self.opp = opp_params

    # -- Opportunity Cost (Invest-the-Difference) -------------------------

    def invest_the_difference(
        self,
        base_payment: float,
        alt_payment: float,
        lump_sum: Optional[LumpSumSpec] = None,
    ) -> np.ndarray:
        """Simulate a brokerage account funded by payment deltas.

        Each month the difference ``base_payment - alt_payment`` is invested
        at ``equity_cagr``.  If a lump_sum is specified, that amount is also
        invested in the corresponding month instead of prepaying the mortgage.

        Args:
            base_payment: Higher (e.g. fixed) monthly payment.
            alt_payment: Lower (e.g. variable) monthly payment.
            lump_sum: Amount that *could* have been invested.

        Returns:
            1-D array of portfolio value at each month-end.
        """
        n = self.mp.term_months
        monthly_r = (1 + self.opp.equity_cagr) ** (1 / 12) - 1
        portfolio = np.zeros(n, dtype=np.float64)
        value = 0.0
        delta = base_payment - alt_payment

        ls_idx = (lump_sum.month - 1) if lump_sum else -1
        ls_amt = lump_sum.amount if lump_sum else 0.0

        for t in range(n):
            contribution = delta
            if t == ls_idx:
                contribution += ls_amt
            value = (value + contribution) * (1 + monthly_r)
            portfolio[t] = value

        return portfolio

    # -- Break-even Inflation (Fisher Equation) ---------------------------

    @staticmethod
    def breakeven_inflation(nominal_rate: float, real_rate: float = 0.0) -> float:
        """Fisher Equation: i ò r + p  ?  p = i - r.

        When ``real_rate=0`` the break-even inflation equals the nominal rate
        (i.e., the rate at which the real cost of debt is zero).

        Args:
            nominal_rate: Annual nominal mortgage rate.
            real_rate: Target real cost of debt (default 0 ? break-even).

        Returns:
            Break-even annual inflation rate p.
        """
        return nominal_rate - real_rate

    # -- Convexity of Lump-Sum Prepayments --------------------------------

    def lump_sum_convexity(
        self,
        annual_rate: float,
        monthly_payment: float,
        ls_amounts: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Measure non-linear interest savings from increasing lump sums.

        For each lump-sum size, computes total interest paid over the term,
        then derives the marginal and second-order (convexity) savings.

        Args:
            annual_rate: Flat annual rate assumption.
            monthly_payment: Monthly payment.
            ls_amounts: Array of lump-sum sizes to evaluate.

        Returns:
            DataFrame with columns LumpSum, TotalInterest, MarginalSaving,
            Convexity.
        """
        if ls_amounts is None:
            ls_amounts = np.arange(0, 60_001, 5_000, dtype=np.float64)

        engine = AmortizationEngine(self.mp)
        flat_rates = np.full(self.mp.term_months, annual_rate)
        results: List[Dict[str, float]] = []

        for ls in ls_amounts:
            spec = LumpSumSpec(amount=float(ls), month=12) if ls > 0 else None
            df = engine.amortize(flat_rates, monthly_payment, lump_sum=spec)
            results.append({"LumpSum": ls, "TotalInterest": df["Interest"].sum()})

        out = pd.DataFrame(results)
        out["MarginalSaving"] = -out["TotalInterest"].diff().fillna(0)
        out["Convexity"] = out["MarginalSaving"].diff().fillna(0)
        return out


# ------------------------------------------------------------------------------
# 5. VISUALIZATION SUITE
# ------------------------------------------------------------------------------


class ExecutiveDashboard:
    """Four-pane institutional dashboard.

    Panes:
        1. Executive Summary Table
        2. Probability Fan Chart (Vasicek rate percentiles)
        3. Equity Trajectory (4 strategies)
        4. Sensitivity Waterfall (Alpha vs Fixed Baseline)

    Args:
        mortgage_params: Core mortgage terms.
        vasicek_params: Rate-model calibration.
        opp_params: Equity-portfolio assumptions.
        seed: RNG seed for reproducibility.
    """

    # Colour palette ó institutional blue / green / amber / red
    _C = {
        "fixed": "#1f4e79",
        "std_var": "#d4820a",
        "hedged": "#2e7d32",
        "invest": "#6a1b9a",
        "stress": "#c62828",
        "fan_fill": "#90caf9",
        "fan_med": "#0d47a1",
        "bg": "#fafafa",
        "grid": "#e0e0e0",
    }

    def __init__(
        self,
        mortgage_params: MortgageParams = MortgageParams(),
        vasicek_params: VasicekParams = VasicekParams(),
        opp_params: OpportunityCostParams = OpportunityCostParams(),
        seed: Optional[int] = 42,
    ) -> None:
        self.mp = mortgage_params
        self.vp = vasicek_params
        self.opp = opp_params

        self.rate_engine = VasicekRateEngine(vasicek_params, seed=seed)
        self.amort_engine = AmortizationEngine(mortgage_params)
        self.risk = RiskAnalytics(mortgage_params, opp_params)

        # Pre-compute level payments
        self.fixed_pmt = AmortizationEngine.level_payment(
            self.mp.principal, self.mp.fixed_rate, self.mp.total_months
        )
        self.var_pmt = AmortizationEngine.level_payment(
            self.mp.principal, self.mp.var_rate_start, self.mp.total_months
        )

    # -- helper: deterministic scenario schedules -------------------------

    def _build_deterministic_scenarios(
        self, lump_sum: LumpSumSpec
    ) -> Dict[str, pd.DataFrame]:
        """Build the three core deterministic amortization schedules.

        Args:
            lump_sum: Prepayment specification for the hedged strategy.

        Returns:
            Dict mapping strategy label ? amortization DataFrame.
        """
        n = self.mp.term_months
        flat_f = np.full(n, self.mp.fixed_rate)
        flat_v = np.full(n, self.mp.var_rate_start)

        return {
            "Fixed Baseline": self.amort_engine.amortize(flat_f, self.fixed_pmt),
            "Std Variable": self.amort_engine.amortize(flat_v, self.var_pmt),
            "Hedged Var + LS": self.amort_engine.amortize(
                flat_v, self.fixed_pmt, lump_sum=lump_sum
            ),
        }

    def _build_deterministic_scenarios_with_stress(
        self, lump_sum: LumpSumSpec
    ) -> Dict[str, pd.DataFrame]:
        """Build the four core deterministic amortization schedules including stress test.

        Args:
            lump_sum: Prepayment specification for the hedged strategy.

        Returns:
            Dict mapping strategy label ? amortization DataFrame.
        """
        n = self.mp.term_months
        flat_f = np.full(n, self.mp.fixed_rate)
        flat_v = np.full(n, self.mp.var_rate_start)
        
        # Stress scenario: +2% spike starting at month 12
        stress_rates = np.full(n, self.mp.var_rate_start)
        stress_rates[12:] = self.mp.var_rate_start + 0.02  # 2% spike after year 1

        return {
            "Fixed (4.10%)": self.amort_engine.amortize(flat_f, self.fixed_pmt),
            "Variable (3.35%)": self.amort_engine.amortize(flat_v, self.var_pmt),
            "Hedged Variable": self.amort_engine.amortize(
                flat_v, self.fixed_pmt, lump_sum=lump_sum
            ),
            "Stress (Var + 2% Spike)": self.amort_engine.amortize(
                stress_rates, self.fixed_pmt, lump_sum=lump_sum
            ),
        }

    # -- PANE 1: Executive Summary Table ----------------------------------

    def _render_summary_table(
        self,
        ax: plt.Axes,
        scenarios: Dict[str, pd.DataFrame],
        invest_terminal: float,
    ) -> None:
        """Render the executive KPI table.

        Args:
            ax: Matplotlib Axes (will be turned off).
            scenarios: Strategy label ? DataFrame.
            invest_terminal: Terminal value of the invest-the-difference portfolio.
        """
        ax.axis("off")

        col_labels = list(scenarios.keys()) + ["Invest ?"]
        row_labels = [
            "Monthly Payment",
            "Total Interest (5Y)",
            "Equity Gained (5Y)",
            "Terminal Balance (Mo 60)",
        ]

        payments = {
            "Fixed Baseline": self.fixed_pmt,
            "Std Variable": self.var_pmt,
            "Hedged Var + LS": self.fixed_pmt,
        }

        cell_text: List[List[str]] = []

        # Row 0 ó Monthly Payment
        row = [f"${payments[k]:,.0f}" for k in scenarios] + ["ó"]
        cell_text.append(row)

        # Row 1 ó Total Interest
        row = [f"${scenarios[k]['Interest'].sum():,.0f}" for k in scenarios]
        row.append("ó")
        cell_text.append(row)

        # Row 2 ó Equity Gained (principal paid + lump sum reduction)
        row = [f"${scenarios[k]['Principal'].sum():,.0f}" for k in scenarios]
        row.append(f"${invest_terminal:,.0f}")
        cell_text.append(row)

        # Row 3 ó Terminal Balance
        row = [f"${scenarios[k].iloc[-1]['Balance']:,.0f}" for k in scenarios]
        row.append("ó")
        cell_text.append(row)

        table = ax.table(
            cellText=cell_text,
            rowLabels=row_labels,
            colLabels=col_labels,
            loc="center",
            cellLoc="center",
        )
        
        # Fix table sizing and font
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.8)
        
        # Adjust cell dimensions - use correct indexing
        cellDict = table.get_celld()
        
        # Set uniform cell dimensions for all existing cells
        for (i, j), cell in cellDict.items():
            cell.set_width(0.18)
            cell.set_height(0.12)
            
        # Style header row
        for j in range(len(col_labels)):
            if (0, j) in cellDict:
                cellDict[(0, j)].set_facecolor("#1f4e79")
                cellDict[(0, j)].set_text_props(color="white", weight="bold", size=8)
            
        # Style row labels
        for i in range(1, len(row_labels) + 1):
            if (i, -1) in cellDict:
                cellDict[(i, -1)].set_facecolor("#e8eaf6")
                cellDict[(i, -1)].set_text_props(weight="bold", size=8)

        ax.set_title(
            "EXECUTIVE QUANT SUMMARY ó CALGARY MORTGAGE PORTFOLIO",
            fontsize=11,
            weight="bold",
            pad=8,
            y=0.95
        )
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.0)

    # -- PANE 2: Probability Fan Chart ------------------------------------

    def _render_fan_chart(
        self,
        ax: plt.Axes,
        paths: np.ndarray,
    ) -> None:
        """Render the stochastic rate fan chart with 10/50/90 percentiles.

        Args:
            ax: Target Axes.
            paths: Shape (n_months, n_paths).
        """
        months = np.arange(1, paths.shape[0] + 1)
        pcts = self.rate_engine.percentiles(paths, (0.10, 0.50, 0.90))

        ax.fill_between(
            months,
            pcts[0.10] * 100,
            pcts[0.90] * 100,
            alpha=0.25,
            color=self._C["fan_fill"],
            label="10thñ90th pctl",
        )
        ax.plot(
            months, pcts[0.50] * 100, color=self._C["fan_med"], lw=2, label="Median"
        )
        ax.axhline(
            self.mp.fixed_rate * 100,
            color=self._C["fixed"],
            ls="--",
            lw=1,
            label=f"Fixed ({self.mp.fixed_rate*100:.2f}%)",
        )
        ax.set_ylabel("Annualised Rate (%)")
        ax.set_xlabel("Month")
        ax.set_title("BoC Rate Paths ó Vasicek Fan Chart", weight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3, color=self._C["grid"])

    # -- PANE 3: Equity Trajectory ----------------------------------------

    def _render_equity_trajectory(
        self,
        ax: plt.Axes,
        scenarios: Dict[str, pd.DataFrame],
        invest_portfolio: np.ndarray,
    ) -> None:
        """Comparative equity build across strategies.

        Equity = Principal - Balance  (i.e., cumulative principal repaid).

        Args:
            ax: Target Axes.
            scenarios: Strategy label ? DataFrame.
            invest_portfolio: 1-D array of invest-the-difference portfolio value.
        """
        colours = {
            "Fixed Baseline": self._C["fixed"],
            "Std Variable": self._C["std_var"],
            "Hedged Var + LS": self._C["hedged"],
        }
        styles = {
            "Fixed Baseline": "--",
            "Std Variable": ":",
            "Hedged Var + LS": "-",
        }

        for label, df in scenarios.items():
            equity = self.mp.principal - df["Balance"].values
            ax.plot(
                df["Month"],
                equity,
                color=colours[label],
                ls=styles[label],
                lw=2,
                label=f"{label}: ${equity[-1]:,.0f}",
            )

        # Invest-the-difference line: equity is Std Var equity + portfolio value
        std_var_equity = (
            self.mp.principal - scenarios["Std Variable"]["Balance"].values
        )
        combined = std_var_equity + invest_portfolio
        ax.plot(
            np.arange(1, self.mp.term_months + 1),
            combined,
            color=self._C["invest"],
            lw=2,
            ls="-.",
            label=f"Invest ? (7% CAGR): ${combined[-1]:,.0f}",
        )

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_ylabel("Cumulative Equity ($)")
        ax.set_xlabel("Month")
        ax.set_title("Equity Trajectory ó Strategy Comparison", weight="bold")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3, color=self._C["grid"])

    # -- PANE 4: Sensitivity Waterfall ------------------------------------

    def _render_waterfall(
        self,
        ax: plt.Axes,
        scenarios: Dict[str, pd.DataFrame],
        invest_terminal: float,
    ) -> None:
        """Alpha waterfall ó net-worth gain vs Fixed Baseline.

        Alpha = (Equity_strategy - Equity_fixed) + (Interest_fixed - Interest_strategy).

        Args:
            ax: Target Axes.
            scenarios: Strategy label ? DataFrame.
            invest_terminal: Terminal value of the invest-the-difference portfolio.
        """
        fixed_interest = scenarios["Fixed Baseline"]["Interest"].sum()
        fixed_equity = self.mp.principal - scenarios["Fixed Baseline"].iloc[-1]["Balance"]

        labels: List[str] = []
        alphas: List[float] = []

        for label, df in scenarios.items():
            if label == "Fixed Baseline":
                continue
            strat_interest = df["Interest"].sum()
            strat_equity = self.mp.principal - df.iloc[-1]["Balance"]
            alpha = (strat_equity - fixed_equity) + (fixed_interest - strat_interest)
            labels.append(label)
            alphas.append(alpha)

        # Invest-the-difference alpha
        std_var_equity = (
            self.mp.principal - scenarios["Std Variable"].iloc[-1]["Balance"]
        )
        std_var_interest = scenarios["Std Variable"]["Interest"].sum()
        invest_alpha = (
            (std_var_equity + invest_terminal - fixed_equity)
            + (fixed_interest - std_var_interest)
        )
        labels.append("Invest ? (7%)")
        alphas.append(invest_alpha)

        colours = [
            self._C["hedged"] if a > 0 else self._C["stress"] for a in alphas
        ]
        bars = ax.bar(labels, alphas, color=colours, alpha=0.85, edgecolor="white")

        for bar, val in zip(bars, alphas):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (200 if val > 0 else -600),
                f"${val:,.0f}",
                ha="center",
                va="bottom" if val > 0 else "top",
                fontsize=10,
                weight="bold",
            )

        ax.axhline(0, color="black", lw=0.8)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_ylabel("Alpha vs Fixed Baseline ($)")
        ax.set_title("Sensitivity Waterfall ó Net-Worth Alpha", weight="bold")
        ax.grid(True, axis="y", alpha=0.3, color=self._C["grid"])

    # -- PUBLIC: Render Full Stochastic Dashboard -------------------------

    def render(
        self,
        n_sims: int = 5_000,
        lump_sum: Optional[LumpSumSpec] = None,
        figsize: Tuple[float, float] = (20, 16),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Compose and render the four-pane executive dashboard.

        Args:
            n_sims: Number of Monte-Carlo paths for the fan chart.
            lump_sum: Prepayment specification (defaults to $25 000 at month 12).
            figsize: Figure dimensions in inches.
            save_path: If provided, saves the figure to this path.
            dpi: Resolution for saved figure.

        Returns:
            The Matplotlib Figure object.
        """
        if lump_sum is None:
            lump_sum = LumpSumSpec()

        # -- data generation -----------------------------------------------
        scenarios = self._build_deterministic_scenarios(lump_sum)

        # Stochastic paths
        paths = self.rate_engine.simulate(
            self.mp.term_months, n_sims, self.mp.var_rate_start
        )

        # Invest-the-difference
        invest_portfolio = self.risk.invest_the_difference(
            self.fixed_pmt, self.var_pmt, lump_sum=lump_sum
        )

        # -- figure layout with better spacing -----------------------------
        fig = plt.figure(figsize=figsize, facecolor=self._C["bg"])
        gs = gridspec.GridSpec(
            2,
            2,
            height_ratios=[1, 1.2],
            width_ratios=[1, 1],
            hspace=0.4,
            wspace=0.3,
            left=0.05,
            right=0.97,
            top=0.92,
            bottom=0.08,
        )

        ax_table = fig.add_subplot(gs[0, 0])
        ax_fan = fig.add_subplot(gs[0, 1])
        ax_equity = fig.add_subplot(gs[1, 0])
        ax_water = fig.add_subplot(gs[1, 1])

        # -- render panes --------------------------------------------------
        self._render_summary_table(ax_table, scenarios, invest_portfolio[-1])
        self._render_fan_chart(ax_fan, paths)
        self._render_equity_trajectory(ax_equity, scenarios, invest_portfolio)
        self._render_waterfall(ax_water, scenarios, invest_portfolio[-1])

        # -- Break-even inflation annotation -------------------------------
        be_fixed = RiskAnalytics.breakeven_inflation(self.mp.fixed_rate)
        be_var = RiskAnalytics.breakeven_inflation(self.mp.var_rate_start)
        fig.text(
            0.5,
            0.98,
            (
                f"Break-even Inflation (Fisher):  Fixed ? p = {be_fixed*100:.2f}%  |  "
                f"Variable ? p = {be_var*100:.2f}%  |  "
                f"Vasicek: ?={self.vp.kappa}, ?={self.vp.theta*100:.1f}%, "
                f"s={self.vp.sigma*100:.1f}%  |  MC Paths: {n_sims:,}"
            ),
            ha="center",
            fontsize=9,
            style="italic",
            color="#555555",
        )

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())

        return fig

    # -- PUBLIC: Render Deterministic Analysis ----------------------------

    def render_deterministic_analysis(
        self,
        lump_sum: Optional[LumpSumSpec] = None,
        figsize: Tuple[float, float] = (16, 12),
        save_path: Optional[str] = None,
        dpi: int = 150,
    ) -> plt.Figure:
        """Render the deterministic 4-panel analysis matching the provided image.

        Args:
            lump_sum: Prepayment specification (defaults to $25 000 at month 12).
            figsize: Figure dimensions in inches.
            save_path: If provided, saves the figure to this path.
            dpi: Resolution for saved figure.

        Returns:
            The Matplotlib Figure object.
        """
        if lump_sum is None:
            lump_sum = LumpSumSpec()

        # Build scenarios including stress test
        scenarios = self._build_deterministic_scenarios_with_stress(lump_sum)
        
        # Create figure with 2x2 layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize, facecolor='white')
        fig.suptitle("Mortgage Strategy Analysis ó Calgary Portfolio", fontsize=16, fontweight='bold', y=0.98)

        # Colors matching the image
        colors = {
            "Fixed (4.10%)": "#1f4e79",
            "Variable (3.35%)": "#ff8c00", 
            "Hedged Variable": "#2e7d32",
            "Stress (Var + 2% Spike)": "#dc143c"
        }

        # Panel 1: Principal Balance Trajectory
        ax1.set_title("Mortgage Principal Balance (Next 60 Months)", fontweight='bold', pad=15)
        
        for label, df in scenarios.items():
            style = '--' if 'Hedged' in label else '-'
            ax1.plot(df['Month'], df['Balance'], label=label, color=colors[label], 
                    linewidth=2.5 if 'Hedged' in label else 2, linestyle=style)
        
        ax1.set_xlabel("Month")
        ax1.set_ylabel("Balance ($)")
        ax1.legend(loc='upper right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

        # Panel 2: Total Interest Paid (Bar Chart)
        ax2.set_title("Total Interest Paid (5-Year Term)", fontweight='bold', pad=15)
        
        labels = list(scenarios.keys())
        interest_values = [scenarios[label]['Interest'].sum() for label in labels]
        
        bars = ax2.bar(range(len(labels)), interest_values, 
                       color=[colors[label] for label in labels], alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, interest_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'${val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels([l.split('(')[0].strip() for l in labels], rotation=45, ha='right')
        ax2.set_ylabel("Interest ($)")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax2.grid(True, axis='y', alpha=0.3)

        # Panel 3: Total Principal Paid (Equity Gain)
        ax3.set_title("Total Principal Paid (Equity Gain)", fontweight='bold', pad=15)
        
        principal_values = [scenarios[label]['Principal'].sum() for label in labels]
        
        bars = ax3.bar(range(len(labels)), principal_values,
                       color=[colors[label] for label in labels], alpha=0.8)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, principal_values)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                    f'${val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax3.set_xticks(range(len(labels)))
        ax3.set_xticklabels([l.split('(')[0].strip() for l in labels], rotation=45, ha='right')
        ax3.set_ylabel("Principal ($)")
        ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax3.grid(True, axis='y', alpha=0.3)

        # Panel 4: Rate Spike Stress Test (Year 2)
        ax4.set_title("Rate Spike Stress Test (Year 2)", fontweight='bold', pad=15)
        
        months = np.arange(0, 61)
        
        # Base rates
        ax4.axhline(self.mp.fixed_rate * 100, color=colors["Fixed (4.10%)"], 
                   linestyle='-', label='Fixed Rate (4.10%)', linewidth=2)
        ax4.axhline(self.mp.var_rate_start * 100, color=colors["Variable (3.35%)"], 
                   linestyle='--', label='Initial Var (3.35%)', linewidth=2)
        
        # Spike scenario
        spike_rates = np.full(61, self.mp.var_rate_start * 100)
        spike_rates[12:] = (self.mp.var_rate_start + 0.02) * 100
        ax4.plot(months, spike_rates, color=colors["Stress (Var + 2% Spike)"], 
                linewidth=3, label='Spike Path (5.35%)')
        
        ax4.set_xlabel("Month")
        ax4.set_ylabel("Annual Rate (%)")
        ax4.set_xlim(0, 60)
        ax4.set_ylim(3.0, 5.5)
        ax4.legend(loc='center right', fontsize=9)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor='white')

        return fig


# ------------------------------------------------------------------------------
# 6. CONVEXITY REPORT (STANDALONE CHART)
# ------------------------------------------------------------------------------


def plot_convexity_report(
    mortgage_params: MortgageParams = MortgageParams(),
    annual_rate: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 6),
) -> plt.Figure:
    """Standalone convexity chart for lump-sum prepayment analysis.

    Args:
        mortgage_params: Core mortgage terms.
        annual_rate: Rate to use (defaults to the variable start rate).
        figsize: Figure dimensions.

    Returns:
        Matplotlib Figure.
    """
    if annual_rate is None:
        annual_rate = mortgage_params.var_rate_start

    pmt = AmortizationEngine.level_payment(
        mortgage_params.principal, annual_rate, mortgage_params.total_months
    )
    risk = RiskAnalytics(mortgage_params)
    df = risk.lump_sum_convexity(annual_rate, pmt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="#fafafa")

    # Total interest curve
    ax1.plot(
        df["LumpSum"],
        df["TotalInterest"],
        "o-",
        color="#1f4e79",
        lw=2,
        markersize=4,
    )
    ax1.set_xlabel("Lump-Sum Amount ($)")
    ax1.set_ylabel("Total Interest Paid ($)")
    ax1.set_title("Interest Cost vs Lump-Sum Size", weight="bold")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    # Marginal saving & convexity
    ax2.bar(
        df["LumpSum"],
        df["MarginalSaving"],
        width=4000,
        color="#2e7d32",
        alpha=0.7,
        label="Marginal Saving",
    )
    ax2_twin = ax2.twinx()
    ax2_twin.plot(
        df["LumpSum"],
        df["Convexity"],
        "s--",
        color="#c62828",
        lw=1.5,
        markersize=4,
        label="Convexity (2nd ?)",
    )
    ax2.set_xlabel("Lump-Sum Amount ($)")
    ax2.set_ylabel("Marginal Interest Saving ($)")
    ax2_twin.set_ylabel("Convexity ($)")
    ax2.set_title("Marginal Saving & Convexity", weight="bold")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax2.legend(loc="upper left", fontsize=8)
    ax2_twin.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------
# 7. DEBUG & DIAGNOSTIC UTILITIES
# ------------------------------------------------------------------------------


def debug_lump_sum():
    """Debug function to test lump sum functionality."""
    print("=" * 60)
    print("DEBUGGING LUMP SUM FUNCTIONALITY")
    print("=" * 60)
    
    params = MortgageParams()
    engine = AmortizationEngine(params)
    
    # Calculate payment
    fixed_pmt = AmortizationEngine.level_payment(
        params.principal, params.fixed_rate, params.total_months
    )
    
    # Create flat rate array
    flat_rates = np.full(params.term_months, params.var_rate_start)
    
    print(f"Principal: ${params.principal:,.2f}")
    print(f"Monthly Payment: ${fixed_pmt:,.2f}")
    print(f"Rate: {params.var_rate_start*100:.2f}%")
    print()
    
    # Test without lump sum
    print("WITHOUT LUMP SUM:")
    df_no_ls = engine.amortize(flat_rates, fixed_pmt, lump_sum=None)
    print(f"Month 11 Balance: ${df_no_ls.iloc[10]['Balance']:,.2f}")
    print(f"Month 12 Balance: ${df_no_ls.iloc[11]['Balance']:,.2f}")
    print(f"Month 13 Balance: ${df_no_ls.iloc[12]['Balance']:,.2f}")
    print(f"Total Interest: ${df_no_ls['Interest'].sum():,.2f}")
    print(f"Total Principal: ${df_no_ls['Principal'].sum():,.2f}")
    print()
    
    # Test with lump sum
    print("WITH $25,000 LUMP SUM AT MONTH 12:")
    lump_sum = LumpSumSpec(amount=25_000, month=12)
    df_with_ls = engine.amortize(flat_rates, fixed_pmt, lump_sum=lump_sum)
    print(f"Month 11 Balance: ${df_with_ls.iloc[10]['Balance']:,.2f}")
    print(f"Month 12 Balance: ${df_with_ls.iloc[11]['Balance']:,.2f}")
    print(f"Month 13 Balance: ${df_with_ls.iloc[12]['Balance']:,.2f}")
    print(f"Month 12 Principal: ${df_with_ls.iloc[11]['Principal']:,.2f}")
    print(f"Total Interest: ${df_with_ls['Interest'].sum():,.2f}")
    print(f"Total Principal: ${df_with_ls['Principal'].sum():,.2f}")
    print()
    
    # Show the difference
    interest_diff = df_no_ls['Interest'].sum() - df_with_ls['Interest'].sum()
    balance_diff = df_no_ls.iloc[-1]['Balance'] - df_with_ls.iloc[-1]['Balance']
    
    print("DIFFERENCES:")
    print(f"Interest Savings: ${interest_diff:,.2f}")
    print(f"Balance Reduction: ${balance_diff:,.2f}")
    
    if abs(interest_diff) < 100:
        print("? WARNING: Lump sum appears to have minimal impact!")
    else:
        print("? Lump sum is working correctly!")
    
    print("=" * 60)


def performance_test() -> None:
    """Test vectorized performance for 10,000+ Monte Carlo paths."""
    import time
    
    print("Running Performance Test...")
    
    params = MortgageParams()
    engine = AmortizationEngine(params)
    rate_engine = VasicekRateEngine(seed=42)
    
    # Generate 10,000 paths
    n_paths = 10_000
    print(f"Generating {n_paths:,} Monte Carlo paths...")
    
    start_time = time.time()
    paths = rate_engine.simulate(params.term_months, n_paths, params.var_rate_start)
    
    # Run bulk amortization
    fixed_pmt = AmortizationEngine.level_payment(
        params.principal, params.fixed_rate, params.total_months
    )
    
    balances, interests = engine.amortize_bulk(
        paths, fixed_pmt, lump_sum=LumpSumSpec(amount=25_000, month=12)
    )
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Processed {n_paths:,} paths in {elapsed:.3f} seconds")
    print(f"Performance: {n_paths/elapsed:,.0f} paths/second")
    print(f"Target achieved: {'?' if elapsed < 2.0 else '?'} (< 2 seconds)")
    print()


# ------------------------------------------------------------------------------
# 8. ENTRY POINT
# ------------------------------------------------------------------------------


def main() -> None:
    """Run the full analysis pipeline."""
    print("Initializing Mortgage Risk Engine...")
    print()
    
    # Run debug test first
    debug_lump_sum()
    print()
    
    # Instantiate with defaults (Calgary portfolio)
    params = MortgageParams()
    vasicek = VasicekParams()
    opp = OpportunityCostParams()

    dashboard = ExecutiveDashboard(params, vasicek, opp, seed=42)

    # Render deterministic analysis (matching the provided image)
    print("Rendering Deterministic Analysis Dashboard...")
    fig_deterministic = dashboard.render_deterministic_analysis(
        lump_sum=LumpSumSpec(amount=25_000, month=12),
    )

    # Render full stochastic 4-pane dashboard
    print("Rendering Full Stochastic Dashboard...")
    fig_main = dashboard.render(
        n_sims=5_000,
        lump_sum=LumpSumSpec(amount=25_000, month=12),
    )

    # Convexity side-report
    print("Rendering Convexity Analysis...")
    fig_cvx = plot_convexity_report(params)

    # Print key analytics to console
    risk = RiskAnalytics(params, opp)
    be_fixed = risk.breakeven_inflation(params.fixed_rate)
    be_var = risk.breakeven_inflation(params.var_rate_start)

    print()
    print("=" * 64)
    print("  MORTGAGE RISK ENGINE ó KEY METRICS")
    print("=" * 64)
    print(f"  Fixed Payment ............ ${dashboard.fixed_pmt:>10,.2f}")
    print(f"  Variable Payment ......... ${dashboard.var_pmt:>10,.2f}")
    print(f"  Payment Delta ............ ${dashboard.fixed_pmt - dashboard.var_pmt:>10,.2f}")
    print(f"  Break-even p (Fixed) ..... {be_fixed*100:>10.2f}%")
    print(f"  Break-even p (Variable) .. {be_var*100:>10.2f}%")
    print("=" * 64)
    print()

    plt.show()


if __name__ == "__main__":
    main()
    
    # Uncomment to run performance test
    # performance_test()