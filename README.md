# Mortgage Rate Strategy

An institutional-grade web application for comparing fixed vs. variable mortgage strategies using Monte Carlo rate simulation, vectorized amortization, and opportunity-cost analytics.

## Features

- **Scenario modelling** — Fixed, Variable, Hedged Variable (+ lump-sum), and Stress (+2%) strategies
- **Vasicek SDE Monte Carlo** — Stochastic interest rate fan charts (up to 10 000 paths)
- **Amortization engine** — Full schedules with cumulative interest/principal tracking
- **Risk analytics** — Opportunity cost (invest-the-difference), break-even inflation (Fisher equation), lump-sum convexity
- **Chart exports** — PNG, JPEG, and PDF for every chart; CSV and PDF for the amortization table

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

Open [http://localhost:5000](http://localhost:5000).

## Project Structure

```
MortgageMgt/
├── mortgage_analysis.py   # Core engine (Vasicek, amortization, risk analytics)
├── app.py                 # Flask web server + REST API
├── requirements.txt
├── templates/
│   └── index.html         # Single-page frontend (Bootstrap 5 + Chart.js)
└── static/
    ├── css/style.css
    └── js/app.js
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/` | Serve the web application |
| `POST` | `/api/calculate` | Run full analysis (MC + deterministic) |
| `POST` | `/api/amortize` | Return amortization schedule for one strategy |

Both POST endpoints accept JSON with mortgage parameters (see `app.py` `_parse_params` for the full schema).

## Default Parameters (Calgary Portfolio)

| Parameter | Value |
|-----------|-------|
| Principal | $360,404 CAD |
| Amortization | 20 years |
| Term | 60 months |
| Fixed rate | 4.10% |
| Variable rate | 3.35% |
| Lump sum | $25,000 at month 12 |
| Equity CAGR | 7.0% |
