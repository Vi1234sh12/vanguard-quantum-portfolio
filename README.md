# Q-MAT: Quantum Multi-Asset Terminal

> **Vanguard Challenge Project**
>
> A hybrid quantum-classical architecture for discrete portfolio optimization, bridging **Markowitz Portfolio Theory** and **Ising Spin-Glass/QUBO** formulations.

---

## 📖 Project Overview

Traditional **Markowitz Portfolio Optimization** assumes that portfolio weights are continuous, meaning capital can be divided infinitely among assets. In real institutional portfolios, however, investments must satisfy practical constraints such as:

- Discrete lot sizes
- Cardinality constraints
- Portfolio turnover limits
- Sector exposure limits
- Liquidity requirements

Introducing these constraints transforms the classical **Quadratic Programming (QP)** problem into a **Mixed-Integer Quadratic Program (MIQP)**, making it **NP-Hard**.

Q-MAT addresses this challenge by converting discrete portfolio optimization into a **Quadratic Unconstrained Binary Optimization (QUBO)** problem.

The optimization pipeline consists of:

1. **QUBO Construction**
   - Encode portfolio allocation into a **78-variable binary optimization problem** (13 assets × 6-bit precision).

2. **Discrete Optimization**
   - Solve the QUBO using **D-Wave Ocean's `neal` simulated annealer**.

3. **Continuous Projection**
   - Refine the solution using **SciPy SLSQP** while enforcing all institutional constraints.

4. **AI Compliance Assistant**
   - Generate an executive summary explaining:
     - Binding constraints
     - Portfolio trade-offs
     - Allocation rationale

---

## 🏗 Architecture

Q-MAT follows a hybrid quantum-classical optimization pipeline, transforming financial market data into a quantum-compatible optimization problem before producing an institutionally feasible portfolio with AI-generated explanations.

```text
                 Historical Market Data
                          │
                          ▼
        Black-Litterman Expected Returns
                          │
                          ▼
           PCA Covariance Estimation
                          │
                          ▼
      QUBO Construction (78 Binary Variables)
                          │
                          ▼
     D-Wave neal Simulated Annealing
          (Global Discrete Optimization)
                          │
                          ▼
      Classical SLSQP Optimization
     (Continuous Constraint Projection)
                          │
                          ▼
      Institutionally Feasible Portfolio
                          │
                          ▼
        AI Compliance Memo Generation
                          │
                          ▼
     Bloomberg-Style Terminal Dashboard
```

### Pipeline Overview

| Stage | Description |
|-------|-------------|
| 📈 **Historical Market Data** | Collects historical ETF prices and market information for portfolio construction. |
| 📊 **Black-Litterman Model** | Computes expected returns by combining market equilibrium with investor views. |
| 📉 **PCA Covariance Estimation** | Builds a robust covariance matrix for portfolio risk estimation. |
| ⚛️ **QUBO Construction** | Converts the constrained portfolio optimization problem into a 78-variable QUBO formulation. |
| 🔥 **Simulated Annealing** | Uses D-Wave Ocean's `neal` package to search for high-quality discrete portfolio allocations. |
| 🎯 **SLSQP Refinement** | Projects the discrete solution onto continuous feasible space while enforcing institutional constraints. |
| ✅ **Portfolio Validation** | Ensures compliance with budget, sector limits, turnover, liquidity, and position constraints. |
| 🤖 **AI Compliance Memo** | Automatically explains portfolio decisions, constraint trade-offs, and optimization outcomes. |
| 🖥 **Terminal Dashboard** | Presents results through an interactive Bloomberg-inspired web interface. |

---

# ✨ Key Features

## Quantum Optimization

- 78-variable QUBO formulation
- Ising Spin Hamiltonian mapping
- Binary encoding (6-bit precision per asset)
- Simulated Annealing using D-Wave Ocean (`neal`)

## Hybrid Optimization

- Global discrete search
- Continuous portfolio refinement using SLSQP
- Guaranteed feasible allocations

## Financial Modeling

- Black-Litterman expected return model
- PCA factor covariance estimation
- Geometric Brownian Motion (GBM)
- Portfolio risk analytics

## Institutional Constraints

Supports realistic investment constraints:

- Budget constraint
- Position limits
- Sector exposure caps
- Portfolio turnover
- Equity stress limits
- Liquidity requirements

## AI Co-Pilot

Automatically generates:

- Executive investment memo
- Constraint explanations
- Portfolio trade-off analysis
- Investment rationale

## Interactive Dashboard

Bloomberg-inspired terminal interface featuring:

- Portfolio allocation charts
- Efficient frontier visualization
- Risk metrics
- Live optimization pipeline
- AI-generated compliance reports

---

# 🛠 Tech Stack

## Optimization

- D-Wave Ocean SDK (`neal`)
- PyQUBO
- SciPy Optimization (SLSQP)

## Backend

- Python
- Flask
- PostgreSQL

## Frontend

- Jinja2
- Bootstrap 5
- Chart.js
- KaTeX

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/Vi1234sh12/vanguard-quantum-portfolio.git
cd vanguard-quantum-portfolio
```

## Create Virtual Environment

```bash
python3 -m venv venv
```

### Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Run Application

```bash
python main.py
```

Open your browser:

```
http://127.0.0.1:5000
```

---

# 📂 Project Structure

```text
vanguard-quantum-portfolio/

├── backend/                 # Flask backend, APIs, and web application
├── docs/                    # Project documentation
├── figures/                 # Images, diagrams, and research figures
├── notebooks/               # Research notebooks and experiments
├── Presentation/            # Final presentation materials
├── src/                     # Core optimization and financial models
│
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---
# 📚 References

## Research

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance.
- Lucas, A. (2014). *Ising formulations of many NP problems*. Frontiers in Physics.
- Farhi, E., Goldstone, J., & Gutmann, S. (2014). *Quantum Approximate Optimization Algorithm (QAOA)*.
- Black, F., & Litterman, R. (1991). *Global Asset Allocation with Equities, Bonds, and Currencies.*

---

## Data

The project uses publicly available historical ETF market data together with synthetic market scenarios generated using **Geometric Brownian Motion (GBM)**.

No proprietary or confidential financial datasets are used.

---

# 📄 License

This project was developed for the **Vanguard Challenge** as part of an academic and research initiative.

It is intended solely for educational and research purposes.

---
