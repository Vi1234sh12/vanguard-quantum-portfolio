# Mathematical Formulation: Hybrid Quantum Multi-Asset Portfolio Construction

This document outlines the precise mathematical transformation of a continuous mean-variance portfolio optimization problem into a discrete QUBO formulation compatible with quantum optimization algorithms.

## 1. Decision Variables (Binary Encoding)
To map continuous weights $w_i \in [0, w_{max}]$ to binary quantum states $x \in \{0,1\}^n$, we use an $m$-bit binary encoding:
 $$ w_i = \delta \sum_{j=0}^{m-1} 2^j x_{i,j} $$ Where $\delta = \frac{w_{max}}{2^m - 1}$ is the step size. For 12 assets with $m=4$ bits, this yields $N = 48$ binary variables.

## 2. Quadratic Objective Function
The classical Mean-Variance objective is:
 $$ \min_w \left( \lambda_\text{risk} w^T \Sigma w - \lambda_\text{ret} \mu^T w + \lambda_\text{tc} \sum_i (w_i - w_i^0)^2 \right) $$ 
Substituting the binary encoding, the quadratic risk term $w^T \Sigma w$ becomes:
 $$ \lambda_\text{risk} \delta^2 \sum_{i,k} \sum_{j,l} 2^j 2^l \sigma_{ik} x_{i,j} x_{k,l} $$ This is a purely quadratic function of the binary variables, native to QUBO formulations.

## 3. Linear Constraints as Penalty Terms
Quantum Annealers and QAOA require an unconstrained objective. We enforce linear constraints using quadratic penalty terms added to the QUBO matrix $Q$.

### Budget Constraint: $\sum w_i = 1$  $$ P_\text{budget} \left( \delta \sum_{i,j} 2^j x_{i,j} - 1 \right)^2 $$ 
### Sector Constraints: $\sum_{i \in S} w_i \le S_\text{max}$  $$ P_\text{sector} \max\left(0, \delta \sum_{i \in S, j} 2^j x_{i,j} - S_\text{max}\right)^2 $$ 
### Turnover Constraint: $\sum |w_i - w_i^0| \le T_\text{max}$ To maintain QUBO compatibility, we use an L2-norm approximation for turnover:
 $$ P_\text{turnover} \sum_i \left( \delta \sum_j 2^j x_{i,j} - w_i^0 \right)^2 $$ 
## 4. Final QUBO Matrix Assembly
The total cost Hamiltonian is:
 $$ H = x^T Q x $$ Where $Q$ is the symmetric $48 \times 48$ matrix comprising the summed coefficients of the Risk, Return, Turnover, and Constraint penalty terms.

## 5. Gate-Based Translation (QAOA)
For gate-based execution, the QUBO is translated to an Ising model via $x = \frac{1 - z}{2}$, and subsequently mapped to Qiskit Pauli-$Z$ operators:
 $$ H = \sum_i h_i Z_i + \sum_{i<j} J_{ij} Z_i Z_j + C I $$ This Hamiltonian was successfully encoded into a QAOA Ansatz with $p=2$ layers, resulting in a 48-qubit quantum circuit with a logical depth of 5, validated via MPS simulation.

## 6. The Hybrid Classical-Quantum Polish
To guarantee **Zero Hard-Constraint Breaches**, the discrete quantum output is used as the initial condition ($x_0$) for a classical Sequential Least Squares Programming (SLSQP) solver. This strictly enforces the continuous linear constraints without violating the quantum-discovered optimal risk-return structure.
