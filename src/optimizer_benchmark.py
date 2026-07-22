import numpy as np
from scipy.optimize import minimize
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize as pymoo_minimize
from pymoo.termination import get_termination
import matplotlib.pyplot as plt
import os

def run_slsqp(quantum_weights, data, config):
    """Standard Classical Polisher"""
    mu = data['mu']
    Sigma = data['Sigma']
    w0 = data['w0']
    sectors = data['sectors']
    n_assets = data['n_assets']

    def objective(w): 
        return -np.dot(w, mu) + config['LAMBDA_RISK'] * np.dot(w, np.dot(Sigma, w))

    # FIX: Removed hard return constraint
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: config['MAX_TURNOVER'] - np.sum(np.sqrt((w - w0)**2 + 1e-6))},
        {'type': 'ineq', 'fun': lambda w: config['EQUITY_STRESS_CAP'] - np.sum(w[sectors['Equities']])}
    ]
    for sec, indices in sectors.items():
        constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices: config['MAX_SECTOR'] - np.sum(w[idx])})

    # FIX: Apply liquidity caps to SLSQP benchmark
    liq_caps = data.get('liquidity_caps', [config['max_weight']] * n_assets)
    bounds = [(0, min(config['max_weight'], liq_caps[i])) for i in range(n_assets)]
    w_init = 0.8 * quantum_weights + 0.2 * (np.ones(n_assets) / n_assets)
    res = minimize(objective, w_init, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
    return res.x

def run_slsqp_frontier(data, config):
    """Generates an Efficient Frontier by looping SLSQP across target returns"""
    mu = data['mu']
    Sigma = data['Sigma']
    w0 = data['w0']
    sectors = data['sectors']
    n_assets = data['n_assets']
    
    def objective(w): 
        return np.dot(w, np.dot(Sigma, w)) 


    # FIX: Apply liquidity caps to SLSQP benchmark
    liq_caps = data.get('liquidity_caps', [config['max_weight']] * n_assets)
    bounds = [(0, min(config['max_weight'], liq_caps[i])) for i in range(n_assets)]

    frontier_risks = []
    frontier_returns = []
    
    target_returns = np.linspace(0.02, 0.12, 20)
    
    for target_ret in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w: config['MAX_TURNOVER'] - np.sum(np.sqrt((w - w0)**2 + 1e-6))},
            {'type': 'ineq', 'fun': lambda w: config['EQUITY_STRESS_CAP'] - np.sum(w[sectors['Equities']])},
            {'type': 'ineq', 'fun': lambda w, tr=target_ret: np.dot(w, mu) - tr}
        ]
        for sec, indices in sectors.items():
            constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices: config['MAX_SECTOR'] - np.sum(w[idx])})
            
        w_init = np.ones(n_assets) / n_assets 
        res = minimize(objective, w_init, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if res.success:
            port_ret = np.dot(res.x, mu)
            port_vol = np.sqrt(np.dot(res.x, np.dot(Sigma, res.x)))
            frontier_risks.append(port_vol)
            frontier_returns.append(port_ret)
            
    plt.figure(figsize=(10, 6))
    plt.plot(frontier_risks, frontier_returns, 'b--o', label='SLSQP Classical Frontier', markersize=5)
    plt.title("Classical Efficient Frontier (SLSQP)", fontsize=14, fontweight='bold')
    plt.xlabel("Expected Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/chart_slsqp_frontier.png', dpi=300)
    plt.close()
    print("   ✅ Saved: figures/chart_slsqp_frontier.png")

def run_nsga2(quantum_weights, data, config):
    """Advanced Multi-Objective Polisher (Generates Pareto Frontier)"""
    mu = data['mu']
    Sigma = data['Sigma']
    w0 = data['w0']
    sectors = data['sectors']
    n_assets = data['n_assets']

    class PortfolioProblem(Problem):
        def __init__(self):
            # FIX: Reduced constraint count to 3 + len(sectors) because hard return constraint was removed
            super().__init__(n_var=n_assets, n_obj=2, n_constr=3 + len(sectors),
                             xl=np.zeros(n_assets), xu=np.ones(n_assets) * config['max_weight'])

        def _evaluate(self, w, out, *args, **kwargs):
            risk = np.array([np.sqrt(np.dot(wi, np.dot(Sigma, wi))) for wi in w])
            ret = np.array([-np.dot(wi, mu) for wi in w])
            out["F"] = np.column_stack([risk, ret])
            
            g_budget = np.array([np.sum(wi) - 1.0 for wi in w])
            g_turnover = np.array([np.sum(np.sqrt((wi - w0)**2 + 1e-5)) - config['MAX_TURNOVER'] for wi in w])
            g_equity = np.array([np.sum(wi[sectors['Equities']]) - config['EQUITY_STRESS_CAP'] for wi in w])
            
            # FIX: Removed g_return hard constraint
            constraints = [g_budget, g_turnover, g_equity]
            for sec, indices in sectors.items():
                constraints.append(np.array([np.sum(wi[indices]) - config['MAX_SECTOR'] for wi in w]))
                
            out["G"] = np.column_stack(constraints)

    problem = PortfolioProblem()
    algorithm = NSGA2(pop_size=200)
    termination = get_termination("n_gen", 200)
    
    res = pymoo_minimize(problem, algorithm, termination, seed=42, verbose=False)
    
    if res.F is None or len(res.X) == 0:
        print("   ⚠️ NSGA-II found no feasible solutions. Reverting to SLSQP weights.")
        return run_slsqp(quantum_weights, data, config)
    
    sharpe_ratios = -res.F[:, 1] / res.F[:, 0]
    best_idx = np.argmax(sharpe_ratios)
    
    plt.figure(figsize=(10, 6))
    plt.scatter(res.F[:, 0], -res.F[:, 1], color='blue', alpha=0.6, label='NSGA-II Frontier')
    plt.scatter(res.F[best_idx, 0], -res.F[best_idx, 1], color='red', s=200, marker='*', edgecolor='black', label='Max Sharpe (Chosen)')
    
    cml_x = np.linspace(0, max(res.F[:, 0])*1.2, 100)
    cml_y = sharpe_ratios[best_idx] * cml_x
    plt.plot(cml_x, cml_y, 'k--', label='Capital Market Line (CML)')
    
    plt.title("Quantum-Enhanced Efficient Frontier & Capital Market Line", fontsize=14, fontweight='bold')
    plt.xlabel("Expected Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/chart_nsga2_frontier.png', dpi=300)
    plt.close()
    print("   ✅ Saved: figures/chart_nsga2_frontier.png")
    
    return res.X[best_idx]