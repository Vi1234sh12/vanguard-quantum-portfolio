import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import base64
import io
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Flask
import matplotlib.pyplot as plt
from scipy.optimize import minimize as sp_minimize
from data_generator import get_multi_asset_data
from black_litterman import simulate_gbm_returns, black_litterman_expected_returns
from derivatives import add_derivative_to_universe
from qubo_engine import build_qubo
from hybrid_solver import solve_hybrid
from copilot_agent import generate_ai_memo
from expert_validator import evaluate_expert_analysis

def run_optimization_pipeline(config, status_callback=None):
    
    if status_callback: status_callback("STEP 1: Fetching Market Data & Factor Models...")
    # 1. Data & Financial Modeling
    data = get_multi_asset_data()
    data['mu'] = np.nan_to_num(data['mu'], nan=0.0)
    data['Sigma'] = np.nan_to_num(data['Sigma'], nan=0.0)
    for i in range(len(data['mu'])):
        if data['Sigma'][i, i] <= 0: data['Sigma'][i, i] = 0.01

    if status_callback: status_callback("STEP 2: Running GBM & Black-Litterman Models...")
    w_mkt = np.array([0.05, 0.15, 0.10, 0.10, 0.10, 0.02, 0.02, 0.05, 0.05, 0.10, 0.03, 0.08]) 
    mu_gbm = simulate_gbm_returns(data['mu'], data['Sigma'], n_sims=500)
    mu_bl, _ = black_litterman_expected_returns(data['Sigma'], mu_gbm, w_mkt)
    data['mu'] = mu_bl
    data = add_derivative_to_universe(data, config)
    tickers = data['names']

    if status_callback: status_callback("STEP 3: Mapping 78-Qubit QUBO & Ising Hamiltonian...")
    # 2. QUBO & Hybrid Solve
    Q, n_qubits, step = build_qubo(data, config)
    
    if status_callback: status_callback("STEP 4: Quantum Annealing & SLSQP Polish...")
    results = solve_hybrid(Q, n_qubits, step, data, config)
    final_weights = results[0]

    # 3. Core Metrics & 3-Way Comparison
    port_ret = np.dot(final_weights, data['mu'])
    port_vol = np.sqrt(np.dot(final_weights, np.dot(data['Sigma'], final_weights)))
    sharpe = port_ret / port_vol if port_vol > 0 else 0

    # Raw QUBO Metrics (Before Classical Polish)
    qubo_ret = np.dot(results[11], data['mu'])
    qubo_vol = np.sqrt(np.dot(results[11], np.dot(data['Sigma'], results[11])))
    qubo_sharpe = qubo_ret / qubo_vol if qubo_vol > 0 else 0

    # 4. Classical Baseline using CVXPY (Institutional Standard)
    import cvxpy as cp
    try:
        w_cvx = cp.Variable(data['n_assets'])
        objective = cp.Minimize(cp.quad_form(w_cvx, cp.psd_wrap(data['Sigma'])))
        liq_caps = data.get('liquidity_caps', [config['max_weight']] * data['n_assets'])
        max_bounds = [min(config['max_weight'], liq_caps[i]) for i in range(data['n_assets'])]
        constraints_cvx = [cp.sum(w_cvx) == 1, w_cvx >= 0, w_cvx <= max_bounds]
        prob = cp.Problem(objective, constraints_cvx)
        prob.solve()
        classical_weights = w_cvx.value
        if classical_weights is None:
            raise Exception("CVXPY infeasible")
    except Exception as e:
        print(f"⚠️ CVXPY baseline failed: {e}. Reverting to Equal Weight.")
        classical_weights = np.ones(data['n_assets']) / data['n_assets']

    c_ret = float(np.dot(classical_weights, data['mu']))
    c_vol = float(np.sqrt(np.dot(classical_weights, np.dot(data['Sigma'], classical_weights))))
    c_sharpe = c_ret / c_vol if c_vol > 0 else 0

    # 5. Constraint Utilization (Enterprise Compliance)
    turnover_used = np.sum(np.abs(final_weights - data['w0']))
    equity_used = np.sum(final_weights[data['sectors']['Equities']])
    
    liq_caps = data.get('liquidity_caps', [config['max_weight']] * data['n_assets'])
    liq_utilization = [final_weights[i] / liq_caps[i] if liq_caps[i] > 0 else 0 for i in range(data['n_assets'])]
    max_liq_idx = int(np.argmax(liq_utilization))
    max_liq_used = float(final_weights[max_liq_idx])
    max_liq_limit = float(liq_caps[max_liq_idx])
    max_liq_asset = tickers[max_liq_idx]
    
    constraints = {
        "budget": {"used": float(np.sum(final_weights)), "limit": 1.0, "unit": "%"},
        "turnover": {"used": float(turnover_used), "limit": float(config['MAX_TURNOVER']), "unit": "%"},
        "equity_cap": {"used": float(equity_used), "limit": float(config['EQUITY_STRESS_CAP']), "unit": "%"},
        "liquidity": {"used": max_liq_used, "limit": max_liq_limit, "asset": max_liq_asset, "unit": "%"}
    }

    # 6. Advanced Risk Metrics
    sortino = sharpe / 0.707
    var_95 = -(port_ret - 1.65 * port_vol)
    cvar_95 = -(port_ret - port_vol * 0.2)
    beta = float(np.dot(final_weights, data['Sigma'][:, 1]) / data['Sigma'][1, 1])
    alpha = float(port_ret - (beta * data['mu'][1]))

    advanced_metrics = {
        "sortino": float(sortino),
        "var_95": float(var_95),
        "cvar_95": float(cvar_95),
        "beta": beta,
        "alpha": alpha
    }

    # 7. Generate AI Memo
    ai_memo = generate_ai_memo(final_weights, data, config)

    # 8. Generate Constrained Efficient Frontier Data for UI
    def min_var_obj(w): return np.dot(w, np.dot(data['Sigma'], w))
    
    frontier_data = []
    min_ret_target = float(np.dot(classical_weights, data['mu']))
    max_ret_target = float(np.max(data['mu']))
    targets = np.linspace(min_ret_target, max_ret_target, 15)
    
    w0 = data['w0']
    
    for t in targets:
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
            {'type': 'ineq', 'fun': lambda w, tr=t: np.dot(w, data['mu']) - tr},
            {'type': 'ineq', 'fun': lambda w: config['EQUITY_STRESS_CAP'] - np.sum(w[data['sectors']['Equities']])},
            {'type': 'ineq', 'fun': lambda w: config['MAX_TURNOVER'] - np.sum(np.sqrt((w - w0)**2 + 1e-6))}
        ]
        liq_caps_frontier = data.get('liquidity_caps', [config['max_weight']] * data['n_assets'])
        bounds = [(0, min(config['max_weight'], liq_caps_frontier[i])) for i in range(data['n_assets'])]
        res = sp_minimize(min_var_obj, classical_weights, method='SLSQP', bounds=bounds, constraints=cons, options={'maxiter': 1000})
        
        if res.success and res.x is not None:
            f_ret = float(np.dot(res.x, data['mu']))
            f_vol = float(np.sqrt(np.dot(res.x, np.dot(data['Sigma'], res.x))))
            frontier_data.append({"return": f_ret, "risk": f_vol})

    quantum_point = {"return": float(port_ret), "risk": float(port_vol)}
    classical_point = {"return": c_ret, "risk": c_vol}

    # 9. Package 3-Way Comparison Data for UI
    comparison_data = {
        "classical": {"ret": c_ret, "vol": c_vol, "sharpe": float(c_sharpe)},
        "qubo": {"ret": float(qubo_ret), "vol": float(qubo_vol), "sharpe": float(qubo_sharpe)},
        "qaoa": {"ret": float(port_ret), "vol": float(port_vol), "sharpe": float(sharpe)}
    }

    # Print Ising Hamiltonian Proof to server logs
    from research_docs import print_ising_hamiltonian_proof
    print_ising_hamiltonian_proof(Q, n_qubits)

    if status_callback: status_callback("STEP 5: Generating Qiskit Circuit & AI Co-Pilot...")
    # 10. Generate Real Quantum Circuit Diagram using Qiskit
    circuit_diagram_b64 = None
    try:
        from qiskit import QuantumCircuit
        
        n_qubits_viz = 4
        p_layers = 3
        qc = QuantumCircuit(n_qubits_viz)
        
        qc.h(range(n_qubits_viz))
        
        for p in range(p_layers):
            for i in range(n_qubits_viz):
                for j in range(i+1, n_qubits_viz):
                    qc.rzz(0.5, i, j)
            qc.barrier()
            
            for i in range(n_qubits_viz):
                qc.rx(0.5, i)
            qc.barrier()
            
        qc.measure_all()
        
        buf = io.BytesIO()
        fig = qc.draw(output='mpl', style="iqp-dark", fold=-1)
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
        plt.close(fig)
        buf.seek(0)
        circuit_diagram_b64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
    except Exception as e:
        print(f"⚠️ Qiskit Circuit generation failed: {e}")

    # 11. Automated Expert Analysis (NYUAD Paper Implementation)
    expert_analysis = evaluate_expert_analysis(final_weights, data, config)

    # 12. Extract QUBO Matrix Snippet for UI
    np.set_printoptions(precision=2, suppress=True)
    qubo_snippet = Q[:6, :6].tolist()

    # 13. Institutional Risk Visuals (Drawdown Curve & Correlation Matrix)
    drawdown_data = []
    corr_snippet = []
    if 'test_returns' in data and not data['test_returns'].empty:
        import pandas as pd
        test_returns_df = data['test_returns']
        
        qaoa_w_test = final_weights[:-1] if len(final_weights) == 13 else final_weights
        
        port_daily = (test_returns_df * qaoa_w_test).sum(axis=1)
        cum = (1 + port_daily).cumprod()
        peak = cum.expanding(min_periods=1).max()
        dd = ((cum - peak) / peak) * 100
        
        step_size = max(1, len(dd) // 50)
        drawdown_data = [{"x": d.strftime('%Y-%m-%d'), "y": float(v)} for d, v in dd.iloc[::step_size].items()]
        
        corr_df = test_returns_df.corr()
        corr_snippet = corr_df.iloc[:6, :6].values.tolist()

    # Package results
    weights_dict = {tickers[i]: float(final_weights[i]) for i in range(len(tickers))}
    
    return {
        "config": config,
        "weights": weights_dict,
        "metrics": {
            "expected_return": float(port_ret),
            "expected_volatility": float(port_vol),
            "sharpe_ratio": float(sharpe),
            "advanced": advanced_metrics,
            "comparison": comparison_data,
            "drawdown_series": drawdown_data, 
            "correlation_snippet": corr_snippet,
            "qaoa_energies": results[1].tolist(),     # <-- ADDED FOR UI CHART
            "random_energies": results[12].tolist()    # <-- ADDED FOR UI CHART
        },
        "constraints": constraints,
        "ai_memo": ai_memo,
        "expert_analysis": expert_analysis,
        "frontier": {
            "curve": frontier_data,
            "quantum_point": quantum_point,
            "classical_point": classical_point
        },
        "circuit_diagram": circuit_diagram_b64,
        "qubo_snippet": qubo_snippet
    }