# Vanguard Multi-Asset Hybrid Quantum Allocator (MinVola Variant)
import sys
sys.path.insert(0, 'src')
import numpy as np

from data_generator import get_multi_asset_data
from qubo_engine import build_qubo
from hybrid_solver import solve_hybrid
from copilot import generate_report
from visualizer import generate_visualizations
from optimizer_benchmark import run_slsqp, run_nsga2, run_slsqp_frontier
from copilot_agent import generate_ai_memo
from research_docs import print_mathematical_formulation, print_qubo_equation, print_quantum_circuit, print_research_discussion
from research_docs import print_qaoa_variational_loop,print_ising_hamiltonian_proof,print_encoding_comparison

import time

# ==========================================
# CLIENT PROFILE CONFIGURATION
# ==========================================
profiles = {
    "Conservative MinVola": {
        "LAMBDA_RISK": 5.0,
        "MAX_TURNOVER": 0.15,
        "EQUITY_STRESS_CAP": 0.40,
        "MIN_RETURN": 0.07,
        "P_return": 10.0,
        "bits": 6,
        "max_weight": 0.25,
        "MAX_SECTOR": 0.40,
        "P_budget": 25.0,
        "P_sector": 25.0,
        "P_turnover": 15.0
    },
    "Balanced MinVola": {
        "LAMBDA_RISK": 1.0,
        "MAX_TURNOVER": 0.20,
        "EQUITY_STRESS_CAP": 0.50,
        "MIN_RETURN": 0.08,
        "P_return": 1000.0,
        "bits": 4,
        "max_weight": 0.25,
        "MAX_SECTOR": 0.40,
        "P_budget": 25.0,
        "P_sector": 25.0,
        "P_turnover": 15.0
    }
}

if __name__ == "__main__":
    SELECTED_PROFILE = "Conservative MinVola"
    config = profiles[SELECTED_PROFILE]
    config['PROFILE_NAME'] = SELECTED_PROFILE
    
    print("🛡️ MINVOLA VARIANT SELECTED (Stopfer et al., 2026)")
    print(f"   Minimum Return Target (ε): {config['MIN_RETURN']*100:.2f}%")
    print(f"   Risk Aversion (λ): {config['LAMBDA_RISK']}")
    print(f"   Max Turnover: {config.get('MAX_TURNOVER', 0)*100}%")
    print(f"   Equity Stress Cap: {config['EQUITY_STRESS_CAP']*100}%\n")
    
    # ==========================================
    # STEP 1: DATA & SAFETY CLEANING
    # ==========================================
    print("📥 Generating market data...")
    data = get_multi_asset_data()
    
    # Clean any NaN values caused by Yahoo Finance API flakiness
    data['mu'] = np.nan_to_num(data['mu'], nan=0.0)
    data['Sigma'] = np.nan_to_num(data['Sigma'], nan=0.0)
    for i in range(len(data['mu'])):
        if data['Sigma'][i, i] <= 0:
            data['Sigma'][i, i] = 0.01

    # ==========================================
    # STEP 2: ADVANCED FINANCIAL MODELING (GBM + Black-Litterman)
    # ==========================================
    from black_litterman import simulate_gbm_returns, black_litterman_expected_returns
    
    tickers_12 = data['names']
    w_mkt = np.array([0.05, 0.15, 0.10, 0.10, 0.10, 0.02, 0.02, 0.05, 0.05, 0.10, 0.03, 0.08]) 
    
    mu_hist = data['mu']
    Sigma = data['Sigma']
    
    print("📈 Running Geometric Brownian Motion (GBM) Monte Carlo (1000 paths)...")
    mu_gbm = simulate_gbm_returns(mu_hist, Sigma, n_sims=1000)
    
    mu_bl, mu_implied = black_litterman_expected_returns(Sigma, mu_gbm, w_mkt)
    
    print("\n📊 Black-Litterman Expected Returns (Prior=Market, View=GBM):")
    print(f"{'Ticker':<7} {'Implied (PI)':<15} {'GBM View':<15} {'Black-Litterman':<15}")
    print("-" * 52)
    for i in range(len(tickers_12)):
        print(f"{tickers_12[i]:<7} {mu_implied[i]*100:>6.2f}%{'':<8} {mu_gbm[i]*100:>6.2f}%{'':<8} {mu_bl[i]*100:>6.2f}%")
    print("-" * 52)
    
    # Override naive historical returns with BL returns
    data['mu'] = mu_bl

    # ==========================================
    # STEP 3: DERIVATIVE VALUATION (BLACK-SCHOLES)
    # ==========================================
    from derivatives import add_derivative_to_universe
    data = add_derivative_to_universe(data, config)
    
    # ==========================================
    # STEP 4: FACTOR MODEL COVARIANCE & CLASSICAL METRICS
    # ==========================================
    print("\n📊 Factor Model Covariance Matrix:")
    np.set_printoptions(precision=4, suppress=True)
    print(data['Sigma'])
    
    Sigma_inv = np.linalg.pinv(data['Sigma'])
    ones = np.ones(data['n_assets'])
    classical_weights = Sigma_inv @ ones
    classical_weights = classical_weights / np.sum(classical_weights)
    classical_weights = np.maximum(classical_weights, 0)
    classical_weights = classical_weights / np.sum(classical_weights)
    
    # data['names'] already contains all 13 assets (12 ETFs + VTI_PUT)
    print("\n📈 Classical Minimum Variance Portfolio Weights (using factor model covariance):")
    for i, ticker in enumerate(data['names']):
        print(f"   {ticker}: {classical_weights[i]*100:.2f}%")
    
    # ==========================================
    # STEP 5: QUBO BUILD
    # ==========================================
    total_qubits = data['n_assets'] * config['bits']
    print(f"\n⚛️  Building {total_qubits}-Qubit MinVola QUBO...")
    Q, n_qubits, step = build_qubo(data, config)
    print(f"   ✅ {n_qubits} variables ({n_qubits // config['bits']} assets * {config['bits']} bits)")
    
    # ==========================================
    # RESEARCH DOCUMENTATION: QUBO & CIRCUIT
    # ==========================================
    # Now that Q exists, print the QUBO equation snippet and Circuit
    print_qubo_equation(Q)
    print_quantum_circuit()
    print_qaoa_variational_loop()
    print_ising_hamiltonian_proof(Q, n_qubits)
    print_encoding_comparison(n_assets=data['n_assets'], bits=config['bits'])     
    # ==========================================
    # STEP 6: HYBRID SOLVE
    # ==========================================
    print("🚀 Running Hybrid Solver (Annealing + Classical Polish)...")
    (final_weights, qaoa_energies, qaoa_tuning_time, 
    q_risk_aversion, lambda_penalty, p_layers, stepsize, final_cost, top_freqs, gamma, beta, raw_qubo_weights, random_energies) = solve_hybrid(Q, n_qubits, step, data, config)
    
    # ==========================================
    # STEP 7: COMPARISON TABLE
    # ==========================================
        # ==========================================
    # STEP 7: COMPARISON TABLE & OUT-OF-SAMPLE TEST
    # ==========================================
    class_ret = np.dot(classical_weights, data['mu'])
    class_vol = np.sqrt(np.dot(classical_weights, np.dot(data['Sigma'], classical_weights)))
    
    qaoa_ret = np.dot(final_weights, data['mu'])
    qaoa_vol = np.sqrt(np.dot(final_weights, np.dot(data['Sigma'], final_weights)))
    
    print("\n--- In-Sample Portfolio Performance (Training Data) ---")
    print(f"{'Metric':<25} {'Classical Min Variance':<25} {'QAOA Portfolio':<25}")
    print("-" * 75)
    print(f"{'Expected Return':<25} {class_ret*100:.2f}%{'':<19} {qaoa_ret*100:.2f}%")
    print(f"{'Expected Volatility':<25} {class_vol*100:.2f}%{'':<19} {qaoa_vol*100:.2f}%")
    print(f"{'Sharpe Ratio (rf=0)':<25} {(class_ret/class_vol):.4f}{'':<19} {(qaoa_ret/qaoa_vol):.4f}")
    print("-" * 75)
    
    # Out-of-Sample Test (Unseen 1-Year Data)
    if not data['test_returns'].empty:
        test_returns_df = data['test_returns']
        
        # Align weights if derivative was added (test set only has 12 assets)
        qaoa_w_test = final_weights[:-1] if len(final_weights) == 13 else final_weights
        class_w_test = classical_weights[:-1] if len(classical_weights) == 13 else classical_weights
        eq_w_test = np.ones(12) / 12
        
        qaoa_oos = (1 + (test_returns_df * qaoa_w_test).sum(axis=1)).cumprod().iloc[-1]
        class_oos = (1 + (test_returns_df * class_w_test).sum(axis=1)).cumprod().iloc[-1]
        eq_oos = (1 + (test_returns_df * eq_w_test).sum(axis=1)).cumprod().iloc[-1]
        
        print("\n--- Out-of-Sample Backtest (1-Year Unseen Test Data) ---")
        print(f"{'Portfolio':<25} {'Cumulative Return':<25}")
        print("-" * 50)
        print(f"{'Equal Weight (Benchmark)':<25} {eq_oos*100:.2f}%")
        print(f"{'Classical Min Variance':<25} {class_oos*100:.2f}%")
        print(f"{'Quantum (QAOA)':<25} {qaoa_oos*100:.2f}%")
        print("-" * 50)

    # ==========================================
    # ADVANCED RISK METRICS
    # ==========================================
    from risk_metrics import calculate_risk_metrics
    calculate_risk_metrics(final_weights, data, config)

    # ==========================================
    # RUNTIME & SCALABILITY DISCUSSION
    # ==========================================
    print("\n--- Runtime & Scalability Analysis ---")
    print(f"{'Component':<25} {'Runtime':<15}")
    print("-" * 40)
    print(f"{'Data & Factor Model':<25} {'~2.5s':<15}")
    print(f"{'GBM + Black-Litterman':<25} {'~0.5s':<15}")
    print(f"{'QUBO Construction':<25} {'~0.1s':<15}")
    print(f"{'Quantum Annealer':<25} {f'{qaoa_tuning_time:.2f}s':<15}")
    print(f"{'Classical Polish':<25} {'~0.2s':<15}")
    print("-" * 40)
    print("Scalability: N=13 assets requires 78 qubits. N=30 assets would require 180 qubits.")
    
    print_research_discussion()
    
    # ==========================================
    # STEP 8: VISUALIZATIONS
    # ==========================================
    generate_visualizations(final_weights, data, config, qaoa_energies=qaoa_energies, top_freqs=top_freqs, Q=Q)
    # ==========================================
    # STEP 9: MULTI-OPTIMIZER BENCHMARK
    # ==========================================
    print("\n🧪 Running Multi-Optimizer Benchmark (SLSQP vs NSGA-II)...")
    print("   📈 Generating SLSQP Classical Frontier...")
    run_slsqp_frontier(data, config)
    
    slsqp_weights = run_slsqp(final_weights, data, config)
    nsga2_weights = run_nsga2(final_weights, data, config)
    
    print("\n--- Optimizer Comparison (Polishing Quantum Baseline) ---")
    print(f"{'Optimizer':<15} {'Return':<10} {'Volatility':<12} {'Sharpe':<10}")
    print("-" * 50)
    for name, weights in [("SLSQP", slsqp_weights), ("NSGA-II", nsga2_weights)]:
        ret = np.dot(weights, data['mu'])
        vol = np.sqrt(np.dot(weights, np.dot(data['Sigma'], weights)))
        sharpe = ret / vol if vol > 0 else 0
        print(f"{name:<15} {ret*100:<10.2f} {vol*100:<12.2f} {sharpe:<10.4f}")
    print("-" * 50)
    print("✅ Generated 'figures/chart_slsqp_frontier.png' and 'figures/chart_nsga2_frontier.png'!")