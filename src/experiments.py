import sys
sys.path.insert(0, 'src')
import numpy as np
import time
from data_generator import get_multi_asset_data
from black_litterman import simulate_gbm_returns, black_litterman_expected_returns
from derivatives import add_derivative_to_universe
from qubo_engine import build_qubo
from hybrid_solver import solve_hybrid
import copy


# Baseline configuration for all experiments
BASE_CONFIG = {
    "LAMBDA_RISK": 5.0, "MAX_TURNOVER": 0.15, "EQUITY_STRESS_CAP": 0.40,
    "MIN_RETURN": 0.07, "P_return": 10.0, "bits": 6, "max_weight": 0.25,
    "MAX_SECTOR": 0.40, "P_budget": 25.0, "P_sector": 25.0, "P_turnover": 15.0,
    "PROFILE_NAME": "Experiment"
}

def evaluate(weights, data):
    ret = np.dot(weights, data['mu'])
    vol = np.sqrt(np.dot(weights, np.dot(data['Sigma'], weights)))
    sharpe = ret / vol if vol > 0 else 0
    return ret, vol, sharpe

def run_pipeline(data, config):
    Q, n_qubits, step = build_qubo(data, config)
    # We only need the final weights for the ablation, so we unpack the first element of the tuple
    results = solve_hybrid(Q, n_qubits, step, data, config)
    return results[0]

import copy

def ablation_study():
    print("\n" + "="*75)
    print("🔬 ABLATION STUDY: Isolating Component Contributions")
    print("="*75)
    
    print("📥 Fetching base data...")
    data_base = get_multi_asset_data()
    data_base['mu'] = np.nan_to_num(data_base['mu'], nan=0.0)
    data_base['Sigma'] = np.nan_to_num(data_base['Sigma'], nan=0.0)
    for i in range(len(data_base['mu'])):
        if data_base['Sigma'][i, i] <= 0: data_base['Sigma'][i, i] = 0.01

    experiments = {
        "Full Model (BL+GBM+Put)": None,
        "No Black-Litterman (Hist Mean)": None,
        "No Derivative (No Put)": None
    }

    # 1. Full Model
    print("\nRunning: Full Model...")
    data_full = copy.deepcopy(data_base)
    mu_gbm = simulate_gbm_returns(data_full['mu'], data_full['Sigma'], n_sims=500)
    mu_bl, _ = black_litterman_expected_returns(data_full['Sigma'], mu_gbm, np.array([0.05, 0.15, 0.10, 0.10, 0.10, 0.02, 0.02, 0.05, 0.05, 0.10, 0.03, 0.08]))
    data_full['mu'] = mu_bl
    data_full = add_derivative_to_universe(data_full, BASE_CONFIG)
    experiments["Full Model (BL+GBM+Put)"] = run_pipeline(data_full, BASE_CONFIG)
    
    # 2. No Black-Litterman (Use Historical Mean, but keep Put)
    print("Running: No Black-Litterman...")
    data_nobl = copy.deepcopy(data_base)
    data_nobl = add_derivative_to_universe(data_nobl, BASE_CONFIG)
    experiments["No Black-Litterman (Hist Mean)"] = run_pipeline(data_nobl, BASE_CONFIG)

    # 3. No Derivative (Use BL, but no Put)
    print("Running: No Derivative...")
    data_noput = copy.deepcopy(data_base)
    mu_gbm = simulate_gbm_returns(data_noput['mu'], data_noput['Sigma'], n_sims=500)
    mu_bl, _ = black_litterman_expected_returns(data_noput['Sigma'], mu_gbm, np.array([0.05, 0.15, 0.10, 0.10, 0.10, 0.02, 0.02, 0.05, 0.05, 0.10, 0.03, 0.08]))
    data_noput['mu'] = mu_bl
    experiments["No Derivative (No Put)"] = run_pipeline(data_noput, BASE_CONFIG)

    # Output Table
        # Output Table
    print("\n--- Ablation Results (Target Return = 7%) ---")
    print(f"{'Configuration':<35} {'Return':<10} {'Volatility':<12} {'Sharpe':<10}")
    print("-" * 67)
    for name, weights in experiments.items():
        if "No Derivative" in name:
            ret, vol, sharpe = evaluate(weights, data_noput)
        elif "No Black-Litterman" in name:
            ret, vol, sharpe = evaluate(weights, data_nobl)
        else:
            ret, vol, sharpe = evaluate(weights, data_full)
        print(f"{name:<35} {ret*100:<10.2f} {vol*100:<12.2f} {sharpe:<10.4f}")
    print("-" * 67)


def sensitivity_analysis():
    print("\n" + "="*75)
    print("📈 SENSITIVITY ANALYSIS: Impact of Risk Aversion (λ)")
    print("="*75)
    
    print("📥 Fetching base data...")
    data = get_multi_asset_data()
    data['mu'] = np.nan_to_num(data['mu'], nan=0.0)
    data['Sigma'] = np.nan_to_num(data['Sigma'], nan=0.0)
    for i in range(len(data['mu'])):
        if data['Sigma'][i, i] <= 0: data['Sigma'][i, i] = 0.01

    mu_gbm = simulate_gbm_returns(data['mu'], data['Sigma'], n_sims=500)
    mu_bl, _ = black_litterman_expected_returns(data['Sigma'], mu_gbm, np.array([0.05, 0.15, 0.10, 0.10, 0.10, 0.02, 0.02, 0.05, 0.05, 0.10, 0.03, 0.08]))
    data['mu'] = mu_bl
    data = add_derivative_to_universe(data, BASE_CONFIG)

    lambdas = [1.0, 3.0, 5.0, 10.0]
    print("\n--- Sensitivity Results (No Hard Return Constraint) ---")
    print(f"{'Risk Aversion (λ)':<20} {'Return':<10} {'Volatility':<12} {'Sharpe':<10}")
    print("-" * 52)
    
    for lam in lambdas:
        config = BASE_CONFIG.copy()
        config['LAMBDA_RISK'] = lam
        # Remove the hard return constraint so λ can act as an economic dial
        config['MIN_RETURN'] = 0.0 
        
        print(f"Running λ = {lam}...")
        weights = run_pipeline(data, config)
        ret, vol, sharpe = evaluate(weights, data)
        print(f"\r{lam:<20.1f} {ret*100:<10.2f} {vol*100:<12.2f} {sharpe:<10.4f}")
    print("-" * 52)

if __name__ == "__main__":
    ablation_study()
    sensitivity_analysis()