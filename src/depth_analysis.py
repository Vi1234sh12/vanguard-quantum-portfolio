import sys
sys.path.insert(0, 'src')
import numpy as np
from data_generator import get_multi_asset_data
from black_litterman import simulate_gbm_returns, black_litterman_expected_returns
from derivatives import add_derivative_to_universe
from qubo_engine import build_qubo
from hybrid_solver import solve_hybrid
from config import BASE_CONFIG

def analyze_depth_sensitivity():
    print("\n" + "="*50)
    print("🔬 QAOA DEPTH SENSITIVITY ANALYSIS")
    print("="*50)
    
    data = get_multi_asset_data()
    # ... (add standard BL and Derivative prep here) ...
    
    depths = [1, 2, 3, 4, 5]
    print(f"{'Depth (p)':<10} {'Return':<10} {'Volatility':<12} {'Sharpe':<10}")
    print("-" * 42)
    
    for p in depths:
        config = BASE_CONFIG.copy()
        # Note: You will need to pass 'p' to your hybrid_solver to actually change the QAOA layers.
        # For now, this demonstrates the research output format.
        # results = solve_hybrid(Q, n_qubits, step, data, config) 
        # mock output for demonstration:
        print(f"{p:<10} {'7.20%':<10} {'10.15%':<12} {'0.7094':<10}")

if __name__ == "__main__":
    analyze_depth_sensitivity()