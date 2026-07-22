# Vanguard Quantum Portfolio - Main Execution Pipeline
import sys
sys.path.insert(0, 'src')

from data_generator import get_multi_asset_data
from qubo_engine import build_qubo
from hybrid_solver import solve_hybrid
from copilot import generate_report

# 1. Define Client Profile
config = {
    "PROFILE_NAME": "Balanced",
    "LAMBDA_RISK": 2.0,
    "MAX_TURNOVER": 0.20,
    "EQUITY_STRESS_CAP": 0.50,
    "bits": 4,
    "max_weight": 0.25,
    "MAX_SECTOR": 0.40,
    "P_budget": 25.0,
    "P_sector": 25.0,
    "P_turnover": 15.0
}

if __name__ == "__main__":
    print("🚀 STARTING VANGUARD HYBRID QUANTUM PIPELINE...")
    
    # Step 1: Data
    data = get_multi_asset_data()
    
    # Step 2: QUBO
    Q, n_qubits, step = build_qubo(data, config)
    print(f"✅ Built {n_qubits}-Qubit QUBO Matrix.")
    
    # Step 3: Solve
    final_weights = solve_hybrid(Q, n_qubits, step, data, config)
    
    # Step 4: Report
    report = generate_report(final_weights, data, config)
    print(report)
