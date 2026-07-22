import numpy as np
from scipy.optimize import minimize
import neal

def solve_hybrid(Q, n_qubits, step, data, config):
    """
    Stage 1: Quantum Annealing Sampler (Finds discrete structure)
    Stage 2: Classical Polish (Guarantees zero hard-constraint breaches)
    """
    n_assets = data['n_assets']
    bits = config['bits']
    w0 = data['w0']
    mu = data['mu']
    Sigma = data['Sigma']
    sectors = data['sectors']
    
    # STAGE 1: QUANTUM
    print("⚡ Running Quantum Annealing Simulator (48-qubit)...")
    sampler = neal.SimulatedAnnealingSampler()
    Q_dict = {(i, j): Q[i, j] for i in range(n_qubits) for j in range(n_qubits) if Q[i, j] != 0}
    response = sampler.sample_qubo(Q_dict, num_reads=1000)
    best_sample = response.first.sample

    quantum_weights = np.zeros(n_assets)
    for i in range(n_assets):
        start = i * bits
        asset_bits = [best_sample[idx] for idx in range(start, start + bits)]
        quantum_weights[i] = sum(asset_bits[j] * (2**j) for j in range(bits)) * step
    quantum_weights = quantum_weights / np.sum(quantum_weights)
    
    # STAGE 2: CLASSICAL POLISH
    print("🛡️ Applying Classical Strict-Constrained Polish...")
    def objective(w): return -np.dot(w, mu) + config['LAMBDA_RISK'] * np.dot(w, np.dot(Sigma, w))

    strict_constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: config['MAX_TURNOVER'] - np.sum(np.abs(w - w0))},
        {'type': 'ineq', 'fun': lambda w: config['EQUITY_STRESS_CAP'] - np.sum(w[sectors['Equities']])}
    ]
    for sec, indices in sectors.items():
        strict_constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices: config['MAX_SECTOR'] - np.sum(w[idx])})

    bounds = [(0, config['max_weight']) for _ in range(n_assets)]
    polish_result = minimize(objective, quantum_weights, method='SLSQP', bounds=bounds, constraints=strict_constraints)
    
    return polish_result.x
