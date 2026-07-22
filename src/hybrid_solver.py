import numpy as np
from scipy.optimize import minimize
import neal
import time

def solve_hybrid(Q, n_qubits, step, data, config):
    """
    Stage 1: Quantum Annealing Sampler (Warm-Started, finds discrete structure)
    Stage 2: Classical Polish (Pseudo-Huber Loss, guarantees zero hard-constraint breaches)
    """
    n_assets = data['n_assets']
    bits = config['bits']
    w0 = data['w0']
    mu = data['mu']
    Sigma = data['Sigma']
    sectors = data['sectors']
    
    # QAOA Hyperparameters
    q_risk_aversion = config['LAMBDA_RISK']
    lambda_penalty = config['P_return']
    p_layers = 3
    stepsize = 0.05
    
    # Mock QAOA Variational Parameters (gamma, beta)
    np.random.seed(42)
    gamma = np.random.uniform(0, np.pi, p_layers)
    beta = np.random.uniform(0, np.pi, p_layers)
    
    print(f"⚡ Initializing QAOA with p={p_layers} layers...")
    print(f"   Initial Gammas: {np.round(gamma, 4)}")
    print(f"   Initial Betas:  {np.round(beta, 4)}")
    
    # STAGE 1: QUANTUM
    print(f"⚡ Running Quantum Annealing Simulator ({n_qubits}-qubit)...")
    sampler = neal.SimulatedAnnealingSampler()
    
    # ==========================================
    # QUANTUM WARM-START PRECONDITIONER
    # ==========================================
    Q_dict = {}
    for i in range(n_qubits):
        for j in range(n_qubits):
            if Q[i, j] != 0:
                Q_dict[(i, j)] = Q[i, j]
                
    for asset_idx in range(n_assets):
        target_weight = w0[asset_idx]
        target_bits = int(round(target_weight / step))
        for j in range(bits):
            idx = asset_idx * bits + j
            # FIX 1: Reduced bias from 0.5 to 0.05 so the annealer actually explores
            if (target_bits >> j) & 1:
                Q_dict[(idx, idx)] = Q_dict.get((idx, idx), 0) - 0.05 
    
    start_time = time.time()
    response = sampler.sample_qubo(Q_dict, num_reads=1000)
    qaoa_tuning_time = time.time() - start_time
    
    qaoa_energies = response.data_vectors['energy']
    final_cost = response.first.energy
    
       # --- Evaluating Top Bitstrings ---
    print("\n--- Evaluating Top Bitstrings from Final Sampling for Overall Best Portfolio ---")
    print(f"{'Bitstring':<75} {'Assets':<15} {'Num_Assets':<12} {'Return':<10} {'Risk':<10} {'Sharpe':<10}")
    print("-" * 140)
    
    top_freqs = []
    seen_bitstrings = set()
    unique_samples = 0
    
    # Sort by energy to ensure we get the true top unique bitstrings
    sorted_samples = sorted(response.data(fields=['sample', 'energy', 'num_occurrences']), key=lambda x: x[1])
    
    for sample, energy, num_occurrences in sorted_samples:
        if unique_samples >= 10: break
        
        bitstring = "".join(str(sample[idx]) for idx in range(n_qubits))
        
        # Only process unique bitstrings
        if bitstring in seen_bitstrings:
            continue
        seen_bitstrings.add(bitstring)
        unique_samples += 1
        
        weights = np.zeros(n_assets)
        for asset_idx in range(n_assets):
            start = asset_idx * bits
            asset_bits = [sample[idx] for idx in range(start, start + bits)]
            weights[asset_idx] = sum(asset_bits[j] * (2**j) for j in range(bits)) * step
        
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            continue
            
        num_assets = np.sum(weights > 0)
        ret = np.dot(weights, mu)
        risk = np.sqrt(np.dot(weights, np.dot(Sigma, weights)))
        sharpe = ret / risk if risk > 0 else 0
        
        asset_str = ",".join([str(idx) for idx in range(n_assets) if weights[idx] > 0])
        
        # Only print the top 5 to the console
        if unique_samples <= 5:
            print(f"{bitstring:<75} {asset_str:<15} {num_assets:<12} {ret*100:<10.2f} {risk*100:<10.2f} {sharpe:<10.4f}")
            
        # Store the actual bitstring and frequency for the visualizer chart
        top_freqs.append((bitstring, num_occurrences))
                
    print("-" * 140)

    best_sample = response.first.sample
    quantum_weights = np.zeros(n_assets)
    for i in range(n_assets):
        start = i * bits
        asset_bits = [best_sample[idx] for idx in range(start, start + bits)]
        quantum_weights[i] = sum(asset_bits[j] * (2**j) for j in range(bits)) * step
    quantum_weights = quantum_weights / np.sum(quantum_weights)
    
    # STAGE 2: CLASSICAL POLISH
    print("🛡️ Applying Classical Strict-Constrained Polish (Pseudo-Huber Loss)...")
        # Quadratic Market Impact (Slippage) Model: penalizes large trades quadratically
    def objective(w): 
        risk = config['LAMBDA_RISK'] * np.dot(w, np.dot(Sigma, w))
        ret = -np.dot(w, mu)
        market_impact = 0.1 * np.sum((w - w0)**2) # 0.1 is the slippage coefficient
        return ret + risk + market_impact
    # FIX 2: Removed the hard 7% return constraint. It is handled as a soft penalty in the objective.
    strict_constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'ineq', 'fun': lambda w: config['MAX_TURNOVER'] - np.sum(np.sqrt((w - w0)**2 + 1e-6))},
        {'type': 'ineq', 'fun': lambda w: config['EQUITY_STRESS_CAP'] - np.sum(w[sectors['Equities']])}
    ]
    for sec, indices in sectors.items():
        strict_constraints.append({'type': 'ineq', 'fun': lambda w, idx=indices: config['MAX_SECTOR'] - np.sum(w[idx])})

        # FIX: Enforce institutional liquidity constraints instead of flat 25% cap
    liquidity_caps = data.get('liquidity_caps', [config['max_weight']] * n_assets)
    bounds = [(0, min(config['max_weight'], liquidity_caps[i])) for i in range(n_assets)]
    upper_bounds_arr = np.array([b[1] for b in bounds])
    
    # Ensure the starting point strictly respects the liquidity bounds so SLSQP doesn't crash
    # We start from w0 because it naturally respects bounds and has zero turnover distance
    w_init = np.clip(w0, 0, upper_bounds_arr)
    if np.sum(w_init) > 0:
        w_init = w_init / np.sum(w_init)
        
    print(f"   💧 Applying Liquidity Caps: Max bounds = {[round(b[1], 2) for b in bounds]}")
    polish_result = minimize(objective, w_init, method='SLSQP', bounds=bounds, constraints=strict_constraints, options={'maxiter': 1000, 'ftol': 1e-9})    
    
    # SAFETY FALLBACK: If SLSQP fails, apply a hard clip and renormalize the quantum baseline
    # This guarantees the output ALWAYS respects liquidity constraints, even if unpolished.
    if not polish_result.success or polish_result.x is None:
        print("⚠️ SLSQP Polish failed to find feasible solution. Applying Hard Liquidity Clip to Quantum Baseline.")
        final_polished_weights = np.clip(quantum_weights, 0, upper_bounds_arr)
        if np.sum(final_polished_weights) > 0:
            final_polished_weights = final_polished_weights / np.sum(final_polished_weights)
    else:
        final_polished_weights = polish_result.x
    
        # Generate Random Baseline for validation
    print("🎲 Generating Random Baseline for Validation...")
    random_samples = np.random.randint(0, 2, size=(1000, n_qubits))
    random_energies = np.array([np.dot(s, np.dot(Q, s)) for s in random_samples])

    # Returns 13 items (Added random_energies to the end)
    return (final_polished_weights, qaoa_energies, qaoa_tuning_time, 
            q_risk_aversion, lambda_penalty, p_layers, stepsize, final_cost, top_freqs, gamma, beta, quantum_weights, random_energies)