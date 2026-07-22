import numpy as np

def build_qubo(data, config):
    """
    QUBO Builder targeting the MinVola (Minimize Volatility) variant.
    Based on the formulation from Stopfer et al. (2026) Section 4.
    
    Objective: Minimize σ²(ω)
    Subject to: μ'(ω) ≥ ε
               Σ ω_i = 1
               0 ≤ ω_i ≤ u_i
    """
    mu = data['mu']
    Sigma = data['Sigma']
    w0 = data['w0']
    sectors = data['sectors']
    n_assets = data['n_assets']
    bits = config['bits']
    max_weight = config['max_weight']
    step = max_weight / (2**bits - 1)
    n_vars = n_assets * bits
    
    Q = np.zeros((n_vars, n_vars))
    
    LAMBDA_RISK = config['LAMBDA_RISK']
    P_return = config['P_return']      # Penalty for μ'w ≥ ε
    P_budget = config['P_budget']
    P_sector = config['P_sector']
    P_turnover = config['P_turnover']
    MIN_RETURN = config['MIN_RETURN']      # The ε parameter (e.g., 0.07 for 7% minimum return)
    MAX_SECTOR = config['MAX_SECTOR']
    
    bit_values = np.array([2**j for j in range(bits)])
    
    # ==========================================
    # 1. MINIMIZE VOLATILITY (Quadratic Term)
    # σ²(ω) = Σ_i Σ_j w_i * σ_ij * w_j
    # Expanded for binary variables:
    # step² * Σ_i Σ_k Σ_j Σ_l 2^j * 2^l * σ_ik * x_{i,j} * x_{k,l}
    # ==========================================
    for i in range(n_assets):
        for k in range(n_assets):
            for j in range(bits):
                for l in range(bits):
                    idx_ij = i * bits + j
                    idx_kl = k * bits + l
                    
                    coeff = LAMBDA_RISK * (step**2) * bit_values[j] * bit_values[l] * Sigma[i, k]
                    
                    if idx_ij == idx_kl:
                        Q[idx_ij, idx_kl] += coeff
                    else:
                        Q[idx_ij, idx_kl] += coeff / 2
                        Q[idx_kl, idx_ij] += coeff / 2
    
    # ==========================================
    # 2. MINIMUM RETURN PENALTY (Stopfer Section 4)
    # φ * (μ'w - ε)² 
    # Expanded: φ * [ (step * Σ_j 2^j x_{i,j} - ε ]²
    # ==========================================
    for i in range(n_assets):
        for j in range(bits):
            idx = i * bits + j
            # (step * Σ_j 2^j x_{i,j})² = step² * Σ_j Σ_l 2^j 2^l x_{i,j} x_{i,l}
            for l in range(bits):
                idx_il = i * bits + l
                coeff = P_return * (step**2) * bit_values[j] * bit_values[l]
                if idx == idx_il:
                    Q[idx_il, idx_il] += coeff
                else:
                    Q[idx_il, idx_lj] += coeff / 2
                    Q[idx_lj, idx_il] += coeff / 2
            
            # -2 * ε * (step * Σ_j 2^j x_{i,j})
            Q[idx, idx] -= 2 * P_return * MIN_RETURN * step * bit_values[j]
    
    # ==========================================
    # 3. BUDGET PENALTY: (Σw_i - 1)²
    # ==========================================
    for i in range(n_assets):
        for k in range(n_assets):
            for j in range(bits):
                for l in range(bits):
                    idx_ij = i * bits + j
                    idx_kl = k * bits + l
                    
                    coeff = P_budget * (step**2) * bit_values[j] * bit_values[l]
                    
                    if idx_ij == idx_kl:
                        Q[idx_ij, idx_kl] += coeff
                    else:
                        Q[idx_ij, idx_kl] += coeff / 2
                        Q[idx_kl, idx_ij] += coeff / 2
        
        for j in range(bits):
            idx = i * bits + j
            Q[idx, idx] -= 2 * P_budget * step * bit_values[j]
    
    # ==========================================
    # 4. TURNOVER PENALTY (L2 approximation)
    # TC = Σ_i (w_i - w_i^0)²
    # ==========================================
    for i in range(n_assets):
        for j in range(bits):
            for l in range(bits):
                idx_ij = i * bits + j
                idx_il = i * bits + l
                
                coeff = P_turnover * (step**2) * bit_values[j] * bit_values[l]
                if idx_ij == idx_il:
                    Q[idx_ij, idx_il] += coeff
                else:
                    Q[idx_ij, idx_il] += coeff/2
                    Q[idx_il, idx_ij] += coeff/2
                    
        for j in range(bits):
            idx = i * bits + j
            Q[idx, idx] -= 2 * P_turnover * w0[i] * step * bit_values[j]
    
    # ==========================================
    # 5. SECTOR PENALTIES
    # ==========================================
    for sec, indices in sectors.items():
        for i in indices:
            for k in indices:
                for j in range(bits):
                    for l in range(bits):
                        idx_ij = i * bits + j
                        idx_kl = k * bits + l
                        
                        coeff = P_sector * (step**2) * bit_values[j] * bit_values[l]
                        if idx_ij == idx_kl:
                            Q[idx_ij, idx_kl] += coeff
                        else:
                            Q[idx_ij, idx_kl] += coeff / 2
                            Q[idx_kl, idx_ij] += coeff / 2
            
        for i in indices:
            for j in range(bits):
                idx = i * bits + j
                Q[idx, idx] -= 2 * P_sector * MAX_SECTOR * step * bit_values[j]
    
    return Q
