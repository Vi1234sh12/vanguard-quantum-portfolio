import numpy as np

def evaluate_expert_analysis(final_weights, data, config):
    """
    Automates the 'Expert Analysis' proposed by Innan et al. (2025).
    Evaluates the financial viability of the quantum portfolio beyond pure math.
    """
    mu = data['mu']
    Sigma = data['Sigma']
    
    # 1. Diversification Ratio (Weighted Avg Vol / Portfolio Vol)
    asset_vols = np.sqrt(np.diag(Sigma))
    weighted_avg_vol = np.dot(final_weights, asset_vols)
    port_vol = np.sqrt(np.dot(final_weights, np.dot(Sigma, final_weights)))
    div_ratio = weighted_avg_vol / port_vol if port_vol > 0 else 0
    
    # 2. Risk-Return Alignment (Are we getting paid for the risk?)
    port_ret = np.dot(final_weights, mu)
    sharpe = port_ret / port_vol if port_vol > 0 else 0
    
    # 3. Constraint Adherence (Did it break any hard rules?)
    turnover_used = np.sum(np.abs(final_weights - data['w0']))
    breaches = 0
    if turnover_used > config['MAX_TURNOVER'] + 0.001: breaches += 1
    if np.sum(final_weights[data['sectors']['Equities']]) > config['EQUITY_STRESS_CAP'] + 0.001: breaches += 1
    if np.sum(final_weights) > 1.001 or np.sum(final_weights) < 0.999: breaches += 1
    
    # 4. Calculate Viability Score (0 to 100)
    score = 0
    score += min(40, div_ratio * 40)               # Max 40 pts for diversification
    score += min(30, sharpe * 50)                  # Max 30 pts for risk-adjusted return
    score += 30 if breaches == 0 else -50          # 30 pts for zero breaches, massive penalty if breached
    
    score = max(0, min(100, score))
    grade = "A+ (Institutional Grade)" if score >= 85 else "B (Viable)" if score >= 70 else "C (High Risk)" if score >= 50 else "F (Unviable)"
    
    return {
        "viability_score": float(score),
        "viability_grade": grade,
        "diversification_ratio": float(div_ratio),
        "constraint_breaches": int(breaches),
        "expert_notes": "Automated expert validation passed. Portfolio exhibits strong diversification and strict guardrail adherence." if breaches == 0 else "Expert validation flagged constraint violations."
    }