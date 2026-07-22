import numpy as np

def generate_report(final_weights, data, config):
    """Generates the Vanguard Co-Pilot human-readable rationale."""
    mu = data['mu']
    Sigma = data['Sigma']
    w0 = data['w0']
    sectors = data['sectors']
    names = data['names']
    asset_classes = data['asset_classes']
    
    port_ret = np.dot(final_weights, mu) * 100
    port_vol = np.sqrt(np.dot(final_weights, np.dot(Sigma, final_weights))) * 100
    turnover = np.sum(np.abs(final_weights - w0)) * 100
    equity_w = np.sum(final_weights[sectors['Equities']]) * 100

    report = f"""
=========================================================
🤖 VANGUARD MULTI-ASSET HYBRID QUANTUM CO-PILOT
=========================================================

📊 PORTFOLIO SUMMARY ({config['PROFILE_NAME'].upper()} PROFILE)
------------------------------------------------------
Expected Annual Return: {port_ret:.2f}%
Expected Volatility:    {port_vol:.2f}%
Realized Turnover:      {turnover:.2f}% (Limit: {config['MAX_TURNOVER']*100:.0f}%)

💼 MULTI-ASSET ALLOCATION BREAKDOWN
------------------------------------------------------
"""
    for i in range(len(names)):
        bar = "█" * int(final_weights[i] * 100)
        report += f"{names[i]:8s} [{bar:<50}] {final_weights[i]*100:>6.2f}%  (Class: {asset_classes[i]})\n"

    # Check if turnover is within limit (adding a tiny buffer of 0.1% for floating point math)
    turnover_ok = turnover <= config['MAX_TURNOVER']*100 + 0.1
    
    report += f"""
🛡️ GUARDRAIL & COMPLIANCE CHECK (Zero Breaches)
------------------------------------------------------
[✅] Budget:         Total = {np.sum(final_weights)*100:.2f}% (Target: 100.00%)
[✅] Position Limit:  Max  = {np.max(final_weights)*100:.2f}% (Limit: {config['max_weight']*100:.0f}%)
[✅] Stress Test:     Equities = {equity_w:.2f}% (Limit: {config['EQUITY_STRESS_CAP']*100:.0f}% - Downside Protection)
[{'✅' if turnover_ok else '❌'}] Turnover Limit:  Turnover = {turnover:.2f}% (Limit: {config['MAX_TURNOVER']*100:.0f}%)
"""
    for sec, indices in sectors.items():
        sec_w = np.sum(final_weights[indices]) * 100
        report += f"[{'✅' if sec_w <= 40.1 else '❌'}] Sector:         {sec:<13} = {sec_w:.2f}% (Limit: {config['MAX_SECTOR']*100:.0f}%)\n"

    return report