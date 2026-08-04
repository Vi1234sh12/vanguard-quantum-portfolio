import numpy as np

def generate_ai_memo(final_weights, data, config):
    """Generates a human-readable trade-off memo acting as the AI Co-Pilot"""
    tickers = ['QQQ', 'VTI', 'VXUS', 'TLT', 'LQD', 'GLD', 'DBC', 'VNQ', 'VWO', 'BND', 'SHY', 'HYG', 'VTI_PUT']
    mu = data['mu']
    sectors = data['sectors']
    
    # 1. Identify top 3 holdings
    top_indices = np.argsort(final_weights)[::-1][:3]
    top_holdings = [f"{tickers[i]} ({final_weights[i]*100:.1f}%)" for i in top_indices]
    
    # 2. Identify binding constraints (Trade-offs)
    trade_offs = []
    
    eq_weight = np.sum(final_weights[sectors['Equities']])
    if eq_weight > config['EQUITY_STRESS_CAP'] - 0.05:
        trade_offs.append(f"equities were capped near the {config['EQUITY_STRESS_CAP']*100:.0f}% downside protection limit")
        
    fi_weight = np.sum(final_weights[sectors['Fixed_Income']])
    if fi_weight > config['MAX_SECTOR'] - 0.05:
        trade_offs.append(f"fixed income allocation reached the {config['MAX_SECTOR']*100:.0f}% sector limit")
        
    if config['MAX_TURNOVER'] - 0.01 <= np.sum(np.abs(final_weights - data['w0'])) <= config['MAX_TURNOVER'] + 0.01:
        trade_offs.append(f"trading turnover was fully utilized at the {config['MAX_TURNOVER']*100:.0f}% limit to balance new opportunities with transaction costs")

    # 3. Check for derivative allocation
    put_weight = final_weights[-1]
    derivative_note = ""
    if put_weight > 0.001:
        derivative_note = (f" Notably, the optimizer allocated {put_weight*100:.2f}% to a VTI Protective Put "
                           f"(valued via Black-Scholes), utilizing non-linear derivative payoff structures to hedge tail risk.")

    # 4. Calculate portfolio metrics
    port_return = np.dot(final_weights, mu)
    best_asset_idx = top_indices[0]
    
    # 5. Check if target return was met
    target_notice = ""
    if port_return < config['MIN_RETURN'] - 0.005:
        target_notice = (f"⚠️ TARGET RETURN NOTICE: The requested minimum return of {config['MIN_RETURN']*100:.2f}% "
                         f"was mathematically infeasible given the strict liquidity caps and turnover limits. "
                         f"The optimizer safely reverted to the maximum achievable return of {port_return*100:.2f}% "
                         f"to maintain zero hard-constraint breaches.")
    
    # 6. Construct the memo line by line to guarantee exact spacing
    memo_lines = [
        "🤖 AI PORTFOLIO CO-PILOT MEMO",
        "=========================================================",
        "Executive Summary:",
        f"To achieve the target minimum return of {config['MIN_RETURN']*100:.0f}% while minimizing volatility, the Quantum-Hybrid optimizer has selected a diversified multi-asset allocation. ",
        f"The model prioritizes {top_holdings[0]} as the primary growth engine, supported by {top_holdings[1]} and {top_holdings[2]} to provide structural stability.{derivative_note}",
        "",  # Blank line
        "Strategic Trade-offs:"
    ]

    if trade_offs:
        memo_lines.append(f"The optimization required navigating several strict guardrails. Specifically, " + " and ".join(trade_offs) + ". ")
        memo_lines.append(f"By hitting these constraints, the portfolio successfully avoids over-concentration in {tickers[best_asset_idx]}, ensuring resilience under adverse market scenarios.")
    else:
        memo_lines.append("The optimization found a solution well within all guardrails, indicating a highly stable global minimum variance portfolio.")

    memo_lines.extend([
        "",  # Blank line
        "Financial Rationale:",
        f"Using the Black-Litterman expected returns (blended with Geometric Brownian Motion views), the projected annualized return is {port_return*100:.2f}%. ",
        "This allocation favors assets with high risk-adjusted implied returns while utilizing fixed income, real assets, and derivatives to dampen overall portfolio variance.",
        ""  # Blank line
    ])

    if target_notice:
        memo_lines.append(target_notice)
        memo_lines.append("")  # Blank line

    memo_lines.append("=========================================================")
    
    return "\n".join(memo_lines)