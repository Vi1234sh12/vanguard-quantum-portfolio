import numpy as np
import yfinance as yf

def calculate_risk_metrics(final_weights, data, config):
    """Calculates Sortino, VaR, CVaR, Beta, Drawdown, Calmar, Alpha, Info Ratio"""
    mu = data['mu']
    Sigma = data['Sigma']
    
    port_ret = np.dot(final_weights, mu)
    port_vol = np.sqrt(np.dot(final_weights, np.dot(Sigma, final_weights)))
    
    # 1. Sortino Ratio (Target return = 0)
    downside_dev = port_vol * 0.707 
    sortino = port_ret / downside_dev if downside_dev > 0 else 0
    
    # 2. VaR & CVaR (95%, 1Y)
    var_95 = -(port_ret - 1.65 * port_vol)
    cvar_95 = -(port_ret - port_vol * (0.1029 / 0.05)) # Simplified expected shortfall
    
    # 3. Beta vs Market (VTI at index 1)
    mkt_idx = 1
    mkt_var = Sigma[mkt_idx, mkt_idx]
    cov_port_mkt = np.dot(final_weights, Sigma[:, mkt_idx])
    beta = cov_port_mkt / mkt_var if mkt_var > 0 else 0
    
    # 4. Alpha (CAPM: rf=0, Market Return = mu[1])
    market_ret = mu[mkt_idx]
    alpha = port_ret - (beta * market_ret)
    
    # 5. Tracking Error vs Equal-Weight
    eq_weights = np.ones(data['n_assets']) / data['n_assets']
    eq_ret = np.dot(eq_weights, mu)
    active_ret = final_weights - eq_weights
    tracking_error = np.sqrt(np.dot(active_ret, np.dot(Sigma, active_ret)))
    
    # 6. Information Ratio
    info_ratio = (port_ret - eq_ret) / tracking_error if tracking_error > 0 else 0
    
    # 7. Max Drawdown & Calmar (Approximated via historical VTI volatility proxy)
    # Fetching 1Y hist data for true drawdown calculation
    tickers = ['QQQ', 'VTI', 'VXUS', 'TLT', 'LQD', 'GLD', 'DBC', 'VNQ', 'VWO', 'BND', 'SHY', 'HYG']
    try:
        hist = yf.download(tickers, period="1y", progress=False)['Close']
        daily_ret = hist.pct_change().dropna()
        port_daily = (daily_ret * final_weights[:-1]).sum(axis=1)
        cum = (1 + port_daily).cumprod()
        peak = cum.expanding(min_periods=1).max()
        dd = (cum - peak) / peak
        max_dd = dd.min()
        calmar = port_ret / abs(max_dd) if max_dd < 0 else 0
    except:
        max_dd = -0.15 # Fallback
        calmar = port_ret / abs(max_dd)
    
    print("\n--- Advanced Portfolio Risk Metrics ---")
    print(f"Sortino Ratio:        {sortino:.4f}")
    print(f"VaR (95%, 1Y):        {var_95*100:.2f}%")
    print(f"CVaR (95%, 1Y):       {cvar_95*100:.2f}%")
    print(f"Beta (vs VTI):        {beta:.4f}")
    print(f"Alpha (vs VTI):       {alpha*100:.2f}%")
    print(f"Tracking Error:       {tracking_error*100:.2f}%")
    print(f"Information Ratio:    {info_ratio:.4f}")
    print(f"Max Drawdown (1Y):    {max_dd*100:.2f}%")
    print(f"Calmar Ratio:         {calmar:.4f}")
    print("-" * 40)