import scipy.stats as si
import numpy as np

def black_scholes_put(S, K, T, r, sigma):
    """
    Calculates the price of a European Put Option using the Black-Scholes formula.
    S: Spot price
    K: Strike price
    T: Time to maturity (in years)
    r: Risk-free rate
    sigma: Volatility of the underlying asset
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = (K * np.exp(-r * T) * si.norm.cdf(-d2) - S * si.norm.cdf(-d1))
    return put_price

def add_derivative_to_universe(data, config):
    """
    Appends a protective put option on the broad market (VTI) to the asset universe.
    """
    n_assets = data['n_assets']
    mu = data['mu']
    Sigma = data['Sigma']
    w0 = data['w0']
    sectors = data['sectors']
    
    # Fetch existing names and asset classes, defaulting if they don't exist
    names = data.get('names', [f'Asset_{i+1}' for i in range(n_assets)])
    asset_classes = data.get('asset_classes', ['ETF'] * n_assets)
    
    # Assume VTI is index 1
    vti_idx = 1 
    vti_vol = np.sqrt(Sigma[vti_idx, vti_idx])
    
    # Black-Scholes Parameters (1-year ATM Put)
    S = 100.0       # Normalized spot price
    K = 95.0        # 5% Out-of-the-money strike (downside protection)
    T = 1.0         # 1 year to maturity
    r = 0.05        # 5% risk-free rate
    
    put_price = black_scholes_put(S, K, T, r, vti_vol)
    
    # Expected Payoff of the Put (simplified expected return for optimizer)
    put_expected_return = -0.05  # Loses 5% on average (premium decay)
    put_volatility = 0.25        # 25% annualized volatility
    
    # 1. Expand Expected Returns vector
    mu_new = np.append(mu, put_expected_return)
    
    # 2. Expand Covariance Matrix
    Sigma_new = np.zeros((n_assets + 1, n_assets + 1))
    Sigma_new[:n_assets, :n_assets] = Sigma
    
    # Put option has strong negative correlation to VTI (-0.8) and zero to others
    Sigma_new[vti_idx, n_assets] = -0.8 * vti_vol * put_volatility
    Sigma_new[n_assets, vti_idx] = -0.8 * vti_vol * put_volatility
    Sigma_new[n_assets, n_assets] = put_volatility ** 2
    
    # 3. Expand initial weights
    w0_new = np.append(w0, 0.0)
    
    # 3.5 Expand Liquidity Caps for the new derivative
    # The synthetic put option is fully liquid up to the max weight limit
    if 'liquidity_caps' in data:
        data['liquidity_caps'] = np.append(data['liquidity_caps'], config['max_weight'])

    # 4. Expand Sectors
    sectors_new = sectors.copy()
    sectors_new['Alternatives'].append(n_assets) # Categorize put as an Alternative/Hedge
    
    # 5. Expand Names and Asset Classes for the Co-Pilot Report
    names_new = names + ['VTI_PUT']
    asset_classes_new = asset_classes + ['Derivative']
    
    # Update data dictionary
    data['mu'] = mu_new
    data['Sigma'] = Sigma_new
    data['w0'] = w0_new
    data['sectors'] = sectors_new
    data['n_assets'] = n_assets + 1
    data['names'] = names_new
    data['asset_classes'] = asset_classes_new
    
    print(f"\n🕯️ Derivative Valuation (Black-Scholes):")
    print(f"   Added VTI Protective Put (Strike={K}, T={T}yr, Vol={vti_vol*100:.1f}%)")
    print(f"   Black-Scholes Put Price: ${put_price:.2f}")
    
    return data