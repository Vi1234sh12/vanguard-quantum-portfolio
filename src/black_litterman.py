import numpy as np

def simulate_gbm_returns(mu_hist, Sigma, n_sims=1000, T=252):
    """
    Geometric Brownian Motion (GBM) Monte Carlo Simulation.
    Assumes mu_hist and Sigma are ANNUALIZED.
    """
    n_assets = len(mu_hist)
    dt = 1.0 / 252.0  # Daily time step
    
    # Convert annual inputs to daily for the simulation
    mu_daily = mu_hist * dt
    Sigma_daily = Sigma * dt
    
    # Cholesky decomposition for correlated random walks
    # Adding a tiny value to the diagonal for numerical stability
    L = np.linalg.cholesky(Sigma_daily + np.eye(n_assets) * 1e-8)
    
    expected_final_prices = np.zeros(n_assets)
    initial_price = 1.0  # S0 = 1.0
    
    for _ in range(n_sims):
        # Random normal shocks (correlated)
        Z = np.random.normal(0, 1, (T, n_assets))
        daily_shocks = Z @ L.T
        
        # GBM daily log returns: (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
        # Here daily_shocks is already sigma*sqrt(dt)*Z
        drift = (mu_daily - 0.5 * np.diag(Sigma_daily))
        log_returns = drift + daily_shocks
        
        # Cumulative log return over T days
        cum_log_return = np.sum(log_returns, axis=0)
        
        # Final price S_T = S_0 * exp(cum_log_return)
        final_price = initial_price * np.exp(cum_log_return)
        expected_final_prices += final_price
    
    # Expected simple return E[R] = E[S_T/S_0 - 1]
    expected_returns = (expected_final_prices / n_sims) - 1.0
    
    # SAFETY CLIP: Prevent extreme outliers from breaking the QUBO
    expected_returns = np.clip(expected_returns, -0.50, 1.50) # Cap between -50% and +150%
    
    return expected_returns

def black_litterman_expected_returns(Sigma, mu_gbm, w_mkt, tau=0.1, delta=2.5):
    """
    Calculates Black-Litterman Expected Returns.
    Prior = Global Market Weights (w_mkt)
    View = GBM Simulated Returns (mu_gbm)
    Increased tau to 0.1 to give higher confidence to forward-looking GBM views,
    making the 7% return target mathematically feasible.
    """
    n = len(w_mkt)
    
    # 1. Reverse Optimization: Calculate Implied Equilibrium Returns (Pi)
    Pi = delta * np.dot(Sigma, w_mkt)
    
    # 2. Define Views (P = Identity, Q = GBM simulated returns)
    P = np.eye(n)
    Q = mu_gbm
    
    # 3. Calculate View Uncertainty (Omega) - Idzorek's method
    Omega = np.diag(np.diag(tau * P @ Sigma @ P.T))
    
    # 4. Black-Litterman Master Formula
    tau_Sigma = tau * Sigma
    tau_Sigma_P_T = tau_Sigma @ P.T
    middle_term = np.linalg.inv(P @ tau_Sigma_P_T + Omega)
    
    mu_bl = Pi + tau_Sigma_P_T @ middle_term @ (Q - P @ Pi)
    
    return mu_bl, Pi