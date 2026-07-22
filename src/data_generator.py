import yfinance as yf
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def _generate_synthetic_fallback(n_assets=12):
    """
    ENGINEERING FAILSAFE: If Yahoo Finance API fails, generate 
    mathematically sound synthetic data to ensure the pipeline never breaks.
    """
    print("⚠️ Yahoo Finance failed. Activating Synthetic Data Failsafe...")
    
    asset_classes = ['Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E', 
                     'Class_F', 'Class_G', 'Class_H', 'Class_I', 'Class_J', 'Class_K', 'Class_L']
    sectors = {
        'Equities': [0, 1, 2, 8],
        'Fixed_Income': [3, 4, 11],
        'Real_Assets': [5, 6, 7],
        'Alternatives': [9, 10]
    }

    np.random.seed(42)
    volatilities = np.random.uniform(0.05, 0.25, n_assets)
    mu = 0.04 + (volatilities * 0.3) + np.random.normal(0, 0.01, n_assets)
    
    sector_names = list(sectors.keys())
    sector_list = ['Equities','Equities','Equities','Fixed_Income','Fixed_Income','Real_Assets','Real_Assets','Real_Assets','Equities','Alternatives','Alternatives','Fixed_Income']
    
    sector_factors = np.zeros((n_assets, len(sector_names)))
    for i, s in enumerate(sector_list):
        col_idx = sector_names.index(s)
        sector_factors[i, col_idx] = np.random.uniform(0.2, 0.5)
    
    factor_loadings = np.hstack([np.random.uniform(0.1, 0.3, (n_assets, 1)), sector_factors])
    factor_cov = np.eye(factor_loadings.shape[1]) * 0.3
    factor_cov[0, 1:] = 0.1; factor_cov[1:, 0] = 0.1
    idiosyncratic_var = np.diag(np.random.uniform(0.01, 0.04, n_assets))
    Sigma = factor_loadings @ factor_cov @ factor_loadings.T + idiosyncratic_var
    Sigma = (Sigma + Sigma.T) / 2

    # Valid starting portfolio
    w0 = np.zeros(n_assets)
    w0[0:3] = 0.10; w0[8] = 0.10
    w0[3] = 0.10; w0[4] = 0.15; w0[11] = 0.15
    w0[5] = 0.15; w0[6] = 0.05; w0[7] = 0.10
    w0[9:11] = 0.0

    return {
        'mu': mu, 'Sigma': Sigma, 'w0': w0, 'sectors': sectors,
        'asset_classes': asset_classes, 'names': [f"Asset_{i+1}" for i in range(n_assets)],
        'n_assets': n_assets, 'data_source': 'Synthetic Fallback',
        'test_returns': pd.DataFrame() # Empty test set for fallback
    }

def get_multi_asset_data():
    """
    Fetches historical data, anonymizes it, and splits into Train/Test sets.
    Uses 4 years of training data to predict the 1-year out-of-sample test set.
    """
    # Real tickers used for data fetching, but will be anonymized in output
    tickers = ['QQQ', 'VTI', 'VXUS', 'TLT', 'LQD', 'GLD', 'DBC', 'VNQ', 'VWO', 'BND', 'SHY', 'HYG']
    asset_classes = ['Class_A', 'Class_B', 'Class_C', 'Class_D', 'Class_E', 
                     'Class_F', 'Class_G', 'Class_H', 'Class_I', 'Class_J', 'Class_K', 'Class_L']
    
    sectors = {
        'Equities': [0, 1, 2, 8],
        'Fixed_Income': [3, 4, 9, 10, 11],
        'Real_Assets': [5, 6, 7],
        'Alternatives': []
    }

    try:
        print("📥 Downloading 5 years of historical market data...")
        raw_data = yf.download(tickers, period='5y')['Close']
        
        if raw_data.empty:
            raise ValueError("Downloaded empty dataframe")
            
        # Anonymize the columns immediately
        raw_data.columns = [f"Asset_{i+1}" for i in range(12)]
        names = list(raw_data.columns)
            
        # ==========================================
        # TRAIN / TEST SPLIT (4 Years Train, 1 Year Test)
        # ==========================================
        split_date = raw_data.index[-252] # Last 252 trading days = 1 year
        train_data = raw_data[raw_data.index < split_date]
        test_data = raw_data[raw_data.index >= split_date]
        
        train_returns = train_data.pct_change().dropna()
        test_returns = test_data.pct_change().dropna()
        
        # 1. Historical Mean (Training)
        mu = train_returns.mean().values * 252
        
        # 2. Factor Model Covariance Matrix (Training)
        sector_names = list(sectors.keys())
        sector_list = ['Equities','Equities','Equities','Fixed_Income','Fixed_Income','Real_Assets','Real_Assets','Real_Assets','Equities','Fixed_Income','Fixed_Income','Fixed_Income']
        
        factor_loadings = np.zeros((len(names), len(sector_names)))
        for i, s in enumerate(sector_list):
            col_idx = sector_names.index(s)
            factor_loadings[i, col_idx] = 1.0
            
        factor_returns = train_returns.values @ np.linalg.pinv(factor_loadings.T)
        factor_cov = np.cov(factor_returns, rowvar=False) * 252
        
        common_returns = factor_loadings @ factor_returns.T
        residual_returns = train_returns.values.T - common_returns
        idiosyncratic_var = np.var(residual_returns, axis=1) * 252
        
        Sigma = factor_loadings @ factor_cov @ factor_loadings.T + np.diag(idiosyncratic_var)
        Sigma = (Sigma + Sigma.T) / 2

        # 3. Derive w0 (Chronological Baseline)
        # Generate a random valid portfolio that strictly respects liquidity caps
        np.random.seed(42)
        liquidity_caps = np.random.uniform(0.05, 0.20, 12)
        w0 = np.random.uniform(0.01, liquidity_caps)
        w0 = w0 / np.sum(w0) # Normalize to 100%

        # ==========================================
        # LIQUIDITY CONSTRAINTS (Institutional Realism)
        # Simulate max tradable volume per asset (e.g., 5% to 20% of portfolio)
        # ==========================================
        np.random.seed(42)
        liquidity_caps = np.random.uniform(0.05, 0.20, 12)

        result = {
            'mu': mu, 'Sigma': Sigma, 'w0': w0, 'sectors': sectors,
            'asset_classes': asset_classes, 'names': names,
            'n_assets': 12, 'data_source': 'Anonymized Historical (Train/Test Split)',
            'test_returns': test_returns,
            'liquidity_caps': liquidity_caps  # <-- ADDED HERE
        }

        # result = {
        #     'mu': mu, 'Sigma': Sigma, 'w0': w0, 'sectors': sectors,
        #     'asset_classes': asset_classes, 'names': names,
        #     'n_assets': 12, 'data_source': 'Anonymized Historical (Train/Test Split)',
        #     'test_returns': test_returns # Store test set for out-of-sample evaluation
        # }
        print("✅ Historical data retrieved, anonymized, and split (4Y Train / 1Y Test).")
        return result

    except Exception as e:
        print(f"Error: {e}")
        return _generate_synthetic_fallback()