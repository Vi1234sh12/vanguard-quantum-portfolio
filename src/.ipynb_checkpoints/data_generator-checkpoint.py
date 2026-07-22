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
    
    asset_classes = ['US_Tech', 'US_Broad', 'Intl_Equity', 'Long_Bond', 'Corp_Bond', 
                     'Gold', 'Commodities', 'Real_Estate', 'EM_Equity', 'Alt_Risk', 'Cash', 'High_Yield']
    sectors = {
        'Equities': [0, 1, 2, 8],
        'Fixed_Income': [3, 4, 11],
        'Real_Assets': [5, 6, 7],
        'Alternatives': [9, 10]
    }

    np.random.seed(42)
    volatilities = np.random.uniform(0.05, 0.25, n_assets)
    mu = 0.04 + (volatilities * 0.3) + np.random.normal(0, 0.01, n_assets)
    
    # Realistic factor model covariance
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

    np.random.seed(42)
    w0 = np.random.dirichlet(np.ones(n_assets))
    w0 = np.clip(w0, 0.02, 0.15)
    w0 = w0 / np.sum(w0)

    return {
        'mu': mu, 'Sigma': Sigma, 'w0': w0, 'sectors': sectors,
        'asset_classes': asset_classes, 'names': [f"Asset_{i+1}" for i in range(n_assets)],
        'n_assets': n_assets, 'data_source': 'Synthetic Fallback'
    }

def get_multi_asset_data():
    """
    Fetches real 12-asset multi-class data and anonymizes it.
    Includes robust error handling for API failures.
    """
    tickers = ['QQQ', 'SPY', 'EFA', 'TLT', 'LQD', 'GLD', 'DBC', 'VNQ', 'EEM', 'AGG', 'SHY', 'HYG']
    asset_classes = ['US_Tech', 'US_Broad', 'Intl_Equity', 'Long_Bond', 'Corp_Bond', 
                     'Gold', 'Commodities', 'Real_Estate', 'EM_Equity', 'Total_Bond', 'Short_Treasury', 'High_Yield']
    
    sectors = {
        'Equities': [0, 1, 2, 8],
        'Fixed_Income': [3, 4, 9, 10, 11],
        'Real_Assets': [5, 6, 7],
        'Alternatives': []
    }

    try:
        print("📥 Downloading 5 years of real multi-asset market data...")
        # FIX: Modern yfinance requires slicing exactly 'Close' to get just the 12 tickers
        raw_data = yf.download(tickers, period='5y')['Close']
        
        if raw_data.empty:
            raise ValueError("Downloaded empty dataframe")
            
        returns = raw_data.pct_change().dropna()
        mu = returns.mean().values * 252
        Sigma = returns.cov().values * 252

        np.random.seed(42)
        w0 = np.random.dirichlet(np.ones(12))
        w0 = np.clip(w0, 0.02, 0.15)
        w0 = w0 / np.sum(w0)

        result = {
            'mu': mu, 'Sigma': Sigma, 'w0': w0, 'sectors': sectors,
            'asset_classes': asset_classes, 'names': [f"Asset_{i+1}" for i in range(12)],
            'n_assets': 12, 'data_source': 'Live Yahoo Finance'
        }
        print("✅ Live market data retrieved successfully.")
        return result

    except Exception as e:
        # If ANYTHING goes wrong, use fallback
        return _generate_synthetic_fallback()
