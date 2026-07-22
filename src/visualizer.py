import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import yfinance as yf
import textwrap
import os

plt.style.use('ggplot')

def generate_visualizations(final_weights, data, config, qaoa_energies=None, top_freqs=None, Q=None):
    print("\n📊 Generating Quantitative Visualizations...")
    
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Use anonymized names from the data dictionary
    tickers = data['names']
    sectors = data['sectors']
    n_assets = data['n_assets']
    colors = plt.cm.tab20(np.linspace(0, 1, n_assets))
    
    # 1. PIE CHART
    plt.figure(figsize=(10, 7))
    non_zero = final_weights > 0.001
    plt.pie(final_weights[non_zero], labels=[tickers[i] for i in range(n_assets) if non_zero[i]], 
            autopct='%1.1f%%', startangle=140, colors=colors[non_zero])
    plt.title(f"Asset Allocation ({config['PROFILE_NAME']})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('figures/chart_pie_allocation.png', dpi=300)
    plt.close()
    print("   ✅ Saved: figures/chart_pie_allocation.png")

    # 2. BAR CHART
    sector_names = list(sectors.keys())
    sector_weights = [np.sum(final_weights[indices]) for indices in sectors.values()]
    plt.figure(figsize=(10, 6))
    plt.bar(sector_names, sector_weights, color=['#4C72B0', '#55A868', '#C44E52', '#8172B3'])
    plt.axhline(y=config['MAX_SECTOR'], color='red', linestyle='--', label=f"Limit ({config['MAX_SECTOR']*100}%)")
    plt.title("Sector Exposures & Limits", fontsize=14, fontweight='bold')
    plt.ylabel("Weight")
    plt.ylim(0, max(sector_weights) * 1.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/chart_bar_sectors.png', dpi=300)
    plt.close()
    print("   ✅ Saved: figures/chart_bar_sectors.png")

    # 3. LINE GRAPH (Out-of-Sample Test Set)
    real_etf_weights = final_weights[:-1]
    
    # Use the test_returns from the Train/Test split to show out-of-sample performance
    if 'test_returns' in data and not data['test_returns'].empty:
        print("   📈 Using out-of-sample test set for performance tracking...")
        daily_returns = data['test_returns']
    else:
        print("   📈 Fetching 1-year historical data for performance tracking...")
        hist_data = yf.download(tickers[:-1], period="1y", progress=False)['Close']
        if isinstance(hist_data, pd.Series): hist_data = hist_data.to_frame()
        daily_returns = hist_data.pct_change().dropna()
    
    # Dynamically get the column names from the daily_returns dataframe (Anonymized)
    real_etf_tickers = daily_returns.columns.tolist()
    
    classical_weights = np.ones(len(daily_returns.columns)) / len(daily_returns.columns)
    qaoa_cumulative = (1 + (daily_returns * real_etf_weights).sum(axis=1)).cumprod()
    classical_cumulative = (1 + (daily_returns * classical_weights).sum(axis=1)).cumprod()
    
    plt.figure(figsize=(12, 6))
    plt.plot(qaoa_cumulative.index, qaoa_cumulative, label='Quantum (QAOA) Portfolio', color='#E63946', linewidth=2)
    plt.plot(classical_cumulative.index, classical_cumulative, label='Equal-Weight Benchmark', color='#457B9D', linewidth=2)
    plt.title("Out-of-Sample Portfolio Performance (1-Year Test Set)", fontsize=14, fontweight='bold')
    plt.ylabel("Cumulative Return (Growth of $1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/chart_line_performance.png', dpi=300)
    plt.close()
    print("   ✅ Saved: figures/chart_line_performance.png")
    
    # 4. STREAM GRAPH
    asset_wealth = (1 + daily_returns[real_etf_tickers]).cumprod() * real_etf_weights
    plt.figure(figsize=(12, 6))
    plt.stackplot(asset_wealth.index, asset_wealth.T, labels=real_etf_tickers, colors=colors[:-1], alpha=0.8)
    plt.title("Stream Graph: Asset Wealth Contribution Over Time", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=8)
    plt.tight_layout()
    plt.savefig('figures/chart_stream_graph.png', dpi=300)
    plt.close()
    print("   ✅ Saved: figures/chart_stream_graph.png")

    # 5. CANDLESTICK CHART
    # Skipped because data is anonymized and Yahoo Finance cannot fetch 'Asset_1'
    print("   ⚠️ Skipping Candlestick Chart (Anonymized data has no live ticker).")

    # 6. QUBO ENERGY DISTRIBUTION (Mathematician Fix: Histogram, not Line)
    if qaoa_energies is not None:
        plt.figure(figsize=(12, 6))
        plt.hist(qaoa_energies, bins=50, color='#E63946', edgecolor='black', alpha=0.8)
        plt.title("Quantum Annealing: Energy State Distribution (1000 reads)", fontsize=14, fontweight='bold')
        plt.xlabel("QUBO Energy (Cost) [Lower is Better]")
        plt.ylabel("Frequency (Number of Samples)")
        plt.axvline(x=min(qaoa_energies), color='gold', linestyle='--', linewidth=2, label=f'Ground State Energy: {min(qaoa_energies):.2f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig('figures/chart_qaoa_cost_history.png', dpi=300)
        plt.close()
        print("   ✅ Saved: figures/chart_qaoa_cost_history.png")

    # 7. BITSTRING DISTRIBUTION
    if top_freqs is not None:
        plt.figure(figsize=(14, 7))
        labels = [textwrap.fill(lbl, 12) for lbl, freq in top_freqs]
        frequencies = [freq for lbl, freq in top_freqs]
        bars = plt.bar(labels, frequencies, color='mediumseagreen', edgecolor='black')
        plt.title("Distribution of Sampled Bitstrings (Top 10) for Best QAOA Result", fontsize=14, fontweight='bold')
        plt.xlabel("Bitstring (Asset Selection)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plt.savefig('figures/chart_qaoa_bitstring_distribution.png', dpi=300)
        plt.close()
        print("   ✅ Saved: figures/chart_qaoa_bitstring_distribution.png")

    # 8. CORRELATION MATRIX HEATMAP (Mathematician Fix: Correlation, not Covariance)
    plt.figure(figsize=(10, 8))
    # Convert covariance to correlation: Corr = Cov / (sigma_i * sigma_j)
    cov_matrix = data['Sigma'][:-1, :-1]
    volatilities = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(volatilities, volatilities)
    
    plt.imshow(corr_matrix, cmap='RdYlGn_r', interpolation='none', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation Coefficient')
    plt.title("Asset Correlation Matrix Heatmap", fontsize=14, fontweight='bold')
    plt.xticks(range(len(real_etf_tickers)), real_etf_tickers, rotation=45)
    plt.yticks(range(len(real_etf_tickers)), real_etf_tickers)
    plt.tight_layout()
    plt.savefig('figures/chart_covariance_heatmap.png', dpi=300)
    plt.close()
    print("   ✅ Saved: figures/chart_covariance_heatmap.png")

    # 9. QUBO MATRIX HEATMAP (Mathematician Fix: SymLog Scale)
    if Q is not None:
        plt.figure(figsize=(8, 8))
        # Use SymLogNorm to handle the massive scale difference between diagonal and off-diagonal
        norm = mcolors.SymLogNorm(linthresh=0.1, linscale=1, vmin=Q.min(), vmax=Q.max())
        plt.imshow(Q, cmap='viridis', interpolation='none', norm=norm)
        plt.colorbar(label='QUBO Coefficient (SymLog Scale)')
        n_qubits = Q.shape[0]
        plt.title(f"QUBO Matrix Sparsity Structure ({n_qubits}x{n_qubits})", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/chart_qubo_heatmap.png', dpi=300)
        plt.close()
        print("   ✅ Saved: figures/chart_qubo_heatmap.png")

    print("\n✨ All visualizations generated successfully in the 'figures/' folder.\n")