import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Portfolio Efficient Frontier Visualization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and introduction
st.title("Portfolio Efficient Frontier Visualization")
st.markdown("""
This application visualizes the efficient frontier for portfolios consisting of different assets.
* **Two-Asset Frontier**: Explore how correlation affects the efficient frontier for two assets
* **Multi-Asset Frontier**: Visualize the frontier with multiple assets and find optimal portfolios
""")

# Create tabs for main content
tab1, tab2 = st.tabs(["Two-Asset Frontier", "Multi-Asset Frontier"])

# ===== SIDEBAR PARAMETERS =====
st.sidebar.title("Settings")

# Common sidebar parameters
st.sidebar.header("Global Parameters")
rf_rate = st.sidebar.number_input("Risk-free Rate (%)", value=2.0, min_value=0.0, max_value=10.0) / 100

# Plot size controls
st.sidebar.subheader("Plot Settings")
plot_height = st.sidebar.slider("Plot Height", 200, 600, 300, 50)
plot_width = st.sidebar.slider("Plot Width", 300, 1000, 600, 50)
plot_size = (plot_width/100, plot_height/100)  # Convert to inches for matplotlib

# Two-asset specific sidebar parameters
st.sidebar.header("Two-Asset Parameters")
# Asset 1 parameters
st.sidebar.subheader("Asset 1")
mu1 = st.sidebar.number_input("Return (%)", value=10.0, min_value=0.0, max_value=100.0, key="mu1") / 100
sigma1 = st.sidebar.number_input("Volatility (%)", value=12.0, min_value=0.1, max_value=100.0, key="sigma1") / 100

# Asset 2 parameters
st.sidebar.subheader("Asset 2")
mu2 = st.sidebar.number_input("Return (%)", value=15.0, min_value=0.0, max_value=100.0, key="mu2") / 100
sigma2 = st.sidebar.number_input("Volatility (%)", value=24.0, min_value=0.1, max_value=100.0, key="sigma2") / 100

# Short selling and CML options
allow_short = st.sidebar.checkbox("Allow Short Selling", value=True, key="allow_short_2")
show_cml = st.sidebar.checkbox("Show Capital Market Line", value=True, key="show_cml")

# Multi-asset specific sidebar parameters
st.sidebar.header("Multi-Asset Parameters")
n_assets = st.sidebar.number_input("Total assets", min_value=2, max_value=30, value=20, step=1, key="n_assets")
n_simulations = st.sidebar.number_input("Simulations", min_value=100, max_value=10000, value=2000, step=100, key="n_simulations")
alpha = st.sidebar.number_input("Concentration", min_value=0.1, max_value=5.0, value=0.3, step=0.1, key="alpha", 
                              help="Weight concentration parameter (lower = more concentrated)")
allow_short_multi = st.sidebar.checkbox("Allow Short Selling", value=False, key="allow_short_multi")
seed = st.sidebar.number_input("Random Seed", value=42, min_value=1, max_value=1000, key="seed")
show_cml_multi = st.sidebar.checkbox("Show Capital Market Line", value=True, key="show_cml_multi")

# =============== TWO-ASSET FRONTIER TAB ===============
with tab1:
    # Correlation slider in main panel for Two-Asset Frontier
    st.subheader("Correlation between Assets")
    rho = st.slider("Adjust correlation between Asset 1 and Asset 2:", 
                    min_value=-1.0, max_value=1.0, value=0.5, step=0.05, key="rho")
    
    # Generate weights for the frontier
    if allow_short:
        w = np.linspace(-1, 2, 100)  # Allow short positions
    else:
        w = np.linspace(0, 1, 100)  # No short positions
    
    # Calculate the frontier
    cov = rho * sigma1 * sigma2
    portfolio_returns = w*mu1 + (1-w)*mu2
    portfolio_volatility = np.sqrt(
        (w**2)*(sigma1**2) + 
        ((1-w)**2)*(sigma2**2) + 
        2*w*(1-w)*cov
    )
    
    # Calculate Sharpe ratios
    sharpe_ratios = (portfolio_returns - rf_rate) / portfolio_volatility
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_return = portfolio_returns[max_sharpe_idx]
    max_sharpe_vol = portfolio_volatility[max_sharpe_idx]
    max_sharpe_weight = w[max_sharpe_idx]
    
    # Find minimum volatility portfolio
    min_vol_idx = np.argmin(portfolio_volatility)
    min_vol_return = portfolio_returns[min_vol_idx]
    min_vol = portfolio_volatility[min_vol_idx]
    min_vol_weight = w[min_vol_idx]
    
    # Create plot with user-defined size
    fig, ax = plt.subplots(figsize=plot_size)
    
    # Plot the portfolio points colored by Sharpe ratio
    scatter = ax.scatter(portfolio_volatility, portfolio_returns, c=sharpe_ratios, cmap='viridis', s=15, alpha=0.6)
    
    # Plot the frontier line
    ax.plot(portfolio_volatility, portfolio_returns, 'b-', lw=1.5, alpha=0.5)
    
    # Plot the individual assets
    ax.scatter([sigma1], [mu1], color='red', s=60, label='Asset 1')
    ax.scatter([sigma2], [mu2], color='green', s=60, label='Asset 2')
    
    # Plot the risk-free asset
    ax.scatter([0], [rf_rate], color='black', s=60, marker='o', label='Risk-free Asset')
    
    # Plot the special portfolios
    ax.scatter([min_vol], [min_vol_return], color='blue', s=70, marker='*', label='Min Volatility')
    ax.scatter([max_sharpe_vol], [max_sharpe_return], color='purple', s=70, marker='*', label='Max Sharpe Ratio')
    
    # Add Capital Market Line
    if show_cml:
        x = np.linspace(0, max(portfolio_volatility)*1.2, 100)
        y = rf_rate + (max_sharpe_return - rf_rate) / max_sharpe_vol * x
        ax.plot(x, y, 'r--', label='Capital Market Line')
    
    # Add colorbar, labels, title
    cbar = plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
    cbar.ax.tick_params(labelsize=8)  # Smaller colorbar font
    ax.set_xlabel('Volatility (Standard Deviation)', fontsize=8)
    ax.set_ylabel('Expected Return', fontsize=8)
    ax.set_title(f'Two-Asset Efficient Frontier (Correlation = {rho:.2f})', fontsize=9)
    ax.legend(fontsize=7, loc='best')  # Smaller legend font, optimal location
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)  # Smaller tick font
    
    # Set reasonable axis limits to ensure risk-free rate is visible
    ax.set_xlim(0, max(sigma1, sigma2, max(portfolio_volatility)) * 1.1)
    # Ensure y-axis includes risk-free rate
    y_min = min(rf_rate, min(mu1, mu2, min(portfolio_returns))) * 0.9
    y_max = max(mu1, mu2, max(portfolio_returns)) * 1.1
    ax.set_ylim(y_min, y_max)
    
    # Make plot layout tight
    plt.tight_layout()
    
    # Create two columns for plot and details
    col1, col2 = st.columns([1, 1])
    
    # Display the plot
    with col1:
        st.pyplot(fig)
    
    # Display portfolio details
    with col2:
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Minimum Volatility Portfolio")
            st.markdown(f"""
            * Asset 1: {min_vol_weight:.2f} ({min_vol_weight*100:.1f}%)
            * Asset 2: {1-min_vol_weight:.2f} ({(1-min_vol_weight)*100:.1f}%)
            * Return: {min_vol_return:.4f} ({min_vol_return*100:.2f}%)
            * Vol: {min_vol:.4f} ({min_vol*100:.2f}%)
            * Sharpe: {sharpe_ratios[min_vol_idx]:.4f}
            """)
        
        with col_b:
            st.subheader("Maximum Sharpe Ratio Portfolio")
            st.markdown(f"""
            * Asset 1: {max_sharpe_weight:.2f} ({max_sharpe_weight*100:.1f}%)
            * Asset 2: {1-max_sharpe_weight:.2f} ({(1-max_sharpe_weight)*100:.1f}%)
            * Return: {max_sharpe_return:.4f} ({max_sharpe_return*100:.2f}%)
            * Vol: {max_sharpe_vol:.4f} ({max_sharpe_vol*100:.2f}%)
            * Sharpe: {sharpe_ratios[max_sharpe_idx]:.4f}
            """)
        
    # Download portfolio data
    portfolio_data = pd.DataFrame({
        'Weight Asset 1': w,
        'Weight Asset 2': 1-w,
        'Expected Return': portfolio_returns,
        'Volatility': portfolio_volatility,
        'Sharpe Ratio': sharpe_ratios
    })
    
    buffer = io.BytesIO()
    portfolio_data.to_csv(buffer, index=False)
    buffer.seek(0)
    
    st.download_button(
        label="Download Portfolio Data",
        data=buffer,
        file_name="two_asset_frontier.csv",
        mime="text/csv"
    )

# =============== MULTI-ASSET FRONTIER TAB ===============
with tab2:
    # Assets to include slider in main panel for Multi-Asset Frontier
    st.subheader("Portfolio Composition")
    assets_included = st.slider("Number of assets to include in the portfolio:", 
                               min_value=1, max_value=n_assets, value=min(2, n_assets), step=1, key="assets_included")
    
    # Generate asset parameters
    np.random.seed(seed)
    mu = np.random.uniform(0.05, 0.2, n_assets)
    sigma = np.random.uniform(0.1, 0.3, n_assets)
    
    # Generate random correlation matrix
    corr_mat = np.random.uniform(-0.5, 0.7, (n_assets, n_assets))
    corr_mat = (corr_mat + corr_mat.T)/2
    np.fill_diagonal(corr_mat, 1)
    cov_mat = np.outer(sigma, sigma) * corr_mat
    
    # Function to generate weights
    def generate_weights(n, size, alpha=0.3, allow_short=False):
        if allow_short:
            # For short selling, use normal distribution then normalize
            weights = np.random.normal(0, 1, (size, n))
            weights = weights / np.sum(np.abs(weights), axis=1, keepdims=True)
        else:
            # For long-only, use Dirichlet distribution
            weights = np.random.dirichlet(alpha * np.ones(n), size)
        
        # Normalize to ensure sum is 1 (handles floating point errors)
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return weights
    
    # Calculate portfolios
    portfolio_ret = None
    portfolio_vol = None
    sharpe_ratios = None
    min_vol_weights = None
    max_sharpe_weights = None
    
    if assets_included > 1:
        selected = slice(0, assets_included)
        sub_mu = mu[selected]
        sub_cov = cov_mat[selected, :][:, selected]
        
        # Generate random weights
        weights = generate_weights(assets_included, n_simulations, alpha, allow_short_multi)
        
        # Calculate portfolio returns and volatilities
        portfolio_ret = weights @ sub_mu
        portfolio_vol = np.sqrt(np.diag(weights @ sub_cov @ weights.T))
        
        # Calculate Sharpe ratios
        sharpe_ratios = (portfolio_ret - rf_rate) / portfolio_vol
        
        # Find minimum volatility portfolio
        min_vol_idx = np.argmin(portfolio_vol)
        min_vol = portfolio_vol[min_vol_idx]
        min_vol_ret = portfolio_ret[min_vol_idx]
        min_vol_weights = weights[min_vol_idx]
        
        # Find maximum Sharpe ratio portfolio
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_vol = portfolio_vol[max_sharpe_idx]
        max_sharpe_ret = portfolio_ret[max_sharpe_idx]
        max_sharpe_weights = weights[max_sharpe_idx]
    
    # Create a layout for plot and data
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Show correlation matrix in expander to save space
        with st.expander("Show Correlation Matrix", expanded=False):
            fig_corr, ax_corr = plt.subplots(figsize=plot_size)
            sns.heatmap(corr_mat[:assets_included, :assets_included], 
                      annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                      ax=ax_corr, annot_kws={"size": 7})
            ax_corr.set_title("Correlation Matrix of Selected Assets", fontsize=9)
            st.pyplot(fig_corr)
        
        # Create plot for frontier with user-defined size
        fig, ax = plt.subplots(figsize=plot_size)
        
        # Plot all assets
        colors = ['red' if i < assets_included else 'gray' for i in range(n_assets)]
        asset_scatter = ax.scatter(sigma, mu, c=colors, alpha=0.7, s=40)
        
        # Plot the risk-free asset
        ax.scatter([0], [rf_rate], color='black', s=50, marker='o', label='Risk-free')
        
        # Add asset labels for only first few assets to avoid clutter
        max_labels = min(assets_included, 10)  # Limit number of labels
        for i in range(max_labels):
            ax.annotate(f"{i+1}", (sigma[i], mu[i]), 
                      xytext=(3, 3), textcoords='offset points', fontsize=7)
        
        # Plot portfolio cloud
        if assets_included > 1:
            scatter = ax.scatter(portfolio_vol, portfolio_ret, c=sharpe_ratios, cmap='viridis', alpha=0.5, s=5)
            cbar = plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
            cbar.ax.tick_params(labelsize=7)
            ax.scatter([min_vol], [min_vol_ret], color='blue', s=70, marker='*', label='Min Vol')
            ax.scatter([max_sharpe_vol], [max_sharpe_ret], color='purple', s=70, marker='*', label='Max Sharpe')
            
            # Add Capital Market Line
            if show_cml_multi:
                x = np.linspace(0, max(max(portfolio_vol), max(sigma))*1.2, 100)
                y = rf_rate + (max_sharpe_ret - rf_rate) / max_sharpe_vol * x
                ax.plot(x, y, 'r--', label='CML')
        
        ax.set_xlabel('Volatility (Standard Deviation)', fontsize=8)
        ax.set_ylabel('Expected Return', fontsize=8)
        ax.set_title(f'Multi-Asset Frontier (First {assets_included} of {n_assets} assets)', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc='best')
        ax.tick_params(labelsize=7)
        
        # Set reasonable axis limits to include risk-free rate
        ax.set_xlim(0, max(sigma) * 1.1)
        # Ensure y-axis includes risk-free rate
        y_min = min(rf_rate, min(mu)) * 0.9
        y_max = max(mu) * 1.1
        ax.set_ylim(y_min, y_max)
        
        # Make plot layout tight
        plt.tight_layout()
        
        st.pyplot(fig)
    
    with col2:
        # Display asset details in a more compact table
        st.subheader("Asset Information")
        asset_data = []
        for i in range(n_assets):
            status = "✓" if i < assets_included else "✗"
            asset_data.append({
                "No.": i+1, 
                "Status": status,
                "Return": f"{mu[i]:.2%}",
                "Vol": f"{sigma[i]:.2%}"
            })
        
        asset_df = pd.DataFrame(asset_data)
        st.dataframe(asset_df, height=min(180, 35 + 35*min(n_assets, 10)), use_container_width=True)
        
        # Display portfolio details when more than one asset
        if assets_included > 1:
            # Min vol portfolio
            st.subheader("Minimum Volatility Portfolio")
            st.markdown(f"""
            * Return: {min_vol_ret:.2%} | Vol: {min_vol:.2%}
            * Sharpe: {sharpe_ratios[min_vol_idx]:.4f}
            """)
            
            # Show weights in small table
            min_vol_weights_df = pd.DataFrame({
                "Asset": range(1, assets_included + 1),
                "Weight": [f"{w:.2%}" for w in min_vol_weights]
            }).set_index("Asset")
            st.dataframe(min_vol_weights_df, height=min(150, 35 + 25*assets_included), use_container_width=True)
            
            # Max sharpe portfolio
            st.subheader("Maximum Sharpe Ratio Portfolio")
            st.markdown(f"""
            * Return: {max_sharpe_ret:.2%} | Vol: {max_sharpe_vol:.2%}
            * Sharpe: {sharpe_ratios[max_sharpe_idx]:.4f}
            """)
            
            # Show weights in small table
            max_sharpe_weights_df = pd.DataFrame({
                "Asset": range(1, assets_included + 1),
                "Weight": [f"{w:.2%}" for w in max_sharpe_weights]
            }).set_index("Asset")
            st.dataframe(max_sharpe_weights_df, height=min(150, 35 + 25*assets_included), use_container_width=True)
        
            # Download portfolio data
            portfolio_data = pd.DataFrame({
                'Volatility': portfolio_vol,
                'Expected Return': portfolio_ret,
                'Sharpe Ratio': sharpe_ratios
            })
            
            # Add weights columns
            for i in range(assets_included):
                portfolio_data[f'Weight Asset {i+1}'] = weights[:, i]
            
            buffer = io.BytesIO()
            portfolio_data.to_csv(buffer, index=False)
            buffer.seek(0)
            
            st.download_button(
                label="Download Portfolio Data",
                data=buffer,
                file_name="multi_asset_frontier.csv",
                mime="text/csv",
                key="download_multi"
            )

# Add footer with information in expander to save space
with st.expander("About this app", expanded=False):
    st.markdown("""
    This app visualizes the efficient frontier, which shows the optimal portfolios that offer the highest expected return for a defined level of risk.

    Key concepts:
    * **Efficient Frontier**: The set of optimal portfolios that offer the highest expected return for a given level of risk
    * **Minimum Volatility Portfolio**: The portfolio with the lowest possible risk (standard deviation)
    * **Maximum Sharpe Ratio Portfolio**: The portfolio with the highest risk-adjusted return (Sharpe ratio)
    * **Capital Market Line**: A line that shows the risk-return tradeoff when combining the risk-free asset with the optimal risky portfolio

    Created with Streamlit and Matplotlib.
    """)