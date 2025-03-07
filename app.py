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
* **Analytical Solution**: Calculate the exact efficient frontier using matrix operations
* **Black-Litterman Model**: Incorporate investor views with the market equilibrium to improve portfolio allocation
""")

# Create tabs for main content
tab1, tab2, tab3, tab4 = st.tabs(["Two-Asset Frontier", "Multi-Asset Frontier", "Analytical Solution", "Black-Litterman Model"])

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

# =============== ANALYTICAL SOLUTION TAB ===============
with tab3:
    st.subheader("Analytical Efficient Frontier (Four Assets)")
    
    # Instructions
    st.markdown(r"""
    This tab calculates the exact efficient frontier using matrix operations. 
    The efficient frontier equation is:
    
    $$\frac{\sigma^2_p}{1/c} = \frac{(r_p - b/c)^2}{|d|/c^2} + 1$$
    
    Where parameters are derived from the covariance matrix and expected returns.
    """)
    
    # Create a layout with two columns for inputs and main results
    input_col, main_col = st.columns([1, 2])
    
    with input_col:
        # Asset input section
        st.subheader("Asset Parameters")
        
        # Default values for the assets
        default_returns = np.array([0.037, 0.020, 0.026, 0.041])
        default_stdevs = np.array([0.114, 0.147, 0.104, 0.105])
        default_corr = np.array([
            [1.000, 0.239, 0.590, 0.501],
            [0.239, 1.000, 0.262, 0.194],
            [0.590, 0.262, 1.000, 0.458],
            [0.501, 0.194, 0.458, 1.000]
        ])
        
        asset_names = ['A', 'B', 'C', 'D']
        
        # Create inputs for asset returns and volatilities
        st.write("Expected Returns and Standard Deviations:")
        cols = st.columns(2)
        
        # Arrays to store user inputs
        user_returns = np.zeros(4)
        user_stdevs = np.zeros(4)
        
        # Create input fields for each asset
        for i in range(4):
            with cols[i % 2]:
                st.markdown(f"**Asset {asset_names[i]}**")
                user_returns[i] = st.number_input(
                    f"Return", 
                    value=float(default_returns[i]), 
                    min_value=0.0, 
                    max_value=1.0, 
                    step=0.001, 
                    format="%.3f",
                    key=f"ret_{i}"
                )
                user_stdevs[i] = st.number_input(
                    f"Std Dev", 
                    value=float(default_stdevs[i]), 
                    min_value=0.001, 
                    max_value=1.0, 
                    step=0.001, 
                    format="%.3f",
                    key=f"std_{i}"
                )
        
        # Correlation matrix input in expander
        with st.expander("Correlation Matrix", expanded=False):
            # Create a 4x4 matrix of correlation inputs
            user_corr = np.ones((4, 4))  # Initialize with ones on diagonal
            
            # We only need to input the lower triangle since correlation matrix is symmetric
            for i in range(4):
                for j in range(i):
                    # Create a unique key for each correlation input
                    key = f"corr_{i}_{j}"
                    # Create columns for each row
                    if j == 0:  # Start a new row
                        cols = st.columns(4)
                    
                    # Add input in appropriate column
                    with cols[j]:
                        if i > j:  # Only show inputs for lower triangle
                            corr_value = st.number_input(
                                f"{asset_names[i]}-{asset_names[j]}", 
                                value=float(default_corr[i][j]), 
                                min_value=-1.0, 
                                max_value=1.0, 
                                step=0.01,
                                format="%.3f",
                                key=key
                            )
                            user_corr[i, j] = corr_value
                            user_corr[j, i] = corr_value  # Symmetric matrix
        
        # Allow users to toggle short selling and CML
        st.subheader("Options")
        allow_short_analytical = st.checkbox("Allow Short Selling", value=True, key="allow_short_analytical")
        show_cml_analytical = st.checkbox("Show Capital Market Line", value=True, key="show_cml_analytical")
    
    # Calculate the covariance matrix
    cov_matrix = np.outer(user_stdevs, user_stdevs) * user_corr
    
    # Calculate efficient frontier parameters
    def calculate_frontier_params(returns, cov_matrix):
        n = len(returns)
        
        # Create vectors and matrices needed
        ones = np.ones(n)
        
        # Calculate inverse of covariance matrix
        try:
            inv_cov = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            st.error("Error: Covariance matrix is singular. Please check your inputs.")
            return None, None, None, None, None, ones
        
        # Calculate efficient frontier parameters
        a = returns @ inv_cov @ returns
        b = ones @ inv_cov @ returns
        c = ones @ inv_cov @ ones
        d = a * c - b ** 2
        
        return a, b, c, d, inv_cov, ones
    
    # Calculate the parameters
    a, b, c, d, inv_cov, ones_vector = calculate_frontier_params(user_returns, cov_matrix)
    
    if inv_cov is not None:
        # Calculate minimum and maximum returns for the frontier
        min_ret = min(user_returns) * 0.5
        max_ret = max(user_returns) * 1.5
        
        # Calculate minimum variance portfolio
        min_var_return = b / c
        min_var_vol = np.sqrt(1 / c)
        
        # Calculate portfolio weights for minimum variance
        min_var_weights = inv_cov @ (ones_vector * 1 / c)
        
        # Calculate tangency (maximum Sharpe ratio) portfolio if rf_rate is provided
        tangency_return = None
        tangency_vol = None
        tangency_sharpe = None
        weights_tan = None
        
        if 'rf_rate' in locals():
            # Calculate tangency portfolio returns and volatility
            excess_returns = user_returns - rf_rate
            try:
                # Calculate weights
                weights_tan = inv_cov @ excess_returns
                weights_tan = weights_tan / np.sum(weights_tan)  # Normalize
                
                # Apply short-selling constraint if needed
                if not allow_short_analytical:
                    weights_tan = np.maximum(weights_tan, 0)
                    weights_tan = weights_tan / np.sum(weights_tan)  # Re-normalize
                
                # Calculate returns and volatility
                tangency_return = np.sum(weights_tan * user_returns)
                tangency_vol = np.sqrt(weights_tan @ cov_matrix @ weights_tan)
                tangency_sharpe = (tangency_return - rf_rate) / tangency_vol
            except Exception as e:
                st.warning(f"Could not calculate tangency portfolio: {e}")
        
        with main_col:
            # Add a slider for the user to select a target return level
            st.subheader("Select Target Return")
            
            # Ensure target return slider has sensible bounds
            min_slider = min(min(user_returns) * 0.8, min_var_return * 0.8)
            max_slider = max(user_returns) * 1.2
            default_target = min_var_return + (max_slider - min_var_return) / 2
            
            target_return = st.slider(
                "Expected Return", 
                min_value=float(min_slider), 
                max_value=float(max_slider), 
                value=float(default_target),
                step=0.001,
                format="%.3f",
                key="target_return"
            )
            
            # Calculate weights for target return portfolio
            # Formula: w = λ(Σ^(-1)1) + γ(Σ^(-1)μ) where λ = (c - bμₚ)/d and γ = (b - aμₚ)/d
            lam = (c * target_return - b) / d  # Note: this is the original formula, fixed from the existing code
            gamma = - (b * target_return - a) / d  # Fixed from the existing code
            
            
            target_weights = lam * (inv_cov @ user_returns) + gamma * (inv_cov @ ones_vector)
            
            # Apply short-selling constraint if needed
            valid_target = True
            if not allow_short_analytical and np.any(target_weights < 0):
                st.warning("The selected return level requires short selling, which is currently disabled.")
                valid_target = False
            
            # Calculate variance for target return
            target_var = (c * target_return**2 - 2 * b * target_return + a) / d
            
            if target_var <= 0 or np.isnan(target_var) or np.isinf(target_var):
                st.warning("The selected return level results in an invalid portfolio variance.")
                valid_target = False
            else:
                target_vol = np.sqrt(target_var)
                target_sharpe = (target_return - rf_rate) / target_vol if target_vol > 0 else 0
            
            # Create plot for efficient frontier
            fig, ax = plt.subplots(figsize=plot_size)
            
            # Generate points on the efficient frontier
            returns = np.linspace(min_ret, max_ret, 100)
            
            # Calculate variance for each return level
            variances = (c * returns**2 - 2 * b * returns + a) / d
            volatilities = np.sqrt(variances)
            
            # Plot the frontier
            valid_points = ~np.isnan(volatilities) & ~np.isinf(volatilities) & (volatilities > 0)
            ax.plot(volatilities[valid_points], returns[valid_points], 'b-', lw=2, label='Efficient Frontier')
            
            # Plot minimum variance portfolio
            ax.scatter([min_var_vol], [min_var_return], color='blue', s=70, marker='*', label='Min Variance')
            
            # Plot target return portfolio if valid
            if valid_target:
                ax.scatter([target_vol], [target_return], color='orange', s=70, marker='*', label=f'Target Return ({target_return:.3f})')
            
            # Plot tangency portfolio if calculated
            if tangency_return is not None:
                ax.scatter([tangency_vol], [tangency_return], color='purple', s=70, marker='*', label='Max Sharpe Ratio')
                
                # Plot Capital Market Line
                if show_cml_analytical:
                    x_cml = np.linspace(0, max(volatilities[valid_points]) * 1.2, 100)
                    y_cml = rf_rate + (tangency_return - rf_rate) / tangency_vol * x_cml
                    ax.plot(x_cml, y_cml, 'r--', label='Capital Market Line')
            
            # Plot individual assets
            for i in range(len(user_returns)):
                ax.scatter([user_stdevs[i]], [user_returns[i]], color=f'C{i}', s=60, label=f'Asset {asset_names[i]}')
            
            # Plot risk-free asset
            ax.scatter([0], [rf_rate], color='black', s=50, marker='o', label='Risk-free')
            
            ax.set_xlabel('Volatility (Standard Deviation)', fontsize=10)
            ax.set_ylabel('Expected Return', fontsize=10)
            ax.set_title('Analytical Efficient Frontier for Four Assets', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            
            # Set reasonable axis limits to include risk-free rate
            ax.set_xlim(0, max(user_stdevs) * 1.2)
            # Ensure y-axis includes risk-free rate
            y_min = min(rf_rate, min(user_returns)) * 0.9
            y_max = max(user_returns) * 1.3
            ax.set_ylim(y_min, y_max)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Create columns for portfolios
            portfolios_cols = st.columns(3)
            
            # Display minimum variance portfolio details
            with portfolios_cols[0]:
                st.subheader("Minimum Variance Portfolio")
                st.markdown(f"""
                - Return: {min_var_return:.4f} ({min_var_return*100:.2f}%)
                - Volatility: {min_var_vol:.4f} ({min_var_vol*100:.2f}%)
                - Sharpe: {(min_var_return - rf_rate) / min_var_vol:.4f}
                """)
                
                # Create a dataframe for minimum variance weights
                min_var_df = pd.DataFrame({
                    'Asset': asset_names,
                    'Weight': [f"{w:.2%}" for w in min_var_weights]
                }).set_index('Asset')
                st.dataframe(min_var_df, height=150, use_container_width=True)
            
            # Display target return portfolio details if valid
            with portfolios_cols[1]:
                st.subheader("Target Return Portfolio")
                if valid_target:
                    st.markdown(f"""
                    - Return: {target_return:.4f} ({target_return*100:.2f}%)
                    - Volatility: {target_vol:.4f} ({target_vol*100:.2f}%)
                    - Sharpe: {target_sharpe:.4f}
                    """)
                    
                    # Create a dataframe for target weights
                    target_df = pd.DataFrame({
                        'Asset': asset_names,
                        'Weight': [f"{w:.2%}" for w in target_weights]
                    }).set_index('Asset')
                    st.dataframe(target_df, height=150, use_container_width=True)
                else:
                    st.info("Selected return level produces an invalid portfolio with current constraints.")
            
            # Display maximum Sharpe ratio portfolio if available
            with portfolios_cols[2]:
                if tangency_return is not None:
                    st.subheader("Maximum Sharpe Ratio Portfolio")
                    st.markdown(f"""
                    - Return: {tangency_return:.4f} ({tangency_return*100:.2f}%)
                    - Volatility: {tangency_vol:.4f} ({tangency_vol*100:.2f}%)
                    - Sharpe Ratio: {tangency_sharpe:.4f}
                    """)
                    
                    # Create a dataframe for max Sharpe weights
                    max_sharpe_df = pd.DataFrame({
                        'Asset': asset_names,
                        'Weight': [f"{w:.2%}" for w in weights_tan]
                    }).set_index('Asset')
                    st.dataframe(max_sharpe_df, height=150, use_container_width=True)
                else:
                    st.info("Maximum Sharpe ratio portfolio could not be calculated.")
        
        # Display mathematical formulas in expander
        with st.expander("Mathematical Formulation", expanded=False):
            st.subheader("Efficient Frontier Formula")
            st.markdown(r"""
            The equation for the efficient frontier is:

            $$\frac{\sigma^2_p}{1/c} = \frac{(r_p - b/c)^2}{|d|/c^2} + 1$$
            
            This shows the parabolic form of the efficient frontier when properly scaled.
            
            Where:
            - $\sigma^2_p$ is the portfolio variance
            - $r_p$ is the portfolio return
            - $a = \mathbf{\mu}^T \Sigma^{-1} \mathbf{\mu}$
            - $b = \mathbf{1}^T \Sigma^{-1} \mathbf{\mu}$
            - $c = \mathbf{1}^T \Sigma^{-1} \mathbf{1}$
            - $d = ac - b^2$
            
            And:
            - $\mathbf{1}$ is a vector of ones
            - $\mathbf{\mu}$ is the vector of expected returns
            - $\Sigma$ is the covariance matrix
            
            The minimum variance portfolio has return $r_p = b/c$ and is the vertex of the parabola.
            """)
            
            # Display the calculated parameters
            st.subheader("Calculated Parameters")
            st.markdown(f"""
            - a = {a:.6f}
            - b = {b:.6f}
            - c = {c:.6f}
            - d = {d:.6f}
            - b/c = {b/c:.6f} (Global Minimum Variance Return)
            - |d|/c² = {abs(d)/(c**2):.6f}
            """)
        
        # Download portfolio data
        target_returns = np.linspace(min_ret, max_ret, 50)
        portfolio_data = []
        
        for r in target_returns:
            # Calculate weights for this target return
            lam = (c - b * r) / d
            gamma = (b - a * r) / d
            weights = lam * (inv_cov @ ones_vector) + gamma * (inv_cov @ user_returns)
            
            # Skip invalid portfolios
            if np.isnan(weights).any() or np.isinf(weights).any():
                continue
            
            # Apply short-selling constraint if needed
            if not allow_short_analytical and (weights < 0).any():
                continue
            
            # Calculate variance
            variance = (c * r**2 - 2 * b * r + a) / d
            
            # Skip invalid variances
            if variance <= 0 or np.isnan(variance) or np.isinf(variance):
                continue
            
            volatility = np.sqrt(variance)
            
            # Calculate Sharpe ratio
            sharpe = (r - rf_rate) / volatility
            
            # Create a row with weights
            row = {
                'Expected Return': r,
                'Volatility': volatility,
                'Sharpe Ratio': sharpe
            }
            
            # Add weights
            for i, name in enumerate(asset_names):
                row[f'Weight {name}'] = weights[i]
            
            portfolio_data.append(row)
        
        # Create dataframe and download button
        if portfolio_data:
            portfolio_df = pd.DataFrame(portfolio_data)
            
            buffer = io.BytesIO()
            portfolio_df.to_csv(buffer, index=False)
            buffer.seek(0)
            
            st.download_button(
                label="Download Portfolio Data",
                data=buffer,
                file_name="analytical_frontier.csv",
                mime="text/csv",
                key="download_analytical"
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

# =============== BLACK-LITTERMAN MODEL TAB ===============
with tab4:
    st.subheader("Black-Litterman Model")
    
    # Instructions
    st.markdown(r"""
    The Black-Litterman model combines market equilibrium returns with investor views to create an improved 
    estimation of expected returns. This approach helps address some of the limitations of the standard mean-variance optimization:
    
    1. It starts with market equilibrium returns as a neutral prior (using CAPM reverse optimization)
    2. It allows investors to incorporate their own views with different confidence levels
    3. It mitigates issues like input sensitivity and extreme allocations
    
    The model produces a new set of expected returns that can be used in the efficient frontier calculation.
    """)
    
    # Create a layout with columns for inputs and results
    input_col, main_col = st.columns([1, 2])
    
    with input_col:
        # Asset input section
        st.subheader("Asset Parameters")
        
        # Default values for the assets (same as analytical solution tab)
        default_returns = np.array([0.037, 0.020, 0.026, 0.041])
        default_stdevs = np.array([0.114, 0.147, 0.104, 0.105])
        default_corr = np.array([
            [1.000, 0.239, 0.590, 0.501],
            [0.239, 1.000, 0.262, 0.194],
            [0.590, 0.262, 1.000, 0.458],
            [0.501, 0.194, 0.458, 1.000]
        ])
        
        asset_names = ['A', 'B', 'C', 'D']
        
        # Create inputs for asset returns and volatilities
        st.write("Expected Returns and Standard Deviations:")
        cols = st.columns(2)
        
        # Arrays to store user inputs
        user_returns = np.zeros(4)
        user_stdevs = np.zeros(4)
        
        # Create inputs for each asset
        for i in range(4):
            with cols[0]:
                user_returns[i] = st.number_input(
                    f"Return {asset_names[i]} (%)", 
                    value=float(default_returns[i] * 100),
                    step=0.1,
                    format="%.1f",
                    key=f"bl_return_{i}"
                ) / 100
            
            with cols[1]:
                user_stdevs[i] = st.number_input(
                    f"Std {asset_names[i]} (%)", 
                    value=float(default_stdevs[i] * 100),
                    step=0.1,
                    format="%.1f",
                    key=f"bl_std_{i}"
                ) / 100
        
        # Correlation matrix display
        st.write("Correlation Matrix:")
        
        # User inputs for correlation
        user_corr = np.eye(4)  # Initialize with ones on diagonal
        
        # Create correlation matrix inputs
        for i in range(4):
            for j in range(i+1, 4):
                user_corr[i, j] = user_corr[j, i] = st.number_input(
                    f"Corr {asset_names[i]}-{asset_names[j]}", 
                    min_value=-1.0,
                    max_value=1.0,
                    value=float(default_corr[i, j]),
                    step=0.01,
                    format="%.3f",
                    key=f"bl_corr_{i}_{j}"
                )
        
        # Calculate the covariance matrix
        cov_matrix = np.outer(user_stdevs, user_stdevs) * user_corr
        
        # Black-Litterman specific inputs
        st.subheader("Black-Litterman Parameters")
        
        # Option to allow short selling
        allow_short_bl = st.checkbox("Allow Short Selling", value=False, key="allow_short_bl")
        
        # Market capitalization weights (default to equal weights)
        st.write("Market Capitalization Weights (%):")
        market_caps = np.zeros(4)
        total_weight = 0
        
        for i in range(4):
            weight = st.number_input(
                f"Weight {asset_names[i]}", 
                value=25.0,  # Equal weights by default
                min_value=0.0,
                max_value=100.0,
                step=1.0,
                format="%.1f",
                key=f"bl_weight_{i}"
            )
            market_caps[i] = weight
            total_weight += weight
        
        # Normalize weights to sum to 1
        market_caps = market_caps / total_weight if total_weight > 0 else np.ones(4) / 4
        
        # Tau parameter (scalar that adjusts the uncertainty in the prior)
        tau = st.slider(
            "Uncertainty in Prior (τ)", 
            min_value=0.01, 
            max_value=1.0, 
            value=0.05,
            step=0.01,
            format="%.2f"
        )
        
        # Investor views
        st.subheader("Investor Views")
        
        # Allow users to add views
        num_views = st.slider("Number of Views", min_value=0, max_value=3, value=1)
        
        # Initialize view matrices
        P = np.zeros((num_views, 4))  # View pick matrix
        q = np.zeros(num_views)       # View returns
        omega = np.zeros((num_views, num_views))  # Diagonal confidence matrix
        
        # Get views from user
        for v in range(num_views):
            st.write(f"View {v+1}:")
            
            view_col1, view_col2 = st.columns(2)
            
            # For each asset, get the weight in this view
            for i in range(4):
                with view_col1:
                    P[v, i] = st.number_input(
                        f"Weight {asset_names[i]} in View {v+1}", 
                        min_value=-1.0,
                        max_value=1.0,
                        value=1.0 if i == v else 0.0,
                        step=0.1,
                        format="%.1f",
                        key=f"view_weight_{v}_{i}"
                    )
            
            # Get the view's expected return
            with view_col2:
                q[v] = st.number_input(
                    f"Expected Return for View {v+1} (%)",
                    value=float(default_returns[v] * 110),  # 10% higher than default
                    step=0.1,
                    format="%.1f",
                    key=f"view_return_{v}"
                ) / 100
                
                # Confidence in the view (1 = high confidence, 10 = low confidence)
                confidence = st.slider(
                    f"Confidence in View {v+1}",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key=f"view_conf_{v}"
                )
                
                # Convert confidence to omega (lower confidence = higher variance)
                view_variance = (confidence / 10) * (P[v] @ cov_matrix @ P[v])
                omega[v, v] = view_variance
    
    with main_col:
        # Calculate Black-Litterman expected returns
        st.subheader("Black-Litterman Model Results")
        
        # Option to display prior equilibrium returns
        show_prior = st.checkbox("Show Market Equilibrium (Prior) Returns", value=True)
        
        # Display view information in a cleaner format if views exist
        if num_views > 0:
            st.write("Investor Views Summary:")
            views_summary = []
            for v in range(num_views):
                view_str = " + ".join([f"{P[v, i]:.1f} {asset_names[i]}" for i in range(4) if abs(P[v, i]) > 0.001])
                views_summary.append({
                    "View Expression": view_str,
                    "Expected Return (%)": f"{q[v]*100:.1f}%",
                    "Confidence": f"{10-omega[v,v]/np.mean(omega)*5:.1f}/10" if omega[v,v] > 0 else "N/A"
                })
            
            st.dataframe(pd.DataFrame(views_summary), use_container_width=True)
        
        # Calculate CAPM implied equilibrium returns (reverse optimization)
        # Formula: Π = δΣw_mkt where δ is risk aversion and w_mkt are market weights
        risk_aversion = 2.5  # Market price of risk
        pi = risk_aversion * cov_matrix @ market_caps
        
        # Calculate posterior expected returns using Black-Litterman formula
        if num_views > 0:
            # Formula: E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1 Π + P'Ω^-1q]
            try:
                # Calculate precision matrices
                tau_sigma_inv = np.linalg.inv(tau * cov_matrix)
                omega_inv = np.linalg.inv(omega)
                
                # Calculate the posterior precision and returns
                posterior_precision = tau_sigma_inv + P.T @ omega_inv @ P
                posterior_returns = np.linalg.solve(
                    posterior_precision, 
                    tau_sigma_inv @ pi + P.T @ omega_inv @ q
                )
                
                # Calculate posterior covariance (scaled by 1+tau to get the correct variance)
                posterior_covariance = cov_matrix + np.linalg.inv(posterior_precision)
            except np.linalg.LinAlgError:
                st.error("Error: Matrix inversion failed. Please check your inputs, especially your view specifications.")
                posterior_returns = pi  # Default to prior if calculation fails
                posterior_covariance = cov_matrix
        else:
            # If no views, posterior equals prior
            posterior_returns = pi
            posterior_covariance = cov_matrix
        
        # Compare prior and posterior returns
        comparison_data = {
            "Asset": asset_names,
            "Market Implied Returns (%)": [f"{r*100:.2f}" for r in pi],
            "Black-Litterman Returns (%)": [f"{r*100:.2f}" for r in posterior_returns],
            "Original Returns (%)": [f"{r*100:.2f}" for r in user_returns]
        }
        
        st.write("Return Comparison:")
        st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
        
        # Calculate efficient frontier using the Black-Litterman returns
        a_bl, b_bl, c_bl, d_bl, inv_cov_bl, ones_vector = calculate_frontier_params(posterior_returns, posterior_covariance)
        
        if inv_cov_bl is not None:
            # Calculate minimum and maximum returns for the frontier
            min_ret_bl = min(posterior_returns) * 0.5
            max_ret_bl = max(posterior_returns) * 1.5
            
            # Calculate minimum variance portfolio
            min_var_return_bl = b_bl / c_bl
            min_var_vol_bl = np.sqrt(1 / c_bl)
            
            # Calculate efficient frontier graph
            st.subheader("Black-Litterman Efficient Frontier")
            
            # Create plot
            fig, ax = plt.subplots(figsize=plot_size)
            
            # Generate points on the efficient frontier
            returns = np.linspace(min_ret_bl, max_ret_bl, 100)
            
            # Calculate variance for each return level
            variances = (c_bl * returns**2 - 2 * b_bl * returns + a_bl) / d_bl
            volatilities = np.sqrt(np.abs(variances))  # Ensure positive values
            
            # Plot the frontier
            valid_points = ~np.isnan(volatilities) & ~np.isinf(volatilities) & (variances > 0)
            ax.plot(volatilities[valid_points], returns[valid_points], 'b-', lw=2, label='BL Efficient Frontier')
            
            # Plot minimum variance portfolio
            ax.scatter([min_var_vol_bl], [min_var_return_bl], color='blue', s=70, marker='*', label='Min Variance (BL)')
            
            # Calculate and plot tangency portfolio
            if 'rf_rate' in locals():
                # Calculate tangency portfolio
                excess_returns = posterior_returns - rf_rate
                
                # Calculate weights
                weights_bl_tan = inv_cov_bl @ excess_returns
                weights_bl_tan = weights_bl_tan / np.sum(weights_bl_tan)  # Normalize
                
                # Apply short-selling constraint if needed
                if not allow_short_bl:
                    weights_bl_tan = np.maximum(weights_bl_tan, 0)
                    weights_bl_tan = weights_bl_tan / np.sum(weights_bl_tan)  # Re-normalize
                
                # Calculate tangency portfolio return and risk
                tangency_return_bl = weights_bl_tan @ posterior_returns
                tangency_vol_bl = np.sqrt(weights_bl_tan @ posterior_covariance @ weights_bl_tan)
                
                # Plot tangency portfolio
                ax.scatter([tangency_vol_bl], [tangency_return_bl], color='purple', s=70, marker='*', label='Max Sharpe (BL)')
                
                # Plot Capital Market Line
                x_cml = np.linspace(0, max(volatilities[valid_points]) * 1.2, 100)
                y_cml = rf_rate + (tangency_return_bl - rf_rate) / tangency_vol_bl * x_cml
                ax.plot(x_cml, y_cml, 'r--', label='Capital Market Line')
            
            # If user selected to show prior frontier, plot it
            if show_prior:
                # Calculate efficient frontier using market-implied returns
                a_prior, b_prior, c_prior, d_prior, inv_cov_prior, ones_vector = calculate_frontier_params(pi, cov_matrix)
                
                if inv_cov_prior is not None:
                    # Generate points on the prior frontier
                    returns_prior = np.linspace(min(pi) * 0.5, max(pi) * 1.5, 100)
                    
                    # Calculate variance for each return level
                    variances_prior = (c_prior * returns_prior**2 - 2 * b_prior * returns_prior + a_prior) / d_prior
                    volatilities_prior = np.sqrt(np.abs(variances_prior))
                    
                    # Plot the prior frontier
                    valid_points_prior = ~np.isnan(volatilities_prior) & ~np.isinf(volatilities_prior) & (variances_prior > 0)
                    ax.plot(volatilities_prior[valid_points_prior], returns_prior[valid_points_prior], 'g--', lw=1.5, 
                            alpha=0.6, label='Prior Efficient Frontier')
            
            # Plot individual assets
            for i in range(len(posterior_returns)):
                ax.scatter([user_stdevs[i]], [posterior_returns[i]], color=f'C{i}', s=60, label=f'Asset {asset_names[i]} (BL)')
                if show_prior:
                    ax.scatter([user_stdevs[i]], [pi[i]], color=f'C{i}', s=40, alpha=0.5, marker='x')
                
            # Plot risk-free asset
            ax.scatter([0], [rf_rate], color='black', s=50, marker='o', label='Risk-free')
            
            ax.set_xlabel('Volatility (Standard Deviation)', fontsize=10)
            ax.set_ylabel('Expected Return', fontsize=10)
            ax.set_title('Black-Litterman Efficient Frontier', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='best')
            
            # Set reasonable axis limits
            ax.set_xlim(0, max(user_stdevs) * 1.2)
            y_min = min(rf_rate, min(posterior_returns)) * 0.9
            y_max = max(posterior_returns) * 1.3
            ax.set_ylim(y_min, y_max)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Portfolio weights based on Black-Litterman
            st.subheader("Optimal Portfolio Weights")
            
            # Create columns for different optimal portfolios
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("Minimum Variance Weights")
                min_var_weights_bl = inv_cov_bl @ ones_vector / c_bl
                
                # Normalize weights to ensure they sum to 1
                min_var_weights_bl = min_var_weights_bl / np.sum(min_var_weights_bl)
                
                # Create a dataframe for the weights
                min_var_weights_df = pd.DataFrame({
                    'Asset': asset_names,
                    'Weight (%)': [f"{w*100:.2f}" for w in min_var_weights_bl]
                })
                st.dataframe(min_var_weights_df, use_container_width=True)
            
            with col2:
                if 'rf_rate' in locals():
                    st.write("Maximum Sharpe Ratio Weights")
                    
                    # Create a dataframe for the tangency weights
                    tangency_weights_df = pd.DataFrame({
                        'Asset': asset_names,
                        'Weight (%)': [f"{w*100:.2f}" for w in weights_bl_tan]
                    })
                    st.dataframe(tangency_weights_df, use_container_width=True)
                else:
                    st.write("Set risk-free rate to calculate Max Sharpe Ratio")
            
            with col3:
                st.write("Market Weights")
                
                # Create a dataframe for the market weights
                market_weights_df = pd.DataFrame({
                    'Asset': asset_names,
                    'Weight (%)': [f"{w*100:.2f}" for w in market_caps]
                })
                st.dataframe(market_weights_df, use_container_width=True)
            
            # Mathematical explanation of Black-Litterman model
            with st.expander("Black-Litterman Model Details", expanded=False):
                st.markdown(r"""
                ### Black-Litterman Formula
                
                The Black-Litterman model combines two sources of information:
                
                1. **Market Equilibrium (Prior):** $\Pi = \delta \Sigma w_{mkt}$
                   - $\Pi$ is the implied excess return vector
                   - $\delta$ is the risk aversion coefficient
                   - $\Sigma$ is the covariance matrix
                   - $w_{mkt}$ is the market capitalization weights
                
                2. **Investor Views:** Expressed as $P \cdot E[R] = q + \varepsilon$
                   - $P$ is the pick matrix that selects assets for each view
                   - $q$ is the vector of view returns
                   - $\varepsilon$ is the uncertainty in the views, distributed $N(0, \Omega)$
                
                The posterior expected returns are calculated as:
                
                $$E[R] = [(\tau\Sigma)^{-1} + P'\Omega^{-1}P]^{-1} \cdot [(\tau\Sigma)^{-1}\Pi + P'\Omega^{-1}q]$$
                
                where:
                - $\tau$ is a scalar that adjusts the uncertainty in the prior
                - $\Omega$ is the diagonal covariance matrix of view uncertainties
                
                The resulting $E[R]$ represents the Black-Litterman expected returns that can be used
                in standard mean-variance optimization.
                """)
            
            # Download button for Black-Litterman efficient frontier data
            portfolio_data_bl = []
            
            # Generate frontier points
            target_returns_bl = np.linspace(min_ret_bl, max_ret_bl, 50)
            
            for r in target_returns_bl:
                # Calculate weights for this target return
                lam = (c_bl * r - b_bl) / d_bl
                gamma = (b_bl - a_bl * r) / d_bl
                weights = lam * (inv_cov_bl @ posterior_returns) + gamma * (inv_cov_bl @ ones_vector)
                
                # Skip invalid portfolios
                if np.isnan(weights).any() or np.isinf(weights).any():
                    continue
                
                # Apply short-selling constraint if needed
                if not allow_short_bl and (weights < 0).any():
                    continue
                
                # Calculate variance
                variance = (c_bl * r**2 - 2 * b_bl * r + a_bl) / d_bl
                
                # Skip invalid variances
                if variance <= 0 or np.isnan(variance) or np.isinf(variance):
                    continue
                
                volatility = np.sqrt(variance)
                
                # Calculate Sharpe ratio
                sharpe = (r - rf_rate) / volatility if 'rf_rate' in locals() else 0
                
                # Create a row with weights
                row = {
                    'Expected Return': r,
                    'Volatility': volatility,
                    'Sharpe Ratio': sharpe
                }
                
                # Add weights
                for i, name in enumerate(asset_names):
                    row[f'Weight {name}'] = weights[i]
                
                portfolio_data_bl.append(row)
            
            # Create dataframe and download button
            if portfolio_data_bl:
                portfolio_df_bl = pd.DataFrame(portfolio_data_bl)
                
                buffer_bl = io.BytesIO()
                portfolio_df_bl.to_csv(buffer_bl, index=False)
                buffer_bl.seek(0)
                
                st.download_button(
                    label="Download Black-Litterman Portfolio Data",
                    data=buffer_bl,
                    file_name="black_litterman_frontier.csv",
                    mime="text/csv",
                    key="download_bl"
                )