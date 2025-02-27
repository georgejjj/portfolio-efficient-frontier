# Portfolio Efficient Frontier Visualization

An interactive web application for visualizing the efficient frontier of investment portfolios using Streamlit and Python.

![Portfolio Efficient Frontier](https://user-images.githubusercontent.com/your-username/path-to-your-image.png)

## Overview

This application allows investors, students, and financial analysts to visualize and interact with the Modern Portfolio Theory's efficient frontier. Users can explore how different parameters affect optimal portfolio allocation through an intuitive interface.

## Features

- **Two-Asset Portfolio Analysis**: 
  - Adjust expected returns and volatilities for two assets
  - Interactively change correlation between assets and observe the efficient frontier shift
  - View the minimum volatility and maximum Sharpe ratio portfolios
  - See the Capital Market Line and risk-free asset

- **Multi-Asset Portfolio Analysis**:
  - Generate random assets with different risk/return profiles
  - Select how many assets to include in your portfolio
  - Visualize thousands of possible portfolio combinations
  - Identify the optimal portfolios for different risk preferences
  - Export portfolio data for further analysis

- **Interactive Visualization**:
  - Adjust plot sizes
  - View detailed portfolio metrics
  - Explore correlation matrices
  - Toggle between allowing/disallowing short selling

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/georgejjj/portfolio-efficient-frontier.git
   cd portfolio-efficient-frontier
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run app.py
   ```

2. The application will open in your default web browser at `http://localhost:8501`

3. Use the sidebar to adjust global parameters and asset-specific details

4. Switch between tabs to explore two-asset or multi-asset frontier analysis

5. For two-asset analysis:
   - Use the correlation slider in the main panel to see how correlation affects the efficient frontier
   - Observe how the minimum volatility and maximum Sharpe ratio portfolios change

6. For multi-asset analysis:
   - Use the portfolio composition slider to select how many assets to include
   - Explore the correlation matrix to understand relationships between assets
   - See how the cloud of possible portfolios forms the efficient frontier

7. Download portfolio data for further analysis using the download buttons

## Technical Background

### Modern Portfolio Theory

This application is based on Modern Portfolio Theory (MPT) developed by Harry Markowitz in 1952. Key concepts visualized include:

- **Efficient Frontier**: The set of optimal portfolios that offer the highest expected return for a defined level of risk (volatility/standard deviation).

- **Minimum Volatility Portfolio**: The portfolio on the efficient frontier with the lowest possible risk.

- **Maximum Sharpe Ratio Portfolio**: The portfolio that offers the best risk-adjusted return, found at the tangent point between the Capital Market Line and the efficient frontier.

- **Capital Market Line (CML)**: A line drawn from the risk-free rate through the tangency portfolio (maximum Sharpe ratio portfolio), representing the optimal combinations of the risk-free asset and the optimal risky portfolio.

### Mathematical Formulations

- **Portfolio Return**: The weighted sum of individual asset returns.
  
- **Portfolio Volatility**: Determined by both individual asset volatilities and their correlations.

- **Sharpe Ratio**: (Expected Return - Risk-Free Rate) / Portfolio Volatility

## Dependencies

- streamlit: Web application framework
- numpy: Numerical computations
- pandas: Data manipulation
- matplotlib: Data visualization
- seaborn: Enhanced data visualization
- scipy: Scientific computing (optimization)

## File Structure

```
portfolio-efficient-frontier/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── README.md           # This documentation
└── data/               # Optional data files
```

## Requirements

Create a `requirements.txt` file with the following contents:

```
streamlit>=1.10.0
numpy>=1.20.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scipy>=1.7.0
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Harry Markowitz for developing Modern Portfolio Theory
- The Streamlit team for creating an excellent framework for data applications 
