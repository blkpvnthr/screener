# Screener

**Screener** is a Python-based tool designed for analyzing the S&P 500 stocks. It calculates various financial metrics such as expected returns, standard deviation, variance, and covariance, and identifies optimal portfolio weights for minimum variance portfolios (MVP). The project leverages historical data, performs financial analysis, and saves results in structured CSV files for further examination and use.

## Project Structure

The project is divided into several sequential steps, each focusing on a specific aspect of stock analysis. The following is a list of the main files and their functions:

1. **Download SP500 symbols and their historical data**
    - Download historical stock data for S&P 500 symbols.
    - Save the historical data for each stock.

2. **Calculate Expected Return, Standard Deviation, and Variance for each stock**
    - Compute the expected return, standard deviation, and variance for each stock based on historical data.
    - Filter stocks with positive expected returns and save the metrics in a DataFrame.
    - Save the results to `sp500_stock_metrics.csv`.

3. **Analyze Positive Expected Return Stocks**
    - Load the `sp500_stock_metrics.csv` dataset.
    - Filter stocks where the expected return is greater than variance.
    - Add a 'Comparison' column to represent the difference between expected return and variance.
    - Save the filtered dataset to `filtered_positive_returns.csv`.

4. **Optimal Portfolio Candidates Analysis**
    - Load `optimal_portfolio_candidates.csv`.
    - Download historical stock data for selected tickers.
    - Calculate the percentage returns and expected returns for each stock.
    - Calculate the covariance between each pair of stocks and display the covariance matrix.
    - Save the covariance matrix to `covariance_matrix.csv`.

5. **Minimum Variance Portfolio Calculation**
    - Load `optimal_portfolio_candidates.csv` and historical data using yfinance.
    - Calculate expected returns and covariance matrix for each stock.
    - Instantiate the `Efficient Frontier` class with the calculated expected returns and covariance matrix.
    - Determine the Minimum Variance Portfolio (MVP).
    - Save MVP info and optimal weights to `mvp_weights.csv` and `mvp_info.csv`.

6. **Final Consolidation**
    - Consolidate all results, including metrics and MVP information, into a single CSV file for easy access and analysis.

## Getting Started

To run this project, you will need Python 3 and the following packages:

- `yfinance`
- `pandas`
- `numpy`
- `cvxpy`
- `matplotlib` (for visualizations if needed)

### Installation

To install dependencies, run:

```bash
pip install yfinance pandas numpy cvxpy matplotlib
