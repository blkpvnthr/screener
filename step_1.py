import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from datetime import datetime, timedelta

start = datetime.now() - timedelta(days=(365*5))
end = datetime.now()

# Step 1: Download SP500 symbols and their historical data
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

# Step 2: Download historical data for each stock
data = yf.download(sp500_tickers, start=start, end=end)['Adj Close']

# Drop columns where the entire series is NaN
data = data.dropna(axis=1, how='all')

# Step 3: Calculate Returns for each stock
returns = data.pct_change(fill_method=None).dropna()

# Step 4: Function to calculate expected return, variance, and standard deviation
def calculate_metrics(stock_returns):
    expected_return = stock_returns.mean()
    variance = stock_returns.var()
    std_dev = stock_returns.std()
    return expected_return, std_dev, variance

# Calculate Expected Return, Standard Deviation, and Variance for each stock
metrics = []
for symbol in returns.columns:
    stock_returns = returns[symbol]
    if stock_returns.dropna().shape[0] > 1:  # Ensure we have enough data points
        expected_return, std_dev, variance = calculate_metrics(stock_returns)
        metrics.append([symbol, expected_return, std_dev, variance])

# Save the results in a DataFrame
df_metrics = pd.DataFrame(metrics, columns=['Symbol', 'Expected Return', 'Standard Deviation', 'Variance'])

# Step 5: Calculate Covariance Matrix
cov_matrix = returns.cov()

# Ensure covariance matrix is symmetric by forcing symmetry
cov_matrix = (cov_matrix + cov_matrix.T) / 2

# Step 6: Calculate Portfolio Weights using PyPortfolioOpt
mean_returns = expected_returns.mean_historical_return(data)
cov_matrix = risk_models.sample_cov(data)

try:
    ef = EfficientFrontier(mean_returns, cov_matrix)
    weights = ef.max_sharpe()  # Maximize the Sharpe ratio for optimal weights
    cleaned_weights = ef.clean_weights()

    # Step 7: Calculate Expected Return and Risk (Standard Deviation) for the portfolio
    portfolio_return, portfolio_std_dev, _ = ef.portfolio_performance()

    # Print Portfolio Details
    print("Optimal Weights:", cleaned_weights)
    print(f"Expected Portfolio Return: {portfolio_return}")
    print(f"Portfolio Risk (Std Dev): {portfolio_std_dev}")
except ValueError as e:
    print(f"Optimization failed: {e}")

# Filter stocks with positive expected returns
df_positive_returns = df_metrics[df_metrics['Expected Return'] > 0]

# Display the metrics and positive expected return stocks
print(df_positive_returns)

# Save the results to CSV
df_metrics.to_csv('sp500_stock_metrics.csv', index=False)
