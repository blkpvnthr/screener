import pandas as pd
import yfinance as yf
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
from datetime import datetime, timedelta

start = datetime.now() - timedelta(days=(365*5))
end = datetime.now()
"""Covariance is a measure the direction of movement from 2 variables. A positive covariance
means that the combination of two stocks in an exclusive portfolio tends to move in the same
direction. A negative covariance shows that two stocks move in opposite directions,in one
direction, if one stock increases in profit the other will decrease in return. 
create a covariance matrix using optimal_portfolio_candidates.csv"""

# Step 1: Load the optimal_portfolio_candidates.csv file
df_candidates = pd.read_csv('optimal_portfolio_candidates.csv')
tickers = df_candidates['ticker'].tolist()

# Step 2: Download historical stock data for the selected tickers
data = yf.download(tickers, start=start, end=end)['Adj Close']

# Step 3: Calculate the percentage returns for each stock
returns = data.pct_change().dropna()

# Step 4: Calculate the expected return (mean of the returns) for each stock
expected_returns = returns.mean()

# Step 5: Function to calculate covariance between two stocks using the given formula
def calculate_covariance(returns_a, returns_b, expected_return_a, expected_return_b):
    n = len(returns_a)  # Number of observations
    covariance = sum((returns_a - expected_return_a) * (returns_b - expected_return_b)) / n
    return covariance

# Step 6: Create an empty DataFrame to store the covariance values
cov_matrix = pd.DataFrame(index=tickers, columns=tickers)

# Step 7: Calculate the covariance for each pair of stocks
for stock_a in tickers:
    for stock_b in tickers:
        returns_a = returns[stock_a]
        returns_b = returns[stock_b]
        expected_return_a = expected_returns[stock_a]
        expected_return_b = expected_returns[stock_b]
        
        cov_matrix.loc[stock_a, stock_b] = calculate_covariance(returns_a, returns_b, expected_return_a, expected_return_b)

# Step 8: Display the covariance matrix
cov_matrix = cov_matrix.astype(float).round(6)  # Convert to float and round to 6 decimal places
print(cov_matrix)
print("\nCovariance matrix saved to custom_covariance_matrix.csv\n")

# Step 9: Save the covariance matrix to a CSV file
cov_matrix.to_csv('custom_covariance_matrix.csv')