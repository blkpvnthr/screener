import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, risk_models, expected_returns
from datetime import datetime, timedelta

start = datetime.now() - timedelta(days=(365*5))
end = datetime.now()
# Step 1: Load the optimal_portfolio_candidates.csv file
df_candidates = pd.read_csv('optimal_portfolio_candidates.csv')
tickers = df_candidates['ticker'].tolist()

# Step 2: Download historical stock data for the selected tickers using yfinance
data = yf.download(tickers, start=start, end=end)['Adj Close']

# Step 3: Calculate the expected returns and covariance matrix
expected_returns = expected_returns.mean_historical_return(data)
cov_matrix = risk_models.sample_cov(data)

# Step 4: Instantiate the Efficient Frontier class using the expected returns and covariance matrix
ef = EfficientFrontier(expected_returns, cov_matrix)

# Step 5: Find the Minimum Variance Portfolio (MVP)
weights = ef.min_volatility()  # This finds the minimum volatility (MVP) portfolio
cleaned_weights = ef.clean_weights()  # Clean the weights for better readability
min_volatility = ef.portfolio_performance()  # Get the performance of the minimum volatility portfolio

# Step 6: Print the Minimum Variance Portfolio
print("Minimum Variance Portfolio (MVP):")
print(f"Expected return: {min_volatility[0]:.4f}")
print(f"Portfolio volatility (risk): {min_volatility[1]:.4f}")
print(f"Sharpe ratio: {min_volatility[2]:.4f}")

# Step 7: Save MVP Info and Optimal Weights to CSV

# Minimum Variance Portfolio (MVP) information
mvp_info = {
    'Metric': ['Expected return', 'Portfolio volatility (risk)', 'Sharpe ratio'],
    'Value': [min_volatility[0], min_volatility[1], min_volatility[2]]
}

# Create a DataFrame for the MVP metrics
df_mvp_info = pd.DataFrame(mvp_info)

# Optimal Weights for the Minimum Variance Portfolio
optimal_weights = {
    'Ticker': [ticker for ticker, weight in cleaned_weights.items() if weight > 0],
    'Weights': [weight * 100 for ticker, weight in cleaned_weights.items() if weight > 0]
}

# Create a DataFrame for the optimal weights
df_optimal_weights = pd.DataFrame(optimal_weights)

# Step 8: Sort the optimal weights by the "Weights" column in descending order
df_optimal_weights = df_optimal_weights.sort_values(by='Weights', ascending=False).reset_index(drop=True)

# Stocks for The Optimal Portfolio with Index
optimal_portfolio = {
    'Index': list(range(1, len(df_optimal_weights) + 1)),
    'Ticker': df_optimal_weights['Ticker'],
    'Weights': df_optimal_weights['Weights']
}

# Create a DataFrame for the stocks in the optimal portfolio
df_optimal_portfolio = pd.DataFrame(optimal_portfolio)

# Save the MVP info and optimal weights to CSV files
df_mvp_info.to_csv('mvp_info.csv', index=False)
df_optimal_weights.to_csv('optimal_weights.csv', index=False)
df_optimal_portfolio.to_csv('optimal_portfolio.csv', index=False)

# If you want to save all in a single CSV file
with open('mvp_and_optimal_portfolio.csv', 'w') as f:
    f.write("Minimum Variance Portfolio (MVP):\n")
    df_mvp_info.to_csv(f, index=False)
    
    f.write("\nOptimal Weights for the Minimum Variance Portfolio (Sorted by Weights):\n")
    df_optimal_weights.to_csv(f, index=False)
    
    f.write("\nStocks for The Optimal Portfolio:\n")
    df_optimal_portfolio.to_csv(f, index=False)

print("\nStocks for The Optimal Portfolio (Sorted by Weights):")
print(df_optimal_portfolio)
