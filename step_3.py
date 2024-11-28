import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

# Define the start and end dates for historical data
start = datetime.now() - timedelta(days=(365 * 5))
end = datetime.now()

# Step 1: Load the optimal_portfolio_candidates.csv file
# Load the CSV file
file_path = 'optimal_portfolio_candidates.csv'
df_candidates = pd.read_csv(file_path)

# Clean the 'ticker' column to remove punctuation
# In this case, there doesn't seem to be punctuation, but if needed:
df_candidates['ticker'] = df_candidates['ticker'].str.replace(r'[^\w\s]', '', regex=True)

# Extract the tickers as a list
tickers = df_candidates['ticker'].tolist()

# Save tickers to a file, separated by new lines
with open('tickers.txt', 'w') as f:
    f.write('\n'.join(tickers))

print("Tickers extracted and saved to 'tickers.txt':")
print("\n".join(tickers))
# Step 2: Download historical stock data for the selected tickers
try:
    data = yf.download(tickers, start=start, end=end)['Adj Close']
except Exception as e:
    print(f"Error downloading data: {e}")
    data = pd.DataFrame()  # Fallback to empty DataFrame

# Drop tickers with insufficient data
data = data.dropna(axis=1, how='any')  # Remove stocks with missing data
tickers = data.columns.tolist()  # Update the ticker list to include only valid tickers

# Step 3: Calculate the percentage returns for each stock
returns = data.pct_change().dropna()

# Step 4: Calculate the expected return (mean of the returns) for each stock
expected_returns = returns.mean()

# Step 5: Function to calculate covariance between two stocks using the given formula
def calculate_covariance(returns_a, returns_b, expected_return_a, expected_return_b):
    n = len(returns_a)
    if n == 0:
        return np.nan  # Return NaN if there is no data to calculate covariance
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
        
        # Safely calculate covariance
        cov_matrix.loc[stock_a, stock_b] = calculate_covariance(returns_a, returns_b, expected_return_a, expected_return_b)

# Step 8: Display the covariance matrix
cov_matrix = cov_matrix.astype(float).round(6)  # Convert to float and round to 6 decimal places
print(cov_matrix)
print("\nCovariance matrix saved to custom_covariance_matrix.csv\n")

# Step 9: Save the covariance matrix to a CSV file
cov_matrix.to_csv('custom_covariance_matrix.csv')
