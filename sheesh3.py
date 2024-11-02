import pandas as pd
import yfinance as yf
import numpy as np
import datetime
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier

# Step 1: Load the optimal portfolio candidates CSV file
optimal_portfolio_candidates = pd.read_csv('optimal_portfolio_candidates.csv')

# Step 2: Sort by 'Expected Returns' and select the top 5 stocks
top_5_stocks = optimal_portfolio_candidates.sort_values(by='Expected Returns', ascending=False).head(5)

# Step 3: Extract the tickers for the top 5 stocks
top_5_tickers = top_5_stocks['ticker'].tolist()

# Step 4: Download data for portfolio analysis, limited to the top 5 stocks
data_for_portfolio = yf.download(top_5_tickers, start='2020-01-01', end='2024-09-18', interval='1mo')['Close'].dropna()
tickers = data_for_portfolio['ticker'].tolist()
# Step 5: Add a new row for the predicted values
d = datetime.datetime.strptime("2024-09-18", "%Y-%m-%d")
new_row = pd.DataFrame(index=[d])
data_for_portfolio_with_prediction = pd.concat([data_for_portfolio, new_row])

# Step 6: Dictionary to store predictions and RMSE for each ticker
predictions_next_month_close_price = {}

# Loop through tickers and predict next month's close price
for ticker in tickers:
    prediction_and_rmse = data_for_portfolio_with_prediction(data_ten_years, ticker)
    
    if prediction_and_rmse[0] is not None:  # Check if prediction is valid
        prediction = prediction_and_rmse[0][0]  # Extract the predicted value
        rmse = prediction_and_rmse[1]
        predictions_next_month_close_price[ticker] = prediction
        print(f"{ticker} \t RMSE = {rmse} \t Next Close Price prediction = {prediction}")
    else:
        print(f"Skipping {ticker} due to insufficient data.")

# Step 6: Insert the predicted close prices for the next month
for ticker in top_5_tickers:
    if ticker in predictions_next_month_close_price:
        data_for_portfolio_with_prediction.loc[d, ticker] = predictions_next_month_close_price[ticker]

# Step 7: Drop columns and rows with entirely missing data
data_for_portfolio_with_prediction.dropna(axis=1, how='all', inplace=True)
data_for_portfolio_with_prediction.dropna(axis=0, how='all', inplace=True)

# Step 8: Ensure there are no missing values before portfolio optimization
if data_for_portfolio_with_prediction.isnull().values.any():
    data_for_portfolio_with_prediction = data_for_portfolio_with_prediction.ffill()

# Step 9: Portfolio based on real data + next month prediction
try:
    mu = mean_historical_return(data_for_portfolio_with_prediction, frequency=12)
    S = CovarianceShrinkage(data_for_portfolio_with_prediction, frequency=12).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

    # weights = ef.max_sharpe()  # Optional: Use max Sharpe ratio for allocation
    weights = ef.min_volatility()  # Minimize volatility for a more conservative portfolio
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    ef.portfolio_performance(verbose=True)

except ValueError as e:
    print(f"Error during portfolio optimization: {e}")
