import yfinance as yf
import pandas as pd
import numpy as np
import os
import datetime
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta

start = datetime.now() - timedelta(days=(365*5))
end = datetime.now()

# Step 1: Load the optimal_portfolio_candidates.csv file
df_candidates = pd.read_csv('optimal_portfolio_candidates.csv')
tickers = df_candidates['ticker'].tolist()

# Step 2: Download historical stock data for the selected tickers
data = yf.download(tickers, start="2020-01-01", end="2024-09-18")['Adj Close']

# Step 4: Get monthly data using yfinance
def get_monthly_data_from_yf(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1mo', progress=False).dropna()
    data['Close-Previous-Month'] = data['Close'].shift(1)
    return data.dropna()

# Get data for all tickers 2012 - 2024
data_ten_years = {}
for ticker in tickers:
    data_ten_years[ticker] = get_monthly_data_from_yf(ticker, '2012-01-01', '2024-09-18')

# Step 5: Function to make prediction for next month's close price
def mlr_predict_close_price(data, ticker):
    dataset = data[ticker]
    
    # Ensure that there is enough data to split
    if len(dataset) == 0:
        print(f"No data available for {ticker}, skipping.")
        return [None, None]
    
    X = dataset[['Close-Previous-Month']].values
    y = dataset['Close'].values
    
    # Ensure there's enough data for train_test_split
    if len(X) < 2:  # At least 2 samples are needed to split the data
        print(f"Not enough data to train-test split for {ticker}, skipping.")
        return [None, None]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0, shuffle=False)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    
    # Predict the next month's price
    next_month_pred = regressor.predict([[X[-1][0]]])  # Use the last available close price for prediction
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    return [next_month_pred, rmse]

# Step 6: Dictionary to store predictions and RMSE for each ticker
predictions_next_month_close_price = {}

# Loop through tickers and predict next month's close price
for ticker in tickers:
    prediction_and_rmse = mlr_predict_close_price(data_ten_years, ticker)
    
    if prediction_and_rmse[0] is not None:  # Check if prediction is valid
        prediction = prediction_and_rmse[0][0]  # Extract the predicted value
        rmse = prediction_and_rmse[1]
        predictions_next_month_close_price[ticker] = prediction
        print(f"{ticker} \t RMSE = {rmse} \t Next Close Price prediction = {prediction}")
    else:
        print(f"Skipping {ticker} due to insufficient data.")

# Step 7: Download data for portfolio analysis, filtering out tickers with failed downloads
valid_tickers = [ticker for ticker in tickers if ticker not in ['BF.B', 'BRK.B']]  # Remove tickers with failed downloads
data_for_portfolio = yf.download(valid_tickers, start='2020-01-01', end='2024-09-18', interval='1mo')['Close'].dropna()

# Step 8: Add a new row for the predicted values
d = datetime.strptime("2024-09-18", "%Y-%m-%d")
new_row = pd.DataFrame(index=[d])
data_for_portfolio_with_prediction = pd.concat([data_for_portfolio, new_row])

# Step 9: Insert the predicted close prices for the next month
for ticker in valid_tickers:
    if ticker in predictions_next_month_close_price:
        data_for_portfolio_with_prediction.loc[d, ticker] = predictions_next_month_close_price[ticker]

# Step 10: Drop columns and rows with entirely missing data
data_for_portfolio_with_prediction.dropna(axis=1, how='all', inplace=True)  # Drop columns with all NaNs (tickers with missing data)
data_for_portfolio_with_prediction.dropna(axis=0, how='all', inplace=True)  # Drop rows with all NaNs

# Step 11: Ensure there are no missing values before portfolio optimization
if data_for_portfolio_with_prediction.isnull().values.any():
    print("Filling missing data in the DataFrame...")
    data_for_portfolio_with_prediction = data_for_portfolio_with_prediction.ffill()

# Step 12: Limit to top 5 stocks based on highest predicted next month's close price
# Sort tickers by the predicted close price for the next month
sorted_tickers_by_prediction = sorted(predictions_next_month_close_price.items(), key=lambda x: x[1], reverse=True)

# Select the top 5 tickers with the highest predicted price
top_5_tickers = [ticker for ticker, _ in sorted_tickers_by_prediction[:5]]

# Filter the portfolio data to include only these top 5 tickers
data_for_portfolio_with_prediction_top_5 = data_for_portfolio_with_prediction[top_5_tickers]

# Step 12: Portfolio based on real data + next month prediction for top 5 stocks
try:
    mu = mean_historical_return(data_for_portfolio_with_prediction_top_5, frequency=12)
    S = CovarianceShrinkage(data_for_portfolio_with_prediction_top_5, frequency=12).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

    weights = ef.max_sharpe()  # Optional: Use max Sharpe ratio for allocation
    #weights = ef.min_volatility()  # Minimize volatility for a more conservative portfolio
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    for ticker, weight in cleaned_weights.items():
        df = yf.download(ticker, start=start, end=end)
        print(f"{ticker}\t{df}")    
    print('- - -')
    ef.portfolio_performance(verbose=True)

except ValueError as e:
    print(f"Error during portfolio optimization: {e}")
    
"""
Minimize volatility for a more conservative portfolio:

OrderedDict({'BKNG': 0.11291, 'AZO': 0.35686, 'MTD': 0.28791, 'TDG': 0.05729, 'ORLY': 0.18503})
- - -
Expected annual return: 20.7%
Annual volatility: 20.5%
Sharpe Ratio: 0.91

--------------------------------------------------------------------------------
Max Sharpe ratio for allocation:
OrderedDict({'BKNG': 0.0679, 'AZO': 0.57803, 'MTD': 0.0908, 'TDG': 0.0, 'ORLY': 0.26327})
- - -
Expected annual return: 23.7%
Annual volatility: 22.0%
Sharpe Ratio: 0.99

"""