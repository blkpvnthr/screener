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


# Step 1: Scrape the S&P 500 constituents table from Wikipedia
def scrape_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise Exception("Failed to load page")
    
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the table containing the S&P 500 component stocks
    table = soup.find('table', {'id': 'constituents'})
    
    tickers = []
    if table:
        # Loop through each row in the table
        rows = table.find_all('tr')[1:]  # Skip the header row
        for row in rows:
            ticker = row.find_all('td')[0].text.strip()
            tickers.append(ticker)
    
    return tickers

# Get tickers from S&P 500
tickers = scrape_sp500_tickers()

# Step 2: List to keep track of unavailable tickers
unavailable_tickers = []

# Step 3: Function to download stock data using yfinance and save it in the correct folder
def download_stock_data(ticker, start_date, end_date, folder):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if not stock_data.empty:
            # Save the data to a CSV file in the appropriate folder
            file_name = os.path.join(folder, f"{ticker}_historical_data.csv")
            stock_data.to_csv(file_name)
            print(f"Downloaded and saved data for {ticker} in {folder}")
        else:
            print(f"No data found for {ticker}. Adding to unavailable list.")
            unavailable_tickers.append(ticker)
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        unavailable_tickers.append(ticker)

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
d = datetime.datetime.strptime("2024-09-18", "%Y-%m-%d")
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

# Step 12: Portfolio based on real data + next month prediction
try:
    mu = mean_historical_return(data_for_portfolio_with_prediction, frequency=12)
    S = CovarianceShrinkage(data_for_portfolio_with_prediction, frequency=12).ledoit_wolf()
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))

    # weights = ef.max_sharpe()  # Optional: Use max Sharpe ratio for allocation
    weights = ef.min_volatility()  # Minimize volatility for a more conservative portfolio
    cleaned_weights = ef.clean_weights()
    print(cleaned_weights)
    print('- - -')
    ef.portfolio_performance(verbose=True)

except ValueError as e:
    print(f"Error during portfolio optimization: {e}")