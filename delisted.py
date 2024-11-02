import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# Step 1: Load the delisted tickers from data/delisted/delisted.csv
delisted_file_path = 'data/delisted/delisted.csv'

# Ensure the directory exists
if not os.path.exists(delisted_file_path):
    raise FileNotFoundError(f"File not found: {delisted_file_path}")

# Load the CSV
delisted_df = pd.read_csv(delisted_file_path)
tickers = delisted_df['Ticker'].tolist()

# Step 2: Define the function to fetch historical stock data from the EOD Historical Data API
api_token = '66e732112b4d75.52559694'  # Replace with your actual API key

def download_eod_data(symbol, start_date, end_date, api_token):
    url = f"https://eodhd.com/api/eod/{symbol}?from={start_date}&to={end_date}&period=d&api_token={api_token}&fmt=json"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error fetching data for {symbol}: {response.status_code}")
        return None

# Step 3: Create a directory to save the historical data for each ticker
output_dir = 'data/delisted/historical_data'
os.makedirs(output_dir, exist_ok=True)

# Step 4: Iterate over the tickers and download their historical data
for ticker in tickers:
    print(f"Fetching data for {ticker}...")

    # Set the date range (one year of data)
    end_date = datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')

    # Fetch the data from the EOD API
    stock_data = download_eod_data(ticker, start_date, end_date, api_token)
    
    # Step 5: Save the data to a CSV file if the data was fetched successfully
    if stock_data:
        output_file = os.path.join(output_dir, f"{ticker}_eod_data.csv")
        pd.DataFrame(stock_data).to_csv(output_file, index=False)
        print(f"Saved historical data for {ticker} to {output_file}")
    else:
        print(f"No data for {ticker}")

print("Data download complete.")
