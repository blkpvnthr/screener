import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import yfinance as yf
import os

# Step 1: Scrape the S&P 500 constituent changes table from Wikipedia
url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#Selected_changes_to_the_list_of_S&P_500_components'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')

# Locate the "Selected changes" table - assuming it's the second wikitable on the page
tables = soup.find_all('table', {'class': 'wikitable'})
changes_table = tables[1]  # The second table contains the constituent changes

# Step 2: Parse the table to extract relevant data (Date, Added Ticker, Removed Ticker, etc.)
data = []
for row in changes_table.find_all('tr')[1:]:
    cols = [col.text.strip() for col in row.find_all('td')]  # Extract text from all columns in the row
    
    if len(cols) < 6:
        # Skip rows that don't have the expected 6 columns
        continue
    
    # Step 3: Extract relevant columns for tickers, securities, and dates
    try:
        date = cols[0]               # Date of change
        added_ticker = cols[1]        # Ticker of the added company
        added_security = cols[2]      # Name of the added company
        removed_ticker = cols[3]      # Ticker of the removed company
        removed_security = cols[4]    # Name of the removed company
        reason = cols[5]              # Reason for the change
        
        # Clean and parse the dates (if they are valid)
        date_dt = pd.to_datetime(date, errors='coerce')
        
        data.append([date_dt, added_ticker, added_security, removed_ticker, removed_security, reason])

    except Exception as e:
        print(f"Error parsing row: {e}")
        continue

# Step 4: Create a DataFrame to store the parsed data
sp500_changes_df = pd.DataFrame(data, columns=['Date', 'Added Ticker', 'Added Security', 'Removed Ticker', 'Removed Security', 'Reason'])

# Step 5: Save the DataFrame to a CSV file for future reference
sp500_changes_df.to_csv('data/sp500_changes.csv', index=False)

# Step 6: Create directories for added and removed tickers if they don't already exist
os.makedirs('added_tickers', exist_ok=True)
os.makedirs('removed_tickers', exist_ok=True)

# Step 7: List to keep track of unavailable tickers
unavailable_tickers = []

# Step 8: Function to download stock data using yfinance and save it in the correct folder
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
            unavailable_tickers.append(ticker)  # Track unavailable tickers
    except Exception as e:
        print(f"Failed to download data for {ticker}: {e}")
        unavailable_tickers.append(ticker)  # Track unavailable tickers in case of failure

# Step 9: Set to keep track of tickers that have already been processed
processed_tickers = set()

# Step 10: Iterate over the DataFrame and download historical data for both Added and Removed tickers
for index, row in sp500_changes_df.iterrows():
    change_date = row['Date']  # 'Date' column is used for both added and removed tickers
    
    # Ensure there is a valid date
    if pd.isna(change_date):
        continue

    # Download historical data for the 'Removed Ticker' first (if it exists)
    removed_ticker = row['Removed Ticker']
    if pd.notna(removed_ticker) and removed_ticker not in processed_tickers:
        end_date = change_date
        start_date = end_date - timedelta(days=365)  # Limit to one year before the removal date
        download_stock_data(removed_ticker, start_date, end_date, 'removed_tickers')
        processed_tickers.add(removed_ticker)

    # Download historical data for the 'Added Ticker' (if it hasn't been processed as removed)
    added_ticker = row['Added Ticker']
    if pd.notna(added_ticker) and added_ticker not in processed_tickers:
        end_date = datetime.today()  # Use today's date for current tickers
        start_date = end_date - timedelta(days=365)  # Limit to one year before the added date
        download_stock_data(added_ticker, start_date, end_date, 'added_tickers')
        processed_tickers.add(added_ticker)

# Step 11: Save the unavailable tickers to a CSV file
if unavailable_tickers:
    os.makedirs('data/delisted', exist_ok=True)
    unavailable_df = pd.DataFrame(unavailable_tickers, columns=['Ticker'])
    unavailable_df.to_csv('data/delisted/delisted.csv', index=False)
    print(f"Saved unavailable tickers to data/delisted/delisted.csv")
