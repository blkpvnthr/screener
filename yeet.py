import os
import pandas as pd
import numpy as np

# Define the directory containing OHLC data for all tickers
all_tickers_dir = 'data/all-tickers'
merged_data_file = 'data/merged_sp500_data.csv'

# Function to calculate cumulative return
def calculate_cumulative_return(data, window):
    return (data.pct_change(window) + 1).prod() - 1

# Function to calculate volatility
def calculate_volatility(data, window):
    return data.pct_change().rolling(window=window).std()

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(cumulative_return, volatility):
    # Align cumulative return and volatility
    aligned_cr, aligned_vol = cumulative_return.align(volatility, join='inner')
    
    # Check if the aligned DataFrames are empty
    if aligned_cr.empty or aligned_vol.empty:
        print("Error: Aligned cumulative return or volatility DataFrame is empty.")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Avoid division by zero by replacing 0 in volatility with NaN
    return aligned_cr / np.where(aligned_vol == 0, np.nan, aligned_vol)

# Function to load OHLC data for each ticker
def load_ohlc_data(ticker, all_tickers_dir):
    historical_file = f"{all_tickers_dir}/{ticker}_historical_data.csv"
    try:
        data = pd.read_csv(historical_file)
        if 'date' in data.columns:
            data.rename(columns={'date': 'Date'}, inplace=True)
        # Ensure the 'Date' column is parsed as dates
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None

# Function to screen stocks based on a metric
def screen_stocks(metric_df, top_n=50, ascending=False):
    if isinstance(metric_df, pd.DataFrame) and not metric_df.empty:
        # Sort and select top_n stocks
        return metric_df.apply(lambda col: col.sort_values(ascending=ascending).head(top_n).index)
    else:
        print(f"Error: Expected a non-empty DataFrame but received {type(metric_df)}.")
        return pd.Index([])

# Step 1: Load the list of tickers from 'merged_sp500_data.csv'
def load_tickers_from_file(merged_data_file):
    try:
        # Load the merged data
        merged_data = pd.read_csv(merged_data_file)
        # The column names in the merged data represent the tickers
        tickers = list(merged_data.columns[1:])  # Skip the 'Date' column
        return tickers
    except Exception as e:
        print(f"Error loading tickers from {merged_data_file}: {e}")
        return []

# Load tickers from the merged data file
tickers = load_tickers_from_file(merged_data_file)

# Initialize a dictionary to store calculated metrics for each ticker
metrics_results = {}

# Define window sizes for calculations
window_sizes = [5, 10, 20, 50]

# Step 2: Calculate cumulative return, volatility, and Sharpe ratio for each stock
for ticker in tickers:
    print(f"Processing {ticker}...")

    # Load OHLC data for the ticker
    stock_data = load_ohlc_data(ticker, all_tickers_dir)
    
    if stock_data is None or stock_data.empty:
        print(f"No data found for {ticker}. Skipping.")
        continue  # Skip if no data
    
    print(f"\nLoaded {ticker} Historical Data:\n", stock_data.head())  # Print head of loaded data for inspection
    
    # Use 'Adj Close' column for calculations
    adj_close_data = stock_data[['Adj Close']].dropna()

    ticker_metrics = {}
    
    # Calculate metrics for each window size
    for window in window_sizes:
        if len(adj_close_data) < window:  # Check if there's enough data for the window
            print(f"Not enough data for window size {window} for {ticker}. Skipping this window.")
            continue
        
        print(f"Calculating metrics for window size: {window} for {ticker}")
        
        cr_df = calculate_cumulative_return(adj_close_data, window)
        volatility_df = calculate_volatility(adj_close_data, window)
        
        # Drop NaNs
        cr_df = cr_df.dropna()
        volatility_df = volatility_df.dropna()
        
        if cr_df.empty or volatility_df.empty:
            print(f"Error: One or more metric DataFrames are empty for window size {window}. Skipping this window for {ticker}.")
            continue
        
        sharpe_df = calculate_sharpe_ratio(cr_df, volatility_df)
        
        if sharpe_df.empty:
            print(f"Error: Sharpe ratio DataFrame is empty for window size {window}. Skipping this window for {ticker}.")
            continue
        
        # Store results for this ticker and window
        ticker_metrics[window] = {
            'Cumulative Return': cr_df,
            'Volatility': volatility_df,
            'Sharpe Ratio': sharpe_df
        }
    
    if ticker_metrics:
        metrics_results[ticker] = ticker_metrics

# Step 3: Screening stocks based on cumulative return, volatility, and Sharpe ratio

# Loop through each window size and apply screening
for window in window_sizes:
    print(f"\nScreening stocks for window size: {window}")
    
    # Collect the data for each ticker
    cumulative_returns = []
    volatility = []
    sharpe_ratios = []
    
    for ticker, ticker_data in metrics_results.items():
        if window in ticker_data:
            cumulative_returns.append(ticker_data[window]['Cumulative Return'])
            volatility.append(ticker_data[window]['Volatility'])
            sharpe_ratios.append(ticker_data[window]['Sharpe Ratio'])
    
    if cumulative_returns and volatility and sharpe_ratios:
        # Concatenate data for each metric
        cr_df = pd.concat(cumulative_returns, axis=1)
        vol_df = pd.concat(volatility, axis=1)
        sr_df = pd.concat(sharpe_ratios, axis=1)
        
        # Perform stock screening for each metric
        top_cr_stocks = screen_stocks(cr_df, top_n=50, ascending=False)  # High cumulative return
        top_vol_stocks = screen_stocks(vol_df, top_n=50, ascending=True)  # Low volatility
        top_sr_stocks = screen_stocks(sr_df, top_n=50, ascending=False)   # High Sharpe ratio
        
        # Print the results (or further process the screened stocks)
        print(f"\nTop 50 stocks by Cumulative Return for window {window}:\n", top_cr_stocks)
        print(f"\nTop 50 stocks by Volatility for window {window}:\n", top_vol_stocks)
        print(f"\nTop 50 stocks by Sharpe Ratio for window {window}:\n", top_sr_stocks)
    else:
        print(f"Error: No data available for window size {window}.")
