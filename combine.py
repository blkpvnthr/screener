import os
import pandas as pd

# Define directories for source data
eod_data_dir = 'data/eod_data/'
historical_data_dir = 'data/historical_data/'
combined_file = 'data/combined.csv'

# Function to process files and extract 'Date' and 'Adjusted Close' columns
def process_files(directory, date_col_name, adjusted_close_col_name):
    all_data = pd.DataFrame()
    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            stock_symbol = filename.split('_')[0]  # Assumes filename starts with stock symbol (e.g., AAPL_eod_data.csv)

            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the required columns exist
                if date_col_name not in df.columns or adjusted_close_col_name not in df.columns:
                    print(f"Warning: Required columns not found in {filename}. Skipping this file.")
                    continue

                # Rename columns to a standard format
                df.rename(columns={date_col_name: 'Date', adjusted_close_col_name: stock_symbol}, inplace=True)
                df = df[['Date', stock_symbol]]  # Keep only 'Date' and adjusted close

                # Append data for this stock to the combined dataframe
                if all_data.empty:
                    all_data = df
                else:
                    all_data = pd.merge(all_data, df, on='Date', how='outer')  # Merge on 'Date'

            except Exception as e:
                print(f"Error processing file {filename}: {e}")
                continue

    return all_data

# Process EOD data (adjusted close is 'adjusted_close', date is 'date')
eod_data = process_files(eod_data_dir, 'date', 'adjusted_close')

# Process historical data (adjusted close is 'Adj Close', date is 'Date')
historical_data = process_files(historical_data_dir, 'Date', 'Adj Close')

# Combine both datasets
combined_df = pd.concat([eod_data, historical_data]).drop_duplicates(subset=['Date']).reset_index(drop=True)

# Convert 'Date' to pandas datetime format
combined_df['Date'] = pd.to_datetime(combined_df['Date'], format='%Y-%m-%d', errors='coerce')

# Drop rows with invalid dates
combined_df.dropna(subset=['Date'], inplace=True)

# Sort by 'Date'
combined_df.sort_values(by='Date', inplace=True)

# Find the earliest date in the combined dataset
earliest_date = combined_df['Date'].min()

# Add a 'Day' column starting from 1 (for the earliest date found)
combined_df['Day'] = (combined_df['Date'] - earliest_date).dt.days + 1

# Reorder columns to have 'Day', 'Date', and then the stock data
combined_df['Date'] = combined_df['Date'].dt.strftime('%d/%m/%Y')  # Format Date as DD/MM/YYYY
combined_df = combined_df[['Day', 'Date'] + [col for col in combined_df.columns if col not in ['Day', 'Date']]]

# Save the combined dataset to a CSV file
combined_df.to_csv(combined_file, index=False)

print(f"Combined dataset created and saved to {combined_file}, with earliest date as {earliest_date.date()}.")
