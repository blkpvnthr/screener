import pandas as pd

"""Using sp500_stock_metrics.csv define candidates for optimal portfolio using 
comparison between expected returns and the risk from each stock. 
If the expected returns from the stock bigger than the
variance from the stock, then the stock will become the candidate for optimal
portfolio and will be used in the next steps."""

# Step 1: Load the sp500_stock_metrics.csv dataset
df_metrics = pd.read_csv('sp500_stock_metrics.csv')

# Step 2: Filter the stocks where Expected Returns > Variance
df_filtered = df_metrics[df_metrics['Expected Return'] > df_metrics['Variance']]

# Step 3: Create a new 'Comparison' column to represent the comparison between Expected Return and Variance
df_filtered['Comparison'] = df_filtered.apply(lambda row: f"{row['Expected Return']:.6f} > {row['Variance']:.6f}", axis=1)

# Step 4: Create the final DataFrame with the required columns: 'index', 'Symbol', 'Expected Returns', 'Comparison', and 'Variance'
df_final = df_filtered[['Symbol', 'Expected Return', 'Comparison', 'Variance']].reset_index(drop=True)

# Rename columns for clarity
df_final = df_final.rename(columns={'Symbol': 'ticker', 'Expected Return': 'Expected Returns'})

# Add an index column starting from 1
df_final.index += 1

# Display the resulting DataFrame
print(df_final)

# Save the filtered dataset to a new CSV file
df_final.to_csv('optimal_portfolio_candidates.csv', index_label='index')