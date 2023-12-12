# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 00:09:22 2023

@author: zhang
"""

import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv("C:/Users/zhang/Downloads/CFAR/10.18/cleaning_data2018-2022.csv")

df_rf = pd.read_csv("C:/Users/zhang/Downloads/CFAR/10.2/F-F_Research_Data_Factors.CSV",skiprows = 3)

df_rf = df_rf.rename(columns ={ 'Unnamed: 0': "DATE"})

df_rf = df_rf[['DATE','RF']]

# Clean DATE column
df_rf['DATE'] = df_rf['DATE'].str.strip()  # Remove leading and trailing spaces

# Fill NaN values with placeholder
placeholder = "INVALID"
df_rf['DATE'].fillna(placeholder, inplace=True)

# Keep only rows where 'DATE' has numeric values
df_rf = df_rf[df_rf['DATE'].str.isnumeric()]

# Convert DATE to datetime format and then to the desired string format
df_rf['DATE'] = pd.to_datetime(df_rf['DATE'], format='%Y%m', errors='coerce')
df_rf['DATE'] = df_rf['DATE'].dt.strftime('%Y-%m')

# Clean RF column
df_rf['RF'] = df_rf['RF'].str.strip()  # Remove leading and trailing spaces
df_rf['RF'] = pd.to_numeric(df_rf['RF'], errors='coerce')  # Convert to numeric, set invalid entries to NaN

# Drop NaN values
df_rf.dropna(subset=['DATE', 'RF'], inplace=True)

df = df.iloc[:,1:]

df_sorted = df.sort_values(by=['DATE', 'SLOPE'], ascending=[True, True])

df['DATE'] = pd.to_datetime(df['DATE'])

df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month

def filter_tickers_within_2_std(df):
    # Calculate the natural logarithm of CAP
    df['LOG_CAP'] = np.log(df['CAP'])

    # Compute monthly log mean and log std of CAP across all tickers
    df['LOG_CAP_MEAN'] = df.groupby(['YEAR', 'MONTH'])['LOG_CAP'].transform('mean')
    df['LOG_CAP_STD'] = df.groupby(['YEAR', 'MONTH'])['LOG_CAP'].transform('std')

    # Define the bounds
    lower_bound = df['LOG_CAP_MEAN'] - 2 * df['LOG_CAP_STD']
    upper_bound = df['LOG_CAP_MEAN'] + 2 * df['LOG_CAP_STD']

    # Filter the dataframe
    df_filtered = df[(df['LOG_CAP'] >= lower_bound) & (df['LOG_CAP'] <= upper_bound)]

    return df_filtered

# Apply the function to df_with_portfolios
df = filter_tickers_within_2_std(df)


def divide_into_portfolios(group):
    group['PORTFOLIO'] = pd.qcut(group['SLOPE'], q=5, labels=False)
    return group

# Apply the function to each YEAR-MONTH group
df_with_portfolios = df.groupby(['YEAR', 'MONTH']).apply(divide_into_portfolios)

# Convert column names to uppercase
df_with_portfolios.columns = df_with_portfolios.columns.str.upper()

# Get unique combinations of years and months in the dataset
year_month_combinations = df_with_portfolios[['YEAR', 'MONTH']].drop_duplicates()

# Initialize empty dictionaries to store results
port1_next_month_dict = {}
port5_next_month_dict = {}

# Iterate through all year-month combinations
for index, row in year_month_combinations.iterrows():
    current_year = row['YEAR']
    current_month = row['MONTH']

    # Define the next year and month
    next_year = current_year + 1 if current_month == 12 else current_year
    next_month = (current_month % 12) + 1
    
    
    # Get the tickers in portfolio 1 and portfolio 5 for the current month
    port1_tickers = df_with_portfolios[(df_with_portfolios['PORTFOLIO'] == 0) & (df_with_portfolios['YEAR'] == current_year) & (df_with_portfolios['MONTH'] == current_month)]['TICKER'].unique()
    port5_tickers = df_with_portfolios[(df_with_portfolios['PORTFOLIO'] == 4) & (df_with_portfolios['YEAR'] == current_year) & (df_with_portfolios['MONTH'] == current_month)]['TICKER'].unique()

    # Check if the next month exists in the data
    if (next_year, next_month) in zip(df_with_portfolios['YEAR'], df_with_portfolios['MONTH']):
        # Filter the DataFrame to get the next month's data
        next_month_data = df_with_portfolios[(df_with_portfolios['YEAR'] == next_year) & (df_with_portfolios['MONTH'] == next_month)]

        # Check if the tickers in portfolio 1 have data in the next month
        port1_next_month = next_month_data[next_month_data['TICKER'].isin(port1_tickers)]

        # Check if the tickers in portfolio 5 have data in the next month
        port5_next_month = next_month_data[next_month_data['TICKER'].isin(port5_tickers)]

        # Store the results in dictionaries
        port1_next_month_dict[(current_year, current_month)] = port1_next_month
        port5_next_month_dict[(current_year, current_month)] = port5_next_month
        
# Sort the results by year and month
port1_next_month_dict = dict(sorted(port1_next_month_dict.items()))
port5_next_month_dict = dict(sorted(port5_next_month_dict.items()))

strategy_returns = {}

# Calculate strategy returns: Port1 - Port5
for (year, month), port1_data in port1_next_month_dict.items():
    port5_data = port5_next_month_dict[(year, month)]

    # Calculate the monthly returns for each portfolio
    port1_monthly_return = port1_data['MTHRET'].mean()
    port5_monthly_return = port5_data['MTHRET'].mean()

    # Calculate the strategy return: Port1 - Port5
    strategy_return = port1_monthly_return - port5_monthly_return

    # Store the strategy return
    strategy_returns[(year, month)] = strategy_return

# Convert the results to a DataFrame
strategy_returns_df = pd.DataFrame(list(strategy_returns.items()), columns=['Year_Month', 'Strategy_Return'])

# Create a date column with the desired format
strategy_returns_df['Date'] = strategy_returns_df['Year_Month'].apply(lambda x: f"{x[0]}-{x[1]:02d}")

# Add one month to each date
strategy_returns_df['Date'] = pd.to_datetime(strategy_returns_df['Date']) + pd.DateOffset(months=1)

# Format the date column to show only the year and month
strategy_returns_df['Date'] = strategy_returns_df['Date'].dt.strftime('%Y-%m')

# Reorder the columns
strategy_returns_df = strategy_returns_df[['Date', 'Strategy_Return']]

strategy_returns_df["Strategy_Return"] = strategy_returns_df["Strategy_Return"]*100

strategy_returns_df.columns = strategy_returns_df.columns.str.upper()
merged_df = pd.merge(df_rf,strategy_returns_df, on = 'DATE', how = 'inner' )
sharpe_ratio = (merged_df['STRATEGY_RETURN'].mean()-merged_df['RF'].mean())/merged_df['STRATEGY_RETURN'].std()

print(strategy_returns_df.describe())

# Extract the "Strategy_Return" column as a NumPy array
returns = strategy_returns_df['STRATEGY_RETURN'].values

# Perform a one-sample t-test assuming a population mean of 0
t_statistic, p_value = stats.ttest_1samp(returns, 0)

# Print the t-statistic and p-value
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)
print("sharpe_ratio:",sharpe_ratio)
