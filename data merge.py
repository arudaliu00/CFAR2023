# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import os

# Define the directory path where your CSV files are stored
directory_path = 'C:/Users/zhang/Downloads/CFAR/10.18/2018-2022'

# Get a list of all files in the directory
all_files = os.listdir(directory_path)

# Filter out option and stock files
option_files = [f for f in all_files if 'option_' in f]
stock_files = [f for f in all_files if 'stock_' in f]

# Sort the files to ensure they are processed in order
option_files.sort()
stock_files.sort()

# Check if the number of option and stock files are the same
if len(option_files) != len(stock_files):
    raise ValueError("Mismatch in the number of option and stock files.")

# List to store all processed dataframes
all_processed_dfs = []

# Loop through each pair of option and stock files
for i in range(len(option_files)):
    option_file_path = os.path.join(directory_path, option_files[i])
    stock_file_path = os.path.join(directory_path, stock_files[i])

    # Read the CSV files using pandas
    dtype = {"COLUMN_NAME_WITH_MIXED_TYPES": str}
    df_option = pd.read_csv(option_file_path, dtype=dtype)
    df_stock_monthly = pd.read_csv(stock_file_path)

    # Convert all column names to uppercase
    df_option.columns = df_option.columns.str.upper()
    df_stock_monthly.columns = df_stock_monthly.columns.str.upper()
    df_stock_monthly['DATE'] = pd.to_datetime(df_stock_monthly['MTHCALDT'])


    df_stock_monthly['CAP'] = abs(df_stock_monthly['MTHPRC']) * df_stock_monthly['SHROUT']

    # Select the desired columns
    df_option = df_option[["DATE",  "DELTA", "IMPL_VOLATILITY", "CP_FLAG", "TICKER"]]


    # Filter Rows Where DATE is the Last Day of the Month
    df_option['DATE'] = pd.to_datetime(df_option['DATE'])
    
    # Calculate the absolute value of 'DELTA' and filter rows
    df_option['DELTA_ABS'] = df_option['DELTA'].abs()
    
    # Filter rows where DELTA_ABS is equal to 50
    df_option_filtered = df_option[df_option['DELTA_ABS'] == 50]
    
    
    df_option_filtered = df_option_filtered.dropna(subset=['IMPL_VOLATILITY'])

    def calculate_slope(group):
          put_iv = group[group['CP_FLAG'] == 'P']['IMPL_VOLATILITY'].values[0]
          call_iv = group[group['CP_FLAG'] == 'C']['IMPL_VOLATILITY'].values[0]
          return put_iv - call_iv
 
    # Calculate the slopes for each group and reset the index
    slopes = df_option_filtered.groupby(['TICKER', 'DATE']).apply(calculate_slope).reset_index()
    
    # Rename the calculated column to 'SLOPE'
    slopes.rename(columns={0: 'SLOPE'}, inplace=True)

    # Merge the calculated slopes with the original DataFrame
    df_option_filtered = df_option_filtered.merge(slopes, on=['TICKER', 'DATE'], how='inner')

    # Select and sort columns
    df_option_filtered = df_option_filtered[['TICKER', 'DATE', 'SLOPE']]
    df_option_filtered.sort_values(by=["TICKER", "DATE"], inplace=True)

    # Drop duplicates based on DATE, TICKER, and SLOPE columns
    df_option_filtered.drop_duplicates(subset=["TICKER", "DATE", "SLOPE"], inplace=True)

    # Merge the dataframes
    joined_df = pd.merge(df_stock_monthly, df_option_filtered, left_on=["TICKER", "DATE"],
                         right_on=["TICKER", "DATE"], how="inner")
    
    
    # Drop the redundant columns and keep only desired columns
    cleaned_df = joined_df[["TICKER", "DATE", "MTHRET", "CAP", "SLOPE"]]

    # Sort the dataframe
    sorted_df = cleaned_df.sort_values(by=["TICKER", "DATE"])

    # Append the processed dataframe to the list
    all_processed_dfs.append(sorted_df)


# Concatenate all processed dataframes into one
final_df = pd.concat(all_processed_dfs, ignore_index=True)

# Assuming final_df is your DataFrame

def sort_by_date(group):
    return group.sort_values(by='DATE', ascending=True)

final_df_sorted = final_df.groupby('TICKER', group_keys=False).apply(sort_by_date).reset_index(drop=True)


print("All files processed and merged.")
final_df_sorted.to_csv('C:/Users/zhang/Downloads/CFAR/10.18/2018-2022.csv')