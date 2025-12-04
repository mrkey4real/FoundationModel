# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 13:47:13 2025

@author: 90829
"""

import pandas as pd
from pathlib import Path

target_house = 'East'
YEAR = 2025
file_name = f"{target_house}_{YEAR}_csv"

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / f"{file_name}"
OUTPUT_FILE = SCRIPT_DIR / f"merged_{target_house}_labview.csv"
LOG_FILE = SCRIPT_DIR / "merge_log.txt"

def merge():
    def log(msg):
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    
    # Clear log file
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("Starting merge process...\n")
    
    # Read all CSV files
    csv_files = list(RAW_DIR.glob("*.csv"))
    log(f"Found {len(csv_files)} CSV files")
    
    # Load and concatenate all files
    dfs = []
    for csv_file in csv_files:
        log(f"Reading: {csv_file.name}")
        df = pd.read_csv(csv_file)
        dfs.append(df)
    
    # Filter out empty dataframes before concat
    dfs = [df for df in dfs if not df.empty]
    combined = pd.concat(dfs, ignore_index=True)
    log(f"Total rows after concatenation: {len(combined)}")
    
    # Parse time column (filter out invalid rows first)
    # Some rows have numeric values instead of datetime strings
    valid_time_mask = combined["Time"].astype(str).str.contains(r"^\d{1,2}/\d{1,2}/\d{4}", na=False)
    invalid_count = (~valid_time_mask).sum()
    log(f"Found {invalid_count} rows with invalid time format, removing them")
    combined = combined[valid_time_mask].copy()
    
    # Use mixed format parsing since data has multiple time formats
    # (both 12-hour with AM/PM and 24-hour formats)
    combined["Time"] = pd.to_datetime(combined["Time"], format="mixed", dayfirst=False)
    
    # Remove any rows where time parsing failed
    nat_count = combined["Time"].isna().sum()
    if nat_count > 0:
        log(f"Removed {nat_count} rows with unparseable time")
        combined = combined.dropna(subset=["Time"])
    
    # Round to nearest minute (floor to minute)
    combined["Time"] = combined["Time"].dt.floor("min")
    
    # Remove duplicates (keep first occurrence for each minute)
    combined = combined.drop_duplicates(subset=["Time"], keep="first")
    log(f"Rows after removing duplicates: {len(combined)}")
    
    # Create complete time range for the entire year (1-minute intervals)
    start_time = pd.Timestamp(f"{YEAR}-01-01 00:00:00")
    end_time = pd.Timestamp(f"{YEAR}-12-31 23:59:00")
    full_time_range = pd.date_range(start=start_time, end=end_time, freq="1min")
    log(f"Full year has {len(full_time_range)} minutes")
    
    # Create template DataFrame with full time range
    template = pd.DataFrame({"Time": full_time_range})
    
    # Merge data with template (left join to keep all time points)
    merged = template.merge(combined, on="Time", how="left")
    
    # Sort by time (ascending)
    merged = merged.sort_values("Time").reset_index(drop=True)
    
    # Save to CSV
    merged.to_csv(OUTPUT_FILE, index=False)
    log(f"Merged file saved to: {OUTPUT_FILE}")
    log(f"Total rows: {len(merged)}")
    
    # Count rows with actual data
    data_cols = merged.columns[1:]
    rows_with_data = merged.dropna(subset=data_cols, how="all").shape[0]
    log(f"Data coverage: {rows_with_data} rows with data")
    log("Done!")


merge()
#%% statistic
# ============ Monthly Missing Rate Statistics ============
# Load merged file directly (no need to re-run merge)
merged = pd.read_csv(OUTPUT_FILE, parse_dates=["Time"])
data_cols = merged.columns[1:]

merged["Month"] = merged["Time"].dt.month

print(f"\n===== Monthly {target_house.upper()} House Missing Rate Statistics in {YEAR}=====")
for month in range(1, 13):
    month_data = merged[merged["Month"] == month]
    total_rows = len(month_data)
    if total_rows == 0:
        print(f"Month {month:02d}: No data")
        continue
    
    # Count rows where all data columns are NaN (missing)
    missing_rows = month_data[data_cols].isna().all(axis=1).sum()
    missing_rate = missing_rows / total_rows * 100
    
    print(f"Month {month:02d}: {missing_rate:6.2f}% missing ({missing_rows:,}/{total_rows:,} rows)")

# Overall summary
total_missing = merged[data_cols].isna().all(axis=1).sum()
overall_rate = total_missing / len(merged) * 100
print(f"\nOverall: {overall_rate:.2f}% missing ({total_missing:,}/{len(merged):,} rows)")

merged.drop(columns=["Month"], inplace=True)