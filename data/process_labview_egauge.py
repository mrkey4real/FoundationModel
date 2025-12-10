# -*- coding: utf-8 -*-
"""
Created on Tue Dec  2 14:04:57 2025

@author: 90829
"""

import pandas as pd
from pathlib import Path

# ============== Configuration ==============
target_house = "East"
egauge_path = Path("./east_egauge_2025-1-1_2025-12-1_1min.csv")
labview_path = Path(f"./merged_{target_house}_labview.csv")

start_time = "2025-01-01 00:00:00"
end_time = "2025-12-01 00:00:00"
resample_interval = "15min"  # 1min, 5min, 15min, 1h, etc.

output_path = Path(f"./merged_{target_house}_labview_egauge_{resample_interval}.csv")

def normalize_to_1min(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """
    Normalize dataframe to 1-minute intervals.
    This ensures consistent time index before merging.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col)
    df = df.sort_index()
    
    # Remove duplicate indices by keeping the first occurrence
    df = df[~df.index.duplicated(keep='first')]
    
    # Resample to 1-minute intervals (using mean for numeric columns)
    # Forward fill for small gaps, leave NaN for larger gaps
    df_1min = df.resample('1min').mean()
    
    return df_1min


def create_continuous_timeindex(start: str, end: str, freq: str = '1min') -> pd.DatetimeIndex:
    """Create a continuous datetime index from start to end."""
    return pd.date_range(start=start, end=end, freq=freq)


def process_and_merge():
    print(f"Loading egauge data from: {egauge_path}")
    egauge_df = pd.read_csv(egauge_path)
    print(f"  Shape: {egauge_df.shape}")
    print(f"  Time column: 'Date & Time'")
    print(f"  Time range: {egauge_df['Date & Time'].iloc[0]} to {egauge_df['Date & Time'].iloc[-1]}")
    
    print(f"\nLoading labview data from: {labview_path}")
    labview_df = pd.read_csv(labview_path)
    print(f"  Shape: {labview_df.shape}")
    print(f"  Time column: 'Time'")
    print(f"  Time range: {labview_df['Time'].iloc[0]} to {labview_df['Time'].iloc[-1]}")
    
    # Step 1: Normalize both datasets to 1-minute intervals
    print("\n=== Step 1: Normalizing to 1-minute intervals ===")
    egauge_1min = normalize_to_1min(egauge_df, 'Date & Time')
    print(f"  egauge normalized: {egauge_1min.shape}")
    
    labview_1min = normalize_to_1min(labview_df, 'Time')
    print(f"  labview normalized: {labview_1min.shape}")
    
    # Step 2: Create continuous time index for the specified range
    print(f"\n=== Step 2: Creating continuous time index ===")
    print(f"  Range: {start_time} to {end_time}")
    continuous_index = create_continuous_timeindex(start_time, end_time, '1min')
    print(f"  Total timestamps: {len(continuous_index)}")
    
    # Step 3: Reindex both datasets to the continuous index
    print("\n=== Step 3: Reindexing to continuous time index ===")
    egauge_reindexed = egauge_1min.reindex(continuous_index)
    labview_reindexed = labview_1min.reindex(continuous_index)
    
    egauge_valid = egauge_reindexed.notna().any(axis=1).sum()
    labview_valid = labview_reindexed.notna().any(axis=1).sum()
    print(f"  egauge valid timestamps: {egauge_valid}")
    print(f"  labview valid timestamps: {labview_valid}")
    
    # Step 4: Add prefix to columns to distinguish sources
    egauge_reindexed = egauge_reindexed.add_prefix('egauge_')
    labview_reindexed = labview_reindexed.add_prefix('labview_')
    
    # Step 5: Merge the two datasets
    print("\n=== Step 4: Merging datasets ===")
    merged_df = pd.concat([egauge_reindexed, labview_reindexed], axis=1)
    merged_df.index.name = 'timestamp'
    print(f"  Merged shape (1min): {merged_df.shape}")
    
    # Step 6: Resample to the desired interval
    print(f"\n=== Step 5: Resampling to {resample_interval} ===")
    merged_resampled = merged_df.resample(resample_interval).mean()
    print(f"  Resampled shape: {merged_resampled.shape}")
    
    # Step 7: Save to CSV
    print(f"\n=== Step 6: Saving to {output_path} ===")
    merged_resampled.to_csv(output_path)
    print(f"  Done! Final shape: {merged_resampled.shape}")
    
    # Summary statistics
    print("\n=== Summary ===")
    total_cols = merged_resampled.shape[1]
    egauge_cols = sum(1 for c in merged_resampled.columns if c.startswith('egauge_'))
    labview_cols = sum(1 for c in merged_resampled.columns if c.startswith('labview_'))
    print(f"  Total columns: {total_cols}")
    print(f"  egauge columns: {egauge_cols}")
    print(f"  labview columns: {labview_cols}")
    
    # Show missing data statistics
    missing_pct = merged_resampled.isna().mean() * 100
    print(f"\n  Missing data statistics (top 10 columns with most missing):")
    for col, pct in missing_pct.nlargest(10).items():
        print(f"    {col}: {pct:.1f}%")
    
    return merged_resampled


if __name__ == "__main__":
    merged = process_and_merge()
