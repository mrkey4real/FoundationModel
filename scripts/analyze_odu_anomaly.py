"""
Deep dive into ODU temperature sensor anomalies.
These sensors show extreme negative values (-59423) which are clearly sensor errors.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    return df

def analyze_odu_temps(df):
    """Analyze ODU temperature sensor patterns."""
    print("="*80)
    print("ODU TEMPERATURE SENSOR DEEP ANALYSIS")
    print("="*80)

    odu_cols = ['labview_ODU CoilOut', 'labview_ODU CompSuc', 'labview_ODU CoilInt',
                'labview_ODU CoilIn', 'labview_ODU CompDis', 'labview_ODU Coil2', 'labview_ODU Coil3']

    for col in odu_cols:
        if col not in df.columns:
            continue

        data = df[col].dropna()
        print(f"\n### {col} ###")

        # Physical range check
        reasonable_min = -50  # Refrigerant could be cold
        reasonable_max = 150  # Compressor discharge can be hot

        valid_mask = (data >= reasonable_min) & (data <= reasonable_max)
        valid_data = data[valid_mask]
        invalid_data = data[~valid_mask]

        print(f"  Total data points: {len(data)}")
        print(f"  Valid range [{reasonable_min}, {reasonable_max}]: {len(valid_data)} ({100*len(valid_data)/len(data):.1f}%)")
        print(f"  Invalid: {len(invalid_data)} ({100*len(invalid_data)/len(data):.1f}%)")

        if len(valid_data) > 0:
            print(f"  Valid data range: [{valid_data.min():.1f}, {valid_data.max():.1f}]")
            print(f"  Valid data mean: {valid_data.mean():.1f}")

        if len(invalid_data) > 0:
            print(f"  Invalid values sample: {invalid_data.head(5).tolist()}")

        # Check if invalid values are concentrated in time
        if len(invalid_data) > 0:
            invalid_times = invalid_data.index
            print(f"  Invalid data time range: {invalid_times.min()} to {invalid_times.max()}")

    # Special analysis for Coil2 and Coil3 which have different patterns
    print("\n" + "="*80)
    print("SPECIAL: ODU Coil2 and Coil3 Analysis")
    print("="*80)

    for col in ['labview_ODU Coil2', 'labview_ODU Coil3']:
        if col in df.columns:
            data = df[col].dropna()
            print(f"\n{col}:")
            print(f"  Range: [{data.min():.1f}, {data.max():.1f}]")
            print(f"  These show very high values (>1000), likely different sensor type or miscalibration")

            # Check distribution
            print(f"  Percentiles:")
            for p in [25, 50, 75, 90, 95, 99]:
                print(f"    {p}th: {data.quantile(p/100):.1f}")

def analyze_sensor_error_patterns(df):
    """Identify if sensor errors follow a pattern."""
    print("\n" + "="*80)
    print("SENSOR ERROR PATTERN ANALYSIS")
    print("="*80)

    # Check ODU sensors for correlated errors
    odu_cols = ['labview_ODU CoilOut', 'labview_ODU CompSuc', 'labview_ODU CoilInt',
                'labview_ODU CoilIn', 'labview_ODU CompDis']

    # Filter to common valid timestamps
    odu_df = df[odu_cols].dropna()

    # Check for simultaneous extreme values
    extreme_threshold = -1000
    extreme_mask = (odu_df < extreme_threshold).any(axis=1)
    extreme_count = extreme_mask.sum()

    print(f"\nTimestamps with ANY ODU sensor < {extreme_threshold}: {extreme_count}")
    print(f"  ({100*extreme_count/len(odu_df):.1f}% of valid data)")

    if extreme_count > 0:
        # Check if all sensors go extreme together
        all_extreme = (odu_df < extreme_threshold).all(axis=1).sum()
        print(f"\nTimestamps where ALL ODU sensors are extreme: {all_extreme}")
        print("  This suggests a common cause (communication failure, power loss, etc.)")

        # Sample of extreme timestamps
        extreme_samples = odu_df[extreme_mask].head(10)
        print("\n  Sample of extreme values:")
        print(extreme_samples.to_string())

def recommend_data_cleaning(df):
    """Recommend data cleaning strategy for ODU sensors."""
    print("\n" + "="*80)
    print("DATA CLEANING RECOMMENDATIONS FOR ODU SENSORS")
    print("="*80)

    odu_cols = ['labview_ODU CoilOut', 'labview_ODU CompSuc', 'labview_ODU CoilInt',
                'labview_ODU CoilIn', 'labview_ODU CompDis']

    print("""
    Recommendation: Replace extreme values with NaN

    Physical constraints for refrigerant cycle temperatures:
    - Compressor Suction: -10 to 30 degC (low side)
    - Compressor Discharge: 40 to 120 degC (high side)
    - Condenser Coil In/Out: 30 to 70 degC
    - Evaporator Coil In/Out: -10 to 30 degC

    Conservative approach: Set valid range [-50, 150] degC
    Any value outside this range is clearly sensor error.
    """)

    # Count data to be cleaned
    for col in odu_cols:
        if col in df.columns:
            data = df[col].dropna()
            invalid = ((data < -50) | (data > 150)).sum()
            print(f"  {col}: {invalid} values to be set to NaN ({100*invalid/len(data):.1f}%)")

    print("\n  For ODU Coil2 and Coil3:")
    print("  These appear to be different sensors or have different calibration.")
    print("  Recommend: EXCLUDE from analysis until calibration is verified.")

if __name__ == '__main__':
    data_path = Path(r"E:\MOIRAI\data\merged_East_labview_egauge_1min.csv")
    df = load_data(data_path)

    analyze_odu_temps(df)
    analyze_sensor_error_patterns(df)
    recommend_data_cleaning(df)
