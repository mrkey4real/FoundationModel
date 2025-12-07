"""
Comprehensive data quality analysis for merged LabVIEW + eGauge dataset.
Analyzes:
1. Constant columns (no variation)
2. Missing data patterns
3. Data validity and physical ranges
4. HVAC logic chain coverage
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def load_data(file_path):
    """Load the merged dataset."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    return df

def analyze_missing_data(df):
    """Analyze missing data patterns for each column."""
    missing_stats = []
    for col in df.columns:
        total = len(df)
        missing = df[col].isna().sum()
        missing_pct = (missing / total) * 100

        # Check if missing is concentrated or distributed
        if missing > 0 and missing < total:
            # Find first and last valid index
            valid_mask = df[col].notna()
            first_valid = df.index[valid_mask][0] if valid_mask.any() else None
            last_valid = df.index[valid_mask][-1] if valid_mask.any() else None
        else:
            first_valid = None
            last_valid = None

        missing_stats.append({
            'column': col,
            'missing_count': missing,
            'missing_pct': missing_pct,
            'first_valid': first_valid,
            'last_valid': last_valid
        })

    return pd.DataFrame(missing_stats).sort_values('missing_pct', ascending=False)

def analyze_constant_columns(df):
    """Find columns with no variation (constant or near-constant)."""
    constant_stats = []
    for col in df.columns:
        valid_data = df[col].dropna()
        if len(valid_data) == 0:
            constant_stats.append({
                'column': col,
                'is_constant': True,
                'unique_values': 0,
                'std': np.nan,
                'reason': 'all_missing'
            })
        else:
            unique_count = valid_data.nunique()
            std = valid_data.std()
            is_constant = unique_count <= 1 or (std < 1e-10)

            constant_stats.append({
                'column': col,
                'is_constant': is_constant,
                'unique_values': unique_count,
                'std': std,
                'min': valid_data.min(),
                'max': valid_data.max(),
                'mean': valid_data.mean(),
                'reason': 'no_variation' if is_constant else 'ok'
            })

    return pd.DataFrame(constant_stats)

def analyze_physical_validity(df):
    """Check if values are within expected physical ranges."""
    # Define expected ranges for known column types
    physical_ranges = {
        # Temperature columns (degC) - should be -50 to 100
        'Temperature': (-50, 100),
        'Temp': (-50, 100),
        'degC': (-50, 100),

        # Humidity columns (%) - should be 0 to 100
        'Humidity': (0, 100),
        'RH': (0, 100),

        # Power columns (kW) - should be >= 0 typically
        'kW': (-10, 50),  # Allow small negative for measurement noise
        'Power': (-10, 50),

        # Voltage (V) - should be around 120V
        'Voltage': (100, 140),
        '[V]': (100, 140),

        # Irradiance (W/m2) - should be 0 to 1500
        'Irradiance': (0, 1500),
        'W/m2': (0, 1500),

        # Pressure (kPa) - should be around 101 kPa
        'Pressure': (80, 110),
        'kPa': (80, 110),

        # Wind Speed (m/s)
        'Wind Speed': (0, 50),

        # CO2 (ppm)
        'CO2': (300, 5000),
    }

    validity_stats = []
    for col in df.columns:
        valid_data = df[col].dropna()
        if len(valid_data) == 0:
            validity_stats.append({
                'column': col,
                'physical_check': 'no_data',
                'out_of_range_pct': np.nan
            })
            continue

        # Find matching physical range
        expected_range = None
        for key, range_val in physical_ranges.items():
            if key.lower() in col.lower():
                expected_range = range_val
                break

        if expected_range is None:
            validity_stats.append({
                'column': col,
                'physical_check': 'no_rule',
                'min': valid_data.min(),
                'max': valid_data.max(),
                'out_of_range_pct': np.nan
            })
        else:
            low, high = expected_range
            out_of_range = ((valid_data < low) | (valid_data > high)).sum()
            out_of_range_pct = (out_of_range / len(valid_data)) * 100

            validity_stats.append({
                'column': col,
                'physical_check': 'pass' if out_of_range_pct < 1 else 'fail',
                'expected_range': f"[{low}, {high}]",
                'actual_min': valid_data.min(),
                'actual_max': valid_data.max(),
                'out_of_range_pct': out_of_range_pct
            })

    return pd.DataFrame(validity_stats)

def categorize_columns(df):
    """Categorize columns by their role in HVAC logic chain."""
    categories = {
        'outdoor_weather': [],
        'outdoor_unit': [],
        'indoor_unit': [],
        'indoor_environment': [],
        'electrical_power': [],
        'control_setpoint': [],
        'heat_flux': [],
        'irradiance': [],
        'air_quality': [],
        'other': []
    }

    # Categorization rules
    for col in df.columns:
        col_lower = col.lower()

        # Outdoor weather
        if any(x in col_lower for x in ['outdoor', 'wind', 'precipitation', 'dew point',
                                         'global radiation', 'air temperature', 'relative humidity [%]',
                                         'relative air pressure']):
            if 'unit' not in col_lower:
                categories['outdoor_weather'].append(col)
                continue

        # Outdoor unit
        if 'odu' in col_lower or 'outdoor unit' in col_lower:
            categories['outdoor_unit'].append(col)
            continue

        # Indoor unit
        if 'idu' in col_lower or 'indoor unit' in col_lower:
            categories['indoor_unit'].append(col)
            continue

        # Indoor environment (room temperatures, humidity in rooms)
        if any(x in col_lower for x in ['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6',
                                         'a1', 'a2', 'a3', 'a5', 'a6', 'a7',
                                         'b1', 'b2', 'b3', 'b5', 'b6', 'b7',
                                         'c1', 'c2', 'c3', 'c5', 'c6', 'c7',
                                         'd1', 'd2', 'd3', 'd5', 'd6', 'd7',
                                         'e1', 'e2', 'e3', 'e5', 'e6', 'e7',
                                         'f1', 'f2', 'f3', 'f5', 'f6', 'f7',
                                         'g1', 'g2', 'g3', 'g5', 'g6', 'g7',
                                         'h1', 'h2', 'h3', 'h5', 'h6', 'h7',
                                         'thermostat', 'bglobe', 'bath', 'kitchen',
                                         'living', 'bedroom', 'master']):
            if 'rh' in col_lower or 'humidity' in col_lower:
                categories['indoor_environment'].append(col)
            elif 'irradiance' in col_lower or 'pyranometer' in col_lower:
                categories['irradiance'].append(col)
            else:
                categories['indoor_environment'].append(col)
            continue

        # Electrical power/energy
        if any(x in col_lower for x in ['usage', 'generation', 'power', 'kw', 'kva',
                                         'voltage', '[a]', '[v]', 'water heater',
                                         'lights', 's1', 's2', 's3', 's4', 's5',
                                         's6', 's7', 's8', 's9', 's10', 's11', 's12',
                                         's13', 's14', 'main']):
            categories['electrical_power'].append(col)
            continue

        # Heat flux
        if 'hf' in col_lower or 'heat flux' in col_lower or '_top' in col_lower or '_bot' in col_lower:
            categories['heat_flux'].append(col)
            continue

        # Air quality
        if any(x in col_lower for x in ['co2', 'tvoc', 'return rh']):
            categories['air_quality'].append(col)
            continue

        # Control
        if any(x in col_lower for x in ['dr mode', 'off']):
            categories['control_setpoint'].append(col)
            continue

        # Default
        categories['other'].append(col)

    return categories

def generate_report(df, output_dir):
    """Generate comprehensive data quality report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("DATA QUALITY ANALYSIS REPORT")
    print("="*80)

    # 1. Missing data analysis
    print("\n### 1. Missing Data Analysis ###")
    missing_df = analyze_missing_data(df)

    # Completely missing columns
    fully_missing = missing_df[missing_df['missing_pct'] >= 99.9]
    print(f"\nColumns with >99.9% missing data ({len(fully_missing)}):")
    for _, row in fully_missing.iterrows():
        print(f"  - {row['column']}: {row['missing_pct']:.1f}%")

    # Mostly missing columns (>50%)
    mostly_missing = missing_df[(missing_df['missing_pct'] >= 50) & (missing_df['missing_pct'] < 99.9)]
    print(f"\nColumns with 50-99.9% missing data ({len(mostly_missing)}):")
    for _, row in mostly_missing.iterrows():
        print(f"  - {row['column']}: {row['missing_pct']:.1f}%")

    # Good columns (<10% missing)
    good_columns = missing_df[missing_df['missing_pct'] < 10]
    print(f"\nColumns with <10% missing data ({len(good_columns)}):")

    # 2. Constant column analysis
    print("\n### 2. Constant Column Analysis ###")
    constant_df = analyze_constant_columns(df)

    constant_cols = constant_df[constant_df['is_constant'] == True]
    print(f"\nConstant or near-constant columns ({len(constant_cols)}):")
    for _, row in constant_cols.iterrows():
        if row['reason'] == 'all_missing':
            print(f"  - {row['column']}: ALL MISSING")
        else:
            print(f"  - {row['column']}: value={row['mean']:.4f} (std={row['std']:.2e})")

    # 3. Physical validity check
    print("\n### 3. Physical Validity Check ###")
    validity_df = analyze_physical_validity(df)

    failed_physical = validity_df[validity_df['physical_check'] == 'fail']
    print(f"\nColumns failing physical range check ({len(failed_physical)}):")
    for _, row in failed_physical.iterrows():
        print(f"  - {row['column']}: expected {row['expected_range']}, "
              f"actual [{row['actual_min']:.2f}, {row['actual_max']:.2f}], "
              f"{row['out_of_range_pct']:.1f}% out of range")

    # 4. HVAC Logic Chain Analysis
    print("\n### 4. HVAC Logic Chain Analysis ###")
    categories = categorize_columns(df)

    for cat_name, cols in categories.items():
        if len(cols) > 0:
            print(f"\n{cat_name.upper()} ({len(cols)} columns):")
            for col in cols:
                missing_pct = missing_df[missing_df['column'] == col]['missing_pct'].values[0]
                const_info = constant_df[constant_df['column'] == col]
                is_const = const_info['is_constant'].values[0] if len(const_info) > 0 else False

                status = "[CONST]" if is_const else ("[MISSING]" if missing_pct > 50 else "[OK]")
                print(f"  {status} {col}: {missing_pct:.1f}% missing")

    # 5. Summary recommendations
    print("\n### 5. RECOMMENDATIONS ###")

    # Columns to DROP
    drop_columns = []
    keep_columns = []

    for col in df.columns:
        missing_pct = missing_df[missing_df['column'] == col]['missing_pct'].values[0]
        const_info = constant_df[constant_df['column'] == col]
        is_const = const_info['is_constant'].values[0] if len(const_info) > 0 else False

        if missing_pct >= 99.9:
            drop_columns.append((col, "100% missing"))
        elif is_const and missing_pct < 99:
            drop_columns.append((col, "constant value"))
        elif missing_pct >= 80:
            drop_columns.append((col, f"{missing_pct:.1f}% missing"))
        else:
            keep_columns.append(col)

    print(f"\nColumns to DROP ({len(drop_columns)}):")
    for col, reason in drop_columns:
        print(f"  - {col}: {reason}")

    print(f"\nColumns to KEEP ({len(keep_columns)}):")
    for col in keep_columns:
        missing_pct = missing_df[missing_df['column'] == col]['missing_pct'].values[0]
        print(f"  [OK] {col}: {missing_pct:.1f}% missing")

    # Save results
    missing_df.to_csv(output_dir / 'missing_data_analysis.csv', index=False)
    constant_df.to_csv(output_dir / 'constant_columns_analysis.csv', index=False)
    validity_df.to_csv(output_dir / 'physical_validity_analysis.csv', index=False)

    # Save recommendations
    recommendations = {
        'drop_columns': [col for col, _ in drop_columns],
        'keep_columns': keep_columns,
        'categories': categories
    }

    with open(output_dir / 'recommendations.json', 'w') as f:
        json.dump(recommendations, f, indent=2, default=str)

    print(f"\n\nDetailed reports saved to {output_dir}")

    return missing_df, constant_df, validity_df, categories, keep_columns

if __name__ == '__main__':
    data_path = Path(r"E:\MOIRAI\data\merged_East_labview_egauge_1min.csv")
    output_dir = Path(r"E:\MOIRAI\data\quality_analysis")

    df = load_data(data_path)
    missing_df, constant_df, validity_df, categories, keep_columns = generate_report(df, output_dir)
