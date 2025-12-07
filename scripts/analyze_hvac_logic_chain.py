"""
Deeper analysis of HVAC logic chain and data quality issues.
Focus on:
1. Understanding what A1-H7 columns represent
2. Analyzing ODU/IDU temperature sensors
3. Checking RH sensor validity
4. Understanding egauge current sensors S1-S14
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"Data shape: {df.shape}")
    return df

def analyze_temperature_sensors(df):
    """Analyze room temperature sensors A1-H7 and ODU/IDU temperatures."""
    print("\n" + "="*80)
    print("TEMPERATURE SENSOR ANALYSIS")
    print("="*80)

    # Room temperature sensors (A-H represent different locations, 1-7 are sensor positions)
    room_prefixes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    sensor_positions = ['1', '2', '3', '5', '6', '7']  # Note: no 4

    print("\n### Room Temperature Sensors (A-H x 1-7) ###")
    print("These appear to be thermocouple readings at different room positions")

    room_temp_cols = []
    for prefix in room_prefixes:
        for pos in sensor_positions:
            col = f'labview_{prefix}{pos}'
            if col in df.columns:
                room_temp_cols.append(col)

    if room_temp_cols:
        room_temps = df[room_temp_cols].dropna(how='all')
        print(f"\nRoom temperature columns: {len(room_temp_cols)}")
        print(f"Valid data points: {len(room_temps)} ({100*len(room_temps)/len(df):.1f}%)")

        # Summary statistics
        stats = df[room_temp_cols].describe()
        print(f"\nTemperature range across all room sensors:")
        print(f"  Min: {stats.loc['min'].min():.1f} degC")
        print(f"  Max: {stats.loc['max'].max():.1f} degC")
        print(f"  Mean: {stats.loc['mean'].mean():.1f} degC")

        # Check for unreasonable values
        for col in room_temp_cols:
            data = df[col].dropna()
            if len(data) > 0:
                if data.min() < 10 or data.max() > 40:
                    print(f"  WARNING: {col} has suspicious range [{data.min():.1f}, {data.max():.1f}]")

    # ODU (Outdoor Unit) temperature sensors
    print("\n### ODU Temperature Sensors ###")
    odu_cols = [c for c in df.columns if 'ODU' in c and 'off' not in c.lower()]
    for col in odu_cols:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"  {col}: [{data.min():.1f}, {data.max():.1f}] degC, mean={data.mean():.1f}")

    # IDU (Indoor Unit) temperature sensors
    print("\n### IDU Temperature Sensors ###")
    idu_cols = [c for c in df.columns if 'IDU' in c and 'off' not in c.lower() and 'kW' not in c and 'kVA' not in c]
    for col in idu_cols:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"  {col}: [{data.min():.1f}, {data.max():.1f}] degC, mean={data.mean():.1f}")

def analyze_humidity_sensors(df):
    """Analyze RH sensors and identify problematic ones."""
    print("\n" + "="*80)
    print("HUMIDITY SENSOR ANALYSIS")
    print("="*80)

    rh_cols = [c for c in df.columns if 'RH' in c or 'Humidity' in c]

    print("\n### All RH Sensors ###")
    for col in rh_cols:
        data = df[col].dropna()
        if len(data) > 0:
            in_range = ((data >= 0) & (data <= 100)).sum()
            in_range_pct = 100 * in_range / len(data)
            status = "OK" if in_range_pct > 95 else "PROBLEMATIC"
            print(f"  {status}: {col}")
            print(f"    Range: [{data.min():.1f}, {data.max():.1f}]%, Mean: {data.mean():.1f}%")
            print(f"    Valid (0-100%): {in_range_pct:.1f}%")

def analyze_power_sensors(df):
    """Analyze power sensors and understand the logic chain."""
    print("\n" + "="*80)
    print("POWER/ELECTRICAL SENSOR ANALYSIS")
    print("="*80)

    # Main power and submeters
    print("\n### Main Power and Submeters ###")
    power_cols = [c for c in df.columns if 'kW' in c and 'egauge' in c]
    for col in power_cols:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"  {col}:")
            print(f"    Range: [{data.min():.4f}, {data.max():.4f}] kW")
            print(f"    Mean: {data.mean():.4f} kW")
            if data.min() < -0.1:
                print(f"    NOTE: Negative values detected (possible bidirectional flow or noise)")

    # Current sensors S1-S14
    print("\n### Current Sensors (S1-S14) ###")
    current_cols = [c for c in df.columns if c.startswith('egauge_S') and '[A]' in c]
    for col in current_cols:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"  {col}: [{data.min():.3f}, {data.max():.3f}] A, mean={data.mean():.3f}")

    # Check ODU/IDU power
    print("\n### HVAC Unit Power ###")
    hvac_power_cols = ['egauge_Indoor Unit [kW]', 'egauge_Outdoor Unit [kW]']
    for col in hvac_power_cols:
        if col in df.columns:
            data = df[col].dropna()
            print(f"  {col}:")
            print(f"    Range: [{data.min():.4f}, {data.max():.4f}] kW")
            print(f"    Non-zero time: {100*(data > 0.01).sum()/len(data):.1f}%")

def analyze_irradiance_sensors(df):
    """Analyze irradiance/pyranometer sensors."""
    print("\n" + "="*80)
    print("IRRADIANCE SENSOR ANALYSIS")
    print("="*80)

    # Pyranometer voltage readings
    print("\n### Pyranometer Voltage Readings ###")
    pyra_v_cols = [c for c in df.columns if 'Pyranometer [V]' in c]
    for col in pyra_v_cols:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"  {col}:")
            print(f"    Range: [{data.min():.4f}, {data.max():.4f}] V")
            print(f"    NOTE: These are raw voltage signals, not actual irradiance")

    # Irradiance calculated values
    print("\n### Calculated Irradiance ###")
    irr_cols = [c for c in df.columns if 'Irradiance' in c and 'Pyranometer Irradiance' not in c]
    for col in irr_cols:
        data = df[col].dropna()
        if len(data) > 0:
            valid = ((data >= 0) & (data <= 1500)).sum()
            print(f"  {col}:")
            print(f"    Range: [{data.min():.1f}, {data.max():.1f}] W/m2")
            print(f"    Valid (0-1500): {100*valid/len(data):.1f}% - LIKELY BAD CALIBRATION")

def analyze_heat_flux_sensors(df):
    """Analyze heat flux sensors."""
    print("\n" + "="*80)
    print("HEAT FLUX SENSOR ANALYSIS")
    print("="*80)

    hf_cols = [c for c in df.columns if 'HF' in c or 'Heat Flux' in c.lower()]
    print(f"\nHeat flux columns: {len(hf_cols)}")

    for col in hf_cols:
        data = df[col].dropna()
        if len(data) > 0:
            print(f"  {col}:")
            print(f"    Range: [{data.min():.2f}, {data.max():.2f}] W/m^2")
            print(f"    Mean: {data.mean():.2f} W/m^2")

def analyze_hvac_logic_chain(df):
    """Analyze the complete HVAC logic chain."""
    print("\n" + "="*80)
    print("HVAC LOGIC CHAIN ANALYSIS")
    print("="*80)

    print("""
    HVAC Logic Chain:

    1. OUTDOOR CONDITIONS (Boundary)
       - Outdoor Temperature, Humidity, Solar Radiation, Wind

    2. OUTDOOR UNIT (Heat Rejection/Absorption)
       - ODU Power, Compressor Temps, Condenser/Evaporator Temps

    3. REFRIGERANT CYCLE
       - Compressor Discharge/Suction, Coil In/Out temps

    4. INDOOR UNIT (Heat Delivery)
       - IDU Power, Supply Air Temp, Return Air Temp

    5. INDOOR ENVIRONMENT (Controlled Space)
       - Room Temperatures (A-H zones), Humidity, CO2

    Key Physical Relationships:
    - ODU Power + IDU Power = Total HVAC Power
    - Heat removed from indoor = COP * Compressor Power
    - Room temp responds to supply air temp with delay
    - Outdoor temp affects condenser efficiency
    """)

    # Check data availability for each stage
    print("\n### Data Availability by Logic Chain Stage ###")

    stages = {
        'Stage 1 - Outdoor': [
            'labview_Air Temperature [degC]',
            'labview_Relative Humidity [%]',
            'labview_Wind Speed [m/s]'
        ],
        'Stage 2 - ODU': [
            'egauge_Outdoor Unit [kW]',
            'labview_ODU CompDis',
            'labview_ODU CompSuc',
            'labview_ODU CoilIn',
            'labview_ODU CoilOut'
        ],
        'Stage 3 - IDU': [
            'egauge_Indoor Unit [kW]',
            'labview_IDU CoilIn',
            'labview_IDU CoilOut',
            'labview_IDU Return Air'
        ],
        'Stage 4 - Indoor': [
            'labview_Thermostat',
            'labview_A1', 'labview_B1', 'labview_C1',
            'labview_CO2 [ppm]'
        ]
    }

    for stage_name, cols in stages.items():
        print(f"\n{stage_name}:")
        for col in cols:
            if col in df.columns:
                missing_pct = 100 * df[col].isna().sum() / len(df)
                data = df[col].dropna()
                if len(data) > 0:
                    print(f"  [OK] {col}: {missing_pct:.1f}% missing, range=[{data.min():.2f}, {data.max():.2f}]")
                else:
                    print(f"  [EMPTY] {col}")
            else:
                print(f"  [MISSING] {col} not in dataset")

def generate_final_recommendations(df):
    """Generate final column recommendations for MOIRAI training."""
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS FOR MOIRAI TRAINING")
    print("="*80)

    # Categorize columns
    keep_essential = []    # Must keep - core HVAC logic
    keep_optional = []     # Good to have
    drop_list = []         # Should drop

    for col in df.columns:
        missing_pct = 100 * df[col].isna().sum() / len(df)
        data = df[col].dropna()

        # Check if constant
        is_constant = False
        if len(data) > 0:
            is_constant = data.std() < 1e-10 or data.nunique() <= 1

        # Drop rules
        if missing_pct >= 80:
            drop_list.append((col, f"high missing ({missing_pct:.0f}%)"))
        elif is_constant:
            drop_list.append((col, "constant value"))
        # Check for physically invalid data
        elif 'RH' in col and len(data) > 0:
            invalid_pct = 100 * ((data < 0) | (data > 100)).sum() / len(data)
            if invalid_pct > 50:
                drop_list.append((col, f"invalid RH values ({invalid_pct:.0f}% out of range)"))
            elif missing_pct < 50:
                keep_optional.append(col)
        elif 'Irradiance [W/m2]' in col and 'Pyranometer Irradiance' not in col:
            # These calculated irradiance columns have bad calibration
            drop_list.append((col, "bad calibration"))
        # Essential columns
        elif any(x in col for x in ['Outdoor Unit', 'Indoor Unit', 'Main Power',
                                     'ODU', 'IDU', 'Air Temperature', 'Thermostat']):
            if missing_pct < 50:
                keep_essential.append(col)
        elif any(x in col for x in ['A1', 'B1', 'C1', 'D1', 'E1', 'F1', 'G1', 'H1',
                                     'Voltage', 'CO2', 'TVOC']):
            if missing_pct < 50:
                keep_essential.append(col)
        elif missing_pct < 50:
            keep_optional.append(col)
        else:
            drop_list.append((col, f"high missing ({missing_pct:.0f}%)"))

    print("\n### ESSENTIAL COLUMNS (Core HVAC Logic) ###")
    for col in sorted(keep_essential):
        print(f"  + {col}")

    print(f"\n### OPTIONAL COLUMNS ({len(keep_optional)} columns) ###")
    print("  (Additional detail, not strictly necessary for basic logic)")

    print("\n### COLUMNS TO DROP ###")
    for col, reason in sorted(drop_list):
        print(f"  - {col}: {reason}")

    # Save final column list
    final_columns = keep_essential + keep_optional
    print(f"\n### SUMMARY ###")
    print(f"  Essential columns: {len(keep_essential)}")
    print(f"  Optional columns: {len(keep_optional)}")
    print(f"  Total to keep: {len(final_columns)}")
    print(f"  Columns to drop: {len(drop_list)}")

    return keep_essential, keep_optional, drop_list

if __name__ == '__main__':
    data_path = Path(r"E:\MOIRAI\data\merged_East_labview_egauge_1min.csv")
    df = load_data(data_path)

    analyze_temperature_sensors(df)
    analyze_humidity_sensors(df)
    analyze_power_sensors(df)
    analyze_irradiance_sensors(df)
    analyze_heat_flux_sensors(df)
    analyze_hvac_logic_chain(df)

    keep_essential, keep_optional, drop_list = generate_final_recommendations(df)

    # Save to file
    output = {
        'essential': keep_essential,
        'optional': keep_optional,
        'drop': [col for col, _ in drop_list]
    }

    import json
    with open(r"E:\MOIRAI\data\quality_analysis\final_column_selection.json", 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nColumn selection saved to E:\\MOIRAI\\data\\quality_analysis\\final_column_selection.json")
