#!/usr/bin/env python
"""
BuildingFM Data Preparation Pipeline

将 CSV 数据转换为 uni2ts 训练所需的格式。
支持 "All-in-Target" 策略：所有变量都放入 target，让模型学习完整的物理因果链。
"""

import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# ============================================================================
# Configuration - 在这里修改配置参数
# ============================================================================

# =============================================================================
# Frequency Configuration - 修改这里会自动调整所有相关参数
# =============================================================================
DATA_FREQ = '15min'  # Options: '1min', '5min', '15min', '30min', '1H', etc.

# Auto-calculated time steps
_freq_minutes = pd.Timedelta(DATA_FREQ).total_seconds() / 60
STEPS_PER_HOUR = int(60 / _freq_minutes)
STEPS_PER_DAY = int(24 * 60 / _freq_minutes)
STEPS_PER_WEEK = int(7 * 24 * 60 / _freq_minutes)

# Window configuration
# 原配置: WINDOW_DAYS=14, STRIDE_DAYS=7 → 只有30个样本，数据量严重不足
# 新配置: WINDOW_DAYS=3, STRIDE_HOURS=6 → 预计~1300个样本，充足的训练数据
#
# 设计理由:
# - 3天窗口 (288 timesteps @ 15min) 足以捕获日周期模式
# - 6小时步长提供充足的数据量，同时保持样本多样性
# - 目标：训练样本数 > 1000，让14M参数模型有足够数据学习
WINDOW_DAYS = 3          # 每个样本覆盖3天 (对于15min数据 = 288步)
STRIDE_HOURS = 6         # 每6小时滑动一次 (对于15min数据 = 24步)

CONFIG = {
    'schema_path': '../config/hvac_schema.yaml',
    'input_csv': '../data/final_essential_merged_East_labview_egauge_15min.csv',
    'output_dir': '../data/buildingfm_processed_15min',
    'resample_freq': DATA_FREQ,
    'window_size': WINDOW_DAYS * STEPS_PER_DAY,      # 3天 = 288步
    'stride': STRIDE_HOURS * STEPS_PER_HOUR,         # 6小时 = 24步
    'train_ratio': 0.7,        # 训练集比例
    'min_valid_ratio': 0.3,    # 每个窗口最小有效数据比例
    'output_format': 'both',   # 'arrow', 'jsonl', 'both'
}

# ============================================================================


def load_schema(schema_path: Path) -> Dict:
    """Load HVAC schema from YAML file."""
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema = yaml.safe_load(f)

    # Normalize variable keys to integers
    if 'variables' in schema:
        schema['variables'] = {int(k): v for k, v in schema['variables'].items()}

    return schema


def load_and_validate_csv(
    csv_path: Path,
    schema: Dict,
    resample_freq: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load CSV and validate against schema.

    Returns:
        df: DataFrame with timestamp index and columns ordered by schema ID
        stats: Data quality statistics
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=['timestamp'], index_col='timestamp')
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Time range: {df.index[0]} to {df.index[-1]}")

    # Build column mapping from schema
    schema_vars = schema['variables']
    col_mapping = {}  # var_id -> original_column
    id_to_name = {}   # var_id -> schema_name

    for var_id, var_info in schema_vars.items():
        var_id = int(var_id)
        col_mapping[var_id] = var_info['original_column']
        id_to_name[var_id] = var_info['name']

    # Validate all schema columns exist in data
    missing_cols = []
    available_cols = []
    for var_id, orig_col in col_mapping.items():
        if orig_col in df.columns:
            available_cols.append((var_id, orig_col))
        else:
            missing_cols.append((var_id, orig_col))

    if missing_cols:
        print(f"\n  Warning: {len(missing_cols)} schema columns not found in data:")
        for var_id, col in missing_cols[:5]:
            print(f"    ID {var_id}: {col}")
        if len(missing_cols) > 5:
            print(f"    ... and {len(missing_cols) - 5} more")

    print(f"  Found {len(available_cols)} / {len(col_mapping)} schema columns")

    # Resample if requested
    if resample_freq:
        print(f"\n  Resampling to {resample_freq}...")
        # Define aggregation rules based on variable type
        agg_rules = {}
        for var_id, orig_col in available_cols:
            var_info = schema_vars[var_id]
            unit = var_info.get('unit', '')

            # Determine aggregation method
            if unit in ['binary', 'status', 'category']:
                agg_rules[orig_col] = 'last'  # Take last value for categorical
            elif 'power' in var_info['name'].lower() or 'current' in var_info['name'].lower():
                agg_rules[orig_col] = 'mean'  # Average power
            elif 'setpoint' in var_info['name'].lower():
                agg_rules[orig_col] = 'last'  # Setpoint is step function
            else:
                agg_rules[orig_col] = 'mean'  # Default to mean

        df = df.resample(resample_freq).agg(agg_rules)
        print(f"  Resampled to {len(df)} rows")

    # Apply physical range validation
    print("\n  Applying physical range validation...")
    clipped_counts = {}
    for var_id, orig_col in available_cols:
        if orig_col not in df.columns:
            continue
        var_info = schema_vars[var_id]
        phys_range = var_info.get('physical_range', [None, None])
        low, high = phys_range

        col_data = df[orig_col].copy()
        original_valid = col_data.notna().sum()

        if low is not None:
            invalid_low = (col_data < low).sum()
            if invalid_low > 0:
                col_data[col_data < low] = np.nan
                clipped_counts[orig_col] = clipped_counts.get(orig_col, 0) + invalid_low

        if high is not None:
            invalid_high = (col_data > high).sum()
            if invalid_high > 0:
                col_data[col_data > high] = np.nan
                clipped_counts[orig_col] = clipped_counts.get(orig_col, 0) + invalid_high

        df[orig_col] = col_data

    if clipped_counts:
        print(f"  Clipped {len(clipped_counts)} columns with out-of-range values")

    # Compute statistics
    stats = {
        'total_rows': len(df),
        'total_columns': len(available_cols),
        'time_start': str(df.index[0]),
        'time_end': str(df.index[-1]),
        'missing_columns': [col for _, col in missing_cols],
        'missing_rate_per_column': {},
        'clipped_counts': clipped_counts,
    }

    for var_id, orig_col in available_cols:
        if orig_col in df.columns:
            missing_rate = df[orig_col].isna().mean()
            stats['missing_rate_per_column'][orig_col] = float(missing_rate)

    return df, stats, available_cols


def create_target_array(
    df: pd.DataFrame,
    schema: Dict,
    available_cols: List[Tuple[int, str]],
    num_variates: int
) -> np.ndarray:
    """
    Create target array in shape (num_variates, time_steps).

    Follows "All-in-Target" strategy: all variables go into target.
    Missing variables are filled with NaN.
    """
    time_steps = len(df)
    target = np.full((num_variates, time_steps), np.nan, dtype=np.float32)

    for var_id, orig_col in available_cols:
        if orig_col in df.columns:
            target[var_id, :] = df[orig_col].values.astype(np.float32)

    return target


def split_into_windows(
    df: pd.DataFrame,
    target: np.ndarray,
    window_size: int,
    stride: int,
    min_valid_ratio: float = 0.5
) -> List[Dict]:
    """
    Split data into overlapping windows for training.

    Args:
        df: DataFrame with timestamp index
        target: Target array (num_variates, time_steps)
        window_size: Number of time steps per window
        stride: Step between windows
        min_valid_ratio: Minimum ratio of non-NaN values required

    Returns:
        List of sample dicts in uni2ts format
    """
    samples = []
    num_variates, total_steps = target.shape

    for start_idx in range(0, total_steps - window_size + 1, stride):
        end_idx = start_idx + window_size

        window_target = target[:, start_idx:end_idx]

        # Check validity: require min_valid_ratio of data points
        valid_ratio = np.isfinite(window_target).mean()
        if valid_ratio < min_valid_ratio:
            continue

        # Get start timestamp
        start_time = df.index[start_idx]

        sample = {
            'item_id': f"east_{start_time.strftime('%Y%m%d_%H%M')}",
            'start': start_time,
            'target': window_target,  # (num_variates, window_size)
            'freq': pd.infer_freq(df.index[:100]) or '1min',
        }
        samples.append(sample)

    return samples


def save_to_arrow(
    samples: List[Dict],
    output_path: Path,
    split: str = 'train'
):
    """Save samples to Arrow format for uni2ts."""
    output_path.mkdir(parents=True, exist_ok=True)

    # Convert to Arrow-compatible format
    records = []
    for sample in tqdm(samples, desc=f"Converting {split}"):
        record = {
            'item_id': sample['item_id'],
            'start': sample['start'].isoformat(),
            'freq': sample['freq'],
            'target': sample['target'].tolist(),  # Convert ndarray to nested list
        }
        records.append(record)

    # Save as parquet (Arrow format)
    df = pd.DataFrame(records)
    output_file = output_path / f"{split}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"  Saved {len(records)} samples to {output_file}")

    return output_file


def save_gluonts_format(
    samples: List[Dict],
    output_path: Path,
    split: str = 'train'
):
    """
    Save in GluonTS ListDataset-compatible JSONL format.
    This is the native format for uni2ts/GluonTS.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / f"{split}.jsonl"

    with open(output_file, 'w') as f:
        for sample in tqdm(samples, desc=f"Saving {split}"):
            record = {
                'item_id': sample['item_id'],
                'start': sample['start'].isoformat(),
                'freq': sample['freq'],
                'target': sample['target'].tolist(),
            }
            f.write(json.dumps(record) + '\n')

    print(f"  Saved {len(samples)} samples to {output_file}")
    return output_file


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def create_metadata(
    schema: Dict,
    stats: Dict,
    output_path: Path,
    num_variates: int,
    freq: str
):
    """Create metadata file for the dataset."""
    metadata = {
        'dataset_name': 'buildingfm_east',
        'created_at': datetime.now().isoformat(),
        'schema_version': schema.get('schema_version', '1.0'),
        'building_id': schema.get('building_id', 'East'),
        'num_variates': num_variates,
        'freq': freq,
        'data_stats': stats,
        'variable_groups': schema.get('groups', {}),
    }

    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, cls=NumpyEncoder)
    print(f"  Saved metadata to {metadata_file}")


def main():
    # Setup paths from CONFIG
    schema_path = Path(CONFIG['schema_path'])
    input_path = Path(CONFIG['input_csv'])
    output_path = Path(CONFIG['output_dir'])

    # Load schema
    print("\n" + "="*60)
    print("BuildingFM Data Preparation")
    print("="*60)
    schema = load_schema(schema_path)

    # Determine number of variates from schema
    num_variates = max(int(k) for k in schema['variables'].keys()) + 1
    print(f"Schema defines {num_variates} variables (IDs 0-{num_variates-1})")

    # Load and validate data
    df, stats, available_cols = load_and_validate_csv(
        input_path, schema, resample_freq=CONFIG['resample_freq']
    )

    # Create target array
    print("\nCreating target array...")
    target = create_target_array(df, schema, available_cols, num_variates)
    print(f"  Target shape: {target.shape} (variates x time_steps)")
    print(f"  Overall missing rate: {np.isnan(target).mean():.2%}")

    # Per-group missing rate
    print("\n  Missing rate by group:")
    for group_name, group_info in schema.get('groups', {}).items():
        id_range = group_info['id_range']
        group_data = target[id_range[0]:id_range[1]+1, :]
        missing_rate = np.isnan(group_data).mean()
        print(f"    {group_name}: {missing_rate:.2%}")

    # Split into windows
    print(f"\nSplitting into windows (size={CONFIG['window_size']}, stride={CONFIG['stride']})...")
    samples = split_into_windows(
        df, target,
        window_size=CONFIG['window_size'],
        stride=CONFIG['stride'],
        min_valid_ratio=CONFIG['min_valid_ratio']
    )
    print(f"  Created {len(samples)} valid windows")

    # Train/Val/Test split
    n_samples = len(samples)
    n_train = int(n_samples * CONFIG['train_ratio'])
    n_val = int(n_samples * (1 - CONFIG['train_ratio']) / 2)
    n_test = n_samples - n_train - n_val

    # Chronological split (not random)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:n_train + n_val]
    test_samples = samples[n_train + n_val:]

    print(f"\n  Split: {n_train} train, {n_val} val, {n_test} test")

    # Determine frequency
    freq = CONFIG['resample_freq'] or '1min'

    # Save data
    print("\nSaving processed data...")

    if CONFIG['output_format'] in ['arrow', 'both']:
        arrow_path = output_path / 'arrow'
        save_to_arrow(train_samples, arrow_path, 'train')
        save_to_arrow(val_samples, arrow_path, 'val')
        save_to_arrow(test_samples, arrow_path, 'test')

    if CONFIG['output_format'] in ['jsonl', 'both']:
        jsonl_path = output_path / 'jsonl'
        save_gluonts_format(train_samples, jsonl_path, 'train')
        save_gluonts_format(val_samples, jsonl_path, 'val')
        save_gluonts_format(test_samples, jsonl_path, 'test')

    # Save metadata
    create_metadata(schema, stats, output_path, num_variates, freq)

    print("\n" + "="*60)
    print("Data preparation complete!")
    print("="*60)
    print(f"Output directory: {output_path}")
    print(f"Total samples: {len(samples)}")
    print(f"Variables: {num_variates}")
    print(f"Window size: {CONFIG['window_size']} steps")


if __name__ == '__main__':
    main()