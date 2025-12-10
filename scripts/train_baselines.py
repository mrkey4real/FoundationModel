#!/usr/bin/env python
"""
BuildingFM Baseline Training Script

Train traditional ML baselines (XGBoost) for comparison with MOIRAI.
This script is SEPARATE from evaluation - train once, evaluate many times.

Features:
    - Lag features: lag_1, lag_60, lag_1440, lag_10080 (1min, 1hr, 1day, 1week)
    - Time features: hour, dayofweek, month

Output:
    - outputs/baselines/xgboost_{group_name}.joblib

Usage:
    python scripts/train_baselines.py
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import datasets

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path('../data/buildingfm_processed_15min')
OUTPUT_DIR = Path('../outputs/baselines_15min')

# Variable groups to train (same as evaluation)
TRAIN_GROUPS = {
    'Main Power': {'id_range': (10, 11), 'unit': 'kW'},
    'ODU Power': {'id_range': (12, 13), 'unit': 'kW'},
    'IDU Power': {'id_range': (30, 33), 'unit': 'kW'},
    'Zone Temps': {'id_range': (50, 61), 'unit': '°C'},
    'IAQ': {'id_range': (98, 101), 'unit': 'ppm/ug'},
}

# =============================================================================
# Frequency Configuration - ONLY CHANGE THIS!
# =============================================================================
DATA_FREQ = '15min'  # Options: '1min', '5min', '15min', '30min', '1H', etc.

# Auto-calculated time steps (DO NOT modify manually)
_freq_minutes = pd.Timedelta(DATA_FREQ).total_seconds() / 60
STEPS_PER_HOUR = int(60 / _freq_minutes)
STEPS_PER_DAY = int(24 * 60 / _freq_minutes)
STEPS_PER_WEEK = int(7 * 24 * 60 / _freq_minutes)

# Feature engineering (lag steps - auto-calculated)
LAG_STEPS = [1, STEPS_PER_HOUR, STEPS_PER_HOUR * 6, STEPS_PER_DAY]  # 1step, 1hr, 6hr, 1day

# XGBoost parameters
# 自动检测设备: 如果有GPU则使用GPU加速
XGBOOST_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',  # 'hist' 支持CPU和GPU
    'device': 'cuda' if __import__('torch').cuda.is_available() else 'cpu',
}

# Training settings
MAX_TRAIN_SAMPLES = 500000  # Limit samples per variable for memory


# =============================================================================
# Feature Engineering
# =============================================================================

def create_lag_features(data: np.ndarray, lag_steps: List[int]) -> np.ndarray:
    """
    Create lag features for time series data.
    
    Args:
        data: 1D array of time series values
        lag_steps: List of lag offsets
    
    Returns:
        2D array of shape (valid_length, num_lags)
    """
    n = len(data)
    max_lag = max(lag_steps)
    
    if n <= max_lag:
        return np.array([]).reshape(0, len(lag_steps))
    
    valid_length = n - max_lag
    features = np.zeros((valid_length, len(lag_steps)), dtype=np.float32)
    
    for i, lag in enumerate(lag_steps):
        features[:, i] = data[max_lag - lag : n - lag]
    
    return features


def create_time_features(timestamps: pd.DatetimeIndex, start_idx: int) -> np.ndarray:
    """
    Create time-based features.
    
    Args:
        timestamps: DatetimeIndex of full series
        start_idx: Starting index (after max lag)
    
    Returns:
        2D array of shape (valid_length, 3) with [hour, dayofweek, month]
    """
    valid_ts = timestamps[start_idx:]
    features = np.zeros((len(valid_ts), 3), dtype=np.float32)
    
    features[:, 0] = valid_ts.hour
    features[:, 1] = valid_ts.dayofweek
    features[:, 2] = valid_ts.month
    
    return features


def prepare_training_data(
    hf_dataset: datasets.Dataset,
    var_ids: List[int],
    max_samples: int = MAX_TRAIN_SAMPLES
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training data for XGBoost from HuggingFace dataset.
    
    Args:
        hf_dataset: HuggingFace dataset with 'target' field
        var_ids: List of variable indices to include
        max_samples: Maximum number of samples to use
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
    """
    max_lag = max(LAG_STEPS)
    
    all_X = []
    all_y = []
    
    for sample in hf_dataset:
        target = np.array(sample['target'])  # (num_variates, seq_len)
        start_ts = pd.Timestamp(sample['start'])
        freq = sample['freq']
        
        # Create timestamp index
        seq_len = target.shape[1]
        timestamps = pd.date_range(start=start_ts, periods=seq_len, freq=freq)
        
        for var_id in var_ids:
            if var_id >= target.shape[0]:
                continue
            
            var_data = target[var_id, :]
            
            # Skip if too much NaN
            if np.isnan(var_data).mean() > 0.5:
                continue
            
            # Fill NaN with forward fill then backward fill
            var_data = pd.Series(var_data).ffill().bfill().values
            
            # Create features
            lag_features = create_lag_features(var_data, LAG_STEPS)
            
            if len(lag_features) == 0:
                continue
            
            time_features = create_time_features(timestamps, max_lag)
            
            # Combine features
            X = np.hstack([lag_features, time_features])
            y = var_data[max_lag:]
            
            # Remove NaN rows
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            all_X.append(X)
            all_y.append(y)
        
        # Check if we have enough samples
        total_samples = sum(len(y) for y in all_y)
        if total_samples >= max_samples:
            break
    
    if len(all_X) == 0:
        return np.array([]).reshape(0, len(LAG_STEPS) + 3), np.array([])
    
    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    
    # Subsample if too large
    if len(y) > max_samples:
        indices = np.random.choice(len(y), max_samples, replace=False)
        X = X[indices]
        y = y[indices]
    
    return X, y


# =============================================================================
# Training Functions
# =============================================================================

def train_xgboost_for_group(
    hf_dataset: datasets.Dataset,
    group_name: str,
    var_ids: List[int],
    output_dir: Path
) -> Path:
    """
    Train XGBoost model for a variable group.
    
    Args:
        hf_dataset: Training dataset
        group_name: Name of variable group
        var_ids: Variable indices in this group
        output_dir: Directory to save model
    
    Returns:
        Path to saved model
    """
    from xgboost import XGBRegressor
    
    print(f"  Preparing data for {group_name}...")
    X, y = prepare_training_data(hf_dataset, var_ids)
    
    if len(y) < 100:
        print(f"    WARNING: Not enough data ({len(y)} samples), skipping")
        return None
    
    print(f"    Training samples: {len(y):,}")
    print(f"    Features: {X.shape[1]} (lags: {len(LAG_STEPS)}, time: 3)")
    
    # Train model
    print(f"    Training XGBoost...")
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X, y)
    
    # Save model
    safe_name = group_name.replace(' ', '_').lower()
    model_path = output_dir / f'xgboost_{safe_name}.joblib'
    joblib.dump(model, model_path)
    print(f"    Saved: {model_path}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_names = [f'lag_{l}' for l in LAG_STEPS] + ['hour', 'dayofweek', 'month']
    print(f"    Feature importance:")
    for name, imp in sorted(zip(feature_names, importance), key=lambda x: -x[1])[:5]:
        print(f"      {name}: {imp:.3f}")
    
    return model_path


def train_combined_model(
    hf_dataset: datasets.Dataset,
    output_dir: Path
) -> Path:
    """
    Train a single XGBoost model on all variables combined.
    This is simpler and often works well for general forecasting.
    """
    from xgboost import XGBRegressor
    
    print(f"\n  Training combined model on all variables...")
    
    # Collect all variable IDs
    all_var_ids = []
    for group_info in TRAIN_GROUPS.values():
        id_start, id_end = group_info['id_range']
        all_var_ids.extend(range(id_start, id_end + 1))
    
    X, y = prepare_training_data(hf_dataset, all_var_ids)
    
    if len(y) < 100:
        print(f"    WARNING: Not enough data ({len(y)} samples), skipping")
        return None
    
    print(f"    Training samples: {len(y):,}")
    
    model = XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X, y)
    
    # Save
    model_path = output_dir / 'xgboost_model.joblib'
    joblib.dump(model, model_path)
    print(f"    Saved: {model_path}")
    
    return model_path


# =============================================================================
# Main
# =============================================================================

def main():
    """Main training function."""
    
    print("=" * 60)
    print("BuildingFM Baseline Training (XGBoost)")
    print("=" * 60)
    
    # 检测设备
    import torch
    device_available = torch.cuda.is_available()
    if device_available:
        print(f"\n设备: GPU - {torch.cuda.get_device_name(0)}")
        print(f"  XGBoost将使用GPU加速训练")
    else:
        print(f"\n设备: CPU")
        print(f"  XGBoost将使用CPU训练")
    
    # Setup
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    hf_data_dir = DATA_DIR / 'hf'
    print(f"\nLoading training data from {hf_data_dir}...")
    train_hf = datasets.load_from_disk(str(hf_data_dir / 'buildingfm_train'))
    print(f"  Training samples (sequences): {len(train_hf)}")
    
    # Check data shape
    sample = train_hf[0]
    target = np.array(sample['target'])
    print(f"  Variates: {target.shape[0]}")
    print(f"  Sequence length: {target.shape[1]}")
    
    # Train combined model (main model for evaluation)
    print("\n[1/2] Training combined XGBoost model...")
    combined_path = train_combined_model(train_hf, output_dir)
    
    # Train per-group models (optional, for detailed analysis)
    print("\n[2/2] Training per-group XGBoost models...")
    group_models = {}
    for group_name, group_info in TRAIN_GROUPS.items():
        print(f"\n  Group: {group_name}")
        id_start, id_end = group_info['id_range']
        var_ids = list(range(id_start, id_end + 1))
        
        model_path = train_xgboost_for_group(train_hf, group_name, var_ids, output_dir)
        if model_path:
            group_models[group_name] = str(model_path)
    
    # Save metadata
    metadata = {
        'combined_model': str(combined_path) if combined_path else None,
        'group_models': group_models,
        'lag_steps': LAG_STEPS,
        'xgboost_params': XGBOOST_PARAMS,
        'feature_names': [f'lag_{l}' for l in LAG_STEPS] + ['hour', 'dayofweek', 'month'],
    }
    
    with open(output_dir / 'baselines_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModels saved to: {output_dir}")
    print(f"  - Combined model: xgboost_model.joblib")
    print(f"  - Per-group models: xgboost_{{group}}.joblib")
    print(f"  - Metadata: baselines_metadata.json")
    print("\nRun evaluate_models.py to compare with MOIRAI models.")


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    main()

