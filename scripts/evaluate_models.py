#!/usr/bin/env python
"""
BuildingFM Model Evaluation Script (Refactored)

Evaluation Framework for MOIRAI Foundation Model on HVAC Time Series.

Models evaluated:
    - Seasonal Naive: Periodic baseline (same time yesterday)
    - XGBoost: Traditional ML baseline (load from pretrained)
    - MOIRAI Small (Zero-shot): Pretrained foundation model
    - MOIRAI Small (Fine-tuned): Domain-adapted model
    - MOIRAI Base (Zero-shot): Larger pretrained model
    - MOIRAI Base (Fine-tuned): Larger domain-adapted model

Evaluation Tasks:
    - Task 1: Standard Forecasting (time extrapolation)
    - Task 2: Fill-in-the-Blank (causal chain validation)
    - Task 3: OOD Stress Test (extreme condition generalization)

Metrics:
    - Primary: SMAPE, CRPS, MSIS (probabilistic metrics)
    - Secondary: Smoothed MAE, Consistency Score

IMPORTANT - Point Prediction Strategy:
    MOIRAI uses probabilistic forecasting (outputs distribution, not point).
    For point metrics, the OFFICIAL method is:
        1. Sample N times from the distribution
        2. Take MEDIAN as point prediction (for L1 loss optimality)

Usage:
    python scripts/evaluate_models.py
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
import datasets
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from uni2ts.model.moirai import MoiraiModule
from uni2ts.distribution import (
    MixtureOutput,
    StudentTOutput,
    NormalFixedScaleOutput,
    NegativeBinomialOutput,
    LogNormalOutput,
)

# =============================================================================
# Configuration
# =============================================================================

# =============================================================================
# Model Paths - 与 train_buildingfm.py 中的 model_name 对应
# =============================================================================
# 命名格式: moirai_{size}_{pattern}
#   - size: small | base | large
#   - pattern: full | freeze_ffn | head_only
#
# 目录结构:
#   outputs/buildingfm_15min/
#     ├── moirai_small_full/          <- 全量微调 small
#     ├── moirai_small_freeze_ffn/    <- 冻结FFN small
#     ├── moirai_small_head_only/     <- 只训练head small
#     ├── moirai_base_full/           <- 全量微调 base
#     ├── moirai_base_freeze_ffn/     <- 冻结FFN base
#     ├── moirai_base_head_only/      <- 只训练head base
#     │   ├── checkpoints/last.ckpt
#     │   └── baseline_untrained.pt
#     └── ...

MODEL_OUTPUT_DIR = Path('../outputs/buildingfm_15min')
BASELINES_DIR = Path('../outputs/baselines_15min')

# =============================================================================
# 评估模型配置 - 支持多种微调策略对比
# =============================================================================
# 模型尺寸
MODEL_SIZES = ['small', 'base']

# 微调策略
FINETUNE_PATTERNS = ['full', 'freeze_ffn', 'head_only']

# 自动生成所有模型目录 (基础命名，不含lr)
def get_model_dir(size: str, pattern: str) -> Path:
    return MODEL_OUTPUT_DIR / f'moirai_{size}_{pattern}'


def discover_all_models(output_dir: Path) -> Dict[str, List[Path]]:
    """
    自动发现所有 moirai_* 模型目录，按 size+pattern 分组。

    Returns:
        Dict mapping 'size_pattern' -> list of model directories
        Example: {'small_freeze_ffn': [moirai_small_freeze_ffn, moirai_small_freeze_ffn_1e05, ...]}
    """
    discovered = {}

    for model_dir in output_dir.glob('moirai_*'):
        if not model_dir.is_dir():
            continue

        # 检查是否有checkpoint
        ckpt_dir = model_dir / 'checkpoints'
        if not ckpt_dir.exists():
            continue

        best_ckpts = list(ckpt_dir.glob('best-*.ckpt'))
        if not best_ckpts:
            continue

        # 解析名称: moirai_{size}_{pattern}[_{lr}][_{timestamp}]
        name_parts = model_dir.name.split('_')
        if len(name_parts) < 3:
            continue

        size = name_parts[1]  # small or base

        # 确定pattern (可能是复合词如 freeze_ffn 或 head_only)
        pattern = None
        for p in FINETUNE_PATTERNS:
            p_parts = p.split('_')
            if name_parts[2:2+len(p_parts)] == p_parts:
                pattern = p
                break

        if pattern is None:
            continue

        key = f'{size}_{pattern}'
        if key not in discovered:
            discovered[key] = []
        discovered[key].append(model_dir)

    return discovered


def find_best_model_for_group(model_dirs: List[Path]) -> Optional[Tuple[Path, float]]:
    """
    从同一 size+pattern 的多个模型中选择最佳的 (最低val_loss)。

    Returns:
        (best_model_dir, best_val_loss) or None
    """
    best_dir = None
    best_loss = float('inf')

    for model_dir in model_dirs:
        # 尝试读取训练历史
        history_sources = [
            model_dir / 'csv_logs' / 'version_0' / 'metrics.csv',
            model_dir / 'training_history.csv',
        ]

        val_loss = None
        for history_file in history_sources:
            if not history_file.exists():
                continue

            try:
                df = pd.read_csv(history_file)

                if 'val/PackedNLLLoss' in df.columns:
                    val_df = df[df['val/PackedNLLLoss'].notna()]
                    if len(val_df) > 0:
                        val_loss = val_df['val/PackedNLLLoss'].min()
                        break
                elif 'val_loss' in df.columns:
                    valid = df['val_loss'].dropna()
                    if len(valid) > 0:
                        val_loss = valid.min()
                        break
            except Exception:
                continue

        if val_loss is not None and val_loss < best_loss:
            best_loss = val_loss
            best_dir = model_dir

    if best_dir is not None:
        return (best_dir, best_loss)
    return None

# 显示名称映射 (用于图表)
DISPLAY_NAMES = {
    'small_full': 'Small-Full',
    'small_freeze_ffn': 'Small-FreezeFNN',
    'small_head_only': 'Small-HeadOnly',
    'base_full': 'Base-Full',
    'base_freeze_ffn': 'Base-FreezeFNN',
    'base_head_only': 'Base-HeadOnly',
}

# 颜色方案 (专业配色)
MODEL_COLORS = {
    # Baselines
    'Seasonal Naive': '#7f8c8d',
    'XGBoost': '#f39c12',
    # Zero-shot
    'Small (Zero-shot)': '#85c1e9',
    'Base (Zero-shot)': '#f1948a',
    # Full finetune
    'Small-Full': '#2980b9',
    'Base-Full': '#c0392b',
    # Freeze FFN
    'Small-FreezeFNN': '#27ae60',
    'Base-FreezeFNN': '#8e44ad',
    # Head only
    'Small-HeadOnly': '#16a085',
    'Base-HeadOnly': '#d35400',
}

# Data path
DATA_DIR = Path('../data/buildingfm_processed_15min')

# Output
OUTPUT_DIR = Path('../outputs/evaluation_15min')

# =============================================================================
# Plot Style Configuration - 更大更美观的图表
# =============================================================================
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 180,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 15,
})

# =============================================================================
# Frequency Configuration - ONLY CHANGE THIS!
# =============================================================================
DATA_FREQ = '15min'  # Options: '1min', '5min', '15min', '30min', '1H', etc.

# Auto-calculated time steps (DO NOT modify manually)
_freq_minutes = pd.Timedelta(DATA_FREQ).total_seconds() / 60
STEPS_PER_HOUR = int(60 / _freq_minutes)
STEPS_PER_DAY = int(24 * 60 / _freq_minutes)
STEPS_PER_WEEK = int(7 * 24 * 60 / _freq_minutes)

# =============================================================================
# Window Configuration - MUST ALIGN WITH prepare_buildingfm_data.py!
# =============================================================================
# prepare_buildingfm_data.py: WINDOW_DAYS = 3, window_size = 96 * 3 = 288 (3 days)
# CONSTRAINT: CONTEXT_LENGTH + PREDICTION_LENGTH <= window_size
SAMPLE_WINDOW_SIZE = STEPS_PER_DAY * 3  # Must match prepare_buildingfm_data.py

# Prediction settings (expressed in human-readable units)
# 设计理由:
# - 训练窗口为3天，评估窗口需要匹配
# - 2天context + 1天prediction = 3天，刚好匹配训练窗口
CONTEXT_DAYS = 2            # 2 days of history (192 steps @ 15min)
PREDICTION_DAYS = 1         # 1 day forecast horizon (96 steps @ 15min)
CONTEXT_LENGTH = CONTEXT_DAYS * STEPS_PER_DAY
PREDICTION_LENGTH = PREDICTION_DAYS * STEPS_PER_DAY

# Validation: ensure we don't exceed sample window
assert CONTEXT_LENGTH + PREDICTION_LENGTH <= SAMPLE_WINDOW_SIZE, \
    f"CONTEXT({CONTEXT_LENGTH}) + PREDICTION({PREDICTION_LENGTH}) > WINDOW({SAMPLE_WINDOW_SIZE})!"

# PATCH_SIZE: target ~8 hours per patch for good daily pattern capture
PATCH_SIZE = 8
NUM_SAMPLES = 20            # Reduced from 50 - still good for CRPS/MSIS
SEASONAL_PERIOD = STEPS_PER_DAY  # Daily seasonality (auto-calculated)

# Evaluation settings
MAX_EVAL_SAMPLES = 30        # Reduced from 100 - faster evaluation
CONFIDENCE_LEVEL = 0.95     # For MSIS calculation

# Variable groups from metadata
EVAL_GROUPS = {
    'Main Power': {'id_range': (10, 11), 'unit': 'kW'},
    'ODU Power': {'id_range': (12, 13), 'unit': 'kW'},
    'IDU Power': {'id_range': (30, 33), 'unit': 'kW'},
    'Zone Temps': {'id_range': (50, 61), 'unit': '°C'},  # Subset for efficiency
    'IAQ': {'id_range': (98, 101), 'unit': 'ppm/ug'},
}

# OOD thresholds for stress testing 
# Note: Thresholds based on actual data distribution (use percentiles)
# Will be dynamically adjusted based on test data if needed
OOD_THRESHOLDS = {
    'high_temp': 22.0,      # Outdoor temp > 22°C (upper quartile)
    'low_temp': 17.0,       # Outdoor temp < 17°C (lower quartile)
    'high_load': 0.5,       # Main power > 50% of max (high HVAC load)
}

# Variable indices for fill-in-the-blank task
WEATHER_VAR_IDS = list(range(0, 8))       # Weather variables
ODU_POWER_VAR_IDS = list(range(12, 14))   # ODU power (to be masked/predicted)
ZONE_TEMP_VAR_IDS = list(range(50, 62))   # Zone temperatures


# =============================================================================
# Part 1: Metrics Functions
# =============================================================================

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Symmetric Mean Absolute Percentage Error.
    
    SMAPE = 200 * mean(|y - ŷ| / (|y| + |ŷ|))
    
    Properties:
        - Bounded [0, 200]
        - Symmetric: same penalty for over/under prediction
        - Handles zeros gracefully (unlike MAPE)
        - Good for HVAC: stable during low-load periods
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 10:
        return np.nan
    
    denominator = np.abs(y_true) + np.abs(y_pred)
    # Avoid division by zero: when both are zero, SMAPE contribution is 0
    nonzero_mask = denominator > 1e-8
    
    if nonzero_mask.sum() < 10:
        return 0.0  # All zeros means perfect match
    
    smape = 200.0 * np.mean(
        np.abs(y_true[nonzero_mask] - y_pred[nonzero_mask]) / denominator[nonzero_mask]
    )
    return smape


def calculate_crps(
    y_true: np.ndarray,
    samples: np.ndarray
) -> float:
    """
    Continuous Ranked Probability Score (Vectorized).

    CRPS measures the integral of squared difference between predicted CDF
    and empirical CDF (step function at observed value).

    For samples, CRPS = E[|X - y|] - 0.5 * E[|X - X'|]
    where X, X' are independent samples from the forecast distribution.

    Properties:
        - Proper scoring rule (optimized by true distribution)
        - Rewards sharp forecasts that cover truth
        - Unit is same as target variable

    Args:
        y_true: (T,) ground truth values
        samples: (N, T) samples from predictive distribution
    """
    y_true = y_true.flatten()
    N, T = samples.shape

    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    samples = samples[:, mask]

    if len(y_true) < 10:
        return np.nan

    # E[|X - y|]: mean absolute error of samples to truth
    mae_term = np.mean(np.abs(samples - y_true))

    # E[|X - X'|]: Vectorized computation (O(N) instead of O(N²))
    # Use the identity: E[|X - X'|] = 2 * E[|X - median(X)|] for symmetric distributions
    # Or use sorted samples method: E[|X - X'|] = 2 * sum((2i - N - 1) * X_sorted[i]) / (N * (N-1))
    if N > 1:
        # Sort samples along sample axis, then use vectorized formula
        sorted_samples = np.sort(samples, axis=0)  # (N, T)
        # Weights: (2*i - N - 1) / (N * (N-1)) for i in 0..N-1
        weights = (2 * np.arange(N) - N + 1) / (N * (N - 1))
        diff_term = 2 * np.mean(np.sum(weights[:, None] * sorted_samples, axis=0))
    else:
        diff_term = 0.0

    crps = mae_term - 0.5 * diff_term
    return crps


def calculate_msis(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    seasonal_error: float,
    alpha: float = 0.05
) -> float:
    """
    Mean Scaled Interval Score.
    
    MSIS penalizes:
        1. Wide intervals (poor sharpness)
        2. Non-coverage (truth outside interval)
    
    Formula:
        MSIS = mean((U - L) + (2/α)*(L - y)*1(y < L) + (2/α)*(y - U)*1(y > U)) / seasonal_error
    
    Properties:
        - Scaled by seasonal naive error for comparability
        - Lower is better
        - Balances coverage vs sharpness
    
    Args:
        y_true: Ground truth
        lower: Lower bound of prediction interval (α/2 quantile)
        upper: Upper bound of prediction interval (1 - α/2 quantile)
        seasonal_error: MAE of seasonal naive for scaling
        alpha: Significance level (default 0.05 for 95% interval)
    """
    y_true = y_true.flatten()
    lower = lower.flatten()
    upper = upper.flatten()
    
    mask = ~(np.isnan(y_true) | np.isnan(lower) | np.isnan(upper))
    y_true = y_true[mask]
    lower = lower[mask]
    upper = upper[mask]
    
    if len(y_true) < 10:
        return np.nan
    
    # Interval width
    width = upper - lower
    
    # Penalty for under-prediction (truth below lower bound)
    under_penalty = (2.0 / alpha) * np.maximum(lower - y_true, 0)
    
    # Penalty for over-prediction (truth above upper bound)
    over_penalty = (2.0 / alpha) * np.maximum(y_true - upper, 0)
    
    # Total score
    score = np.mean(width + under_penalty + over_penalty)
    
    # Scale by seasonal error (avoid division by zero)
    if seasonal_error > 1e-8:
        msis = score / seasonal_error
    else:
        msis = score
    
    return msis


def calculate_smoothed_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 15
) -> float:
    """
    Smoothed Mean Absolute Error.
    
    Applies rolling average before computing MAE.
    Useful for presenting trend accuracy to non-technical stakeholders.
    
    Args:
        window: Smoothing window size (default 15 = 15 minutes)
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < window:
        return np.nan
    
    # Rolling mean using convolution
    kernel = np.ones(window) / window
    y_true_smooth = np.convolve(y_true, kernel, mode='valid')
    y_pred_smooth = np.convolve(y_pred, kernel, mode='valid')
    
    return np.mean(np.abs(y_true_smooth - y_pred_smooth))


def calculate_consistency_score(
    predictions: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray]
) -> float:
    """
    Physical Consistency Score.
    
    Measures whether predictions preserve inter-variable correlations.
    
    Score = 1 - mean(|corr_pred(Xi, Xj) - corr_true(Xi, Xj)|) for all pairs
    
    Properties:
        - Range [0, 1], higher is better
        - 1.0 means perfect correlation preservation
        - Important for HVAC: power should correlate with temp deviation
    """
    var_names = list(predictions.keys())
    if len(var_names) < 2:
        return np.nan
    
    corr_diffs = []
    
    for i, var1 in enumerate(var_names):
        for var2 in var_names[i + 1:]:
            pred1 = predictions[var1].flatten()
            pred2 = predictions[var2].flatten()
            true1 = ground_truth[var1].flatten()
            true2 = ground_truth[var2].flatten()
            
            # Align lengths
            min_len = min(len(pred1), len(pred2), len(true1), len(true2))
            pred1, pred2 = pred1[:min_len], pred2[:min_len]
            true1, true2 = true1[:min_len], true2[:min_len]
            
            # Remove NaN pairs
            mask = ~(np.isnan(pred1) | np.isnan(pred2) | np.isnan(true1) | np.isnan(true2))
            if mask.sum() < 10:
                continue
            
            pred1, pred2 = pred1[mask], pred2[mask]
            true1, true2 = true1[mask], true2[mask]
            
            # Calculate correlations
            corr_pred = np.corrcoef(pred1, pred2)[0, 1]
            corr_true = np.corrcoef(true1, true2)[0, 1]
            
            if not (np.isnan(corr_pred) or np.isnan(corr_true)):
                corr_diffs.append(np.abs(corr_pred - corr_true))
    
    if len(corr_diffs) == 0:
        return np.nan
    
    return 1.0 - np.mean(corr_diffs)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    samples: Optional[np.ndarray] = None,
    seasonal_error: float = 1.0
) -> Dict[str, float]:
    """
    Calculate all evaluation metrics.
    
    Args:
        y_true: Ground truth
        y_pred: Point prediction (median of samples)
        samples: (N, T) samples from predictive distribution (for CRPS/MSIS)
        seasonal_error: MAE of seasonal naive for MSIS scaling
    
    Returns:
        Dictionary of all metrics
    """
    metrics = {
        'SMAPE': calculate_smape(y_true, y_pred),
        'SmoothedMAE': calculate_smoothed_mae(y_true, y_pred),
    }
    
    # MAE for backward compatibility
    mask = ~(np.isnan(y_true.flatten()) | np.isnan(y_pred.flatten()))
    if mask.sum() >= 10:
        metrics['MAE'] = np.mean(np.abs(y_true.flatten()[mask] - y_pred.flatten()[mask]))
    else:
        metrics['MAE'] = np.nan
    
    # Probabilistic metrics (require samples)
    if samples is not None and len(samples) > 1:
        metrics['CRPS'] = calculate_crps(y_true, samples)
        
        # Calculate prediction intervals from samples
        alpha = 1 - CONFIDENCE_LEVEL
        lower = np.percentile(samples, alpha / 2 * 100, axis=0)
        upper = np.percentile(samples, (1 - alpha / 2) * 100, axis=0)
        metrics['MSIS'] = calculate_msis(y_true, lower, upper, seasonal_error, alpha)
        
        # Coverage rate (auxiliary)
        y_flat = y_true.flatten()
        lower_flat = lower.flatten()
        upper_flat = upper.flatten()
        mask = ~(np.isnan(y_flat) | np.isnan(lower_flat) | np.isnan(upper_flat))
        if mask.sum() > 0:
            coverage = np.mean((y_flat[mask] >= lower_flat[mask]) & (y_flat[mask] <= upper_flat[mask]))
            metrics['Coverage'] = coverage * 100  # As percentage
    else:
        metrics['CRPS'] = np.nan
        metrics['MSIS'] = np.nan
        metrics['Coverage'] = np.nan
    
    return metrics


# =============================================================================
# Part 2: Baseline Models
# =============================================================================

class SeasonalNaiveModel:
    """
    Seasonal Naive Baseline.
    
    Predicts future values as same time from previous period (e.g., yesterday).
    For 1-minute data with daily seasonality: y_t = y_{t-1440}
    """
    
    def __init__(self, seasonal_period: int = SEASONAL_PERIOD):
        self.seasonal_period = seasonal_period
    
    def predict(
        self,
        context: np.ndarray,
        prediction_length: int
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate seasonal naive forecast.
        
        Args:
            context: Historical data (context_length,)
            prediction_length: Number of steps to forecast
        
        Returns:
            point_pred: Point prediction
            samples: None (deterministic model)
        """
        # For seasonal naive, we repeat the pattern from (seasonal_period) steps ago
        # If context is shorter than seasonal_period, use available history
        
        predictions = np.zeros(prediction_length)
        
        for t in range(prediction_length):
            # Index in history that corresponds to same time in previous period
            history_idx = len(context) - self.seasonal_period + t
            
            if history_idx >= 0 and history_idx < len(context):
                predictions[t] = context[history_idx]
            else:
                # Fallback to last value if seasonal period exceeds context
                predictions[t] = context[-1]
        
        return predictions, None


class XGBoostModel:
    """
    XGBoost Baseline (loads pretrained model).
    
    Features: lag_1, lag_60, lag_360, lag_1440, hour, dayofweek, month
    Must match train_baselines.py feature engineering!
    """
    
    # Lag steps: auto-calculated from global frequency settings
    LAG_STEPS = [1, STEPS_PER_HOUR, STEPS_PER_HOUR * 6, STEPS_PER_DAY]  # 1step, 1hr, 6hr, 1day
    
    def __init__(self, model_path: Optional[Path] = None):
        self.model = None
        self.model_path = model_path
        
        if model_path is not None and model_path.exists():
            import joblib
            self.model = joblib.load(model_path)
    
    def predict(
        self,
        context: np.ndarray,
        prediction_length: int,
        timestamps: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Generate XGBoost forecast.
        
        Note: XGBoost requires iterative prediction for multi-step forecast.
        """
        if self.model is None:
            return np.full(prediction_length, np.nan), None
        
        # Fill NaN in context
        context_clean = np.nan_to_num(context, nan=0.0)
        
        predictions = np.zeros(prediction_length)
        extended = np.concatenate([context_clean, np.zeros(prediction_length)])
        
        for t in range(prediction_length):
            current_idx = len(context_clean) + t
            
            # Create lag features (must match training)
            lag_features = []
            for lag in self.LAG_STEPS:
                if current_idx >= lag:
                    lag_features.append(extended[current_idx - lag])
                else:
                    lag_features.append(0.0)
            
            # Time features
            if timestamps is not None and t < len(timestamps):
                hour = timestamps[t].hour
                dayofweek = timestamps[t].dayofweek
                month = timestamps[t].month
            else:
                hour = ((current_idx // 60) % 24)
                dayofweek = ((current_idx // 1440) % 7)
                month = 6
            
            # Feature vector: [lag_1, lag_60, lag_360, lag_1440, hour, dayofweek, month]
            features = np.array([lag_features + [hour, dayofweek, month]])
            
            pred = self.model.predict(features)[0]
            predictions[t] = pred
            extended[current_idx] = pred
        
        return predictions, None


# =============================================================================
# Part 3: MOIRAI Model Interface (Preserved from original)
# =============================================================================

def create_distr_output():
    """Create the distribution output for MOIRAI"""
    return MixtureOutput(
        components=[
            StudentTOutput(),
            NormalFixedScaleOutput(),
            NegativeBinomialOutput(),
            LogNormalOutput(),
        ]
    )


def load_model_from_baseline(checkpoint_path: Path, device: str = 'cpu') -> MoiraiModule:
    """Load MoiraiModule from baseline .pt file"""
    state = torch.load(checkpoint_path, map_location=device)
    d_model = state['config']['d_model']
    num_layers = state['config']['num_layers']
    patch_sizes = state['config']['patch_sizes']
    max_seq_len = state['config']['max_seq_len']
    
    module = MoiraiModule(
        distr_output=create_distr_output(),
        d_model=d_model,
        num_layers=num_layers,
        patch_sizes=patch_sizes,
        max_seq_len=max_seq_len,
        attn_dropout_p=0.0,
        dropout_p=0.1,
        scaling=True,
    )
    
    module.load_state_dict(state['model_state_dict'])
    return module


def load_model_from_checkpoint(checkpoint_path: Path, device: str = 'cpu') -> MoiraiModule:
    """Load MoiraiModule from Lightning checkpoint (finetuned).
    
    Requires baseline_untrained.pt in parent directory for architecture config.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    state_dict = checkpoint['state_dict']
    module_state = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}
    
    # Load architecture from local baseline
    baseline_path = checkpoint_path.parent.parent / 'baseline_untrained.pt'
    assert baseline_path.exists(), f"Missing baseline: {baseline_path}"
    
    baseline_state = torch.load(baseline_path, map_location=device)
    config = baseline_state['config']
    
    module = MoiraiModule(
        distr_output=create_distr_output(),
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        patch_sizes=config['patch_sizes'],
        max_seq_len=config['max_seq_len'],
        attn_dropout_p=0.0,
        dropout_p=0.1,
        scaling=True,
    )
    module.load_state_dict(module_state)
    return module


def prepare_native_input(
    var_data: np.ndarray,
    context_length: int,
    prediction_length: int,
    patch_size: int,
    max_patch_size: int = 128,
    device: str = 'cpu',
    var_id: int = 0
) -> Tuple[torch.Tensor, ...]:
    """
    Prepare input tensors in the same format as MoiraiPretrain training.

    This creates a single-variate sequence with:
    - Context window: observed data (with actual values)
    - Prediction window: actual future values (but masked for prediction)

    The key is that we provide the FULL data (context + future), but mark
    the future region with prediction_mask=True and observed_mask=False.
    This allows the scaler to compute loc/scale from the context correctly.

    Args:
        var_id: The schema ID of the variable (for variate embedding consistency with training)

    Returns tensors matching MoiraiModule.forward() signature.
    """
    total_length = context_length + prediction_length
    var_data = var_data[:total_length].astype(np.float32)
    
    # Handle NaN: replace with 0 for target, track in observed_mask
    nan_mask = np.isnan(var_data)
    var_data_clean = np.nan_to_num(var_data, nan=0.0)
    
    # Ensure data length matches patch boundaries (pad at the BEGINNING if needed)
    num_patches = (total_length + patch_size - 1) // patch_size
    padded_len = num_patches * patch_size
    
    if padded_len > total_length:
        pad_amount = padded_len - total_length
        var_data_clean = np.concatenate([np.zeros(pad_amount, dtype=np.float32), var_data_clean])
        nan_mask = np.concatenate([np.ones(pad_amount, dtype=bool), nan_mask])  # pad region is "missing"
    
    # Calculate context patches accounting for left padding
    context_end_idx = padded_len - prediction_length
    context_patches = context_end_idx // patch_size
    
    # Create target tensor (batch=1, seq_len=num_patches, max_patch_size)
    target = np.zeros((1, num_patches, max_patch_size), dtype=np.float32)
    observed_mask = np.zeros((1, num_patches, max_patch_size), dtype=bool)
    
    for i in range(num_patches):
        start_idx = i * patch_size
        end_idx = start_idx + patch_size
        target[0, i, :patch_size] = var_data_clean[start_idx:end_idx]
        # observed_mask: True if data is in context AND not NaN
        if i < context_patches:
            observed_mask[0, i, :patch_size] = ~nan_mask[start_idx:end_idx]
        # Prediction region: observed_mask stays False
    
    # sample_id: 1 for all (indicating same sample, non-zero for proper scaling)
    sample_id = np.ones((1, num_patches), dtype=np.int64)
    
    # time_id: sequential from 0
    time_id = np.arange(num_patches, dtype=np.int64).reshape(1, -1)

    # variate_id: Use actual schema ID for consistency with training (fixed variate IDs)
    variate_id = np.full((1, num_patches), var_id, dtype=np.int64)
    
    # prediction_mask: 0 for context, 1 for prediction
    prediction_mask = np.zeros((1, num_patches), dtype=bool)
    prediction_mask[0, context_patches:] = True
    
    # patch_size tensor
    patch_size_tensor = np.full((1, num_patches), patch_size, dtype=np.int64)
    
    # Convert to torch tensors
    target = torch.tensor(target, device=device)
    observed_mask = torch.tensor(observed_mask, device=device)
    sample_id = torch.tensor(sample_id, device=device)
    time_id = torch.tensor(time_id, device=device)
    variate_id = torch.tensor(variate_id, device=device)
    prediction_mask = torch.tensor(prediction_mask, device=device)
    patch_size_tensor = torch.tensor(patch_size_tensor, device=device)
    
    return target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor


def predict_moirai(
    module: MoiraiModule,
    var_data: np.ndarray,
    context_length: int,
    prediction_length: int,
    patch_size: int = PATCH_SIZE,
    num_samples: int = NUM_SAMPLES,
    device: str = 'cpu',
    var_id: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make prediction using native MoiraiModule interface.
    Same data format as training.

    Args:
        var_data: Full window data (context_length + prediction_length)
        context_length: Number of timesteps in context
        prediction_length: Number of timesteps to predict
        var_id: Schema ID of the variable (for variate embedding consistency)

    Returns:
        point_pred: Median of samples (official MOIRAI point estimate)
        samples: (num_samples, prediction_length) for probabilistic metrics
    """
    max_patch_size = max(module.patch_sizes)

    target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor = \
        prepare_native_input(var_data, context_length, prediction_length, patch_size, max_patch_size, device, var_id)
    
    module = module.to(device)
    module.eval()
    
    with torch.no_grad():
        distr = module(
            target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor
        )
        
        # Sample from distribution
        samples = distr.sample((num_samples,))  # (num_samples, batch, seq, max_patch)
        
        # Get prediction region
        num_patches = target.shape[1]
        padded_len = num_patches * patch_size
        context_end_idx = padded_len - prediction_length
        context_patches = context_end_idx // patch_size
        
        pred_samples = samples[:, 0, context_patches:, :patch_size]  # (num_samples, pred_patches, patch_size)
        
        # Reshape to (num_samples, prediction_length)
        pred_samples = pred_samples.reshape(num_samples, -1)[:, :prediction_length]
        pred_samples_np = pred_samples.cpu().numpy()
        
        # Use MEDIAN - this is the OFFICIAL MOIRAI design!
        # See: src/uni2ts/model/moirai/pretrain.py line 237-238
        pred_median = np.median(pred_samples_np, axis=0)
    
    return pred_median, pred_samples_np


def prepare_multivariate_input(
    full_data: np.ndarray,
    context_length: int,
    prediction_length: int,
    patch_size: int,
    max_patch_size: int,
    device: str,
    weather_var_ids: List[int],
    target_var_ids: List[int],
    task_type: str = 'forecast'
) -> Tuple[torch.Tensor, ...]:
    """
    Unified multi-variate input preparation - ALL variates input, mask differs by task.

    This matches training where ALL variates are input together!

    Args:
        full_data: (num_variates, time_steps) - ALL variates
        context_length: Number of historical timesteps
        prediction_length: Number of future timesteps (for forecast task)
        weather_var_ids: Weather variables (0-7) - future is known in forecast
        target_var_ids: Variables we want to evaluate/predict
        task_type:
            - 'forecast': Weather future visible, all other vars future masked
            - 'virtual_sensor': Target vars masked entirely, others fully visible

    Mask strategy:
        - forecast: history all visible, future masked except weather
        - virtual_sensor: target vars masked for middle 30% (like training)
    """
    num_variates = full_data.shape[0]
    total_length = context_length + prediction_length

    # Calculate patches
    num_patches_per_var = (total_length + patch_size - 1) // patch_size
    padded_len = num_patches_per_var * patch_size
    pad_amount = padded_len - total_length if padded_len > total_length else 0

    # Context boundary in patches
    context_patches = (context_length + pad_amount + patch_size - 1) // patch_size

    total_patches = num_patches_per_var * num_variates

    # Initialize tensors
    target = np.zeros((1, total_patches, max_patch_size), dtype=np.float32)
    observed_mask = np.zeros((1, total_patches, max_patch_size), dtype=bool)
    sample_id = np.ones((1, total_patches), dtype=np.int64)
    time_id = np.zeros((1, total_patches), dtype=np.int64)
    variate_id = np.zeros((1, total_patches), dtype=np.int64)
    prediction_mask = np.zeros((1, total_patches), dtype=bool)
    patch_size_tensor = np.full((1, total_patches), patch_size, dtype=np.int64)

    # For virtual_sensor task: mask middle 30% of target vars (like training)
    vs_mask_start = int(num_patches_per_var * 0.35)
    vs_mask_end = int(num_patches_per_var * 0.65)

    patch_idx = 0

    # Process ALL variates
    for var_id in range(num_variates):
        var_data = full_data[var_id, :total_length].astype(np.float32)
        nan_mask = np.isnan(var_data)
        var_data_clean = np.nan_to_num(var_data, nan=0.0)

        if pad_amount > 0:
            var_data_clean = np.concatenate([np.zeros(pad_amount, dtype=np.float32), var_data_clean])
            nan_mask = np.concatenate([np.ones(pad_amount, dtype=bool), nan_mask])

        for p in range(num_patches_per_var):
            start_idx = p * patch_size
            end_idx = start_idx + patch_size

            target[0, patch_idx, :patch_size] = var_data_clean[start_idx:end_idx]
            time_id[0, patch_idx] = p
            variate_id[0, patch_idx] = var_id

            # Determine mask based on task type and variable type
            if task_type == 'forecast':
                # Forecast: history visible, future masked (except weather)
                if p < context_patches:
                    # History: all visible
                    observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start_idx:end_idx]
                    prediction_mask[0, patch_idx] = False
                else:
                    # Future: weather visible, others masked
                    if var_id in weather_var_ids:
                        observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start_idx:end_idx]
                        prediction_mask[0, patch_idx] = False
                    else:
                        observed_mask[0, patch_idx, :patch_size] = False
                        prediction_mask[0, patch_idx] = True

            elif task_type == 'virtual_sensor':
                # Virtual sensor: target vars masked in middle 30%, others fully visible
                if var_id in target_var_ids:
                    if vs_mask_start <= p < vs_mask_end:
                        observed_mask[0, patch_idx, :patch_size] = False
                        prediction_mask[0, patch_idx] = True
                    else:
                        observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start_idx:end_idx]
                        prediction_mask[0, patch_idx] = False
                else:
                    # Non-target vars: fully visible
                    observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start_idx:end_idx]
                    prediction_mask[0, patch_idx] = False

            patch_idx += 1

    # Convert to torch
    target = torch.tensor(target, device=device)
    observed_mask = torch.tensor(observed_mask, device=device)
    sample_id = torch.tensor(sample_id, device=device)
    time_id = torch.tensor(time_id, device=device)
    variate_id = torch.tensor(variate_id, device=device)
    prediction_mask = torch.tensor(prediction_mask, device=device)
    patch_size_tensor = torch.tensor(patch_size_tensor, device=device)

    return target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor


def predict_multivariate(
    module: MoiraiModule,
    full_data: np.ndarray,
    context_length: int,
    prediction_length: int,
    weather_var_ids: List[int],
    target_var_ids: List[int],
    task_type: str = 'forecast',
    patch_size: int = PATCH_SIZE,
    num_samples: int = NUM_SAMPLES,
    device: str = 'cpu'
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Unified multi-variate prediction - ALL variates input, returns predictions for targets.

    This matches training where ALL variates are processed together!

    Args:
        full_data: (num_variates, time_steps) - ALL variates
        task_type: 'forecast' or 'virtual_sensor'

    Returns:
        Dict mapping var_id -> (point_pred, samples)
        Only returns predictions for target_var_ids
    """
    max_patch_size = max(module.patch_sizes)
    num_variates = full_data.shape[0]
    total_length = context_length + prediction_length

    tensors = prepare_multivariate_input(
        full_data, context_length, prediction_length,
        patch_size, max_patch_size, device,
        weather_var_ids, target_var_ids, task_type
    )
    target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor = tensors

    module = module.to(device)
    module.eval()

    num_patches_per_var = (total_length + patch_size - 1) // patch_size
    padded_len = num_patches_per_var * patch_size
    pad_amount = padded_len - total_length if padded_len > total_length else 0

    with torch.no_grad():
        distr = module(
            target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor
        )

        samples = distr.sample((num_samples,))  # (num_samples, batch, total_patches, max_patch)

        results = {}

        for var_id in target_var_ids:
            # Find patches for this variable
            start_patch = var_id * num_patches_per_var
            end_patch = start_patch + num_patches_per_var

            var_samples = samples[:, 0, start_patch:end_patch, :patch_size]
            var_samples = var_samples.reshape(num_samples, -1).cpu().numpy()

            # Remove padding and get correct length
            if pad_amount > 0:
                var_samples = var_samples[:, pad_amount:]
            var_samples = var_samples[:, :total_length]

            if task_type == 'forecast':
                # For forecast: return only the prediction region (future)
                pred_samples = var_samples[:, context_length:]
                point_pred = np.median(pred_samples, axis=0)
                results[var_id] = (point_pred, pred_samples)
            else:
                # For virtual_sensor: return masked region (middle 30%)
                mask_start = int(total_length * 0.35)
                mask_end = int(total_length * 0.65)
                pred_samples = var_samples[:, mask_start:mask_end]
                point_pred = np.median(pred_samples, axis=0)
                results[var_id] = (point_pred, pred_samples, mask_start, mask_end)

    return results


def prepare_fill_in_blank_input(
    full_data: np.ndarray,
    observed_var_ids: List[int],
    masked_var_ids: List[int],
    context_length: int,
    prediction_length: int,
    patch_size: int,
    max_patch_size: int,
    device: str,
    mask_start_ratio: float = 0.0,
    mask_end_ratio: float = 1.0
) -> Tuple[torch.Tensor, ...]:
    """
    Prepare input for fill-in-the-blank task.

    In this task:
    - observed_var_ids: Variables that are fully observed (e.g., weather + zone temps)
    - masked_var_ids: Variables to be predicted (e.g., ODU power)
    - mask_start_ratio: Start position of mask as ratio of total length (0.0 = beginning)
    - mask_end_ratio: End position of mask as ratio of total length (1.0 = end)

    与训练对应：
    - 训练: 随机 mask 15-50% 的时间步，在任意位置
    - 评估: 可指定 mask 的位置和长度，但应在训练覆盖范围内

    例如:
    - mask_start_ratio=0.5, mask_end_ratio=0.8 → mask 中间 30% 的时间步
    - mask_start_ratio=0.7, mask_end_ratio=1.0 → mask 最后 30% 的时间步 (forecasting)

    This tests whether the model learned causal chains: Weather → ODU → Indoor
    """
    total_length = context_length + prediction_length
    num_vars = len(observed_var_ids) + len(masked_var_ids)

    # Calculate patches
    num_patches_per_var = (total_length + patch_size - 1) // patch_size
    padded_len = num_patches_per_var * patch_size
    pad_amount = padded_len - total_length if padded_len > total_length else 0

    total_patches = num_patches_per_var * num_vars

    # Calculate which patches to mask based on time ratios
    mask_start_patch = int(num_patches_per_var * mask_start_ratio)
    mask_end_patch = int(num_patches_per_var * mask_end_ratio)
    if mask_end_patch <= mask_start_patch:
        mask_end_patch = mask_start_patch + 1  # At least 1 patch

    # Initialize tensors
    target = np.zeros((1, total_patches, max_patch_size), dtype=np.float32)
    observed_mask = np.zeros((1, total_patches, max_patch_size), dtype=bool)
    sample_id = np.ones((1, total_patches), dtype=np.int64)
    time_id = np.zeros((1, total_patches), dtype=np.int64)
    variate_id = np.zeros((1, total_patches), dtype=np.int64)
    prediction_mask = np.zeros((1, total_patches), dtype=bool)
    patch_size_tensor = np.full((1, total_patches), patch_size, dtype=np.int64)

    patch_idx = 0

    # Process observed variables (fully visible)
    for var_idx, var_id in enumerate(observed_var_ids):
        var_data = full_data[var_id, :total_length].astype(np.float32)
        nan_mask = np.isnan(var_data)
        var_data_clean = np.nan_to_num(var_data, nan=0.0)

        if pad_amount > 0:
            var_data_clean = np.concatenate([np.zeros(pad_amount, dtype=np.float32), var_data_clean])
            nan_mask = np.concatenate([np.ones(pad_amount, dtype=bool), nan_mask])

        for p in range(num_patches_per_var):
            start_idx = p * patch_size
            end_idx = start_idx + patch_size

            target[0, patch_idx, :patch_size] = var_data_clean[start_idx:end_idx]
            observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start_idx:end_idx]
            time_id[0, patch_idx] = p
            variate_id[0, patch_idx] = var_id  # Use actual schema ID, not sequential index!
            prediction_mask[0, patch_idx] = False  # Observed

            patch_idx += 1

    # Process masked variables (partially masked based on time range)
    for var_idx, var_id in enumerate(masked_var_ids):
        var_data = full_data[var_id, :total_length].astype(np.float32)
        nan_mask = np.isnan(var_data)
        var_data_clean = np.nan_to_num(var_data, nan=0.0)

        if pad_amount > 0:
            var_data_clean = np.concatenate([np.zeros(pad_amount, dtype=np.float32), var_data_clean])
            nan_mask = np.concatenate([np.ones(pad_amount, dtype=bool), nan_mask])

        for p in range(num_patches_per_var):
            start_idx = p * patch_size
            end_idx = start_idx + patch_size

            target[0, patch_idx, :patch_size] = var_data_clean[start_idx:end_idx]
            time_id[0, patch_idx] = p
            variate_id[0, patch_idx] = var_id  # Use actual schema ID, not sequential index!

            # Only mask patches within the specified time range
            if mask_start_patch <= p < mask_end_patch:
                observed_mask[0, patch_idx, :patch_size] = False
                prediction_mask[0, patch_idx] = True  # To predict
            else:
                # Outside mask range: treat as observed
                observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start_idx:end_idx]
                prediction_mask[0, patch_idx] = False

            patch_idx += 1

    # Convert to torch
    target = torch.tensor(target, device=device)
    observed_mask = torch.tensor(observed_mask, device=device)
    sample_id = torch.tensor(sample_id, device=device)
    time_id = torch.tensor(time_id, device=device)
    variate_id = torch.tensor(variate_id, device=device)
    prediction_mask = torch.tensor(prediction_mask, device=device)
    patch_size_tensor = torch.tensor(patch_size_tensor, device=device)

    return target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor


def predict_fill_in_blank(
    module: MoiraiModule,
    full_data: np.ndarray,
    observed_var_ids: List[int],
    masked_var_ids: List[int],
    context_length: int,
    prediction_length: int,
    patch_size: int = PATCH_SIZE,
    num_samples: int = NUM_SAMPLES,
    device: str = 'cpu',
    mask_start_ratio: float = 0.0,
    mask_end_ratio: float = 1.0
) -> Dict[int, Tuple[np.ndarray, np.ndarray, int, int]]:
    """
    Predict masked variables given observed variables.

    Args:
        mask_start_ratio: Start of mask as ratio of total length (0.0 = beginning)
        mask_end_ratio: End of mask as ratio of total length (1.0 = end)

    Returns:
        Dict mapping var_id -> (point_pred, samples, mask_start_idx, mask_end_idx)
        point_pred and samples only contain the masked region
    """
    max_patch_size = max(module.patch_sizes)
    total_length = context_length + prediction_length

    target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor = \
        prepare_fill_in_blank_input(
            full_data, observed_var_ids, masked_var_ids,
            context_length, prediction_length, patch_size, max_patch_size, device,
            mask_start_ratio, mask_end_ratio
        )

    module = module.to(device)
    module.eval()

    # Calculate actual mask indices in timesteps
    mask_start_idx = int(total_length * mask_start_ratio)
    mask_end_idx = int(total_length * mask_end_ratio)
    mask_length = mask_end_idx - mask_start_idx

    with torch.no_grad():
        distr = module(
            target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor
        )

        samples = distr.sample((num_samples,))  # (num_samples, batch, total_patches, max_patch)

        # Extract predictions for masked variables
        results = {}
        num_patches_per_var = (total_length + patch_size - 1) // patch_size
        base_patch_idx = len(observed_var_ids) * num_patches_per_var

        for i, var_id in enumerate(masked_var_ids):
            start_patch = base_patch_idx + i * num_patches_per_var
            end_patch = start_patch + num_patches_per_var

            var_samples = samples[:, 0, start_patch:end_patch, :patch_size]
            var_samples = var_samples.reshape(num_samples, -1)[:, :total_length].cpu().numpy()

            # Extract only the masked region for evaluation
            var_samples_masked = var_samples[:, mask_start_idx:mask_end_idx]
            point_pred = np.median(var_samples_masked, axis=0)

            results[var_id] = (point_pred, var_samples_masked, mask_start_idx, mask_end_idx)
    
    return results


# =============================================================================
# Part 4: Evaluation Tasks
# =============================================================================

def find_best_window(var_data: np.ndarray, context_length: int, prediction_length: int, step: int = 100) -> int:
    """Find window start index where prediction region has maximum variance."""
    total_window = context_length + prediction_length
    if len(var_data) < total_window:
        return 0

    best_start = 0
    best_variance = 0

    for start in range(0, len(var_data) - total_window + 1, step):
        pred_region = var_data[start + context_length : start + total_window]
        valid = pred_region[~np.isnan(pred_region)]
        if len(valid) > 10:
            variance = np.max(valid) - np.min(valid)
            if variance > best_variance:
                best_variance = variance
                best_start = start

    return best_start


def task1_standard_forecast(
    models: Dict[str, object],
    test_hf: datasets.Dataset,
    var_ids: List[int],
    max_samples: int,
    device: str
) -> Dict[str, Dict[str, float]]:
    """
    Task 1: Standard Forecasting (MULTI-VARIATE).

    CRITICAL: Input ALL variates together, matching training!
    - History: all variates visible
    - Future: Weather visible (can be forecasted), others masked

    This tests whether the model learned cross-variate relationships.
    """
    results = {name: [] for name in models.keys()}

    seasonal_naive = SeasonalNaiveModel()

    for idx in range(min(len(test_hf), max_samples)):
        sample = test_hf[idx]
        target = np.array(sample['target'])  # (num_variates, time_steps)
        num_variates = target.shape[0]

        # Find best window using first target variable
        first_var = var_ids[0] if var_ids[0] < num_variates else 10
        var_data = target[first_var, :]
        start_idx = find_best_window(var_data, CONTEXT_LENGTH, PREDICTION_LENGTH)
        end_idx = start_idx + CONTEXT_LENGTH + PREDICTION_LENGTH

        if end_idx > target.shape[1]:
            continue

        # Extract window for ALL variates
        full_data = target[:, start_idx:end_idx]  # (num_variates, total_length)

        # Evaluate baselines per-variable (they can't do multi-variate)
        for var_id in var_ids:
            if var_id >= num_variates:
                continue

            var_window = full_data[var_id, :]
            context = var_window[:CONTEXT_LENGTH]
            future_true = var_window[CONTEXT_LENGTH:]

            if np.isnan(context).sum() > CONTEXT_LENGTH * 0.5:
                continue
            if np.all(np.isnan(future_true)):
                continue

            # Seasonal Naive baseline
            sn_pred, _ = seasonal_naive.predict(context, PREDICTION_LENGTH)
            seasonal_mae = np.nanmean(np.abs(future_true - sn_pred))
            if np.isnan(seasonal_mae) or seasonal_mae < 1e-8:
                seasonal_mae = 1.0

            # Seasonal Naive
            if 'Seasonal Naive' in models and models['Seasonal Naive'] is not None:
                metrics = calculate_all_metrics(future_true, sn_pred, None, seasonal_mae)
                results['Seasonal Naive'].append(metrics)

            # XGBoost
            if 'XGBoost' in models and models['XGBoost'] is not None:
                xgb_pred, _ = models['XGBoost'].predict(context, PREDICTION_LENGTH)
                if not (np.isnan(xgb_pred).all() or np.isinf(xgb_pred).any()):
                    metrics = calculate_all_metrics(future_true, xgb_pred, None, seasonal_mae)
                    results['XGBoost'].append(metrics)

        # MOIRAI models: use multi-variate prediction (ALL variates input!)
        moirai_models = {k: v for k, v in models.items()
                        if k not in ['Seasonal Naive', 'XGBoost'] and v is not None}

        for model_name, module in moirai_models.items():
            try:
                # Input ALL variates, get predictions for target vars
                predictions = predict_multivariate(
                    module,
                    full_data,
                    context_length=CONTEXT_LENGTH,
                    prediction_length=PREDICTION_LENGTH,
                    weather_var_ids=WEATHER_VAR_IDS,
                    target_var_ids=var_ids,
                    task_type='forecast',
                    patch_size=PATCH_SIZE,
                    num_samples=NUM_SAMPLES,
                    device=device
                )

                # Calculate metrics for each target variable
                for var_id in var_ids:
                    if var_id not in predictions:
                        continue

                    point_pred, samples = predictions[var_id]
                    future_true = full_data[var_id, CONTEXT_LENGTH:]
                    context = full_data[var_id, :CONTEXT_LENGTH]

                    if np.all(np.isnan(future_true)):
                        continue

                    # Seasonal MAE for scaling
                    sn_pred, _ = seasonal_naive.predict(context, PREDICTION_LENGTH)
                    seasonal_mae = np.nanmean(np.abs(future_true - sn_pred))
                    if np.isnan(seasonal_mae) or seasonal_mae < 1e-8:
                        seasonal_mae = 1.0

                    if not (np.isnan(point_pred).all() or np.isinf(point_pred).any()):
                        metrics = calculate_all_metrics(future_true, point_pred, samples, seasonal_mae)
                        results[model_name].append(metrics)

            except Exception as e:
                print(f"  Warning: {model_name} failed on sample {idx}: {e}")
                continue

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Aggregate metrics
    aggregated = {}
    for model_name, metric_list in results.items():
        if len(metric_list) == 0:
            aggregated[model_name] = {k: np.nan for k in ['SMAPE', 'CRPS', 'MSIS', 'MAE', 'SmoothedMAE', 'Coverage']}
        else:
            aggregated[model_name] = {}
            for key in metric_list[0].keys():
                values = [m[key] for m in metric_list if not np.isnan(m[key])]
                aggregated[model_name][key] = np.mean(values) if values else np.nan

    return aggregated


def task2_fill_in_blank(
    moirai_models: Dict[str, MoiraiModule],
    test_hf: datasets.Dataset,
    max_samples: int,
    device: str
) -> Dict[str, Dict[str, float]]:
    """
    Task 2: Fill-in-the-Blank (Cross-Variate Inference).

    与训练对应：
    - 训练: 随机选择部分 variates，每个 mask 15-50% 的时间步（任意位置）
    - 评估: 选择 ODU Power variates，mask 中间 30% 的时间步

    Given:
    - Weather variables (boundary conditions) - 100% observed
    - Zone temperatures (end results) - 100% observed
    - ODU Power (target) - 前后各 35% observed，中间 30% masked

    Predict:
    - ODU Power 的中间 30% 时间步

    这样评估与训练的 masking 模式一致：
    - 训练覆盖了 "部分 variates 的部分时间步 masked" 的情况
    - 评估是这个模式的一个特定实例

    This validates whether the model learned: Weather → HVAC Power → Indoor Conditions

    NOTE: Only MOIRAI can do this task. XGBoost cannot handle arbitrary masking.
    """
    # Mask configuration - aligned with training (15-50% mask ratio)
    MASK_START_RATIO = 0.35  # Start masking at 35% of the window
    MASK_END_RATIO = 0.65    # End masking at 65% of the window (30% masked)

    results = {name: [] for name in moirai_models.keys()}

    for idx in range(min(len(test_hf), max_samples)):
        sample = test_hf[idx]
        target = np.array(sample['target'])

        # Use shorter window for fill-in-blank (no future prediction)
        total_length = CONTEXT_LENGTH

        if target.shape[1] < total_length:
            continue

        # Find good start position (where ODU power has variance)
        odu_data = target[ODU_POWER_VAR_IDS[0], :]
        best_start = 0
        best_var = 0
        for start in range(0, len(odu_data) - total_length + 1, 100):
            segment = odu_data[start:start + total_length]
            valid = segment[~np.isnan(segment)]
            if len(valid) > 10:
                var = np.max(valid) - np.min(valid)
                if var > best_var:
                    best_var = var
                    best_start = start

        full_data = target[:, best_start:best_start + total_length]

        # Get ground truth for masked region only
        mask_start_idx = int(total_length * MASK_START_RATIO)
        mask_end_idx = int(total_length * MASK_END_RATIO)
        true_odu = {var_id: full_data[var_id, mask_start_idx:mask_end_idx] for var_id in ODU_POWER_VAR_IDS}

        # Evaluate each MOIRAI model
        for model_name, module in moirai_models.items():
            if module is None:
                continue

            predictions = predict_fill_in_blank(
                module, full_data,
                observed_var_ids=WEATHER_VAR_IDS + ZONE_TEMP_VAR_IDS,
                masked_var_ids=ODU_POWER_VAR_IDS,
                context_length=total_length,
                prediction_length=0,  # No future prediction, just fill-in
                patch_size=PATCH_SIZE,
                num_samples=NUM_SAMPLES,
                device=device,
                mask_start_ratio=MASK_START_RATIO,
                mask_end_ratio=MASK_END_RATIO
            )

            # Calculate metrics for masked variables
            for var_id in ODU_POWER_VAR_IDS:
                if var_id not in predictions:
                    continue

                point_pred, samples, _, _ = predictions[var_id]
                y_true = true_odu[var_id]
                
                if np.isnan(point_pred).all():
                    continue
                
                metrics = calculate_all_metrics(y_true, point_pred, samples)
                results[model_name].append(metrics)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Aggregate
    aggregated = {}
    for model_name, metric_list in results.items():
        if len(metric_list) == 0:
            aggregated[model_name] = {k: np.nan for k in ['SMAPE', 'CRPS', 'MSIS', 'MAE']}
        else:
            aggregated[model_name] = {}
            for key in metric_list[0].keys():
                values = [m[key] for m in metric_list if not np.isnan(m[key])]
                aggregated[model_name][key] = np.mean(values) if values else np.nan
    
    return aggregated


def task3_ood_stress_test(
    models: Dict[str, object],
    test_hf: datasets.Dataset,
    var_ids: List[int],
    max_samples: int,
    device: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Task 3: OOD Stress Test.
    
    Filter test samples by extreme conditions:
    - High temperature: outdoor temp > upper quartile
    - Low temperature: outdoor temp < lower quartile  
    - High load: main power > median (HVAC working hard)
    
    Run standard forecasting on these filtered samples to evaluate generalization.
    """
    results = {
        condition: {name: [] for name in models.keys()}
        for condition in ['high_temp', 'low_temp', 'high_load']
    }
    
    seasonal_naive = SeasonalNaiveModel()
    
    # Variable indices for condition detection
    outdoor_temp_id = 0   # First weather variable is outdoor temperature
    main_power_id = 10    # Main power
    
    for idx in range(min(len(test_hf), max_samples)):
        sample = test_hf[idx]
        target = np.array(sample['target'])
        
        # Check each variable for evaluation
        for var_id in var_ids:
            if var_id >= target.shape[0]:
                continue
            
            var_data = target[var_id, :]
            start_idx = find_best_window(var_data, CONTEXT_LENGTH, PREDICTION_LENGTH)
            end_idx = start_idx + CONTEXT_LENGTH + PREDICTION_LENGTH
            
            if end_idx > len(var_data):
                continue
            
            full_window = var_data[start_idx:end_idx]
            context = full_window[:CONTEXT_LENGTH]
            future_true = full_window[CONTEXT_LENGTH:]
            
            if np.isnan(context).sum() > CONTEXT_LENGTH * 0.5:
                continue
            if np.all(np.isnan(future_true)):
                continue
            
            # Determine which OOD conditions apply to this window
            applicable_conditions = []
            
            # Check outdoor temperature
            if outdoor_temp_id < target.shape[0]:
                outdoor_temp = target[outdoor_temp_id, start_idx:end_idx]
                temp_mean = np.nanmean(outdoor_temp)
                if not np.isnan(temp_mean):
                    if temp_mean > OOD_THRESHOLDS['high_temp']:
                        applicable_conditions.append('high_temp')
                    elif temp_mean < OOD_THRESHOLDS['low_temp']:
                        applicable_conditions.append('low_temp')
            
            # Check main power (high HVAC load)
            if main_power_id < target.shape[0]:
                power_vals = target[main_power_id, start_idx:end_idx]
                power_mean = np.nanmean(power_vals)
                power_max = np.nanmax(target[main_power_id, :])
                if not np.isnan(power_mean) and power_max > 0:
                    if power_mean / power_max > OOD_THRESHOLDS['high_load']:
                        applicable_conditions.append('high_load')
            
            if not applicable_conditions:
                continue
            
            # Get seasonal naive error
            sn_pred, _ = seasonal_naive.predict(context, PREDICTION_LENGTH)
            seasonal_mae = np.nanmean(np.abs(future_true - sn_pred))
            if np.isnan(seasonal_mae) or seasonal_mae < 1e-8:
                seasonal_mae = 1.0
            
            # Evaluate models on this OOD sample
            for model_name, model in models.items():
                if model is None:
                    continue
                
                if model_name == 'Seasonal Naive':
                    point_pred, samples = sn_pred, None
                elif model_name == 'XGBoost':
                    point_pred, samples = model.predict(context, PREDICTION_LENGTH)
                else:
                    # MOIRAI models - pass var_id for correct variate embedding
                    point_pred, samples = predict_moirai(
                        model, full_window, CONTEXT_LENGTH, PREDICTION_LENGTH,
                        PATCH_SIZE, NUM_SAMPLES, device, var_id=var_id
                    )

                if np.isnan(point_pred).all() or np.isinf(point_pred).any():
                    continue

                metrics = calculate_all_metrics(future_true, point_pred, samples, seasonal_mae)

                # Add to all applicable conditions
                for condition in applicable_conditions:
                    results[condition][model_name].append(metrics)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Aggregate
    aggregated = {}
    for condition, condition_results in results.items():
        aggregated[condition] = {}
        for model_name, metric_list in condition_results.items():
            if len(metric_list) == 0:
                aggregated[condition][model_name] = {k: np.nan for k in ['SMAPE', 'CRPS', 'MSIS', 'MAE']}
            else:
                aggregated[condition][model_name] = {}
                for key in metric_list[0].keys():
                    values = [m[key] for m in metric_list if not np.isnan(m[key])]
                    aggregated[condition][model_name][key] = np.mean(values) if values else np.nan
    
    return aggregated


# =============================================================================
# Part 5: Visualization Functions
# =============================================================================

def plot_training_curves(output_dir: Path):
    """Plot training curves for all finetune patterns (2 rows × 3 cols)."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    pattern_titles = {
        'full': 'Full Finetune',
        'freeze_ffn': 'Freeze FFN',
        'head_only': 'Head Only',
    }
    
    line_styles = {'Train': ('b-', 0.6), 'Val': ('r-', 1.0)}
    
    for row_idx, size in enumerate(MODEL_SIZES):
        for col_idx, pattern in enumerate(FINETUNE_PATTERNS):
            ax = axes[row_idx, col_idx]
            model_dir = get_model_dir(size, pattern)
            csv_path = model_dir / 'csv_logs' / 'version_0' / 'metrics.csv'
            
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                
                train_df = df[df['train/PackedNLLLoss'].notna()][['epoch', 'train/PackedNLLLoss']]
                val_df = df[df['val/PackedNLLLoss'].notna()][['epoch', 'val/PackedNLLLoss']]
                
                train_loss = train_df.groupby('epoch')['train/PackedNLLLoss'].mean()
                val_loss = val_df.groupby('epoch')['val/PackedNLLLoss'].mean()
                
                ax.plot(train_loss.index, train_loss.values, 'b-', alpha=0.6, label='Train', lw=1.5)
                ax.plot(val_loss.index, val_loss.values, 'r-', label='Val', lw=2.5)
                
                best_epoch = val_loss.idxmin()
                best_loss = val_loss.min()
                ax.scatter([best_epoch], [best_loss], color='#27ae60', s=120, zorder=5, edgecolor='white', lw=2)
                ax.annotate(f'Best: {best_loss:.4f}\nEpoch {int(best_epoch)}',
                           xy=(best_epoch, best_loss), xytext=(best_epoch + 5, best_loss + 0.2),
                           fontsize=10, color='#27ae60', fontweight='bold',
                           arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5))
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes,
                       fontsize=14, color='#bdc3c7')
            
            ax.set_xlabel('Epoch', fontweight='bold')
            ax.set_ylabel('Loss (PackedNLLLoss)', fontweight='bold')
            ax.set_title(f'{size.upper()} - {pattern_titles[pattern]}', fontweight='bold', fontsize=13)
            ax.legend(loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    
    plt.suptitle('Training Curves: All Finetune Strategies', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: training_curves.png")


def plot_metrics_comparison(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """Plot comprehensive metrics comparison across all models and groups."""
    models = list(results.keys())
    groups = list(EVAL_GROUPS.keys())
    metrics = ['SMAPE', 'CRPS', 'MSIS', 'MAE']
    
    # 更大的图表以容纳更多模型
    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    axes = axes.flatten()
    
    # 动态调整bar宽度
    num_models = len(models)
    width = 0.8 / max(num_models, 1)
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        x = np.arange(len(groups))
        
        for i, model in enumerate(models):
            if model not in results:
                continue
            
            values = []
            for group in groups:
                if group in results[model] and metric in results[model][group]:
                    values.append(results[model][group][metric])
                else:
                    values.append(np.nan)
            
            offset = (i - num_models / 2 + 0.5) * width
            color = MODEL_COLORS.get(model, '#95a5a6')
            bars = ax.bar(x + offset, values, width * 0.9, label=model, color=color, alpha=0.85, edgecolor='white', lw=0.5)
        
        ax.set_xlabel('Variable Group', fontweight='bold', fontsize=12)
        ax.set_ylabel(metric, fontweight='bold', fontsize=12)
        ax.set_title(f'{metric} Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=25, ha='right', fontsize=11)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Legend outside plot area
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=9, framealpha=0.9)
    
    plt.suptitle('Task 1: Standard Forecasting - Metrics Comparison', fontweight='bold', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: metrics_comparison.png")


def plot_crps_distribution(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """Plot CRPS comparison grouped by variable group with all MOIRAI models."""
    moirai_models = [m for m in results.keys() if 'Small' in m or 'Base' in m]
    groups = list(EVAL_GROUPS.keys())
    
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Group by variable group, show all models side by side
    x = np.arange(len(groups))
    num_models = len(moirai_models)
    width = 0.8 / max(num_models, 1)
    
    for i, model in enumerate(moirai_models):
        values = []
        for group in groups:
            if model in results and group in results[model]:
                crps = results[model][group].get('CRPS', np.nan)
                values.append(crps)
            else:
                values.append(np.nan)
        
        offset = (i - num_models / 2 + 0.5) * width
        color = MODEL_COLORS.get(model, '#95a5a6')
        bars = ax.bar(x + offset, values, width * 0.9, label=model, color=color, alpha=0.85, edgecolor='white', lw=0.5)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            if not np.isnan(val) and val > 0:
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                           xytext=(0, 3), textcoords='offset points', ha='center', fontsize=8, rotation=45)
    
    ax.set_xlabel('Variable Group', fontweight='bold', fontsize=12)
    ax.set_ylabel('CRPS (lower is better)', fontweight='bold', fontsize=12)
    ax.set_title('CRPS Comparison Across Finetune Strategies', fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'crps_distribution.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: crps_distribution.png")


def plot_fill_in_blank_results(results: Dict[str, Dict[str, float]], output_dir: Path):
    """Plot fill-in-the-blank task results for all MOIRAI models."""
    models = list(results.keys())
    metrics = ['SMAPE', 'CRPS', 'MAE']
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        values = [results[m].get(metric, np.nan) for m in models]
        colors = [MODEL_COLORS.get(m, '#95a5a6') for m in models]
        
        bars = ax.bar(range(len(models)), values, color=colors, alpha=0.85, edgecolor='white', lw=1)
        
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 5), textcoords='offset points', ha='center', fontsize=11, fontweight='bold')
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=35, ha='right', fontsize=10)
        ax.set_ylabel(metric, fontweight='bold', fontsize=12)
        ax.set_title(f'{metric}', fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Task 2: Fill-in-the-Blank (ODU Power Prediction)\nGiven: Weather + Zone Temps → Predict: ODU Power', 
                fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fill_in_blank.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: fill_in_blank.png")


def plot_ood_performance(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """Plot OOD stress test performance for all models."""
    conditions = list(results.keys())
    models = list(next(iter(results.values())).keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(26, 10))
    
    condition_names = {
        'high_temp': f'High Temp (>{OOD_THRESHOLDS["high_temp"]}°C)',
        'low_temp': f'Low Temp (<{OOD_THRESHOLDS["low_temp"]}°C)',
        'high_load': f'High Load (>{OOD_THRESHOLDS["high_load"]*100:.0f}% max)'
    }
    
    for ax_idx, condition in enumerate(conditions):
        ax = axes[ax_idx]
        
        smape_values = [results[condition][m].get('SMAPE', np.nan) for m in models]
        colors = [MODEL_COLORS.get(m, '#95a5a6') for m in models]
        
        bars = ax.bar(range(len(models)), smape_values, color=colors, alpha=0.85, edgecolor='white', lw=1)
        
        for bar, val in zip(bars, smape_values):
            if not np.isnan(val):
                ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=40, ha='right', fontsize=10)
        ax.set_ylabel('SMAPE', fontweight='bold', fontsize=12)
        ax.set_title(condition_names.get(condition, condition), fontweight='bold', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.suptitle('Task 3: OOD Stress Test - SMAPE under Extreme Conditions', fontweight='bold', fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ood_performance.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: ood_performance.png")


def plot_sample_predictions(
    models: Dict[str, object],
    test_hf: datasets.Dataset,
    var_id: int,
    var_name: str,
    output_dir: Path,
    device: str
):
    """
    Plot sample predictions with probabilistic confidence intervals.
    
    For MOIRAI models: shows 90% confidence interval (5th-95th percentile) from samples.
    For baselines: only shows point prediction (no distribution available).
    """
    sample = test_hf[0]
    target = np.array(sample['target'])
    
    # Find good window
    var_data = target[var_id, :]
    start_idx = find_best_window(var_data, CONTEXT_LENGTH, PREDICTION_LENGTH)
    full_window = var_data[start_idx:start_idx + CONTEXT_LENGTH + PREDICTION_LENGTH]
    
    # Dynamic grid layout based on number of models
    num_models = len(models)
    ncols = min(4, num_models)
    nrows = (num_models + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.array(axes).flatten() if num_models > 1 else [axes]
    
    seasonal_naive = SeasonalNaiveModel()
    
    for idx, (model_name, model) in enumerate(models.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        context = full_window[:CONTEXT_LENGTH]
        future_true = full_window[CONTEXT_LENGTH:]
        samples = None  # Will hold distribution samples for MOIRAI
        
        if model_name == 'Seasonal Naive':
            point_pred, _ = seasonal_naive.predict(context, PREDICTION_LENGTH)
        elif model_name == 'XGBoost':
            point_pred, _ = model.predict(context, PREDICTION_LENGTH) if model else (np.full(PREDICTION_LENGTH, np.nan), None)
        elif model is not None:
            # MOIRAI: get both point prediction and samples for confidence interval
            # Pass var_id for correct variate embedding consistency with training
            point_pred, samples = predict_moirai(model, full_window, CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_SIZE, NUM_SAMPLES, device, var_id=var_id)
        else:
            point_pred = np.full(PREDICTION_LENGTH, np.nan)
        
        time_axis = np.arange(len(full_window))
        pred_time_axis = time_axis[CONTEXT_LENGTH:CONTEXT_LENGTH + len(point_pred)]
        
        # Plot context
        ax.plot(time_axis[:CONTEXT_LENGTH], full_window[:CONTEXT_LENGTH], '#3498db', alpha=0.4, label='Context', lw=1.2)
        
        # Plot confidence interval (if samples available)
        pred_color = MODEL_COLORS.get(model_name, '#e74c3c')
        if samples is not None and len(samples) > 1:
            # Calculate percentiles for confidence intervals
            p5 = np.percentile(samples, 5, axis=0)
            p25 = np.percentile(samples, 25, axis=0)
            p75 = np.percentile(samples, 75, axis=0)
            p95 = np.percentile(samples, 95, axis=0)
            
            # Plot 90% interval (5th-95th)
            ax.fill_between(pred_time_axis, p5, p95, alpha=0.2, color=pred_color, label='90% CI')
            # Plot 50% interval (25th-75th)
            ax.fill_between(pred_time_axis, p25, p75, alpha=0.3, color=pred_color, label='50% CI')
        
        # Plot ground truth
        ax.plot(time_axis[CONTEXT_LENGTH:], future_true, '#27ae60', lw=2.5, label='Truth')
        
        # Plot point prediction (median)
        ax.plot(pred_time_axis, point_pred, color=pred_color, linestyle='--', lw=2.0, label='Median')
        
        # Vertical line at prediction start
        ax.axvline(x=CONTEXT_LENGTH, color='#7f8c8d', linestyle=':', alpha=0.7, lw=1.5)
        
        # Title with metrics
        if not np.isnan(point_pred).all():
            smape = calculate_smape(future_true, point_pred)
            title_text = f'{model_name}\nSMAPE = {smape:.2f}'
            if samples is not None:
                crps = calculate_crps(future_true, samples)
                title_text += f' | CRPS = {crps:.3f}'
            ax.set_title(title_text, fontweight='bold', fontsize=11)
        else:
            ax.set_title(f'{model_name}\n(No prediction)', fontweight='bold', fontsize=12, color='#bdc3c7')
        
        ax.set_xlabel('Time Step', fontweight='bold')
        ax.set_ylabel(var_name, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Hide unused axes
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Sample Predictions Comparison: {var_name}', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    safe_name = var_name.replace('/', '_').replace(' ', '_').replace('[', '').replace(']', '')
    plt.savefig(output_dir / f'predictions_{safe_name}.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: predictions_{safe_name}.png")


def plot_finetune_strategy_comparison(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """
    Create a comprehensive comparison chart specifically for finetune strategies.
    Shows Small vs Base across Full/FreezeFNN/HeadOnly with heatmap style.
    """
    # Extract MOIRAI models only
    strategies = ['Full', 'FreezeFNN', 'HeadOnly']
    sizes = ['Small', 'Base']
    metric = 'SMAPE'  # Primary metric
    
    # Build data matrix
    data = np.zeros((len(sizes), len(strategies)))
    
    for i, size in enumerate(sizes):
        for j, strategy in enumerate(strategies):
            model_name = f'{size}-{strategy}'
            if model_name in results:
                # Average across all variable groups
                values = [results[model_name][g].get(metric, np.nan) 
                         for g in results[model_name].keys() if isinstance(results[model_name][g], dict)]
                valid_values = [v for v in values if not np.isnan(v)]
                data[i, j] = np.mean(valid_values) if valid_values else np.nan
    
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Subplot 1: Heatmap of average SMAPE
    ax1 = axes[0]
    im = ax1.imshow(data, cmap='RdYlGn_r', aspect='auto')
    ax1.set_xticks(np.arange(len(strategies)))
    ax1.set_yticks(np.arange(len(sizes)))
    ax1.set_xticklabels(strategies, fontsize=12, fontweight='bold')
    ax1.set_yticklabels(sizes, fontsize=12, fontweight='bold')
    
    # Add value annotations
    for i in range(len(sizes)):
        for j in range(len(strategies)):
            val = data[i, j]
            if not np.isnan(val):
                color = 'white' if val > np.nanmean(data) else 'black'
                ax1.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=14, fontweight='bold', color=color)
    
    ax1.set_title('Average SMAPE by Strategy\n(lower is better)', fontweight='bold', fontsize=13)
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
    cbar.set_label('SMAPE', fontweight='bold')
    
    # Subplot 2: Bar chart comparison by size
    ax2 = axes[1]
    x = np.arange(len(strategies))
    width = 0.35
    
    for i, size in enumerate(sizes):
        values = data[i, :]
        offset = (i - 0.5) * width
        color = '#2980b9' if size == 'Small' else '#c0392b'
        ax2.bar(x + offset, values, width, label=size, color=color, alpha=0.85, edgecolor='white', lw=1)
    
    ax2.set_xlabel('Finetune Strategy', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Average SMAPE', fontweight='bold', fontsize=12)
    ax2.set_title('Strategy Comparison by Model Size', fontweight='bold', fontsize=13)
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, fontsize=11)
    ax2.legend(fontsize=11, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Subplot 3: Improvement over zero-shot
    ax3 = axes[2]
    
    # Get zero-shot baselines
    zs_small = results.get('Small (Zero-shot)', {})
    zs_base = results.get('Base (Zero-shot)', {})
    
    improvements = np.zeros((len(sizes), len(strategies)))
    for i, (size, zs) in enumerate(zip(sizes, [zs_small, zs_base])):
        zs_values = [zs[g].get(metric, np.nan) for g in zs.keys() if isinstance(zs.get(g), dict)]
        zs_avg = np.nanmean(zs_values) if zs_values else np.nan
        
        for j, strategy in enumerate(strategies):
            if not np.isnan(zs_avg) and not np.isnan(data[i, j]):
                # Improvement = (zs - ft) / zs * 100  (positive = better)
                improvements[i, j] = (zs_avg - data[i, j]) / zs_avg * 100
    
    for i, size in enumerate(sizes):
        values = improvements[i, :]
        offset = (i - 0.5) * width
        color = '#27ae60' if size == 'Small' else '#8e44ad'
        bars = ax3.bar(x + offset, values, width, label=size, color=color, alpha=0.85, edgecolor='white', lw=1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ypos = bar.get_height()
                va = 'bottom' if ypos >= 0 else 'top'
                ax3.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width() / 2, ypos),
                           xytext=(0, 3 if ypos >= 0 else -3), textcoords='offset points',
                           ha='center', va=va, fontsize=10, fontweight='bold')
    
    ax3.axhline(y=0, color='#7f8c8d', linestyle='-', lw=1.5)
    ax3.set_xlabel('Finetune Strategy', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Improvement over Zero-shot (%)', fontweight='bold', fontsize=12)
    ax3.set_title('Finetune Gain vs Zero-shot Baseline', fontweight='bold', fontsize=13)
    ax3.set_xticks(x)
    ax3.set_xticklabels(strategies, fontsize=11)
    ax3.legend(fontsize=11, framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    plt.suptitle('Finetune Strategy Comparison: Full vs Freeze-FFN vs Head-Only', fontweight='bold', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'finetune_strategy_comparison.png', dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: finetune_strategy_comparison.png")


def create_summary_table(all_results: Dict, output_dir: Path):
    """Create comprehensive summary table."""
    rows = []
    
    # Task 1 results
    if 'task1' in all_results:
        for model_name, group_results in all_results['task1'].items():
            for group_name, metrics in group_results.items():
                row = {
                    'Task': 'Standard Forecast',
                    'Model': model_name,
                    'Variable Group': group_name,
                    **metrics
                }
                rows.append(row)
    
    # Task 2 results
    if 'task2' in all_results:
        for model_name, metrics in all_results['task2'].items():
            row = {
                'Task': 'Fill-in-Blank',
                'Model': model_name,
                'Variable Group': 'ODU Power',
                **metrics
            }
            rows.append(row)
    
    # Task 3 results
    if 'task3' in all_results:
        for condition, condition_results in all_results['task3'].items():
            for model_name, metrics in condition_results.items():
                row = {
                    'Task': f'OOD ({condition})',
                    'Model': model_name,
                    'Variable Group': 'All',
                    **metrics
                }
                rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)
    
    if 'task1' in all_results:
        print("\n[Task 1: Standard Forecasting]")
        for model_name in all_results['task1'].keys():
            task1_df = df[(df['Task'] == 'Standard Forecast') & (df['Model'] == model_name)]
            if len(task1_df) > 0:
                print(f"  {model_name}:")
                print(f"    Avg SMAPE: {task1_df['SMAPE'].mean():.2f}")
                print(f"    Avg CRPS:  {task1_df['CRPS'].mean():.4f}")
                print(f"    Avg MAE:   {task1_df['MAE'].mean():.4f}")
    
    if 'task2' in all_results:
        print("\n[Task 2: Fill-in-the-Blank]")
        for model_name, metrics in all_results['task2'].items():
            print(f"  {model_name}: SMAPE={metrics.get('SMAPE', np.nan):.2f}, MAE={metrics.get('MAE', np.nan):.4f}")
    
    if 'task3' in all_results:
        print("\n[Task 3: OOD Stress Test]")
        for condition in all_results['task3'].keys():
            print(f"  {condition}:")
            for model_name, metrics in all_results['task3'][condition].items():
                print(f"    {model_name}: SMAPE={metrics.get('SMAPE', np.nan):.2f}")
    
    print("\n" + "=" * 80)
    
    return df


# =============================================================================
# Part 6: Model Loading Helper
# =============================================================================

def load_model_by_name(model_dir_name: str, device: str = 'cpu') -> MoiraiModule:
    """
    根据目录名加载指定模型，自动找 best epoch 对应的 checkpoint。
    """
    model_dir = MODEL_OUTPUT_DIR / model_dir_name
    assert model_dir.exists(), f"Model directory not found: {model_dir}"

    ckpt_dir = model_dir / 'checkpoints'
    assert ckpt_dir.exists(), f"No checkpoints directory in {model_dir}"

    best_ckpts = list(ckpt_dir.glob('best-*.ckpt'))
    assert best_ckpts, f"No best-*.ckpt found in {ckpt_dir}"

    # 从 csv_logs 找 best epoch
    csv_log_path = model_dir / 'csv_logs' / 'version_0' / 'metrics.csv'
    assert csv_log_path.exists(), f"CSV log not found: {csv_log_path}"
    
    df = pd.read_csv(csv_log_path)
    val_df = df[df['val/PackedNLLLoss'].notna()]
    assert len(val_df) > 0, f"No validation loss records in {csv_log_path}"
    
    best_epoch = int(val_df.loc[val_df['val/PackedNLLLoss'].idxmin(), 'epoch'])

    # 找对应 epoch 的 checkpoint
    ckpt_path = None
    for ckpt in best_ckpts:
        for part in ckpt.stem.split('-'):
            if part.startswith('epoch='):
                if int(part.split('=')[1]) == best_epoch:
                    ckpt_path = ckpt
                    break
        if ckpt_path:
            break
    
    assert ckpt_path is not None, f"Checkpoint for epoch {best_epoch} not found. Available: {[c.name for c in best_ckpts]}"

    print(f"  Loading {model_dir_name} from {ckpt_path.name} (best epoch={best_epoch})...")
    return load_model_from_checkpoint(ckpt_path, device)


def list_available_models():
    """列出所有可用的模型目录"""
    print("\n" + "=" * 60)
    print("Available models in:", MODEL_OUTPUT_DIR)
    print("=" * 60)

    for d in sorted(MODEL_OUTPUT_DIR.iterdir()):
        if d.is_dir() and d.name.startswith('moirai_'):
            ckpt_dir = d / 'checkpoints'
            has_ckpt = ckpt_dir.exists() and list(ckpt_dir.glob('best-*.ckpt'))
            status = "✓" if has_ckpt else "✗"
            print(f"  {status} {d.name}")

    print("=" * 60)


# =============================================================================
# Part 7: Main Function
# =============================================================================

def main():
    """Main evaluation function integrating all three tasks."""

    # =========================================================================
    # ★★★ 用户配置区域 - 在这里指定要对比的模型 ★★★
    # =========================================================================
    #
    # 模式选择:
    #   - 'auto': 自动发现所有模型，每个 size+pattern 组合选最佳
    #   - 'manual': 手动指定要对比的模型（使用下面的 MODELS_TO_COMPARE）
    #
    EVAL_MODE = 'manual'  # 'auto' 或 'manual'

    # 手动模式下，指定要对比的模型
    # 格式: { '显示名称': '模型目录名' }
    # 目录名就是 outputs/buildingfm_15min/ 下的文件夹名
    #
    # 示例:
    MODELS_TO_COMPARE = {
        # 'Small-Full-5e6': 'moirai_small_full_5e6',
        # 'Small-Freeze-1e5': 'moirai_small_freeze_ffn_1e5',
        # 'Small-Head-1e4': 'moirai_small_head_only_1e4',
        # 'Base-Full-1e6': 'moirai_base_full_1e6',
        # 'Base-Freeze-5e6': 'moirai_base_freeze_ffn_5e6',
        # 'Base-Head-5e5': 'moirai_base_head_only_5e5',
        'Small-Full-1e5': 'moirai_small_full',
        'Small-Freeze-1e5': 'moirai_small_freeze_ffn',
        'Small-Head--1e5': 'moirai_small_head_only',
    }

    # 是否包含 zero-shot baseline (未微调的原始模型)
    INCLUDE_ZEROSHOT = {
        'small': True,   # 包含 Small (Zero-shot)
        'base': False,   # 包含 Base (Zero-shot)
    }

    # 是否包含传统 baseline
    INCLUDE_SEASONAL_NAIVE = True
    INCLUDE_XGBOOST = True

    # 运行 list_available_models() 可以查看所有可用模型
    # list_available_models()
    # return

    # =========================================================================
    # 以下是评估逻辑，一般不需要修改
    # =========================================================================

    print("=" * 80)
    print("BuildingFM Comprehensive Model Evaluation")
    print("=" * 80)
    print(f"Mode: {EVAL_MODE}")
    print("Tasks: Standard Forecast | Fill-in-Blank | OOD Stress Test")
    print("Metrics: SMAPE | CRPS | MSIS | Smoothed MAE | Consistency")
    print("=" * 80)

    # Setup
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    torch.set_float32_matmul_precision('high')

    # =========================================================================
    # Load Data
    # =========================================================================
    print("\n[1/7] Loading test data...")
    hf_data_dir = DATA_DIR / 'hf'
    test_hf = datasets.load_from_disk(str(hf_data_dir / 'buildingfm_test'))
    print(f"  Test samples: {len(test_hf)}")

    sample = test_hf[0]
    num_variates = np.array(sample['target']).shape[0]
    print(f"  Num variates: {num_variates}")

    # =========================================================================
    # Load Models
    # =========================================================================
    print("\n[2/7] Loading models...")

    models = {}

    # Traditional baselines
    if INCLUDE_SEASONAL_NAIVE:
        models['Seasonal Naive'] = SeasonalNaiveModel()

    if INCLUDE_XGBOOST:
        xgb_path = BASELINES_DIR / 'xgboost_model.joblib'
        if xgb_path.exists():
            print("  Loading XGBoost...")
            models['XGBoost'] = XGBoostModel(xgb_path)
        else:
            print("  XGBoost model not found (skipping)")

    if EVAL_MODE == 'manual':
        # =====================================================================
        # 手动模式: 加载用户指定的模型
        # =====================================================================
        print("\n  Loading specified models...")

        # Load zero-shot baselines if requested
        for size, include in INCLUDE_ZEROSHOT.items():
            if not include:
                continue

            # Find baseline from any available model of this size
            baseline_path = None
            for d in MODEL_OUTPUT_DIR.iterdir():
                if d.is_dir() and d.name.startswith(f'moirai_{size}_'):
                    candidate = d / 'baseline_untrained.pt'
                    if candidate.exists():
                        baseline_path = candidate
                        break

            if baseline_path and baseline_path.exists():
                display_name = f'{size.capitalize()} (Zero-shot)'
                print(f"  Loading {display_name}...")
                models[display_name] = load_model_from_baseline(baseline_path, device)

        # Load specified finetuned models
        for display_name, model_dir_name in MODELS_TO_COMPARE.items():
            module = load_model_by_name(model_dir_name, device)
            models[display_name] = module
            # 自动添加颜色
            if display_name not in MODEL_COLORS:
                import hashlib
                h = int(hashlib.md5(display_name.encode()).hexdigest()[:6], 16)
                MODEL_COLORS[display_name] = f'#{h:06x}'

    else:
        # =====================================================================
        # 自动模式: 原有的自动发现逻辑
        # =====================================================================
        print("\n  Auto-discovering MOIRAI models...")
        discovered = discover_all_models(MODEL_OUTPUT_DIR)

        if discovered:
            print(f"  Found {sum(len(v) for v in discovered.values())} model directories in {len(discovered)} groups")
            for key, dirs in discovered.items():
                print(f"    {key}: {len(dirs)} experiments")
        else:
            print("  No MOIRAI models discovered!")

        # First load zero-shot baselines (one per size)
        for size in MODEL_SIZES:
            baseline_path = None
            for pattern in FINETUNE_PATTERNS:
                key = f'{size}_{pattern}'
                if key in discovered and discovered[key]:
                    candidate = discovered[key][0] / 'baseline_untrained.pt'
                    if candidate.exists():
                        baseline_path = candidate
                        break
            if baseline_path is None:
                baseline_path = get_model_dir(size, 'full') / 'baseline_untrained.pt'

            if baseline_path.exists():
                display_name = f'{size.capitalize()} (Zero-shot)'
                print(f"  Loading {display_name}...")
                models[display_name] = load_model_from_baseline(baseline_path, device)

        # Load best finetuned model for each size+pattern combination
        for size in MODEL_SIZES:
            for pattern in FINETUNE_PATTERNS:
                key = f'{size}_{pattern}'
                display_name = DISPLAY_NAMES.get(key, f'{size.capitalize()}-{pattern}')

                if key not in discovered or not discovered[key]:
                    print(f"  {display_name}: no experiments found (skipping)")
                    continue

                best_result = find_best_model_for_group(discovered[key])

                if best_result is None:
                    print(f"  {display_name}: no valid checkpoints found (skipping)")
                    continue

                best_dir, best_loss = best_result
                ckpt_dir = best_dir / 'checkpoints'

                all_best = list(ckpt_dir.glob('best-*.ckpt'))
                valid_ckpts = []
                for p in all_best:
                    try:
                        epoch = int(p.stem.split('-')[1])
                        if epoch > 2:
                            valid_ckpts.append(p)
                    except (IndexError, ValueError):
                        valid_ckpts.append(p)

                if not valid_ckpts:
                    valid_ckpts = all_best

                best_ckpts = sorted(valid_ckpts, key=lambda p: p.stat().st_mtime, reverse=True)
                ckpt_path = best_ckpts[0] if best_ckpts else None

                if ckpt_path and ckpt_path.exists():
                    print(f"  Loading {display_name} from {best_dir.name}/{ckpt_path.name} (val_loss={best_loss:.4f})...")
                    models[display_name] = load_model_from_checkpoint(ckpt_path, device)
                else:
                    print(f"  {display_name}: checkpoint file not found (skipping)")
    
    # Filter out None models for counting
    loaded_models = {k: v for k, v in models.items() if v is not None}
    print(f"\n  Total loaded: {len(loaded_models)} models")
    for name in loaded_models.keys():
        print(f"    - {name}")
    
    # Store all results
    all_results = {}
    
    # =========================================================================
    # Task 1: Standard Forecasting
    # =========================================================================
    print("\n[3/7] Task 1: Standard Forecasting...")
    
    task1_results = {}
    for group_name, group_info in EVAL_GROUPS.items():
        print(f"  Evaluating: {group_name}")
        id_start, id_end = group_info['id_range']
        var_ids = list(range(id_start, id_end + 1))
        
        group_results = task1_standard_forecast(
            models, test_hf, var_ids, MAX_EVAL_SAMPLES, device
        )
        
        for model_name, metrics in group_results.items():
            if model_name not in task1_results:
                task1_results[model_name] = {}
            task1_results[model_name][group_name] = metrics
            print(f"    {model_name}: SMAPE={metrics['SMAPE']:.2f}, MAE={metrics['MAE']:.4f}")
        
        torch.cuda.empty_cache()
    
    all_results['task1'] = task1_results
    
    # =========================================================================
    # Task 2: Fill-in-the-Blank
    # =========================================================================
    print("\n[4/7] Task 2: Fill-in-the-Blank (Causal Chain Validation)...")
    
    moirai_models = {
        k: v for k, v in models.items()
        if v is not None and isinstance(v, MoiraiModule)
    }
    
    if moirai_models:
        task2_results = task2_fill_in_blank(moirai_models, test_hf, MAX_EVAL_SAMPLES, device)
        all_results['task2'] = task2_results
        
        for model_name, metrics in task2_results.items():
            print(f"  {model_name}: SMAPE={metrics.get('SMAPE', np.nan):.2f}, MAE={metrics.get('MAE', np.nan):.4f}")
    else:
        print("  Skipping (no MOIRAI models loaded)")
        all_results['task2'] = {}
    
    # =========================================================================
    # Task 3: OOD Stress Test
    # =========================================================================
    print("\n[5/7] Task 3: OOD Stress Test...")
    
    # Use a subset of variables for OOD test
    ood_var_ids = list(range(10, 14)) + list(range(50, 55))  # Power + Zone temps
    task3_results = task3_ood_stress_test(models, test_hf, ood_var_ids, MAX_EVAL_SAMPLES, device)
    all_results['task3'] = task3_results
    
    for condition, condition_results in task3_results.items():
        print(f"  {condition}:")
        for model_name, metrics in condition_results.items():
            if metrics.get('SMAPE') is not None and not np.isnan(metrics.get('SMAPE', np.nan)):
                print(f"    {model_name}: SMAPE={metrics['SMAPE']:.2f}")
    
    # =========================================================================
    # Generate Visualizations
    # =========================================================================
    print("\n[6/7] Generating visualizations...")
    
    # Training curves for all finetune strategies
    plot_training_curves(output_dir)
    
    # Metrics comparison (all models)
    plot_metrics_comparison(task1_results, output_dir)
    
    # CRPS distribution for probabilistic models
    if moirai_models:
        plot_crps_distribution(task1_results, output_dir)
    
    # Finetune strategy comparison (key insight chart)
    plot_finetune_strategy_comparison(task1_results, output_dir)
    
    # Fill-in-blank results (MOIRAI only)
    if all_results.get('task2'):
        plot_fill_in_blank_results(all_results['task2'], output_dir)
    
    # OOD stress test results
    if any(all_results['task3'].values()):
        plot_ood_performance(all_results['task3'], output_dir)
    
    # Sample predictions (visual comparison)
    plot_sample_predictions(models, test_hf, 10, 'Main Power [kW]', output_dir, device)
    plot_sample_predictions(models, test_hf, 50, 'Zone A1 Temp [°C]', output_dir, device)
    
    # =========================================================================
    # Save Results
    # =========================================================================
    print("\n[7/7] Saving results...")
    
    summary_df = create_summary_table(all_results, output_dir)
    
    # Save JSON
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nAll results saved to {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    main()
