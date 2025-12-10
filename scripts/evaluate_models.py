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

# Model paths (relative to script directory)
SMALL_MODEL_DIR = Path('../outputs/buildingfm/20251208_173657')
BASE_MODEL_DIR = Path('../outputs/buildingfm/20251208_180818')
BASELINES_DIR = Path('../outputs/baselines')

# Data path
DATA_DIR = Path('../data/buildingfm_processed')

# Output
OUTPUT_DIR = Path('../outputs/evaluation')

# Prediction settings
CONTEXT_LENGTH = 256 * 6    # 1536 timesteps (~25.6 hours at 1min)
PREDICTION_LENGTH = 64 * 6  # 384 timesteps (~6.4 hours at 1min)
PATCH_SIZE = 128            # Patch size for MOIRAI
NUM_SAMPLES = 50            # Number of samples for probabilistic prediction
SEASONAL_PERIOD = 1440      # 1 day = 1440 minutes for Seasonal Naive

# Evaluation settings
MAX_EVAL_SAMPLES = 20       # Number of test samples to evaluate
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
    Continuous Ranked Probability Score.
    
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
    
    # E[|X - X'|]: expected difference between pairs of samples
    # Use Monte Carlo approximation: compare each sample with all others
    diff_term = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            diff_term += np.mean(np.abs(samples[i] - samples[j]))
    diff_term = 2.0 * diff_term / (N * (N - 1)) if N > 1 else 0.0
    
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
    
    # Lag steps must match train_baselines.py
    LAG_STEPS = [1, 60, 360, 1440]  # 1min, 1hr, 6hr, 1day
    
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


def load_model_from_baseline(checkpoint_path: Path) -> MoiraiModule:
    """Load MoiraiModule from baseline .pt file"""
    state = torch.load(checkpoint_path, map_location='cpu')
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


def load_model_from_checkpoint(checkpoint_path: Path) -> MoiraiModule:
    """Load MoiraiModule from Lightning checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    hparams = checkpoint['hyper_parameters']
    module_kwargs = hparams['module_kwargs']
    
    module = MoiraiModule(**module_kwargs)
    
    state_dict = checkpoint['state_dict']
    module_state = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}
    module.load_state_dict(module_state)
    
    return module


def prepare_native_input(
    var_data: np.ndarray,
    context_length: int,
    prediction_length: int,
    patch_size: int,
    max_patch_size: int = 128,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, ...]:
    """
    Prepare input tensors in the same format as MoiraiPretrain training.
    
    This creates a single-variate sequence with:
    - Context window: observed data (with actual values)
    - Prediction window: actual future values (but masked for prediction)
    
    The key is that we provide the FULL data (context + future), but mark
    the future region with prediction_mask=True and observed_mask=False.
    This allows the scaler to compute loc/scale from the context correctly.
    
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
    
    # variate_id: 0 for all (single variate)
    variate_id = np.zeros((1, num_patches), dtype=np.int64)
    
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
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make prediction using native MoiraiModule interface.
    Same data format as training.
    
    Args:
        var_data: Full window data (context_length + prediction_length)
        context_length: Number of timesteps in context
        prediction_length: Number of timesteps to predict
    
    Returns:
        point_pred: Median of samples (official MOIRAI point estimate)
        samples: (num_samples, prediction_length) for probabilistic metrics
    """
    max_patch_size = max(module.patch_sizes)
    
    target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor = \
        prepare_native_input(var_data, context_length, prediction_length, patch_size, max_patch_size, device)
    
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


def prepare_fill_in_blank_input(
    full_data: np.ndarray,
    observed_var_ids: List[int],
    masked_var_ids: List[int],
    context_length: int,
    prediction_length: int,
    patch_size: int,
    max_patch_size: int,
    device: str
) -> Tuple[torch.Tensor, ...]:
    """
    Prepare input for fill-in-the-blank task.
    
    In this task:
    - observed_var_ids: Variables that are fully observed (e.g., weather + zone temps)
    - masked_var_ids: Variables to be predicted in the middle (e.g., ODU power)
    
    This tests whether the model learned causal chains: Weather → ODU → Indoor
    """
    total_length = context_length + prediction_length
    num_vars = len(observed_var_ids) + len(masked_var_ids)
    
    # Calculate patches
    num_patches_per_var = (total_length + patch_size - 1) // patch_size
    padded_len = num_patches_per_var * patch_size
    pad_amount = padded_len - total_length if padded_len > total_length else 0
    
    total_patches = num_patches_per_var * num_vars
    
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
            variate_id[0, patch_idx] = var_idx
            prediction_mask[0, patch_idx] = False  # Observed
            
            patch_idx += 1
    
    # Process masked variables (to be predicted)
    base_var_idx = len(observed_var_ids)
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
            # Masked variables: observed_mask is False, prediction_mask is True
            observed_mask[0, patch_idx, :patch_size] = False
            time_id[0, patch_idx] = p
            variate_id[0, patch_idx] = base_var_idx + var_idx
            prediction_mask[0, patch_idx] = True  # To predict
            
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
    device: str = 'cuda'
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Predict masked variables given observed variables.
    
    Returns:
        Dict mapping var_id -> (point_pred, samples)
    """
    max_patch_size = max(module.patch_sizes)
    total_length = context_length + prediction_length
    
    target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor = \
        prepare_fill_in_blank_input(
            full_data, observed_var_ids, masked_var_ids,
            context_length, prediction_length, patch_size, max_patch_size, device
        )
    
    module = module.to(device)
    module.eval()
    
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
            
            point_pred = np.median(var_samples, axis=0)
            results[var_id] = (point_pred, var_samples)
    
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
    Task 1: Standard Forecasting.
    
    Given past CONTEXT_LENGTH timesteps, predict future PREDICTION_LENGTH timesteps.
    Evaluates time extrapolation ability.
    """
    results = {name: [] for name in models.keys()}
    
    seasonal_naive = SeasonalNaiveModel()
    
    for idx in range(min(len(test_hf), max_samples)):
        sample = test_hf[idx]
        target = np.array(sample['target'])
        
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
            
            # Check for valid data
            if np.isnan(context).sum() > CONTEXT_LENGTH * 0.5:
                continue
            if np.all(np.isnan(future_true)):
                continue
            
            # Get seasonal naive error for MSIS scaling
            sn_pred, _ = seasonal_naive.predict(context, PREDICTION_LENGTH)
            seasonal_mae = np.nanmean(np.abs(future_true - sn_pred))
            if np.isnan(seasonal_mae) or seasonal_mae < 1e-8:
                seasonal_mae = 1.0
            
            # Evaluate each model
            for model_name, model in models.items():
                if model is None:
                    continue
                
                if model_name == 'Seasonal Naive':
                    point_pred, samples = sn_pred, None
                elif model_name == 'XGBoost':
                    point_pred, samples = model.predict(context, PREDICTION_LENGTH)
                else:
                    # MOIRAI models
                    point_pred, samples = predict_moirai(
                        model, full_window, CONTEXT_LENGTH, PREDICTION_LENGTH,
                        PATCH_SIZE, NUM_SAMPLES, device
                    )
                
                if np.isnan(point_pred).all() or np.isinf(point_pred).any():
                    continue
                
                metrics = calculate_all_metrics(future_true, point_pred, samples, seasonal_mae)
                results[model_name].append(metrics)
                
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
    Task 2: Fill-in-the-Blank (Causal Chain Validation).
    
    Given:
    - Weather variables (boundary conditions)
    - Zone temperatures (end results)
    
    Predict:
    - ODU Power (intermediate in causal chain)
    
    This validates whether the model learned: Weather → HVAC Power → Indoor Conditions
    
    NOTE: Only MOIRAI can do this task. XGBoost cannot handle arbitrary masking.
    """
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
        
        # Get ground truth for masked variables
        true_odu = {var_id: full_data[var_id, :] for var_id in ODU_POWER_VAR_IDS}
        
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
                device=device
            )
            
            # Calculate metrics for masked variables
            for var_id in ODU_POWER_VAR_IDS:
                if var_id not in predictions:
                    continue
                
                point_pred, samples = predictions[var_id]
                y_true = true_odu[var_id]
                
                if np.isnan(point_pred).all():
                    continue
                
                metrics = calculate_all_metrics(y_true, point_pred, samples)
                results[model_name].append(metrics)
            
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
                    point_pred, samples = predict_moirai(
                        model, full_window, CONTEXT_LENGTH, PREDICTION_LENGTH,
                        PATCH_SIZE, NUM_SAMPLES, device
                    )
                
                if np.isnan(point_pred).all() or np.isinf(point_pred).any():
                    continue
                
                metrics = calculate_all_metrics(future_true, point_pred, samples, seasonal_mae)
                
                # Add to all applicable conditions
                for condition in applicable_conditions:
                    results[condition][model_name].append(metrics)
            
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
    """Plot training curves for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, (model_dir, title) in enumerate([
        (SMALL_MODEL_DIR, 'Small Model (d_model=384, layers=6)'),
        (BASE_MODEL_DIR, 'Base Model (d_model=768, layers=6)')
    ]):
        csv_path = model_dir / 'csv_logs' / 'version_0' / 'metrics.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            
            train_df = df[df['train/PackedNLLLoss'].notna()][['epoch', 'train/PackedNLLLoss']]
            val_df = df[df['val/PackedNLLLoss'].notna()][['epoch', 'val/PackedNLLLoss']]
            
            train_loss = train_df.groupby('epoch')['train/PackedNLLLoss'].mean()
            val_loss = val_df.groupby('epoch')['val/PackedNLLLoss'].mean()
            
            axes[idx].plot(train_loss.index, train_loss.values, 'b-', alpha=0.7, label='Train', lw=1.5)
            axes[idx].plot(val_loss.index, val_loss.values, 'r-', label='Val', lw=2)
            
            best_epoch = val_loss.idxmin()
            best_loss = val_loss.min()
            axes[idx].scatter([best_epoch], [best_loss], color='green', s=100, zorder=5)
            axes[idx].annotate(f'Best: {best_loss:.4f}\nEpoch {int(best_epoch)}',
                             xy=(best_epoch, best_loss), xytext=(best_epoch + 15, best_loss + 0.3),
                             fontsize=10, color='green', arrowprops=dict(arrowstyle='->', color='green'))
        
        axes[idx].set_xlabel('Epoch')
        axes[idx].set_ylabel('Loss (PackedNLLLoss)')
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].legend()
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: training_curves.png")


def plot_metrics_comparison(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """Plot comprehensive metrics comparison across all models and groups."""
    models = list(results.keys())
    groups = list(EVAL_GROUPS.keys())
    metrics = ['SMAPE', 'CRPS', 'MSIS', 'MAE']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    colors = {
        'Seasonal Naive': '#95a5a6',
        'XGBoost': '#f39c12',
        'Small (Zero-shot)': '#3498db',
        'Small (Fine-tuned)': '#2ecc71',
        'Base (Zero-shot)': '#e74c3c',
        'Base (Fine-tuned)': '#9b59b6',
    }
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        x = np.arange(len(groups))
        width = 0.12
        
        for i, model in enumerate(models):
            if model not in results:
                continue
            
            values = []
            for group in groups:
                if group in results[model] and metric in results[model][group]:
                    values.append(results[model][group][metric])
                else:
                    values.append(np.nan)
            
            offset = (i - len(models) / 2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model, color=colors.get(model, 'gray'), alpha=0.85)
        
        ax.set_xlabel('Variable Group')
        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_comparison.png")


def plot_crps_distribution(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """Plot CRPS distribution as boxplot across variable groups."""
    moirai_models = [m for m in results.keys() if 'MOIRAI' in m or 'Small' in m or 'Base' in m]
    groups = list(EVAL_GROUPS.keys())
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    positions = []
    crps_data = []
    labels = []
    colors = []
    
    color_map = {
        'Small (Zero-shot)': '#3498db',
        'Small (Fine-tuned)': '#2ecc71',
        'Base (Zero-shot)': '#e74c3c',
        'Base (Fine-tuned)': '#9b59b6',
    }
    
    pos = 0
    for group in groups:
        for model in moirai_models:
            if model in results and group in results[model]:
                crps = results[model][group].get('CRPS', np.nan)
                if not np.isnan(crps):
                    positions.append(pos)
                    crps_data.append([crps])  # Boxplot expects list
                    labels.append(f'{model}\n{group}')
                    colors.append(color_map.get(model, 'gray'))
                    pos += 1
        pos += 0.5  # Gap between groups
    
    if crps_data:
        bp = ax.bar(range(len(crps_data)), [d[0] for d in crps_data], color=colors, alpha=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    ax.set_ylabel('CRPS (lower is better)')
    ax.set_title('CRPS by Model and Variable Group', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'crps_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: crps_distribution.png")


def plot_fill_in_blank_results(results: Dict[str, Dict[str, float]], output_dir: Path):
    """Plot fill-in-the-blank task results."""
    models = list(results.keys())
    metrics = ['SMAPE', 'CRPS', 'MAE']
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    colors = ['#2ecc71', '#9b59b6', '#3498db', '#e74c3c']
    
    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        values = [results[m].get(metric, np.nan) for m in models]
        
        bars = ax.bar(range(len(models)), values, color=colors[:len(models)], alpha=0.85)
        
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.annotate(f'{val:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right')
        ax.set_ylabel(metric)
        ax.set_title(f'Fill-in-Blank: {metric}', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Fill-in-the-Blank Task: ODU Power Prediction\n(Given Weather + Zone Temps)', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'fill_in_blank.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: fill_in_blank.png")


def plot_ood_performance(results: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """Plot OOD stress test performance."""
    conditions = list(results.keys())
    models = list(next(iter(results.values())).keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    condition_names = {
        'high_temp': f'High Temp (>{OOD_THRESHOLDS["high_temp"]}°C)',
        'low_temp': f'Low Temp (<{OOD_THRESHOLDS["low_temp"]}°C)',
        'high_load': f'High Load (>{OOD_THRESHOLDS["high_load"]*100:.0f}% max)'
    }
    
    colors = {
        'Seasonal Naive': '#95a5a6',
        'XGBoost': '#f39c12',
        'Small (Zero-shot)': '#3498db',
        'Small (Fine-tuned)': '#2ecc71',
        'Base (Zero-shot)': '#e74c3c',
        'Base (Fine-tuned)': '#9b59b6',
    }
    
    for ax_idx, condition in enumerate(conditions):
        ax = axes[ax_idx]
        
        smape_values = [results[condition][m].get('SMAPE', np.nan) for m in models]
        
        bars = ax.bar(range(len(models)), smape_values, 
                     color=[colors.get(m, 'gray') for m in models], alpha=0.85)
        
        for bar, val in zip(bars, smape_values):
            if not np.isnan(val):
                ax.annotate(f'{val:.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                          xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
        
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('SMAPE')
        ax.set_title(condition_names.get(condition, condition), fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('OOD Stress Test: SMAPE under Extreme Conditions', fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'ood_performance.png', dpi=150, bbox_inches='tight')
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
    """Plot sample predictions for visual comparison."""
    sample = test_hf[0]
    target = np.array(sample['target'])
    
    # Find good window
    var_data = target[var_id, :]
    start_idx = find_best_window(var_data, CONTEXT_LENGTH, PREDICTION_LENGTH)
    full_window = var_data[start_idx:start_idx + CONTEXT_LENGTH + PREDICTION_LENGTH]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    colors = {
        'Seasonal Naive': '#95a5a6',
        'XGBoost': '#f39c12',
        'Small (Zero-shot)': '#3498db',
        'Small (Fine-tuned)': '#2ecc71',
        'Base (Zero-shot)': '#e74c3c',
        'Base (Fine-tuned)': '#9b59b6',
    }
    
    seasonal_naive = SeasonalNaiveModel()
    
    for idx, (model_name, model) in enumerate(models.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]
        
        context = full_window[:CONTEXT_LENGTH]
        future_true = full_window[CONTEXT_LENGTH:]
        
        if model_name == 'Seasonal Naive':
            point_pred, _ = seasonal_naive.predict(context, PREDICTION_LENGTH)
        elif model_name == 'XGBoost':
            point_pred, _ = model.predict(context, PREDICTION_LENGTH) if model else (np.full(PREDICTION_LENGTH, np.nan), None)
        elif model is not None:
            point_pred, _ = predict_moirai(model, full_window, CONTEXT_LENGTH, PREDICTION_LENGTH, PATCH_SIZE, NUM_SAMPLES, device)
        else:
            point_pred = np.full(PREDICTION_LENGTH, np.nan)
        
        time_axis = np.arange(len(full_window))
        
        ax.plot(time_axis[:CONTEXT_LENGTH], full_window[:CONTEXT_LENGTH], 'b-', alpha=0.5, label='Context', lw=1)
        ax.plot(time_axis[CONTEXT_LENGTH:], future_true, 'g-', lw=2, label='Truth')
        ax.plot(time_axis[CONTEXT_LENGTH:CONTEXT_LENGTH + len(point_pred)], point_pred,
               color=colors.get(model_name, 'gray'), linestyle='--', lw=2, label='Prediction')
        
        ax.axvline(x=CONTEXT_LENGTH, color='gray', linestyle=':', alpha=0.5)
        
        if not np.isnan(point_pred).all():
            smape = calculate_smape(future_true, point_pred)
            ax.set_title(f'{model_name}\nSMAPE={smape:.1f}', fontweight='bold')
        else:
            ax.set_title(f'{model_name}\n(No prediction)', fontweight='bold')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel(var_name)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused axes
    for idx in range(len(models), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Sample Predictions: {var_name}', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    safe_name = var_name.replace('/', '_').replace(' ', '_').replace('[', '').replace(']', '')
    plt.savefig(output_dir / f'predictions_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: predictions_{safe_name}.png")


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
# Part 6: Main Function
# =============================================================================

def main():
    """Main evaluation function integrating all three tasks."""
    
    print("=" * 80)
    print("BuildingFM Comprehensive Model Evaluation")
    print("=" * 80)
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
    # Load All Models
    # =========================================================================
    print("\n[2/7] Loading models...")
    
    models = {
        'Seasonal Naive': SeasonalNaiveModel(),
        'XGBoost': None,  # Will be loaded if exists
        'Small (Zero-shot)': None,
        'Small (Fine-tuned)': None,
        'Base (Zero-shot)': None,
        'Base (Fine-tuned)': None,
    }
    
    # Load XGBoost if available
    xgb_path = BASELINES_DIR / 'xgboost_model.joblib'
    if xgb_path.exists():
        print("  Loading XGBoost...")
        models['XGBoost'] = XGBoostModel(xgb_path)
    else:
        print("  XGBoost model not found (skipping)")
    
    # Load MOIRAI models
    small_baseline_path = SMALL_MODEL_DIR / 'baseline_untrained.pt'
    if small_baseline_path.exists():
        print("  Loading Small Zero-shot...")
        models['Small (Zero-shot)'] = load_model_from_baseline(small_baseline_path)
    
    small_ckpt = SMALL_MODEL_DIR / 'checkpoints' / 'last.ckpt'
    if small_ckpt.exists():
        print("  Loading Small Fine-tuned...")
        models['Small (Fine-tuned)'] = load_model_from_checkpoint(small_ckpt)
    
    base_baseline_path = BASE_MODEL_DIR / 'baseline_untrained.pt'
    if base_baseline_path.exists():
        print("  Loading Base Zero-shot...")
        models['Base (Zero-shot)'] = load_model_from_baseline(base_baseline_path)
    
    base_ckpt = BASE_MODEL_DIR / 'checkpoints' / 'last.ckpt'
    if base_ckpt.exists():
        print("  Loading Base Fine-tuned...")
        models['Base (Fine-tuned)'] = load_model_from_checkpoint(base_ckpt)
    
    # Filter out None models for counting
    loaded_models = {k: v for k, v in models.items() if v is not None}
    print(f"  Loaded {len(loaded_models)} models")
    
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
    
    plot_training_curves(output_dir)
    plot_metrics_comparison(task1_results, output_dir)
    
    if moirai_models:
        plot_crps_distribution(task1_results, output_dir)
    
    if all_results.get('task2'):
        plot_fill_in_blank_results(all_results['task2'], output_dir)
    
    if any(all_results['task3'].values()):
        plot_ood_performance(all_results['task3'], output_dir)
    
    # Sample predictions
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
