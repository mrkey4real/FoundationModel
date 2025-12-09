#!/usr/bin/env python
"""
BuildingFM Model Evaluation Script

Compare zero-shot vs finetuned models for Small and Base variants.
Uses the same data processing pipeline as training (MoiraiPretrain transform).

Metrics: MAE, RMSE, CVRMSE, R2, MAPE

IMPORTANT - Point Prediction Strategy:
    MOIRAI uses probabilistic forecasting (outputs distribution, not point).
    For point metrics (MAE/RMSE), the OFFICIAL method is:
        1. Sample N times from the distribution
        2. Take MEDIAN as point prediction
    
    This is consistent with MOIRAI training:
        - Training: NLL Loss to optimize distribution parameters
        - Evaluation: Median of samples for point metrics
    
    See: src/uni2ts/model/moirai/pretrain.py line 237-238
         src/uni2ts/model/moirai/finetune.py line 248-249

Usage:
    python scripts/evaluate_models.py
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

# Data path
DATA_DIR = Path('../data/buildingfm_processed')

# Output
OUTPUT_DIR = Path('../outputs/evaluation')

# Prediction settings
CONTEXT_LENGTH = 256*6   # Number of timesteps as context
PREDICTION_LENGTH = 64*6  # Number of timesteps to predict
PATCH_SIZE = 128         # Patch size for prediction
NUM_SAMPLES = 50        # Number of samples for prediction

# Target variable groups for evaluation
# From metadata: power, zone temps, IAQ are most important for HVAC
EVAL_GROUPS = {
    'Main Power': {'id_range': (10, 11), 'unit': 'kW'},
    'ODU Power': {'id_range': (12, 13), 'unit': 'kW'},
    'IDU Power': {'id_range': (30, 33), 'unit': 'kW'},
    'Zone Temps': {'id_range': (50, 61), 'unit': '°C'},
    'IAQ': {'id_range': (98, 101), 'unit': 'ppm/ug'},
}

# Number of test samples to evaluate
MAX_EVAL_SAMPLES = 20

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate HVAC forecasting metrics"""
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 10:
        return {'MAE': np.nan, 'RMSE': np.nan, 'CVRMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan}

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    mean_true = np.mean(np.abs(y_true))
    cvrmse = (rmse / mean_true * 100) if mean_true > 1e-6 else np.nan

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-6 else np.nan

    nonzero_mask = np.abs(y_true) > 1e-6
    if nonzero_mask.sum() > 10:
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    else:
        mape = np.nan

    return {
        'MAE': mae,
        'RMSE': rmse,
        'CVRMSE': cvrmse,
        'R2': r2,
        'MAPE': mape
    }


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


# =============================================================================
# Native Prediction (Using same format as training)
# =============================================================================

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
    # Context ends at index (padded_len - prediction_length)
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


def predict_native(
    module: MoiraiModule,
    var_data: np.ndarray,
    context_length: int,
    prediction_length: int,
    patch_size: int = PATCH_SIZE,
    num_samples: int = NUM_SAMPLES,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Make prediction using native MoiraiModule interface.
    Same data format as training.
    
    Args:
        var_data: Full window data (context_length + prediction_length)
        context_length: Number of timesteps in context
        prediction_length: Number of timesteps to predict
    """
    max_patch_size = max(module.patch_sizes)
    total_length = context_length + prediction_length
    
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
        
        # Get prediction region - find where prediction_mask is True
        num_patches = target.shape[1]
        padded_len = num_patches * patch_size
        context_end_idx = padded_len - prediction_length
        context_patches = context_end_idx // patch_size
        
        pred_samples = samples[:, 0, context_patches:, :patch_size]  # (num_samples, pred_patches, patch_size)
        
        # Reshape to (num_samples, prediction_length)
        pred_samples = pred_samples.reshape(num_samples, -1)[:, :prediction_length]
        
        # Use MEDIAN - this is the OFFICIAL MOIRAI design!
        # See: src/uni2ts/model/moirai/pretrain.py line 237-238:
        #   if isinstance(metric_func, PackedPointLoss):
        #       pred = distr.sample(torch.Size((self.hparams.num_samples,)))
        #       pred = torch.median(pred, dim=0).values
        #
        # Reason: For probabilistic models with mixture distributions,
        # median is the robust point estimate for L1 loss (MAE/MAPE).
        # Training uses NLL Loss to optimize distribution parameters,
        # but point evaluation uses median of samples.
        pred_median = pred_samples.median(dim=0).values.cpu().numpy()
    
    return pred_median


def evaluate_single_var(
    module: MoiraiModule,
    sample: Dict,
    var_id: int,
    context_length: int,
    prediction_length: int,
    device: str = 'cuda'
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate single variable prediction"""
    target = np.array(sample['target'])
    num_variates, total_length = target.shape
    
    if total_length < context_length + prediction_length:
        return None, None
    
    if var_id >= num_variates:
        return None, None
    
    var_data = target[var_id, :]
    
    # Get full context + prediction window
    full_window = var_data[:context_length + prediction_length]
    future_true = var_data[context_length:context_length + prediction_length]
    
    # Check for valid data (need at least 50% non-NaN in context)
    context_data = var_data[:context_length]
    if np.isnan(context_data).sum() > context_length * 0.5:
        return None, None
    if np.all(np.isnan(future_true)):
        return None, None
    
    # Predict - pass full window so scaler sees correct context values
    future_pred = predict_native(
        module, full_window,  # Pass full window, not just context
        context_length, prediction_length,
        PATCH_SIZE, NUM_SAMPLES, device
    )
    
    return future_true, future_pred


def evaluate_model_on_group(
    module: MoiraiModule,
    test_hf: datasets.Dataset,
    var_ids: List[int],
    max_samples: int = MAX_EVAL_SAMPLES,
    device: str = 'cuda'
) -> Dict[str, float]:
    """Evaluate model on a specific variable group
    
    IMPORTANT: Metrics are computed PER-SEQUENCE then averaged.
    This gives meaningful R² values (not inflated by cross-sequence variance).
    """
    
    all_metrics = []
    errors = 0
    
    for idx in range(min(len(test_hf), max_samples)):
        sample = test_hf[idx]
        
        for var_id in var_ids:
            y_true, y_pred = evaluate_single_var(
                module, sample, var_id,
                CONTEXT_LENGTH, PREDICTION_LENGTH, device
            )
            
            if y_true is not None and y_pred is not None:
                # Skip if prediction has NaN or Inf
                if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                    errors += 1
                    continue
                
                # Calculate metrics for THIS sequence
                seq_metrics = calculate_metrics(y_true, y_pred)
                
                # Only include if metrics are valid
                if not np.isnan(seq_metrics['MAE']):
                    all_metrics.append(seq_metrics)
    
    if errors > 0:
        print(f"({errors} errors)", end=" ")
    
    if len(all_metrics) == 0:
        return {'MAE': np.nan, 'RMSE': np.nan, 'CVRMSE': np.nan, 'R2': np.nan, 'MAPE': np.nan}
    
    # Average metrics across all sequences
    avg_metrics = {}
    for key in ['MAE', 'RMSE', 'CVRMSE', 'R2', 'MAPE']:
        values = [m[key] for m in all_metrics if not np.isnan(m[key])]
        avg_metrics[key] = np.mean(values) if values else np.nan
    
    return avg_metrics


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_training_curves(output_dir: Path):
    """Plot training curves for both models"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    small_csv = SMALL_MODEL_DIR / 'csv_logs' / 'version_0' / 'metrics.csv'
    if small_csv.exists():
        df = pd.read_csv(small_csv)
        
        train_df = df[df['train/PackedNLLLoss'].notna()][['epoch', 'train/PackedNLLLoss']]
        val_df = df[df['val/PackedNLLLoss'].notna()][['epoch', 'val/PackedNLLLoss']]
        
        train_loss = train_df.groupby('epoch')['train/PackedNLLLoss'].mean()
        val_loss = val_df.groupby('epoch')['val/PackedNLLLoss'].mean()
        
        axes[0].plot(train_loss.index, train_loss.values, 'b-', alpha=0.7, label='Train Loss', linewidth=1.5)
        axes[0].plot(val_loss.index, val_loss.values, 'r-', label='Val Loss', linewidth=2)
        
        best_epoch = val_loss.idxmin()
        best_loss = val_loss.min()
        axes[0].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=1)
        axes[0].scatter([best_epoch], [best_loss], color='green', s=100, zorder=5)
        axes[0].annotate(f'Best: {best_loss:.4f}\nEpoch {int(best_epoch)}',
                        xy=(best_epoch, best_loss), 
                        xytext=(best_epoch + 15, best_loss + 0.3),
                        fontsize=10, color='green',
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
        
    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Loss (PackedNLLLoss)', fontsize=11)
    axes[0].set_title('Small Model (d_model=384, layers=6)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(left=0)
    
    base_csv = BASE_MODEL_DIR / 'csv_logs' / 'version_0' / 'metrics.csv'
    if base_csv.exists():
        df = pd.read_csv(base_csv)
        
        train_df = df[df['train/PackedNLLLoss'].notna()][['epoch', 'train/PackedNLLLoss']]
        val_df = df[df['val/PackedNLLLoss'].notna()][['epoch', 'val/PackedNLLLoss']]
        
        train_loss = train_df.groupby('epoch')['train/PackedNLLLoss'].mean()
        val_loss = val_df.groupby('epoch')['val/PackedNLLLoss'].mean()
        
        axes[1].plot(train_loss.index, train_loss.values, 'b-', alpha=0.7, label='Train Loss', linewidth=1.5)
        axes[1].plot(val_loss.index, val_loss.values, 'r-', label='Val Loss', linewidth=2)
        
        best_epoch = val_loss.idxmin()
        best_loss = val_loss.min()
        axes[1].axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, linewidth=1)
        axes[1].scatter([best_epoch], [best_loss], color='green', s=100, zorder=5)
        axes[1].annotate(f'Best: {best_loss:.4f}\nEpoch {int(best_epoch)}',
                        xy=(best_epoch, best_loss), 
                        xytext=(best_epoch + 10, best_loss + 0.3),
                        fontsize=10, color='green',
                        arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))
    
    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss (PackedNLLLoss)', fontsize=11)
    axes[1].set_title('Base Model (d_model=768, layers=6)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'training_curves.png'}")


def plot_metrics_comparison(results: Dict, output_dir: Path):
    """Plot comprehensive metrics comparison"""
    
    models = list(results.keys())
    groups = list(EVAL_GROUPS.keys())
    metrics = ['MAE', 'RMSE', 'CVRMSE', 'R2', 'MAPE']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = {
        'Small (Zero-shot)': '#3498db',
        'Small (Finetuned)': '#2ecc71', 
        'Base (Zero-shot)': '#e74c3c',
        'Base (Finetuned)': '#9b59b6',
    }
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        x = np.arange(len(groups))
        width = 0.2
        
        for i, model in enumerate(models):
            values = []
            for group in groups:
                if group in results[model] and metric in results[model][group]:
                    values.append(results[model][group][metric])
                else:
                    values.append(np.nan)
            
            offset = (i - len(models)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=model, color=colors.get(model, 'gray'), alpha=0.85)
            
            for bar, val in zip(bars, values):
                if not np.isnan(val):
                    height = bar.get_height()
                    if metric == 'R2':
                        ax.annotate(f'{val:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=7, rotation=45)
                    elif val < 10:
                        ax.annotate(f'{val:.2f}',
                                    xy=(bar.get_x() + bar.get_width() / 2, height),
                                    xytext=(0, 3), textcoords="offset points",
                                    ha='center', va='bottom', fontsize=7, rotation=45)
        
        ax.set_xlabel('Variable Group', fontsize=10)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(f'{metric} Comparison', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=30, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        if metric == 'R2':
            ax.set_ylim(-0.5, 1.1)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    axes[5].axis('off')
    axes[5].legend(handles=[plt.Rectangle((0,0),1,1, color=colors[m], alpha=0.85) for m in models],
                   labels=models, loc='center', fontsize=11, ncol=1)
    axes[5].set_title('Legend', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'metrics_comparison.png'}")


def plot_improvement_heatmap(results: Dict, output_dir: Path):
    """Plot heatmap showing improvement from zero-shot to finetuned"""
    
    groups = list(EVAL_GROUPS.keys())
    metrics = ['MAE', 'RMSE', 'CVRMSE', 'MAPE']
    
    small_improv = np.zeros((len(groups), len(metrics)))
    base_improv = np.zeros((len(groups), len(metrics)))
    
    for i, group in enumerate(groups):
        for j, metric in enumerate(metrics):
            small_zs = results['Small (Zero-shot)'].get(group, {}).get(metric, np.nan)
            small_ft = results['Small (Finetuned)'].get(group, {}).get(metric, np.nan)
            base_zs = results['Base (Zero-shot)'].get(group, {}).get(metric, np.nan)
            base_ft = results['Base (Finetuned)'].get(group, {}).get(metric, np.nan)
            
            if not np.isnan(small_zs) and not np.isnan(small_ft) and abs(small_zs) > 1e-6:
                small_improv[i, j] = (small_zs - small_ft) / abs(small_zs) * 100
            else:
                small_improv[i, j] = np.nan
                
            if not np.isnan(base_zs) and not np.isnan(base_ft) and abs(base_zs) > 1e-6:
                base_improv[i, j] = (base_zs - base_ft) / abs(base_zs) * 100
            else:
                base_improv[i, j] = np.nan
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    im1 = axes[0].imshow(small_improv, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=100)
    axes[0].set_xticks(range(len(metrics)))
    axes[0].set_xticklabels(metrics)
    axes[0].set_yticks(range(len(groups)))
    axes[0].set_yticklabels(groups)
    axes[0].set_title('Small Model: Improvement (%)\n(Zero-shot → Finetuned)', fontsize=11, fontweight='bold')
    
    for i in range(len(groups)):
        for j in range(len(metrics)):
            val = small_improv[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 30 else 'black'
                axes[0].text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=10, color=color)
    
    plt.colorbar(im1, ax=axes[0], label='Improvement (%)')
    
    im2 = axes[1].imshow(base_improv, cmap='RdYlGn', aspect='auto', vmin=-50, vmax=100)
    axes[1].set_xticks(range(len(metrics)))
    axes[1].set_xticklabels(metrics)
    axes[1].set_yticks(range(len(groups)))
    axes[1].set_yticklabels(groups)
    axes[1].set_title('Base Model: Improvement (%)\n(Zero-shot → Finetuned)', fontsize=11, fontweight='bold')
    
    for i in range(len(groups)):
        for j in range(len(metrics)):
            val = base_improv[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 30 else 'black'
                axes[1].text(j, i, f'{val:.1f}%', ha='center', va='center', fontsize=10, color=color)
    
    plt.colorbar(im2, ax=axes[1], label='Improvement (%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / 'improvement_heatmap.png'}")


def plot_sample_predictions(
    modules: Dict[str, MoiraiModule],
    test_hf: datasets.Dataset,
    var_id: int,
    var_name: str,
    output_dir: Path,
    device: str = 'cuda'
):
    """Plot sample predictions for visual comparison"""
    
    sample = test_hf[0]
    target = np.array(sample['target'])
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = {
        'Small (Zero-shot)': '#3498db',
        'Small (Finetuned)': '#2ecc71',
        'Base (Zero-shot)': '#e74c3c', 
        'Base (Finetuned)': '#9b59b6',
    }
    
    for idx, (model_name, module) in enumerate(modules.items()):
        ax = axes[idx]
        
        y_true, y_pred = evaluate_single_var(
            module, sample, var_id,
            CONTEXT_LENGTH, PREDICTION_LENGTH, device
        )
        
        full_true = target[var_id, :CONTEXT_LENGTH + PREDICTION_LENGTH]
        time_axis = np.arange(len(full_true))
        
        ax.plot(time_axis[:CONTEXT_LENGTH], full_true[:CONTEXT_LENGTH], 
                'b-', alpha=0.7, label='Context', linewidth=1)
        
        ax.plot(time_axis[CONTEXT_LENGTH:], full_true[CONTEXT_LENGTH:],
                'g-', linewidth=2, label='Ground Truth')
        
        if y_pred is not None:
            pred_time = time_axis[CONTEXT_LENGTH:CONTEXT_LENGTH + len(y_pred)]
            ax.plot(pred_time, y_pred, 
                    color=colors[model_name], linestyle='--', linewidth=2, 
                    label=f'Prediction')
            
            metrics = calculate_metrics(y_true, y_pred)
            title = f'{model_name}\nMAE={metrics["MAE"]:.3f}, RMSE={metrics["RMSE"]:.3f}, R²={metrics["R2"]:.3f}'
        else:
            title = f'{model_name}\n(No prediction available)'
        
        ax.axvline(x=CONTEXT_LENGTH, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel(var_name, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = var_name.replace('/', '_').replace(' ', '_')
    plt.savefig(output_dir / f'predictions_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir / f'predictions_{safe_name}.png'}")


def create_summary_table(results: Dict, output_dir: Path):
    """Create summary table and save to CSV"""
    
    rows = []
    for model_name, model_results in results.items():
        for group_name, metrics in model_results.items():
            row = {
                'Model': model_name,
                'Variable Group': group_name,
                **metrics
            }
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'evaluation_results.csv', index=False)
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS SUMMARY")
    print("="*80)
    
    for model in results.keys():
        model_df = df[df['Model'] == model]
        print(f"\n{model}:")
        print(f"  Avg MAE:    {model_df['MAE'].mean():.4f}")
        print(f"  Avg RMSE:   {model_df['RMSE'].mean():.4f}")
        print(f"  Avg CVRMSE: {model_df['CVRMSE'].mean():.2f}%")
        print(f"  Avg R²:     {model_df['R2'].mean():.4f}")
        print(f"  Avg MAPE:   {model_df['MAPE'].mean():.2f}%")
    
    print("\n" + "="*80)
    return df


# =============================================================================
# Main Evaluation
# =============================================================================

def main():
    """Main evaluation function"""
    
    print("="*80)
    print("BuildingFM Model Evaluation (Native MoiraiModule Interface)")
    print("="*80)
    
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    torch.set_float32_matmul_precision('high')
    
    # Load test dataset
    hf_data_dir = DATA_DIR / 'hf'
    print(f"\nLoading test data from {hf_data_dir}...")
    test_hf = datasets.load_from_disk(str(hf_data_dir / 'buildingfm_test'))
    print(f"Test samples: {len(test_hf)}")
    
    sample = test_hf[0]
    num_variates = np.array(sample['target']).shape[0]
    print(f"Num variates: {num_variates}")
    
    results = {
        'Small (Zero-shot)': {},
        'Small (Finetuned)': {},
        'Base (Zero-shot)': {},
        'Base (Finetuned)': {},
    }
    
    modules = {}
    
    # ==========================================================================
    # 1. Plot training curves
    # ==========================================================================
    print("\n[1/6] Generating training curves...")
    plot_training_curves(output_dir)
    
    # ==========================================================================
    # 2. Load all models
    # ==========================================================================
    print("\n[2/6] Loading models...")
    
    small_baseline_path = SMALL_MODEL_DIR / 'baseline_untrained.pt'
    if small_baseline_path.exists():
        print("  Loading Small Zero-shot...")
        modules['Small (Zero-shot)'] = load_model_from_baseline(small_baseline_path)
    
    small_ckpt = SMALL_MODEL_DIR / 'checkpoints' / 'last.ckpt'
    if small_ckpt.exists():
        print("  Loading Small Finetuned...")
        modules['Small (Finetuned)'] = load_model_from_checkpoint(small_ckpt)
    
    base_baseline_path = BASE_MODEL_DIR / 'baseline_untrained.pt'
    if base_baseline_path.exists():
        print("  Loading Base Zero-shot...")
        modules['Base (Zero-shot)'] = load_model_from_baseline(base_baseline_path)
    
    base_ckpt = BASE_MODEL_DIR / 'checkpoints' / 'last.ckpt'
    if base_ckpt.exists():
        print("  Loading Base Finetuned...")
        modules['Base (Finetuned)'] = load_model_from_checkpoint(base_ckpt)
    
    # ==========================================================================
    # 3. Evaluate on each variable group
    # ==========================================================================
    print("\n[3/6] Evaluating models on variable groups...")
    
    for group_name, group_info in EVAL_GROUPS.items():
        print(f"\n  Evaluating: {group_name}")
        id_start, id_end = group_info['id_range']
        var_ids = list(range(id_start, id_end + 1))
        
        for model_name, module in modules.items():
            print(f"    - {model_name}...", end=" ")
            metrics = evaluate_model_on_group(module, test_hf, var_ids, MAX_EVAL_SAMPLES, device)
            results[model_name][group_name] = metrics
            print(f"MAE={metrics['MAE']:.4f}, R²={metrics['R2']:.4f}")
            
            torch.cuda.empty_cache()
    
    # ==========================================================================
    # 4. Generate comparison plots
    # ==========================================================================
    print("\n[4/6] Generating comparison plots...")
    plot_metrics_comparison(results, output_dir)
    
    # ==========================================================================
    # 5. Generate improvement heatmap
    # ==========================================================================
    print("\n[5/6] Generating improvement heatmap...")
    plot_improvement_heatmap(results, output_dir)
    
    # ==========================================================================
    # 6. Plot sample predictions
    # ==========================================================================
    print("\n[6/6] Plotting sample predictions...")
    plot_sample_predictions(modules, test_hf, 10, 'Main Power [kW]', output_dir, device)
    plot_sample_predictions(modules, test_hf, 50, 'Zone A1 Temp [°C]', output_dir, device)
    
    # ==========================================================================
    # Save results
    # ==========================================================================
    print("\n" + "="*80)
    summary_df = create_summary_table(results, output_dir)
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    
    print(f"\nAll results saved to {output_dir}")
    print("="*80)


if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    main()
