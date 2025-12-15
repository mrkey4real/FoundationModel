#!/usr/bin/env python
"""
Evaluate MOIRAI models on BuildingFM tasks:
- Task 1: standard forecasting
- Task 2: virtual sensor (fill-in-the-blank)
- Task 3: FDD reconstruction
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import datasets
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import unified configuration
from buildingfm_config import (
    cfg,
    WEATHER_VAR_IDS,
    FORECAST_TARGET_IDS,
    ODU_POWER_VAR_IDS,
    STEPS_PER_DAY,
    STEPS_PER_HOUR,
    CONTEXT_LENGTH,
    PREDICTION_LENGTH,
    SEASONAL_PERIOD,
)

from uni2ts.model.moirai import MoiraiModule  # type: ignore
from uni2ts.distribution import (  # type: ignore
    LogNormalOutput,
    MixtureOutput,
    NegativeBinomialOutput,
    NormalFixedScaleOutput,
    StudentTOutput,
)

# ----------------------------------------------------------------------------- #
# Configuration - Uses unified config from buildingfm_config.py
# ----------------------------------------------------------------------------- #
MODEL_SIZES = ["small", "base"]
FINETUNE_PATTERNS = ["full", "freeze_ffn", "head_only"]
DISPLAY_NAMES = {
    "small_full": "Small-Full",
    "small_freeze_ffn": "Small-FreezeFNN",
    "small_head_only": "Small-HeadOnly",
    "base_full": "Base-Full",
    "base_freeze_ffn": "Base-FreezeFNN",
    "base_head_only": "Base-HeadOnly",
}
# Simple runtime toggle (edit here, no CLI arg needed)
USE_XGBOOST_BASELINE = False

# Paths from unified config
DATA_DIR = cfg.data_dir
OUTPUT_DIR = cfg.evaluation_output_dir

# Data frequency and derived values from unified config
DATA_FREQ = cfg.data_freq
_freq_minutes = pd.Timedelta(DATA_FREQ).total_seconds() / 60

# Patch size from unified config
PATCH_SIZE = cfg.patch_size

# Evaluation parameters
NUM_SAMPLES = 20
MAX_EVAL_SAMPLES = 30
CONFIDENCE_LEVEL = 0.95

# Variable IDs are imported from buildingfm_config above:
# - WEATHER_VAR_IDS
# - FORECAST_TARGET_IDS
# - ODU_POWER_VAR_IDS
# - CONTEXT_LENGTH, PREDICTION_LENGTH, STEPS_PER_DAY, STEPS_PER_HOUR, SEASONAL_PERIOD



# ----------------------------------------------------------------------------- #
# Utility helpers
# ----------------------------------------------------------------------------- #
def filter_var_ids(var_ids: List[int], num_variates: int) -> List[int]:
    return [v for v in var_ids if 0 <= v < num_variates]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------- #
# Metrics
# ----------------------------------------------------------------------------- #
def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 1:
        return np.nan
    denom = np.abs(y_true) + np.abs(y_pred)
    nonzero = denom > 1e-8
    if not np.any(nonzero):
        return 0.0
    return 200.0 * np.mean(np.abs(y_true[nonzero] - y_pred[nonzero]) / denom[nonzero])


def calculate_crps(y_true: np.ndarray, samples: np.ndarray) -> float:
    y_true = y_true.flatten()
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    samples = samples[:, mask]
    if len(y_true) == 0:
        return np.nan
    n, _ = samples.shape
    mae_term = np.mean(np.abs(samples - y_true))
    if n > 1:
        sorted_samples = np.sort(samples, axis=0)
        weights = (2 * np.arange(n) - n + 1) / (n * (n - 1))
        diff_term = 2 * np.mean(np.sum(weights[:, None] * sorted_samples, axis=0))
    else:
        diff_term = 0.0
    return float(mae_term - 0.5 * diff_term)


def calculate_weighted_quantile_loss(
    y_true: np.ndarray,
    samples: np.ndarray,
    quantiles: List[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
) -> Dict[str, float]:
    y_true = y_true.flatten()
    mask = ~np.isnan(y_true)
    y_true = y_true[mask]
    samples = samples[:, mask]
    if len(y_true) == 0:
        return {f"wQL_{int(q*100)}": np.nan for q in quantiles} | {"wQL_mean": np.nan}

    wql_results: Dict[str, float] = {}
    denom = np.sum(np.abs(y_true)) + 1e-8

    for q in quantiles:
        q_pred = np.quantile(samples, q, axis=0)
        diff = y_true - q_pred
        loss = np.where(diff >= 0, (1 - q) * diff, -q * diff)
        wql_results[f"wQL_{int(q*100)}"] = float(2 * np.sum(loss) / denom)

    wql_results["wQL_mean"] = float(np.mean(list(wql_results.values())))
    return wql_results


def calculate_msis(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    seasonal_error: float,
    alpha: float,
) -> float:
    y_true = y_true.flatten()
    lower = lower.flatten()
    upper = upper.flatten()
    mask = ~(np.isnan(y_true) | np.isnan(lower) | np.isnan(upper))
    y_true = y_true[mask]
    lower = lower[mask]
    upper = upper[mask]
    if len(y_true) == 0:
        return np.nan
    interval_width = upper - lower
    is_lower = y_true < lower
    is_upper = y_true > upper
    penalties = (2 / alpha) * ((lower - y_true) * is_lower + (y_true - upper) * is_upper)
    msis = np.mean(interval_width + penalties) / max(seasonal_error, 1e-6)
    return float(msis)


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    samples: Optional[np.ndarray] = None,
    seasonal_error: float = 1.0,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {"SMAPE": calculate_smape(y_true, y_pred)}
    mask = ~(np.isnan(y_true.flatten()) | np.isnan(y_pred.flatten()))
    if np.any(mask):
        metrics["MAE"] = float(np.mean(np.abs(y_true.flatten()[mask] - y_pred.flatten()[mask])))
    else:
        metrics["MAE"] = np.nan

    if samples is not None and len(samples) > 1:
        metrics["CRPS"] = calculate_crps(y_true, samples)
        wql = calculate_weighted_quantile_loss(y_true, samples)
        metrics.update(wql)
        alpha = 1 - CONFIDENCE_LEVEL
        lower = np.percentile(samples, alpha / 2 * 100, axis=0)
        upper = np.percentile(samples, (1 - alpha / 2) * 100, axis=0)
        metrics["MSIS"] = calculate_msis(y_true, lower, upper, seasonal_error, alpha)
        y_flat = y_true.flatten()
        lower_flat = lower.flatten()
        upper_flat = upper.flatten()
        mask_cov = ~(np.isnan(y_flat) | np.isnan(lower_flat) | np.isnan(upper_flat))
        metrics["Coverage"] = (
            float(
                np.mean(
                    (y_flat[mask_cov] >= lower_flat[mask_cov])
                    & (y_flat[mask_cov] <= upper_flat[mask_cov])
                )
            )
            * 100
            if np.any(mask_cov)
            else np.nan
        )
    else:
        metrics.update({"CRPS": np.nan, "MSIS": np.nan, "Coverage": np.nan})
        for q in (5, 25, 50, 75, 95):
            metrics[f"wQL_{q}"] = np.nan
        metrics["wQL_mean"] = np.nan

    return metrics


def calculate_fdd_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, samples: Optional[np.ndarray] = None
) -> Dict[str, float]:
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 5:
        return {
            "Recon_MAE": np.nan,
            "Recon_SMAPE": np.nan,
            "Mean_Z": np.nan,
            "Max_Z": np.nan,
            "Anomaly_Rate": np.nan,
        }
    metrics: Dict[str, float] = {
        "Recon_MAE": float(np.mean(np.abs(y_true - y_pred))),
        "Recon_SMAPE": calculate_smape(y_true, y_pred),
    }
    if samples is not None and samples.shape[0] > 1:
        samples = samples[:, mask]
        std = np.maximum(np.std(samples, axis=0), 1e-6)
        z_scores = np.abs(y_true - y_pred) / std
        metrics["Mean_Z"] = float(np.mean(z_scores))
        metrics["Max_Z"] = float(np.max(z_scores))
        metrics["Anomaly_Rate"] = float(np.mean(z_scores > 3.0) * 100)
    else:
        metrics["Mean_Z"] = np.nan
        metrics["Max_Z"] = np.nan
        metrics["Anomaly_Rate"] = np.nan
    return metrics


# ----------------------------------------------------------------------------- #
# Baseline + model loading
# ----------------------------------------------------------------------------- #
class SeasonalNaiveModel:
    def __init__(self, seasonal_period: int = SEASONAL_PERIOD):
        self.seasonal_period = seasonal_period

    def predict(self, context: np.ndarray, prediction_length: int) -> Tuple[np.ndarray, None]:
        context = np.nan_to_num(context, nan=0.0)
        preds = np.zeros(prediction_length)
        for t in range(prediction_length):
            idx = len(context) - self.seasonal_period + t
            if 0 <= idx < len(context):
                preds[t] = context[idx]
            else:
                preds[t] = context[-1]
        return preds, None


def create_distr_output() -> MixtureOutput:
    return MixtureOutput(
        components=[StudentTOutput(), NormalFixedScaleOutput(), NegativeBinomialOutput(), LogNormalOutput()]
    )


def load_model_from_checkpoint(checkpoint_path: Path, device: str = "cpu") -> MoiraiModule:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    module_state = {k.replace("module.", ""): v for k, v in state_dict.items() if k.startswith("module.")}

    baseline_path = checkpoint_path.parent.parent / "baseline_untrained.pt"
    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline: {baseline_path}")

    baseline_state = torch.load(baseline_path, map_location=device)
    config = baseline_state["config"]
    module = MoiraiModule(
        distr_output=create_distr_output(),
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        patch_sizes=config["patch_sizes"],
        max_seq_len=config["max_seq_len"],
        attn_dropout_p=0.0,
        dropout_p=0.1,
        scaling=True,
    )
    module.load_state_dict(module_state)
    return module


def load_model_from_baseline(baseline_path: Path, device: str = "cpu") -> MoiraiModule:
    state = torch.load(baseline_path, map_location=device)
    config = state["config"]
    module = MoiraiModule(
        distr_output=create_distr_output(),
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        patch_sizes=config["patch_sizes"],
        max_seq_len=config["max_seq_len"],
        attn_dropout_p=0.0,
        dropout_p=0.1,
        scaling=True,
    )
    module.load_state_dict(state["model_state_dict"])
    return module


class XGBoostModel:
    """XGBoost baseline (iterative multi-step forecast)."""

    LAG_STEPS = [1, STEPS_PER_HOUR, STEPS_PER_HOUR * 6, STEPS_PER_DAY]

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.model = None
        if model_path is not None and model_path.exists():
            import joblib  # Lazy import

            self.model = joblib.load(model_path)

    def predict(self, context: np.ndarray, prediction_length: int) -> Tuple[np.ndarray, None]:
        if self.model is None:
            return np.full(prediction_length, np.nan), None

        context_clean = np.nan_to_num(context, nan=0.0)
        preds = np.zeros(prediction_length, dtype=np.float32)
        extended = np.concatenate([context_clean, np.zeros(prediction_length, dtype=np.float32)])

        for t in range(prediction_length):
            idx = len(context_clean) + t
            lag_features = []
            for lag in self.LAG_STEPS:
                lag_features.append(extended[idx - lag] if idx >= lag else 0.0)

            hour = (idx // STEPS_PER_HOUR) % 24
            dayofweek = (idx // STEPS_PER_DAY) % 7
            month = 6  # no timestamp available in evaluation
            features = np.array([[*lag_features, hour, dayofweek, month]])
            preds[t] = float(self.model.predict(features)[0])
            extended[idx] = preds[t]

        return preds, None


def discover_models(output_dir: Path) -> List[Path]:
    return [p for p in output_dir.glob("moirai_*") if (p / "checkpoints").exists()]


def pick_best_checkpoint(model_dir: Path) -> Optional[Path]:
    ckpt_dir = model_dir / "checkpoints"
    if not ckpt_dir.exists():
        return None
    best_ckpts = sorted(ckpt_dir.glob("best-*.ckpt"))
    if best_ckpts:
        return best_ckpts[0]
    last_ckpt = ckpt_dir / "last.ckpt"
    return last_ckpt if last_ckpt.exists() else None


def discover_all_models(output_dir: Path) -> Dict[str, List[Path]]:
    """Group discovered moirai experiments by size+pattern."""
    discovered: Dict[str, List[Path]] = {}
    for model_dir in output_dir.glob("moirai_*"):
        if not model_dir.is_dir():
            continue
        ckpt_dir = model_dir / "checkpoints"
        if not ckpt_dir.exists() or not list(ckpt_dir.glob("best-*.ckpt")):
            continue
        parts = model_dir.name.split("_")
        if len(parts) < 3:
            continue
        size = parts[1]
        pattern = "_".join(parts[2:4]) if "_".join(parts[2:4]) in FINETUNE_PATTERNS else parts[2]
        if size not in MODEL_SIZES or pattern not in FINETUNE_PATTERNS:
            continue
        key = f"{size}_{pattern}"
        discovered.setdefault(key, []).append(model_dir)
    return discovered


def find_best_model_for_group(model_dirs: List[Path]) -> Optional[Tuple[Path, float]]:
    """Select model dir with lowest validation loss (fallback to mtime)."""
    best_dir: Optional[Path] = None
    best_loss = float("inf")
    for d in model_dirs:
        val_loss: Optional[float] = None
        csv_path = d / "csv_logs" / "version_0" / "metrics.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                if "val/PackedNLLLoss" in df.columns:
                    vals = df["val/PackedNLLLoss"].dropna()
                    if len(vals) > 0:
                        val_loss = float(vals.min())
            except Exception:
                val_loss = None
        if val_loss is None:
            ckpts = list((d / "checkpoints").glob("best-*.ckpt"))
            if ckpts:
                val_loss = -ckpts[0].stat().st_mtime  # later is better
        if val_loss is None:
            continue
        if val_loss < best_loss:
            best_loss = val_loss
            best_dir = d
    if best_dir is None:
        return None
    return best_dir, best_loss


# ----------------------------------------------------------------------------- #
# Input preparation + prediction
# ----------------------------------------------------------------------------- #
def prepare_multivariate_input(
    full_data: np.ndarray,
    context_length: int,
    prediction_length: int,
    patch_size: int,
    max_patch_size: int,
    device: str,
    weather_var_ids: List[int],
    target_var_ids: List[int],
    task_type: str = "forecast",
) -> Tuple[torch.Tensor, ...]:
    num_variates = full_data.shape[0]
    total_length = context_length + prediction_length
    num_patches_per_var = (total_length + patch_size - 1) // patch_size
    padded_len = num_patches_per_var * patch_size
    pad_amount = padded_len - total_length if padded_len > total_length else 0
    pad_amount = max(0, padded_len - total_length)
    context_end = pad_amount + context_length

    total_patches = num_variates * num_patches_per_var
    target = np.zeros((1, total_patches, max_patch_size), dtype=np.float32)
    observed_mask = np.zeros_like(target, dtype=bool)
    sample_id = np.ones((1, total_patches), dtype=np.int64)
    time_id = np.zeros((1, total_patches), dtype=np.int64)
    variate_id = np.zeros((1, total_patches), dtype=np.int64)
    prediction_mask = np.zeros((1, total_patches), dtype=bool)
    patch_size_tensor = np.full((1, total_patches), patch_size, dtype=np.int64)

    # Virtual sensor: mask a middle slice (default 35%-65%) to probe causal filling
    vs_mask_start = int(total_length * 0.35) + pad_amount
    vs_mask_end = int(total_length * 0.65) + pad_amount

    patch_idx = 0
    for var_id in range(num_variates):
        var_series = full_data[var_id, :total_length].astype(np.float32)
        nan_mask = np.isnan(var_series)
        clean = np.nan_to_num(var_series, nan=0.0)
        if pad_amount:
            clean = np.concatenate([np.zeros(pad_amount, dtype=np.float32), clean])
            nan_mask = np.concatenate([np.ones(pad_amount, dtype=bool), nan_mask])

        for p in range(num_patches_per_var):
            start = p * patch_size
            end = start + patch_size
            target[0, patch_idx, :patch_size] = clean[start:end]
            time_id[0, patch_idx] = p
            variate_id[0, patch_idx] = var_id

            if task_type == "forecast":
                if var_id in weather_var_ids:
                    observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start:end]
                else:
                    if end <= context_end:
                        observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start:end]
                    elif start >= context_end:
                        prediction_mask[0, patch_idx] = True
                    else:
                        cutoff = max(0, context_end - start)
                        observed_mask[0, patch_idx, :cutoff] = ~nan_mask[start : start + cutoff]
                        prediction_mask[0, patch_idx] = True
            else:
                if var_id in target_var_ids:
                    if end <= vs_mask_start or start >= vs_mask_end:
                        observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start:end]
                    elif start >= vs_mask_start and end <= vs_mask_end:
                        prediction_mask[0, patch_idx] = True
                    else:
                        m_start = max(vs_mask_start, start) - start
                        m_end = max(0, min(vs_mask_end, end) - start)
                        observed_mask[0, patch_idx, :m_start] = ~nan_mask[start : start + m_start]
                        observed_mask[0, patch_idx, m_end:patch_size] = ~nan_mask[start + m_end : end]
                        prediction_mask[0, patch_idx] = True
                else:
                    observed_mask[0, patch_idx, :patch_size] = ~nan_mask[start:end]
            patch_idx += 1

    tensors = [
        torch.tensor(target, device=device),
        torch.tensor(observed_mask, device=device),
        torch.tensor(sample_id, device=device),
        torch.tensor(time_id, device=device),
        torch.tensor(variate_id, device=device),
        torch.tensor(prediction_mask, device=device),
        torch.tensor(patch_size_tensor, device=device),
    ]
    return tuple(tensors)


def predict_multivariate(
    module: MoiraiModule,
    full_data: np.ndarray,
    context_length: int,
    prediction_length: int,
    weather_var_ids: List[int],
    target_var_ids: List[int],
    task_type: str = "forecast",
    patch_size: int = PATCH_SIZE,
    num_samples: int = NUM_SAMPLES,
    device: str = "cpu",
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    max_patch_size = max(module.patch_sizes)
    tensors = prepare_multivariate_input(
        full_data,
        context_length,
        prediction_length,
        patch_size,
        max_patch_size,
        device,
        weather_var_ids,
        target_var_ids,
        task_type,
    )
    target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor = tensors

    module = module.to(device)
    module.eval()

    num_variates = full_data.shape[0]
    total_length = context_length + prediction_length
    num_patches_per_var = (total_length + patch_size - 1) // patch_size
    padded_len = num_patches_per_var * patch_size
    pad_amount = max(0, padded_len - total_length)

    with torch.no_grad():
        distr = module(target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size_tensor)
        samples = distr.sample((num_samples,))

    results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for var_id in target_var_ids:
        if var_id >= num_variates:
            continue
        start_patch = var_id * num_patches_per_var
        end_patch = start_patch + num_patches_per_var
        var_samples = samples[:, 0, start_patch:end_patch, :patch_size].reshape(num_samples, -1).cpu().numpy()
        if pad_amount:
            var_samples = var_samples[:, pad_amount:]
        var_samples = var_samples[:, :total_length]

        if task_type == "forecast":
            pred_region = var_samples[:, context_length:]
            point_pred = np.median(pred_region, axis=0)
            results[var_id] = (point_pred, pred_region)
        else:
            mask_start = int(total_length * 0.35)
            mask_end = int(total_length * 0.65)
            masked = var_samples[:, mask_start:mask_end]
            point_pred = np.median(masked, axis=0)
            results[var_id] = (point_pred, masked)
    return results


def find_best_window(var_data: np.ndarray, context_length: int, prediction_length: int, step: int = 100) -> int:
    total = context_length + prediction_length
    if len(var_data) < total:
        return 0
    best_start, best_var = 0, -np.inf
    for start in range(0, len(var_data) - total + 1, step):
        window = var_data[start + context_length : start + total]
        valid = window[~np.isnan(window)]
        if len(valid) < 5:
            continue
        spread = np.max(valid) - np.min(valid)
        if spread > best_var:
            best_var = spread
            best_start = start
    return best_start


# ----------------------------------------------------------------------------- #
# Evaluation tasks
# ----------------------------------------------------------------------------- #
def task1_standard_forecast(
    models: Dict[str, object],
    test_hf: datasets.Dataset,
    target_var_ids: List[int],
    max_samples: int,
    device: str,
) -> Dict[str, Dict[str, float]]:
    seasonal_naive = SeasonalNaiveModel()
    results: Dict[str, List[Dict[str, float]]] = {name: [] for name in models.keys()}
    for idx in range(min(len(test_hf), max_samples)):
        sample = test_hf[idx]
        target = np.array(sample["target"])
        num_variates = target.shape[0]
        valid_targets = filter_var_ids(target_var_ids, num_variates)
        if not valid_targets:
            continue
        window_len = CONTEXT_LENGTH + PREDICTION_LENGTH
        start_idx = find_best_window(target[valid_targets[0]], CONTEXT_LENGTH, PREDICTION_LENGTH)
        end_idx = start_idx + window_len
        if end_idx > target.shape[1]:
            continue
        window = target[:, start_idx:end_idx]
        for model_name, model in models.items():
            try:
                if model_name == "Seasonal Naive":
                    for var_id in valid_targets:
                        series = window[var_id]
                        context = series[:CONTEXT_LENGTH]
                        truth = series[CONTEXT_LENGTH:]
                        point_pred, _ = seasonal_naive.predict(context, PREDICTION_LENGTH)
                        metrics = calculate_all_metrics(
                            truth, point_pred, None, seasonal_error=np.nanmean(np.abs(truth - point_pred))
                        )
                        results[model_name].append(metrics)
                elif isinstance(model, XGBoostModel):
                    for var_id in valid_targets:
                        series = window[var_id]
                        context = series[:CONTEXT_LENGTH]
                        truth = series[CONTEXT_LENGTH:]
                        point_pred, _ = model.predict(context, PREDICTION_LENGTH)
                        metrics = calculate_all_metrics(
                            truth, point_pred, None, seasonal_error=np.nanmean(np.abs(truth - point_pred))
                        )
                        results[model_name].append(metrics)
                elif isinstance(model, MoiraiModule):
                    preds = predict_multivariate(
                        model,
                        window,
                        context_length=CONTEXT_LENGTH,
                        prediction_length=PREDICTION_LENGTH,
                        weather_var_ids=WEATHER_VAR_IDS,
                        target_var_ids=valid_targets,
                        task_type="forecast",
                        patch_size=PATCH_SIZE,
                        num_samples=NUM_SAMPLES,
                        device=device,
                    )
                    for var_id in valid_targets:
                        if var_id not in preds:
                            continue
                        point_pred, samples = preds[var_id]
                        truth = window[var_id, CONTEXT_LENGTH:]
                        seasonal_mae = np.nanmean(
                            np.abs(truth - seasonal_naive.predict(window[var_id, :CONTEXT_LENGTH], PREDICTION_LENGTH)[0])
                        )
                        metrics = calculate_all_metrics(truth, point_pred, samples, seasonal_mae)
                        results[model_name].append(metrics)
            except Exception as exc:
                print(f"  Warning: forecast failed for {model_name} on sample {idx}: {exc}")
                continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregated: Dict[str, Dict[str, float]] = {}
    for name, metric_list in results.items():
        if not metric_list:
            aggregated[name] = {k: np.nan for k in ["SMAPE", "MAE", "CRPS", "MSIS", "wQL_mean"]}
            continue
        keys = set().union(*metric_list)
        aggregated[name] = {k: float(np.nanmean([m.get(k, np.nan) for m in metric_list])) for k in keys}
    return aggregated


def task2_fdd(
    models: Dict[str, object],
    test_hf: datasets.Dataset,
    max_samples: int,
    device: str,
) -> Dict[str, Dict[str, float]]:
    """
    FDD Task: Mask middle 35%-65% of target variables, predict from context,
    then compare with ground truth to detect anomalies.

    Returns both probabilistic metrics (SMAPE, MAE, CRPS, MSIS) and FDD-specific
    metrics (Recon_MAE, Mean_Z, Max_Z, Anomaly_Rate).
    """
    seasonal_naive = SeasonalNaiveModel()
    results: Dict[str, List[Dict[str, float]]] = {name: [] for name in models.keys()}
    for idx in range(min(len(test_hf), max_samples)):
        sample = test_hf[idx]
        target = np.array(sample["target"])
        num_variates = target.shape[0]
        valid_targets = filter_var_ids(ODU_POWER_VAR_IDS, num_variates)
        if not valid_targets:
            continue
        total_len = CONTEXT_LENGTH
        if target.shape[1] < total_len:
            continue
        var_data = target[valid_targets[0]]
        start = find_best_window(var_data, CONTEXT_LENGTH, 0)
        window = target[:, start : start + total_len]
        mask_start = int(total_len * 0.35)
        mask_end = int(total_len * 0.65)
        for model_name, model in models.items():
            try:
                if model_name == "Seasonal Naive":
                    # For Seasonal Naive: use values from one day ago to fill masked region
                    for var_id in valid_targets:
                        series = window[var_id]
                        truth = series[mask_start:mask_end]
                        # Use values from SEASONAL_PERIOD steps earlier as prediction
                        pred_indices = np.arange(mask_start, mask_end) - SEASONAL_PERIOD
                        # Ensure indices are valid (>= 0)
                        pred_indices = np.clip(pred_indices, 0, len(series) - 1)
                        point_pred = np.nan_to_num(series[pred_indices], nan=0.0)
                        # Calculate metrics (no samples for naive baseline)
                        prob_metrics = calculate_all_metrics(truth, point_pred, None)
                        fdd_metrics = calculate_fdd_metrics(truth, point_pred, None)
                        metrics = {**prob_metrics, **fdd_metrics}
                        results[model_name].append(metrics)
                elif isinstance(model, MoiraiModule):
                    # Mask the middle slice (35%-65%) for FDD
                    preds = predict_multivariate(
                        model,
                        window,
                        context_length=CONTEXT_LENGTH,
                        prediction_length=0,
                        weather_var_ids=WEATHER_VAR_IDS,
                        target_var_ids=valid_targets,
                        task_type="virtual_sensor",  # reuse same masking logic
                        patch_size=PATCH_SIZE,
                        num_samples=NUM_SAMPLES,
                        device=device,
                    )
                    for var_id in valid_targets:
                        if var_id not in preds:
                            continue
                        point_pred, samples = preds[var_id]
                        truth = window[var_id, mask_start:mask_end]
                        # Calculate both probabilistic metrics and FDD metrics
                        prob_metrics = calculate_all_metrics(truth, point_pred, samples)
                        fdd_metrics = calculate_fdd_metrics(truth, point_pred, samples)
                        # Merge both
                        metrics = {**prob_metrics, **fdd_metrics}
                        results[model_name].append(metrics)
            except Exception as exc:
                print(f"  Warning: FDD failed for {model_name} on sample {idx}: {exc}")
                continue
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    aggregated: Dict[str, Dict[str, float]] = {}
    for name, metric_list in results.items():
        if not metric_list:
            aggregated[name] = {
                "SMAPE": np.nan, "MAE": np.nan, "CRPS": np.nan, "MSIS": np.nan,
                "Recon_MAE": np.nan, "Recon_SMAPE": np.nan,
                "Mean_Z": np.nan, "Max_Z": np.nan, "Anomaly_Rate": np.nan,
            }
            continue
        keys = set().union(*metric_list)
        aggregated[name] = {k: float(np.nanmean([m.get(k, np.nan) for m in metric_list])) for k in keys}
    return aggregated


# ----------------------------------------------------------------------------- #
# Visualization (task-level + item-level)
# ----------------------------------------------------------------------------- #
def plot_training_curves(model_dirs: List[Path], output_dir: Path) -> None:
    if not model_dirs:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_dir in model_dirs:
        csv_path = model_dir / "csv_logs" / "version_0" / "metrics.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        val_df = df[df["val/PackedNLLLoss"].notna()][["epoch", "val/PackedNLLLoss"]]
        if val_df.empty:
            continue
        grouped = val_df.groupby("epoch")["val/PackedNLLLoss"].mean()
        ax.plot(grouped.index, grouped.values, label=model_dir.name)
    if not ax.has_data():
        plt.close(fig)
        return
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Loss (PackedNLLLoss)")
    ax.set_title("Finetuning Curves")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    ensure_dir(output_dir)
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=160)
    plt.close(fig)


def plot_task_metrics_grid(task_name: str, metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not metrics:
        return
    metric_keys = ["SMAPE", "MAE", "CRPS", "MSIS"]
    models = list(metrics.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, key in zip(axes, metric_keys):
        values = [metrics[m].get(key, np.nan) for m in models]
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, values, color=colors)
        ax.set_title(key)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    rotation=45,
                )
    plt.suptitle(f"{task_name} Metrics Overview", fontweight="bold")
    plt.tight_layout()
    ensure_dir(output_dir)
    plt.savefig(output_dir / f"{task_name.lower()}_metrics_grid.png", dpi=180)
    plt.close(fig)


def plot_fdd_specific_metrics(task_name: str, metrics: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    """Plot FDD-specific metrics: Mean_Z, Max_Z, Anomaly_Rate."""
    if not metrics:
        return
    fdd_keys = ["Recon_MAE", "Mean_Z", "Max_Z", "Anomaly_Rate"]
    models = list(metrics.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    for ax, key in zip(axes, fdd_keys):
        values = [metrics[m].get(key, np.nan) for m in models]
        colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(models)))
        bars = ax.bar(models, values, color=colors)
        ax.set_title(key)
        ax.set_xticklabels(models, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.2, linestyle="--")
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.annotate(
                    f"{val:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    rotation=45,
                )
    plt.suptitle(f"{task_name} FDD-Specific Metrics", fontweight="bold")
    plt.tight_layout()
    ensure_dir(output_dir)
    plt.savefig(output_dir / f"{task_name.lower()}_fdd_metrics.png", dpi=180)
    plt.close(fig)


def plot_single_task_item(
    task_name: str,
    var_label: str,
    model_predictions: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]],
    output_path: Path,
) -> None:
    """Small multiples for one variable across all models."""
    if not model_predictions:
        return

    all_values: List[float] = []
    max_total_len = 0
    for _, entry in model_predictions.items():
        context, truth, pred, samples, *_ = (*entry, None)  # allow optional mask info
        all_values.extend(context[~np.isnan(context)])
        all_values.extend(truth[~np.isnan(truth)])
        all_values.extend(pred[~np.isnan(pred)])
        if len(entry) == 5 and isinstance(entry[4], tuple):
            _, _, _, _, mask_bounds = entry  # type: ignore
            total_len_local = mask_bounds[2]
        else:
            total_len_local = len(context) + len(pred)
        max_total_len = max(max_total_len, total_len_local)

    if not all_values:
        return
    y_min, y_max = np.nanmin(all_values), np.nanmax(all_values)
    margin = (y_max - y_min) * 0.08 if y_max > y_min else 1.0
    ylim = (y_min - margin, y_max + margin)

    n_models = len(model_predictions)
    ncols = min(3, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows))
    axes_arr = np.array(axes).reshape(-1) if n_models > 1 else np.array([axes])

    for ax, (model_name, entry) in zip(axes_arr, model_predictions.items()):
        # Support optional mask bounds: (context, truth, pred, samples, (mask_start, mask_end, total_len))
        if len(entry) == 5 and isinstance(entry[4], tuple):
            context, truth, pred, samples, mask_bounds = entry  # type: ignore
        else:
            context, truth, pred, samples = entry  # type: ignore
            mask_bounds = None

        if mask_bounds:
            mask_start, mask_end, total_len_local = mask_bounds
            ctx_len = len(context)
            total_len = max(total_len_local, max_total_len)
            time_axis = np.arange(total_len)
            ctx_time = time_axis[:ctx_len]
            pred_time = time_axis[mask_start : mask_start + len(pred)]
            mask_region = (mask_start, mask_end)
        else:
            ctx_len = len(context)
            total_len = max_total_len if max_total_len else (ctx_len + len(pred))
            time_axis = np.arange(total_len)
            ctx_time = time_axis[:ctx_len]
            pred_time = time_axis[ctx_len : ctx_len + len(pred)]
            mask_region = None
        ax.plot(ctx_time, context, color="#7f8c8d", alpha=0.6, lw=1.2, label="Context")
        if samples is not None and len(samples) > 1:
            p5 = np.percentile(samples, 5, axis=0)
            p95 = np.percentile(samples, 95, axis=0)
            p25 = np.percentile(samples, 25, axis=0)
            p75 = np.percentile(samples, 75, axis=0)
            ax.fill_between(pred_time, p5, p95, color="#c0392b", alpha=0.1, label="90% CI")
            ax.fill_between(pred_time, p25, p75, color="#c0392b", alpha=0.18, label="50% CI")
        if mask_region:
            ax.axvspan(mask_region[0], mask_region[1], color="#f1c40f", alpha=0.08, label="Masked region")
        ax.plot(pred_time, truth[: len(pred_time)], color="#27ae60", lw=2, label="Truth")
        ax.plot(pred_time, pred[: len(pred_time)], color="#c0392b", linestyle="--", lw=2, label="Pred")
        if mask_region:
            ax.axvline(mask_region[0], color="#95a5a6", linestyle=":", lw=1.0)
            ax.axvline(mask_region[1], color="#95a5a6", linestyle=":", lw=1.0, alpha=0.5)
        else:
            ax.axvline(ctx_len, color="#95a5a6", linestyle=":", lw=1.2)
        ax.set_xlim(0, total_len)
        smape = calculate_smape(truth[: len(pred_time)], pred[: len(pred_time)])
        title = f"{model_name}\nSMAPE={smape:.2f}"
        if samples is not None and len(samples) > 1:
            crps = calculate_crps(truth[: len(pred_time)], samples[:, : len(pred_time)])
            title += f" | CRPS={crps:.3f}"
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylim(ylim)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.tick_params(labelsize=8)

    for ax in axes_arr[n_models:]:
        ax.axis("off")

    fig.suptitle(f"{task_name} - {var_label}", fontsize=13, fontweight="bold")
    handles, labels = axes_arr[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    plt.tight_layout()
    ensure_dir(output_path.parent)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_task_samples(
    models: Dict[str, object], test_hf: datasets.Dataset, output_dir: Path, device: str
) -> None:
    """Item-level plots for representative variables of each task."""
    if not models or len(test_hf) == 0:
        return

    sample = test_hf[0]
    target = np.array(sample["target"])

    # Task 1 sample: forecast
    forecast_ids = filter_var_ids(FORECAST_TARGET_IDS, target.shape[0])
    if forecast_ids:
        var_id = forecast_ids[0]
        total_len = CONTEXT_LENGTH + PREDICTION_LENGTH
        start_idx = find_best_window(target[var_id], CONTEXT_LENGTH, PREDICTION_LENGTH)
        window = target[:, start_idx : start_idx + total_len]
        preds: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Tuple[int, int, int]]] = {}
        seasonal_naive = SeasonalNaiveModel()
        for name, model in models.items():
            try:
                if name == "Seasonal Naive":
                    ctx = window[var_id, :CONTEXT_LENGTH]
                    truth = window[var_id, CONTEXT_LENGTH:]
                    pred, _ = seasonal_naive.predict(ctx, PREDICTION_LENGTH)
                    preds[name] = (ctx, truth, pred, None, (CONTEXT_LENGTH, CONTEXT_LENGTH + len(pred), total_len))
                elif isinstance(model, XGBoostModel):
                    ctx = window[var_id, :CONTEXT_LENGTH]
                    truth = window[var_id, CONTEXT_LENGTH:]
                    pred, _ = model.predict(ctx, PREDICTION_LENGTH)
                    preds[name] = (ctx, truth, pred, None, (CONTEXT_LENGTH, CONTEXT_LENGTH + len(pred), total_len))
                elif isinstance(model, MoiraiModule):
                    res = predict_multivariate(
                        model,
                        window,
                        context_length=CONTEXT_LENGTH,
                        prediction_length=PREDICTION_LENGTH,
                        weather_var_ids=WEATHER_VAR_IDS,
                        target_var_ids=[var_id],
                        task_type="forecast",
                        patch_size=PATCH_SIZE,
                        num_samples=NUM_SAMPLES,
                        device=device,
                    )
                    point_pred, samples = res[var_id]
                    preds[name] = (
                        window[var_id, :CONTEXT_LENGTH],
                        window[var_id, CONTEXT_LENGTH:],
                        point_pred,
                        samples,
                        (CONTEXT_LENGTH, CONTEXT_LENGTH + len(point_pred), total_len),
                    )
            except Exception:
                continue
        if preds:
            plot_single_task_item(
                "Task1 Forecast", f"Var {var_id}", preds, output_dir / "samples" / f"task1_var{var_id}.png"
            )

    # Task 2 sample: FDD (mask middle 35%-65%)
    fdd_ids = filter_var_ids(ODU_POWER_VAR_IDS, target.shape[0])
    if fdd_ids:
        var_id = fdd_ids[0]
        total_len = CONTEXT_LENGTH
        start_idx = find_best_window(target[var_id], CONTEXT_LENGTH, 0)
        window = target[:, start_idx : start_idx + total_len]
        mask_start = int(total_len * 0.35)
        mask_end = int(total_len * 0.65)
        preds_fdd: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Tuple[int, int, int]]] = {}
        for name, model in models.items():
            try:
                if name == "Seasonal Naive":
                    # Seasonal Naive: use values from SEASONAL_PERIOD steps earlier
                    series = window[var_id]
                    truth = series[mask_start:mask_end]
                    pred_indices = np.arange(mask_start, mask_end) - SEASONAL_PERIOD
                    pred_indices = np.clip(pred_indices, 0, len(series) - 1)
                    point_pred = np.nan_to_num(series[pred_indices], nan=0.0)
                    preds_fdd[name] = (
                        window[var_id, :total_len],
                        truth,
                        point_pred,
                        None,
                        (mask_start, mask_end, total_len),
                    )
                elif isinstance(model, MoiraiModule):
                    res = predict_multivariate(
                        model,
                        window,
                        context_length=CONTEXT_LENGTH,
                        prediction_length=0,
                        weather_var_ids=WEATHER_VAR_IDS,
                        target_var_ids=[var_id],
                        task_type="virtual_sensor",  # reuse same masking logic
                        patch_size=PATCH_SIZE,
                        num_samples=NUM_SAMPLES,
                        device=device,
                    )
                    if var_id not in res:
                        continue
                    point_pred, samples = res[var_id]
                    truth = window[var_id, mask_start:mask_end]
                    preds_fdd[name] = (
                        window[var_id, :total_len],  # full context (both sides visible)
                        truth,
                        point_pred,
                        samples,
                        (mask_start, mask_end, total_len),
                    )
            except Exception:
                continue
        if preds_fdd:
            plot_single_task_item(
                "Task2 FDD",
                f"ODU Power (var {var_id})",
                preds_fdd,
                output_dir / "samples" / f"task2_var{var_id}.png",
            )


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="Evaluate MOIRAI models.")
    parser.add_argument("--mode", choices=["auto", "manual"], default="auto", help="auto-discover or manual models")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Model directory names (relative to outputs/buildingfm_15min) when mode=manual",
    )
    parser.add_argument("--max-samples", type=int, default=MAX_EVAL_SAMPLES, help="Number of samples per task")
    parser.add_argument("--run-ood", action="store_true", help="Reserved for future OOD task")
    args = parser.parse_args()

    ensure_dir(OUTPUT_DIR)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("[1/4] Loading test data...")
    test_hf = datasets.load_from_disk(str(DATA_DIR / "hf" / "buildingfm_test"))
    num_variates = np.array(test_hf[0]["target"]).shape[0]
    print(f"  Samples: {len(test_hf)}, variates: {num_variates}")

    print("[2/4] Loading models...")
    models: Dict[str, object] = {}
    model_dirs: List[Path] = []
    output_model_dir = Path("../outputs/buildingfm_lite_15min")

    # Baselines
    models["Seasonal Naive"] = SeasonalNaiveModel()
    if USE_XGBOOST_BASELINE:
        xgb_path = Path("../outputs/baselines_15min") / "xgboost_model.joblib"
        if xgb_path.exists():
            models["XGBoost"] = XGBoostModel(xgb_path)
            print(f"  Loaded XGBoost baseline from {xgb_path.name}")
        else:
            print("  XGBoost baseline not found, skipping.")

    discovered = discover_all_models(output_model_dir)

    if args.mode == "manual" and args.models:
        manual_dirs = [output_model_dir / name for name in args.models]
        for d in manual_dirs:
            ckpt = pick_best_checkpoint(d)
            if ckpt is None:
                continue
            try:
                module = load_model_from_checkpoint(ckpt, device=device)
                models[d.name] = module
                model_dirs.append(d)
                print(f"  Loaded {d.name} ({ckpt.name}) [manual]")
            except Exception as exc:
                print(f"  Skip {d.name}: {exc}")
    else:
        # Zero-shot (un-finetuned) baselines for each size
        for size in MODEL_SIZES:
            baseline_path: Optional[Path] = None
            for pattern in FINETUNE_PATTERNS:
                for d in discovered.get(f"{size}_{pattern}", []):
                    candidate = d / "baseline_untrained.pt"
                    if candidate.exists():
                        baseline_path = candidate
                        break
                if baseline_path:
                    break
            if baseline_path and baseline_path.exists():
                name = f"{size.capitalize()} (Zero-shot)"
                try:
                    models[name] = load_model_from_baseline(baseline_path, device=device)
                    print(f"  Loaded {name} from {baseline_path.parent.name}")
                except Exception as exc:
                    print(f"  Skip zero-shot {size}: {exc}")

        # Best finetuned per size+pattern (small/base x 3)
        for size in MODEL_SIZES:
            for pattern in FINETUNE_PATTERNS:
                key = f"{size}_{pattern}"
                if key not in discovered:
                    continue
                best = find_best_model_for_group(discovered[key])
                if best is None:
                    print(f"  {key}: no valid experiment")
                    continue
                best_dir, best_loss = best
                ckpt = pick_best_checkpoint(best_dir)
                if ckpt is None:
                    print(f"  {best_dir.name}: no checkpoint found")
                    continue
                display_name = DISPLAY_NAMES.get(key, f"{size}-{pattern}")
                try:
                    models[display_name] = load_model_from_checkpoint(ckpt, device=device)
                    model_dirs.append(best_dir)
                    print(f"  Loaded {display_name} from {best_dir.name}/{ckpt.name} (val={best_loss:.4f})")
                except Exception as exc:
                    print(f"  Skip {display_name}: {exc}")

    moirai_models = {k: v for k, v in models.items() if isinstance(v, MoiraiModule)}

    print("[3/3] Running tasks...")
    results: Dict[str, object] = {}

    print("  Task 1: Standard forecast")
    task1 = task1_standard_forecast(models, test_hf, FORECAST_TARGET_IDS, args.max_samples, device)
    results["task1"] = task1

    print("  Task 2: FDD (mask middle 35%-65%)")
    # Include Seasonal Naive baseline in Task 2
    task2_models = {"Seasonal Naive": models["Seasonal Naive"], **moirai_models}
    task2 = task2_fdd(task2_models, test_hf, args.max_samples, device)
    results["task2"] = task2

    output_dir = OUTPUT_DIR
    ensure_dir(output_dir)
    with open(output_dir / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_dir/'evaluation_results.json'}")

    print("[4/4] Plotting...")
    plot_training_curves(model_dirs, output_dir)
    plot_task_metrics_grid("Task1_Forecast", task1, output_dir)
    if task2:
        plot_task_metrics_grid("Task2_FDD", task2, output_dir)
        plot_fdd_specific_metrics("Task2_FDD", task2, output_dir)
    plot_task_samples(models, test_hf, output_dir, device)

    print("Done.")


if __name__ == "__main__":
    main()
