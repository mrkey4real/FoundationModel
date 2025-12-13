#!/usr/bin/env python
"""Direct evaluation comparing zero-shot vs finetuned MOIRAI"""

import sys
sys.path.insert(0, 'E:/MOIRAI/src')

import numpy as np
import torch
from pathlib import Path
import datasets
from uni2ts.model.moirai import MoiraiModule

# Constants
PATCH_SIZE = 32


def prepare_native_input(var_data, context_length, prediction_length, patch_size, max_patch_size, device):
    """Prepare input tensors for MoiraiModule forward pass.

    Key points from evaluate_models.py:
    1. Replace NaN with 0 in target
    2. Pad at BEGINNING if needed
    3. observed_mask: True only for context patches that are NOT NaN
    4. prediction_mask: True only for prediction patches
    """
    total_length = context_length + prediction_length
    var_data = var_data[:total_length].astype(np.float32)

    # Handle NaN: replace with 0, track in nan_mask
    nan_mask = np.isnan(var_data)
    var_data_clean = np.nan_to_num(var_data, nan=0.0)

    # Pad at BEGINNING if needed
    num_patches = (total_length + patch_size - 1) // patch_size
    padded_len = num_patches * patch_size

    if padded_len > total_length:
        pad_amount = padded_len - total_length
        var_data_clean = np.concatenate([np.zeros(pad_amount, dtype=np.float32), var_data_clean])
        nan_mask = np.concatenate([np.ones(pad_amount, dtype=bool), nan_mask])  # pad region is "missing"

    # Calculate context patches accounting for left padding
    context_end_idx = padded_len - prediction_length
    context_patches = context_end_idx // patch_size

    # Create target tensor
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

    # IDs
    sample_id = np.ones((1, num_patches), dtype=np.int64)
    time_id = np.arange(num_patches, dtype=np.int64).reshape(1, -1)
    variate_id = np.zeros((1, num_patches), dtype=np.int64)

    # Prediction mask: True only for prediction patches
    prediction_mask = np.zeros((1, num_patches), dtype=bool)
    prediction_mask[0, context_patches:] = True

    patch_size_tensor = np.full((1, num_patches), patch_size, dtype=np.int64)

    # Convert to tensors
    return (
        torch.tensor(target, device=device),
        torch.tensor(observed_mask, device=device),
        torch.tensor(sample_id, device=device),
        torch.tensor(time_id, device=device),
        torch.tensor(variate_id, device=device),
        torch.tensor(prediction_mask, device=device),
        torch.tensor(patch_size_tensor, device=device),
    )


def predict_moirai(module, var_data, context_length, prediction_length, patch_size=PATCH_SIZE, num_samples=20, device='cpu'):
    """Make prediction using MOIRAI module."""
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

        # Get prediction region (patches where prediction_mask=True)
        num_patches = target.shape[1]
        total_length = context_length + prediction_length
        padded_len = num_patches * patch_size
        context_end_idx = padded_len - prediction_length
        context_patches = context_end_idx // patch_size

        # Extract prediction patches
        pred_samples = samples[:, 0, context_patches:, :patch_size]  # (num_samples, pred_patches, patch_size)

        # Reshape and take only prediction_length steps
        pred_samples = pred_samples.reshape(num_samples, -1)[:, :prediction_length]
        pred_samples_np = pred_samples.cpu().numpy()

        # Use MEDIAN as point estimate (official MOIRAI approach)
        pred_median = np.median(pred_samples_np, axis=0)

    return pred_median, pred_samples_np


def compute_smape(pred, actual):
    """Symmetric Mean Absolute Percentage Error"""
    mask = ~(np.isnan(pred) | np.isnan(actual))
    if mask.sum() == 0:
        return np.nan
    p, a = pred[mask], actual[mask]
    denom = (np.abs(p) + np.abs(a)) / 2
    denom = np.where(denom < 1e-8, 1e-8, denom)
    return np.mean(np.abs(p - a) / denom) * 100


def compute_mae(pred, actual):
    """Mean Absolute Error"""
    mask = ~(np.isnan(pred) | np.isnan(actual))
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs(pred[mask] - actual[mask]))


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Load test data
    data_dir = Path('E:/MOIRAI/data/buildingfm_processed_15min')
    test_hf = datasets.load_from_disk(str(data_dir / 'hf/buildingfm_test'))
    print(f'Test samples: {len(test_hf)}')

    # Load models
    print('\nLoading zero-shot model (HuggingFace)...')
    zs_module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-small')

    print('Loading finetuned model...')
    model_dir = Path('E:/MOIRAI/outputs/buildingfm_15min/moirai_small_head_only_1e4')
    ckpt_path = list(model_dir.glob('checkpoints/best-*.ckpt'))[-1]
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ft_module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-small')
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('module.')}
    ft_module.load_state_dict(state_dict)

    # Evaluation parameters
    context_length = 144  # 1.5 days
    prediction_length = 48  # 0.5 days

    print('\nStarting evaluation...')

    # Evaluate multiple variable groups
    var_groups = {
        'Main Power': [0],
        'ODU Power': [2, 3, 4],
        'Zone Temps': [78, 79, 80],
    }

    for group_name, var_ids in var_groups.items():
        print(f'\n{"="*60}')
        print(f'Evaluating: {group_name}')
        print(f'{"="*60}')

        zs_smapes, ft_smapes = [], []
        zs_maes, ft_maes = [], []

        num_eval = min(30, len(test_hf))

        for i in range(num_eval):
            sample = test_hf[i]
            target = np.array(sample['target'], dtype=np.float32)

            for var_id in var_ids:
                if var_id >= target.shape[0]:
                    continue

                var_data = target[var_id, :]

                # Check data validity
                if np.isnan(var_data[:context_length]).mean() > 0.5:
                    continue

                actual = var_data[context_length:context_length + prediction_length]

                # Zero-shot prediction
                zs_pred, _ = predict_moirai(
                    zs_module, var_data, context_length, prediction_length,
                    patch_size=PATCH_SIZE, num_samples=20, device=device
                )

                # Finetuned prediction
                ft_pred, _ = predict_moirai(
                    ft_module, var_data, context_length, prediction_length,
                    patch_size=PATCH_SIZE, num_samples=20, device=device
                )

                # Compute metrics
                zs_smape = compute_smape(zs_pred, actual)
                ft_smape = compute_smape(ft_pred, actual)
                zs_mae = compute_mae(zs_pred, actual)
                ft_mae = compute_mae(ft_pred, actual)

                if not np.isnan(zs_smape) and not np.isnan(ft_smape):
                    zs_smapes.append(zs_smape)
                    ft_smapes.append(ft_smape)
                    zs_maes.append(zs_mae)
                    ft_maes.append(ft_mae)

        if len(zs_smapes) > 0:
            print(f'\nEvaluated {len(zs_smapes)} valid predictions')

            print(f'\nZero-shot model:')
            print(f'  SMAPE: {np.mean(zs_smapes):.2f} +/- {np.std(zs_smapes):.2f}')
            print(f'  MAE:   {np.mean(zs_maes):.4f} +/- {np.std(zs_maes):.4f}')

            print(f'\nFinetuned model:')
            print(f'  SMAPE: {np.mean(ft_smapes):.2f} +/- {np.std(ft_smapes):.2f}')
            print(f'  MAE:   {np.mean(ft_maes):.4f} +/- {np.std(ft_maes):.4f}')

            # Improvement
            smape_improve = (np.mean(zs_smapes) - np.mean(ft_smapes)) / np.mean(zs_smapes) * 100
            mae_improve = (np.mean(zs_maes) - np.mean(ft_maes)) / np.mean(zs_maes) * 100

            print(f'\nImprovement (positive = finetuned is better):')
            print(f'  SMAPE: {smape_improve:+.1f}%')
            print(f'  MAE:   {mae_improve:+.1f}%')

            # Per-sample comparison
            smape_diff = np.array(ft_smapes) - np.array(zs_smapes)
            better_ratio = (smape_diff < 0).sum() / len(smape_diff) * 100
            print(f'\nFinetuned better on {better_ratio:.1f}% of samples')


if __name__ == '__main__':
    main()
