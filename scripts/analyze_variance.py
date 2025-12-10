"""Analyze data variance to find better evaluation windows"""
import datasets
import numpy as np
import matplotlib.pyplot as plt

test_hf = datasets.load_from_disk('C:/MOIRAI/data/buildingfm_processed/hf/buildingfm_test')
sample = test_hf[0]
target = np.array(sample['target'])

print(f"Sample 0: {sample['item_id']}")
print(f"Shape: {target.shape} (variables x timesteps)")
print(f"Total timesteps: {target.shape[1]} = {target.shape[1]*5/60:.1f} hours = {target.shape[1]*5/60/24:.1f} days")

# Current settings
CONTEXT_LENGTH = 256*6  # 1536
PREDICTION_LENGTH = 64*6  # 384
total_window = CONTEXT_LENGTH + PREDICTION_LENGTH  # 1920

print(f"\nCurrent evaluation window:")
print(f"  Context: {CONTEXT_LENGTH} steps = {CONTEXT_LENGTH*5/60:.1f} hours")
print(f"  Prediction: {PREDICTION_LENGTH} steps = {PREDICTION_LENGTH*5/60:.1f} hours")
print(f"  Total: {total_window} steps = {total_window*5/60:.1f} hours")

# Analyze variance in prediction window for key variables
var_ids = {
    'Main Power (var 10)': 10,
    'Zone Temp (var 50)': 50,
}

print("\n" + "="*60)
print("Variance analysis in PREDICTION window (last 384 steps):")
print("="*60)

for name, vid in var_ids.items():
    data = target[vid, :]

    # Current: prediction window is the last PREDICTION_LENGTH of first total_window
    pred_window = data[CONTEXT_LENGTH:CONTEXT_LENGTH+PREDICTION_LENGTH]

    valid_data = pred_window[~np.isnan(pred_window)]
    if len(valid_data) > 0:
        std = np.std(valid_data)
        range_val = np.max(valid_data) - np.min(valid_data)
        mean_val = np.mean(valid_data)
        print(f"\n{name}:")
        print(f"  Mean: {mean_val:.3f}")
        print(f"  Std:  {std:.3f}")
        print(f"  Range: {range_val:.3f} (max-min)")
        print(f"  CV (std/mean): {std/abs(mean_val)*100:.1f}%")

# Find windows with more variance
print("\n" + "="*60)
print("Finding windows with MORE variance for better evaluation:")
print("="*60)

for name, vid in var_ids.items():
    data = target[vid, :]

    best_start = 0
    best_range = 0

    # Slide window and find max variance region
    for start in range(0, len(data) - total_window, 100):
        pred_window = data[start + CONTEXT_LENGTH : start + total_window]
        valid = pred_window[~np.isnan(pred_window)]
        if len(valid) > 10:
            window_range = np.max(valid) - np.min(valid)
            if window_range > best_range:
                best_range = window_range
                best_start = start

    print(f"\n{name}:")
    print(f"  Best start index: {best_start}")
    print(f"  Best prediction range: {best_range:.3f}")

    # Compare with default (start=0)
    default_pred = data[CONTEXT_LENGTH:CONTEXT_LENGTH+PREDICTION_LENGTH]
    default_valid = default_pred[~np.isnan(default_pred)]
    default_range = np.max(default_valid) - np.min(default_valid) if len(default_valid) > 0 else 0
    print(f"  Default (start=0) range: {default_range:.3f}")
    print(f"  Improvement: {best_range/default_range:.1f}x more variance" if default_range > 0 else "  N/A")

# Plot the full time series to visualize
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

for idx, (name, vid) in enumerate(var_ids.items()):
    ax = axes[idx]
    data = target[vid, :]
    time = np.arange(len(data))

    ax.plot(time, data, 'b-', alpha=0.7, linewidth=0.8)

    # Mark current evaluation window
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Window start')
    ax.axvline(x=CONTEXT_LENGTH, color='orange', linestyle='--', alpha=0.5, label='Prediction start')
    ax.axvline(x=total_window, color='red', linestyle='--', alpha=0.5, label='Window end')

    # Shade prediction region
    ax.axvspan(CONTEXT_LENGTH, total_window, alpha=0.2, color='yellow', label='Prediction region')

    ax.set_xlabel('Time Step (5-min intervals)')
    ax.set_ylabel(name)
    ax.set_title(f'{name} - Full time series with current evaluation window')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('C:/MOIRAI/outputs/evaluation/variance_analysis.png', dpi=150)
print(f"\nSaved: C:/MOIRAI/outputs/evaluation/variance_analysis.png")
