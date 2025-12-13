#!/usr/bin/env python
"""分析训练曲线和评估结果"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_training_log(log_path):
    """分析训练日志"""
    df = pd.read_csv(log_path)

    # 分离train和val loss
    train_df = df[df['train/PackedNLLLoss'].notna()][['epoch', 'step', 'train/PackedNLLLoss']].copy()
    val_df = df[df['val/PackedNLLLoss'].notna()][['epoch', 'step', 'val/PackedNLLLoss']].copy()

    print('=' * 60)
    print('训练损失变化 (Train Loss)')
    print('=' * 60)
    for _, row in train_df.iterrows():
        print(f"Epoch {int(row['epoch']):2d} | Step {int(row['step']):4d} | Loss: {row['train/PackedNLLLoss']:.4f}")

    print()
    print('=' * 60)
    print('验证损失变化 (Val Loss)')
    print('=' * 60)
    for _, row in val_df.iterrows():
        print(f"Epoch {int(row['epoch']):2d} | Step {int(row['step']):4d} | Loss: {row['val/PackedNLLLoss']:.4f}")

    # 计算统计
    train_losses = train_df['train/PackedNLLLoss'].values
    val_losses = val_df['val/PackedNLLLoss'].values

    print()
    print('=' * 60)
    print('统计分析')
    print('=' * 60)

    train_change = (train_losses[-1] - train_losses[0]) / train_losses[0] * 100
    val_change = (val_losses.min() - val_losses[0]) / val_losses[0] * 100

    print(f"Train Loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f} (变化: {train_change:+.1f}%)")
    print(f"Val Loss:   {val_losses[0]:.4f} -> {val_losses.min():.4f} (变化: {val_change:+.1f}%)")

    best_epoch = val_df.loc[val_df['val/PackedNLLLoss'].idxmin(), 'epoch']
    print(f"Best Val Loss: {val_losses.min():.4f} at Epoch {best_epoch:.0f}")

    # 检查过拟合
    print()
    print('过拟合检查:')
    print(f"  最后Train Loss: {train_losses[-1]:.4f}")
    print(f"  最后Val Loss:   {val_losses[-1]:.4f}")
    gap = val_losses[-1] - train_losses[-1]
    print(f"  Gap: {gap:.4f}")
    if gap > 0.1:
        print("  -> 存在过拟合迹象")
    else:
        print("  -> 未见明显过拟合")

    return train_losses, val_losses


def compare_with_old(old_log_path, new_log_path):
    """对比新旧训练结果"""
    print()
    print('=' * 60)
    print('新旧训练对比')
    print('=' * 60)

    old_df = pd.read_csv(old_log_path)
    new_df = pd.read_csv(new_log_path)

    old_train = old_df[old_df['train/PackedNLLLoss'].notna()]['train/PackedNLLLoss'].values
    old_val = old_df[old_df['val/PackedNLLLoss'].notna()]['val/PackedNLLLoss'].values

    new_train = new_df[new_df['train/PackedNLLLoss'].notna()]['train/PackedNLLLoss'].values
    new_val = new_df[new_df['val/PackedNLLLoss'].notna()]['val/PackedNLLLoss'].values

    print(f"{'指标':<20} {'旧(30样本)':<15} {'新(1085样本)':<15} {'变化':<10}")
    print('-' * 60)
    print(f"{'训练步数':<20} {len(old_train):<15} {len(new_train):<15}")
    print(f"{'初始Train Loss':<20} {old_train[0]:<15.4f} {new_train[0]:<15.4f}")
    print(f"{'最终Train Loss':<20} {old_train[-1]:<15.4f} {new_train[-1]:<15.4f}")
    print(f"{'初始Val Loss':<20} {old_val[0]:<15.4f} {new_val[0]:<15.4f}")
    print(f"{'最佳Val Loss':<20} {old_val.min():<15.4f} {new_val.min():<15.4f}")

    old_improvement = (old_val.min() - old_val[0]) / old_val[0] * 100
    new_improvement = (new_val.min() - new_val[0]) / new_val[0] * 100
    print(f"{'Val Loss改善':<20} {old_improvement:<+14.1f}% {new_improvement:<+14.1f}%")


if __name__ == '__main__':
    # 分析新训练结果
    new_log = Path('E:/MOIRAI/outputs/buildingfm_15min/moirai_small_head_only_1e4/csv_logs/version_0/metrics.csv')

    print("\n" + "=" * 60)
    print("新训练结果分析 (1085样本, batch_size=16)")
    print("=" * 60 + "\n")

    analyze_training_log(new_log)

    # 如果有旧结果，进行对比
    old_log = Path('E:/MOIRAI/outputs/buildingfm_15min_old_30samples/moirai_small_head_only_1e4/csv_logs/version_0/metrics.csv')
    if old_log.exists():
        compare_with_old(old_log, new_log)
