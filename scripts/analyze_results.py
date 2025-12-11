#!/usr/bin/env python
"""
MOIRAI 微调实验结果分析脚本

分析所有微调实验的结果，找出最佳配置。

功能:
    1. 读取所有实验的训练历史
    2. 比较不同微调策略的效果
    3. 生成对比图表
    4. 输出最佳配置建议

使用方法:
    python analyze_results.py
    python analyze_results.py --output-dir ../outputs/buildingfm_15min
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
plt.rcParams['font.family'] = 'DejaVu Sans'

# =============================================================================
# 配置
# =============================================================================

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / 'outputs' / 'buildingfm_15min'

# 颜色方案
COLORS = {
    'small_head_only': '#3498db',
    'small_freeze_ffn': '#2ecc71',
    'small_full': '#e74c3c',
    'base_head_only': '#9b59b6',
    'base_freeze_ffn': '#f39c12',
    'base_full': '#1abc9c',
}

PATTERN_MARKERS = {
    'head_only': 'o',
    'freeze_ffn': 's',
    'full': '^',
}


# =============================================================================
# 数据加载
# =============================================================================

def find_all_experiments(output_dir: Path) -> List[Dict]:
    """找到所有已完成的实验"""
    experiments = []

    for model_dir in output_dir.glob('moirai_*'):
        if not model_dir.is_dir():
            continue

        # 解析模型名称: moirai_{size}_{pattern}_{lr}
        parts = model_dir.name.split('_')
        if len(parts) < 3:
            continue

        size = parts[1]  # small or base
        pattern = '_'.join(parts[2:-1]) if len(parts) > 3 else parts[2]

        # 检查是否有训练历史
        history_sources = [
            model_dir / 'training_history.csv',
            model_dir / 'csv_logs' / 'version_0' / 'metrics.csv',
        ]

        history_file = None
        for src in history_sources:
            if src.exists():
                history_file = src
                break

        if history_file is None:
            continue

        # 检查是否有checkpoint
        ckpt_dir = model_dir / 'checkpoints'
        has_checkpoint = ckpt_dir.exists() and len(list(ckpt_dir.glob('*.ckpt'))) > 0

        experiments.append({
            'name': model_dir.name,
            'dir': model_dir,
            'size': size,
            'pattern': pattern,
            'history_file': history_file,
            'has_checkpoint': has_checkpoint,
        })

    return experiments


def load_training_history(history_file: Path) -> pd.DataFrame:
    """加载训练历史"""
    df = pd.read_csv(history_file)

    # 处理不同格式的历史文件
    if 'train/PackedNLLLoss' in df.columns:
        # Lightning CSV logger格式
        train_df = df[df['train/PackedNLLLoss'].notna()][['epoch', 'train/PackedNLLLoss']]
        val_df = df[df['val/PackedNLLLoss'].notna()][['epoch', 'val/PackedNLLLoss']]

        train_loss = train_df.groupby('epoch')['train/PackedNLLLoss'].mean()
        val_loss = val_df.groupby('epoch')['val/PackedNLLLoss'].mean()

        result = pd.DataFrame({
            'epoch': train_loss.index,
            'train_loss': train_loss.values,
        })

        # 合并验证loss
        val_result = pd.DataFrame({
            'epoch': val_loss.index,
            'val_loss': val_loss.values,
        })
        result = result.merge(val_result, on='epoch', how='outer')

    elif 'train_loss' in df.columns:
        # 自定义格式
        result = df[['epoch', 'train_loss', 'val_loss']].copy()
    else:
        raise ValueError(f"Unknown history format: {df.columns.tolist()}")

    return result.sort_values('epoch').reset_index(drop=True)


def get_best_metrics(history: pd.DataFrame) -> Dict:
    """获取最佳指标"""
    if 'val_loss' not in history.columns:
        return {'best_val_loss': np.nan, 'best_epoch': np.nan}

    valid_losses = history['val_loss'].dropna()
    if len(valid_losses) == 0:
        return {'best_val_loss': np.nan, 'best_epoch': np.nan}

    best_idx = valid_losses.idxmin()
    return {
        'best_val_loss': valid_losses[best_idx],
        'best_epoch': history.loc[best_idx, 'epoch'],
        'final_train_loss': history['train_loss'].iloc[-1] if 'train_loss' in history.columns else np.nan,
        'total_epochs': len(history),
    }


# =============================================================================
# 分析函数
# =============================================================================

def analyze_experiments(output_dir: Path) -> pd.DataFrame:
    """分析所有实验"""
    experiments = find_all_experiments(output_dir)

    results = []
    for exp in experiments:
        try:
            history = load_training_history(exp['history_file'])
            metrics = get_best_metrics(history)

            results.append({
                'model_name': exp['name'],
                'size': exp['size'],
                'pattern': exp['pattern'],
                'has_checkpoint': exp['has_checkpoint'],
                **metrics,
            })
        except Exception as e:
            print(f"Warning: Failed to load {exp['name']}: {e}")

    return pd.DataFrame(results)


def print_analysis_report(df: pd.DataFrame):
    """打印分析报告"""
    print("\n" + "=" * 80)
    print("MOIRAI 微调实验分析报告")
    print("=" * 80)

    if len(df) == 0:
        print("\n没有找到已完成的实验!")
        return

    # 按模型大小分组
    for size in ['small', 'base']:
        size_df = df[df['size'] == size]
        if len(size_df) == 0:
            continue

        print(f"\n{'='*40}")
        print(f"  {size.upper()} 模型实验结果")
        print(f"{'='*40}")

        # 按pattern分组
        for pattern in ['head_only', 'freeze_ffn', 'full']:
            pattern_df = size_df[size_df['pattern'] == pattern]
            if len(pattern_df) == 0:
                continue

            print(f"\n  [{pattern}]")
            for _, row in pattern_df.iterrows():
                status = "✓" if row['has_checkpoint'] else "○"
                loss_str = f"{row['best_val_loss']:.4f}" if pd.notna(row['best_val_loss']) else "N/A"
                epoch_str = f"{int(row['best_epoch'])}" if pd.notna(row['best_epoch']) else "N/A"
                print(f"    {status} {row['model_name']}")
                print(f"       Best Val Loss: {loss_str} @ Epoch {epoch_str}")

    # 找出最佳模型
    valid_df = df[df['best_val_loss'].notna()]
    if len(valid_df) > 0:
        best_row = valid_df.loc[valid_df['best_val_loss'].idxmin()]

        print("\n" + "=" * 80)
        print("最佳模型")
        print("=" * 80)
        print(f"  模型: {best_row['model_name']}")
        print(f"  最佳验证Loss: {best_row['best_val_loss']:.4f}")
        print(f"  最佳Epoch: {int(best_row['best_epoch'])}")
        print(f"  配置: {best_row['size']}-{best_row['pattern']}")

        # 建议
        print("\n" + "-" * 40)
        print("建议的下一步:")
        print("-" * 40)

        if best_row['size'] == 'small':
            print("  1. 使用相同配置训练 Base 模型")
            print(f"     - 将 lr 减半: {best_row['pattern']} 策略")
            print("     - epochs 可以减少到 3-5")
        else:
            print("  1. 该配置已经是 Base 模型，可以用于评估")

        print(f"  2. 运行 evaluate_models.py 评估模型效果")
        print(f"  3. 检查 {best_row['model_name']}/checkpoints/ 中的 best-*.ckpt")


def plot_training_curves(output_dir: Path, save_path: Path):
    """绘制训练曲线对比图"""
    experiments = find_all_experiments(output_dir)

    if len(experiments) == 0:
        print("No experiments found for plotting")
        return

    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    plot_configs = [
        ('small', 'head_only', 'Small - Head Only'),
        ('small', 'freeze_ffn', 'Small - Freeze FFN'),
        ('small', 'full', 'Small - Full'),
        ('base', 'head_only', 'Base - Head Only'),
        ('base', 'freeze_ffn', 'Base - Freeze FFN'),
        ('base', 'full', 'Base - Full'),
    ]

    for ax_idx, (size, pattern, title) in enumerate(plot_configs):
        ax = axes[ax_idx]

        # 找到匹配的实验
        matching = [e for e in experiments if e['size'] == size and e['pattern'] == pattern]

        if len(matching) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='#bdc3c7')
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            continue

        # 绘制每个实验
        for exp in matching:
            try:
                history = load_training_history(exp['history_file'])

                label = exp['name'].split('_')[-1]  # 取lr部分作为label

                # 训练loss
                if 'train_loss' in history.columns:
                    ax.plot(history['epoch'], history['train_loss'],
                           alpha=0.5, linestyle='--', label=f'{label} (train)')

                # 验证loss
                if 'val_loss' in history.columns:
                    valid_vals = history[history['val_loss'].notna()]
                    ax.plot(valid_vals['epoch'], valid_vals['val_loss'],
                           linewidth=2, label=f'{label} (val)')

                    # 标记最佳点
                    best_idx = valid_vals['val_loss'].idxmin()
                    best_epoch = valid_vals.loc[best_idx, 'epoch']
                    best_loss = valid_vals.loc[best_idx, 'val_loss']
                    ax.scatter([best_epoch], [best_loss], s=100, zorder=5,
                              edgecolor='white', linewidth=2)

            except Exception as e:
                print(f"Warning: Failed to plot {exp['name']}: {e}")

        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Epoch', fontweight='bold')
        ax.set_ylabel('Loss', fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.suptitle('Training Curves: All Finetune Experiments', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved training curves to: {save_path}")


def plot_comparison_bar(df: pd.DataFrame, save_path: Path):
    """绘制对比柱状图"""
    if len(df) == 0 or df['best_val_loss'].isna().all():
        print("No valid data for comparison plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 按size分组
    for ax_idx, size in enumerate(['small', 'base']):
        ax = axes[ax_idx]
        size_df = df[df['size'] == size].copy()

        if len(size_df) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='#bdc3c7')
            ax.set_title(f'{size.upper()} Model', fontweight='bold')
            continue

        # 排序
        size_df = size_df.sort_values('best_val_loss')

        # 绘制柱状图
        colors = []
        for pattern in size_df['pattern']:
            if pattern == 'head_only':
                colors.append('#3498db')
            elif pattern == 'freeze_ffn':
                colors.append('#2ecc71')
            else:
                colors.append('#e74c3c')

        bars = ax.barh(range(len(size_df)), size_df['best_val_loss'], color=colors, alpha=0.8)

        # 标签
        ax.set_yticks(range(len(size_df)))
        ax.set_yticklabels([n.replace('moirai_', '').replace(f'{size}_', '')
                          for n in size_df['model_name']], fontsize=9)

        # 添加数值标签
        for bar, val in zip(bars, size_df['best_val_loss']):
            if pd.notna(val):
                ax.annotate(f'{val:.4f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                           xytext=(5, 0), textcoords='offset points',
                           ha='left', va='center', fontsize=9)

        ax.set_xlabel('Best Validation Loss', fontweight='bold')
        ax.set_title(f'{size.upper()} Model Comparison', fontweight='bold', fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 图例
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor='#3498db', alpha=0.8, label='head_only'),
            plt.Rectangle((0,0),1,1, facecolor='#2ecc71', alpha=0.8, label='freeze_ffn'),
            plt.Rectangle((0,0),1,1, facecolor='#e74c3c', alpha=0.8, label='full'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.suptitle('Validation Loss Comparison (lower is better)', fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comparison plot to: {save_path}")


def export_summary_csv(df: pd.DataFrame, save_path: Path):
    """导出汇总CSV"""
    if len(df) == 0:
        print("No data to export")
        return

    # 添加排名
    df_sorted = df.copy()
    df_sorted['rank'] = df_sorted['best_val_loss'].rank(method='min', na_option='bottom')
    df_sorted = df_sorted.sort_values('rank')

    df_sorted.to_csv(save_path, index=False)
    print(f"Saved summary to: {save_path}")


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyze MOIRAI finetuning experiments')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR,
                       help='Directory containing experiment outputs')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')

    args = parser.parse_args()

    output_dir = args.output_dir

    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return

    print(f"Analyzing experiments in: {output_dir}")

    # 分析实验
    df = analyze_experiments(output_dir)

    # 打印报告
    print_analysis_report(df)

    # 生成图表
    if not args.no_plots and len(df) > 0:
        analysis_dir = output_dir / 'analysis'
        analysis_dir.mkdir(exist_ok=True)

        plot_training_curves(output_dir, analysis_dir / 'training_curves.png')
        plot_comparison_bar(df, analysis_dir / 'comparison.png')
        export_summary_csv(df, analysis_dir / 'experiment_summary.csv')

        print(f"\nAnalysis outputs saved to: {analysis_dir}")


if __name__ == '__main__':
    main()
