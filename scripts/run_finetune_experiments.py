#!/usr/bin/env python
"""
MOIRAI 批量微调实验脚本

自动运行预定义的超参数组合，无需手动调参。
按照训练时间从短到长排序执行。

使用方法:
    1. 将此脚本放在 scripts/ 目录下
    2. 修改 EXPERIMENTS 配置（如需要）
    3. 运行: python run_finetune_experiments.py
    4. 可选: python run_finetune_experiments.py --only small  # 只运行small模型
    5. 可选: python run_finetune_experiments.py --only base   # 只运行base模型
    6. 可选: python run_finetune_experiments.py --resume      # 跳过已完成的实验
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import numpy as np

# =============================================================================
# 实验配置 - 基于社区最佳实践和论文研究
# =============================================================================

# 注意: batch_size从64改为16，以适应新的数据量
# 新数据准备后预计有~1900个样本，batch_size=16 → 每epoch约119步

EXPERIMENTS: Dict[str, Dict[str, Any]] = {
    # ========================================================================
    # Small 模型 - 细化网格搜索 (14M params, ~1min/实验)
    # ========================================================================

    # === Head Only (只训练输出头，~500K params) ===
    # 可用较高lr，范围: 2e-5 ~ 2e-4
    'small_head_2e5': {
        'pretrained': 'small',
        'pattern': 'head_only',
        'lr': 2e-5,
        'epochs': 25,
        'batch_size': 16,
        'patience': 10,
        'warmup_steps': 50,
        'weight_decay': 0.1,
    },
    'small_head_5e5': {
        'pretrained': 'small',
        'pattern': 'head_only',
        'lr': 5e-5,
        'epochs': 25,
        'batch_size': 16,
        'patience': 10,
        'warmup_steps': 50,
        'weight_decay': 0.1,
    },
    'small_head_7e5': {
        'pretrained': 'small',
        'pattern': 'head_only',
        'lr': 7e-5,
        'epochs': 25,
        'batch_size': 16,
        'patience': 10,
        'warmup_steps': 50,
        'weight_decay': 0.1,
    },
    'small_head_1e4': {
        'pretrained': 'small',
        'pattern': 'head_only',
        'lr': 1e-4,
        'epochs': 25,
        'batch_size': 16,
        'patience': 10,
        'warmup_steps': 50,
        'weight_decay': 0.1,
    },
    'small_head_1.5e4': {
        'pretrained': 'small',
        'pattern': 'head_only',
        'lr': 1.5e-4,
        'epochs': 25,
        'batch_size': 16,
        'patience': 10,
        'warmup_steps': 50,
        'weight_decay': 0.1,
    },
    'small_head_2e4': {
        'pretrained': 'small',
        'pattern': 'head_only',
        'lr': 2e-4,
        'epochs': 25,
        'batch_size': 16,
        'patience': 10,
        'warmup_steps': 50,
        'weight_decay': 0.1,
    },

    # === Freeze FFN (冻结FFN，训练Attention，~3M params) ===
    # 中等lr，范围: 2e-6 ~ 2e-5
    'small_freeze_2e6': {
        'pretrained': 'small',
        'pattern': 'freeze_ffn',
        'lr': 2e-6,
        'epochs': 20,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_freeze_5e6': {
        'pretrained': 'small',
        'pattern': 'freeze_ffn',
        'lr': 5e-6,
        'epochs': 20,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_freeze_7e6': {
        'pretrained': 'small',
        'pattern': 'freeze_ffn',
        'lr': 7e-6,
        'epochs': 20,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_freeze_1e5': {
        'pretrained': 'small',
        'pattern': 'freeze_ffn',
        'lr': 1e-5,
        'epochs': 20,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_freeze_1.5e5': {
        'pretrained': 'small',
        'pattern': 'freeze_ffn',
        'lr': 1.5e-5,
        'epochs': 20,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_freeze_2e5': {
        'pretrained': 'small',
        'pattern': 'freeze_ffn',
        'lr': 2e-5,
        'epochs': 20,
        'batch_size': 16,
        'patience': 8,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },

    # === Full (全参数微调，14M params) ===
    # 需要很低lr (论文建议5e-6~5e-7)，范围: 5e-7 ~ 7e-6
    'small_full_5e7': {
        'pretrained': 'small',
        'pattern': 'full',
        'lr': 5e-7,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_full_1e6': {
        'pretrained': 'small',
        'pattern': 'full',
        'lr': 1e-6,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_full_2e6': {
        'pretrained': 'small',
        'pattern': 'full',
        'lr': 2e-6,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_full_3e6': {
        'pretrained': 'small',
        'pattern': 'full',
        'lr': 3e-6,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_full_5e6': {
        'pretrained': 'small',
        'pattern': 'full',
        'lr': 5e-6,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'small_full_7e6': {
        'pretrained': 'small',
        'pattern': 'full',
        'lr': 7e-6,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },

    # ========================================================================
    # Base 模型实验 (91M params) - 待Small结果确定后再细化
    # ========================================================================

    # Head Only
    'base_head_1e5': {
        'pretrained': 'base',
        'pattern': 'head_only',
        'lr': 1e-5,
        'epochs': 20,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'base_head_2e5': {
        'pretrained': 'base',
        'pattern': 'head_only',
        'lr': 2e-5,
        'epochs': 20,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'base_head_5e5': {
        'pretrained': 'base',
        'pattern': 'head_only',
        'lr': 5e-5,
        'epochs': 20,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'base_head_1e4': {
        'pretrained': 'base',
        'pattern': 'head_only',
        'lr': 1e-4,
        'epochs': 20,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },

    # Freeze FFN
    'base_freeze_1e6': {
        'pretrained': 'base',
        'pattern': 'freeze_ffn',
        'lr': 1e-6,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 150,
        'weight_decay': 0.1,
    },
    'base_freeze_2e6': {
        'pretrained': 'base',
        'pattern': 'freeze_ffn',
        'lr': 2e-6,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 150,
        'weight_decay': 0.1,
    },
    'base_freeze_5e6': {
        'pretrained': 'base',
        'pattern': 'freeze_ffn',
        'lr': 5e-6,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 150,
        'weight_decay': 0.1,
    },
    'base_freeze_1e5': {
        'pretrained': 'base',
        'pattern': 'freeze_ffn',
        'lr': 1e-5,
        'epochs': 15,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 150,
        'weight_decay': 0.1,
    },

    # Full
    'base_full_2e7': {
        'pretrained': 'base',
        'pattern': 'full',
        'lr': 2e-7,
        'epochs': 10,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'base_full_5e7': {
        'pretrained': 'base',
        'pattern': 'full',
        'lr': 5e-7,
        'epochs': 10,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'base_full_1e6': {
        'pretrained': 'base',
        'pattern': 'full',
        'lr': 1e-6,
        'epochs': 10,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    'base_full_2e6': {
        'pretrained': 'base',
        'pattern': 'full',
        'lr': 2e-6,
        'epochs': 10,
        'batch_size': 16,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
}

# 执行顺序（按预估训练时间排序）
EXECUTION_ORDER: List[str] = [
    # Phase 1: Small模型细化搜索 (~20min total)
    # Head Only (6个实验)
    'small_head_2e5',
    'small_head_5e5',
    'small_head_7e5',
    'small_head_1e4',
    'small_head_1.5e4',
    'small_head_2e4',
    # Freeze FFN (6个实验)
    'small_freeze_2e6',
    'small_freeze_5e6',
    'small_freeze_7e6',
    'small_freeze_1e5',
    'small_freeze_1.5e5',
    'small_freeze_2e5',
    # Full (6个实验)
    'small_full_5e7',
    'small_full_1e6',
    'small_full_2e6',
    'small_full_3e6',
    'small_full_5e6',
    'small_full_7e6',

    # Phase 2: Base模型 (基于Small结果选择性运行)
    # Head Only (4个实验)
    'base_head_1e5',
    'base_head_2e5',
    'base_head_5e5',
    'base_head_1e4',
    # Freeze FFN (4个实验)
    'base_freeze_1e6',
    'base_freeze_2e6',
    'base_freeze_5e6',
    'base_freeze_1e5',
    # Full (4个实验)
    'base_full_2e7',
    'base_full_5e7',
    'base_full_1e6',
    'base_full_2e6',
]


# =============================================================================
# 路径配置
# =============================================================================

# 自动检测项目根目录
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'buildingfm_processed_15min'
OUTPUT_DIR = PROJECT_ROOT / 'outputs' / 'buildingfm_15min'
RESULTS_FILE = OUTPUT_DIR / 'experiment_results.csv'


# =============================================================================
# 训练脚本模板
# =============================================================================

TRAIN_SCRIPT_TEMPLATE = '''#!/usr/bin/env python
"""
Auto-generated training script for experiment: {exp_name}
Generated at: {timestamp}
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Override CONFIG for this experiment
CONFIG = {{
    'mode': 'finetune',
    'data_dir': '{data_dir}',
    'output_dir': '{output_dir}',
    'finetune': {{
        'pretrained': '{pretrained}',
        'pattern': '{pattern}',
        'model_name': '{model_name}',
        'epochs': {epochs},
        'lr': {lr},
        'batch_size': {batch_size},
        'patience': {patience},
        'weight_decay': {weight_decay},
        'warmup_steps': {warmup_steps},
    }},
    'hardware': {{
        'num_workers': 0,
        'gpus': 1,
    }},
    'resume_from': None,
}}

# Import and run the training code
if __name__ == '__main__':
    # Re-import with modified CONFIG
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_buildingfm", "{train_script}")
    train_module = importlib.util.module_from_spec(spec)
    
    # Inject our CONFIG
    train_module.CONFIG = CONFIG
    
    # Execute
    spec.loader.exec_module(train_module)
'''


# =============================================================================
# 实验运行器
# =============================================================================

def check_experiment_completed(model_name: str) -> bool:
    """检查实验是否已完成"""
    model_dir = OUTPUT_DIR / model_name
    if not model_dir.exists():
        return False
    
    # 检查是否有checkpoint
    ckpt_dir = model_dir / 'checkpoints'
    if not ckpt_dir.exists():
        return False
    
    # 检查是否有best checkpoint
    best_ckpts = list(ckpt_dir.glob('best-*.ckpt'))
    return len(best_ckpts) > 0


def get_model_name(exp_name: str, config: Dict[str, Any]) -> str:
    """生成模型名称

    从exp_name中提取lr字符串，避免浮点数格式化问题
    例如: small_head_1.5e4 -> moirai_small_head_only_1.5e4
    """
    # 从exp_name中提取lr部分 (最后一个_后面的内容)
    parts = exp_name.split('_')
    lr_str = parts[-1]  # e.g., "1.5e4", "5e6", "2e5"
    return f"moirai_{config['pretrained']}_{config['pattern']}_{lr_str}"


def run_experiment_inline(exp_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """直接在当前进程运行实验（推荐方式）"""
    import torch
    import sys

    model_name = get_model_name(exp_name, config)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"MODEL: {model_name}")
    print(f"CONFIG: {config}")
    print(f"{'='*70}\n")

    # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    start_time = time.time()
    success = False
    error_msg = None
    best_val_loss = None
    best_epoch = None

    try:
        # 添加src到路径（如果还没有）
        src_path = str(SCRIPT_DIR.parent / 'src')
        if src_path not in sys.path:
            sys.path.insert(0, src_path)

        # 直接导入训练模块的finetune函数
        from train_buildingfm import finetune

        # 调用finetune函数
        run_dir = finetune(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR,
            model_name=model_name,
            pretrained_model=config['pretrained'],
            finetune_pattern=config['pattern'],
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            lr=config['lr'],
            weight_decay=config['weight_decay'],
            num_workers=0,
            gpus=1,
            resume=None,
            patience=config['patience'],
            warmup_steps=config.get('warmup_steps'),
        )

        elapsed = time.time() - start_time
        success = True

        # 读取训练结果 - 尝试多个可能的位置
        model_dir = OUTPUT_DIR / model_name
        history_sources = [
            model_dir / 'training_history.csv',
            model_dir / 'csv_logs' / 'version_0' / 'metrics.csv',
        ]

        for history_file in history_sources:
            if history_file.exists():
                df = pd.read_csv(history_file)

                # 处理Lightning CSV格式
                if 'val/PackedNLLLoss' in df.columns:
                    val_df = df[df['val/PackedNLLLoss'].notna()]
                    if len(val_df) > 0:
                        best_val_loss = val_df['val/PackedNLLLoss'].min()
                        best_idx = val_df['val/PackedNLLLoss'].idxmin()
                        best_epoch = val_df.loc[best_idx, 'epoch']
                    break
                # 处理自定义格式
                elif 'val_loss' in df.columns:
                    valid_losses = df['val_loss'].dropna()
                    if len(valid_losses) > 0:
                        best_val_loss = valid_losses.min()
                        best_epoch = df.loc[valid_losses.idxmin(), 'epoch']
                    break

    except Exception as e:
        elapsed = time.time() - start_time
        success = False
        error_msg = str(e)
        print(f"\nERROR in experiment {exp_name}: {e}")
        import traceback
        traceback.print_exc()

    return {
        'experiment': exp_name,
        'model_name': model_name,
        'pretrained': config['pretrained'],
        'pattern': config['pattern'],
        'lr': config['lr'],
        'epochs': config['epochs'],
        'batch_size': config['batch_size'],
        'elapsed_seconds': elapsed,
        'elapsed_minutes': elapsed / 60,
        'success': success,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'error': error_msg,
        'timestamp': datetime.now().isoformat(),
    }


def run_all_experiments(
    experiments: Dict[str, Dict],
    order: List[str],
    only_size: str = None,
    resume: bool = False,
) -> pd.DataFrame:
    """运行所有实验"""
    
    results = []
    
    # 加载已有结果（如果resume=True）
    if resume and RESULTS_FILE.exists():
        existing_df = pd.read_csv(RESULTS_FILE)
        results = existing_df.to_dict('records')
        completed = set(existing_df['experiment'].tolist())
        print(f"Resuming from {len(completed)} completed experiments")
    else:
        completed = set()
    
    # 过滤实验
    filtered_order = []
    for exp_name in order:
        if exp_name not in experiments:
            continue
        if only_size and experiments[exp_name]['pretrained'] != only_size:
            continue
        if resume and exp_name in completed:
            print(f"Skipping {exp_name} (already completed)")
            continue
        filtered_order.append(exp_name)
    
    total = len(filtered_order)
    print(f"\n{'='*70}")
    print(f"RUNNING {total} EXPERIMENTS")
    print(f"{'='*70}\n")
    
    for i, exp_name in enumerate(filtered_order, 1):
        print(f"\n[{i}/{total}] Starting: {exp_name}")
        
        config = experiments[exp_name]
        model_name = get_model_name(exp_name, config)
        
        # 检查是否已完成
        if resume and check_experiment_completed(model_name):
            print(f"  -> Already completed, skipping")
            continue
        
        # 运行实验
        result = run_experiment_inline(exp_name, config)
        results.append(result)
        
        # 保存中间结果
        df = pd.DataFrame(results)
        df.to_csv(RESULTS_FILE, index=False)
        
        # 打印进度
        print(f"\n[{i}/{total}] Completed: {exp_name}")
        print(f"  Time: {result['elapsed_minutes']:.1f} min")
        print(f"  Best Val Loss: {result.get('best_val_loss', 'N/A')}")
        print(f"  Best Epoch: {result.get('best_epoch', 'N/A')}")
        
        # 估算剩余时间
        if i < total:
            avg_time = sum(r['elapsed_minutes'] for r in results if r.get('elapsed_minutes')) / len([r for r in results if r.get('elapsed_minutes')])
            remaining = (total - i) * avg_time
            print(f"  Est. Remaining: {remaining:.0f} min ({remaining/60:.1f} hours)")
    
    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """打印实验总结"""
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80 + "\n")
    
    # 按模型大小分组
    for size in ['small', 'base']:
        size_df = df[df['pretrained'] == size]
        if len(size_df) == 0:
            continue
        
        print(f"\n--- {size.upper()} MODEL ---")
        
        # 按pattern分组
        for pattern in ['head_only', 'freeze_ffn', 'full']:
            pattern_df = size_df[size_df['pattern'] == pattern]
            if len(pattern_df) == 0:
                continue
            
            print(f"\n  {pattern}:")
            for _, row in pattern_df.iterrows():
                status = "✓" if row['success'] else "✗"
                loss_str = f"{row['best_val_loss']:.4f}" if pd.notna(row['best_val_loss']) else "N/A"
                print(f"    {status} lr={row['lr']:.0e}: val_loss={loss_str}, epoch={row.get('best_epoch', 'N/A')}, time={row['elapsed_minutes']:.1f}min")
    
    # 找出最佳模型
    if 'best_val_loss' in df.columns:
        valid_df = df[df['best_val_loss'].notna()]
        if len(valid_df) > 0:
            best_row = valid_df.loc[valid_df['best_val_loss'].idxmin()]
            print(f"\n{'='*80}")
            print(f"BEST MODEL: {best_row['model_name']}")
            print(f"  Val Loss: {best_row['best_val_loss']:.4f}")
            print(f"  Config: {best_row['pretrained']}-{best_row['pattern']}, lr={best_row['lr']:.0e}")
            print(f"{'='*80}")


# =============================================================================
# 主程序
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run MOIRAI finetuning experiments')
    parser.add_argument('--only', choices=['small','base'], help='Only run experiments for this model size')
    parser.add_argument('--resume', action='store_true', help='Skip completed experiments')
    parser.add_argument('--list', action='store_true', help='List all experiments and exit')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    args = parser.parse_args()
    
    # 列出所有实验
    if args.list:
        print("\nAvailable experiments:\n")
        for exp_name in EXECUTION_ORDER:
            if exp_name in EXPERIMENTS:
                config = EXPERIMENTS[exp_name]
                print(f"  {exp_name}")
                print(f"    {config.get('description', 'No description')}")
                print(f"    {config['pretrained']}-{config['pattern']}, lr={config['lr']:.0e}, epochs={config['epochs']}")
                print()
        return
    
    # Dry run
    if args.dry_run:
        print("\nWould run the following experiments:\n")
        for exp_name in EXECUTION_ORDER:
            if exp_name in EXPERIMENTS:
                config = EXPERIMENTS[exp_name]
                if args.only and config['pretrained'] != args.only:
                    continue
                model_name = get_model_name(exp_name, config)
                print(f"  {exp_name} -> {model_name}")
        return
    
    # 确保输出目录存在
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 运行实验
    results_df = run_all_experiments(
        EXPERIMENTS,
        EXECUTION_ORDER,
        only_size=args.only,
        resume=args.resume,
    )
    
    # 打印总结
    if len(results_df) > 0:
        print_summary(results_df)
        print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == '__main__':
    main()
