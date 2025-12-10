#!/usr/bin/env python
"""
BuildingFM Training Script - Spyder Friendly Version

直接修改下面的配置参数，然后在Spyder中运行整个脚本即可。

使用方法:
    1. 首次运行: 设置 RUN_MODE = 'build' 来构建数据集
    2. 从零训练: 设置 RUN_MODE = 'train' 
    3. 微调预训练: 设置 RUN_MODE = 'finetune' (推荐!)
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Generator, Optional, Callable
from functools import partial
from datetime import datetime
import time

import numpy as np
import pandas as pd
import torch
import datasets
from datasets import Features, Sequence, Value

# =============================================================================
# 配置参数 - 只需修改这一个区块！
# =============================================================================

CONFIG = {
    # -------------------------------------------------------------------------
    # 基础设置
    # -------------------------------------------------------------------------
    'mode': 'finetune',        # 'build' | 'train' | 'finetune'
    'data_dir': '../data/buildingfm_processed_15min',
    'output_dir': '../outputs/buildingfm_15min',
    
    # -------------------------------------------------------------------------
    # 微调配置 (mode='finetune' 时使用) - 推荐!
    # -------------------------------------------------------------------------
    'finetune': {
        # 预训练模型: 'small' | 'base' | 'large'
        # small: 14M参数, 4-6GB显存, 快速实验
        # base:  80M参数, 8-12GB显存, 推荐RTX 5070
        # large: 300M参数, 16GB+显存, 追求极致
        'pretrained': 'base',
        
        # 微调策略: 'full' | 'freeze_ffn' | 'head_only'
        # full:       全参数微调 (数据量>10k)
        # freeze_ffn: 冻结FFN层 (数据量1k-10k, 推荐默认)
        # head_only:  只训练输出头 (数据量<1k, 防过拟合)
        'pattern': 'freeze_ffn',
        
        # 输出目录命名: 自动生成为 moirai_{pretrained}_{pattern}
        # 例如: moirai_base_full, moirai_small_freeze_ffn
        # 设为 None 则自动生成，也可手动指定如 'my_custom_name'
        'model_name': None,
        
        # 超参数 - 比从零训练更保守!
        'epochs': 50,          # 微调3-20轮通常足够
        'lr': 5e-5,            # 学习率: 1e-6 ~ 1e-5 (比预训练低!)
        'batch_size': 32,      # 批大小
        'patience': 15,         # Early stopping
        'weight_decay': 0.01,  # 权重衰减
        'warmup_steps': 100,   # Warmup步数 (微调少一些)
    },
    
    # -------------------------------------------------------------------------
    # 从零训练配置 (mode='train' 时使用)
    # -------------------------------------------------------------------------
    'train': {
        # 模型命名 (用于evaluate)
        'model_name': 'moirai_small_scratch',
        
        # 模型架构
        'd_model': 384,        # 隐藏维度: 384(small) | 768(base) | 1024(large)
        'num_layers': 6,       # 层数: 6(small) | 12(base) | 24(large)
        
        # 超参数
        'epochs': 50,          # 训练轮数
        'lr': 1e-4,            # 学习率
        'batch_size': 32,      # 批大小
        'patience': 30,        # Early stopping
        'weight_decay': 0.1,   # 权重衰减
    },
    
    # -------------------------------------------------------------------------
    # 硬件设置
    # -------------------------------------------------------------------------
    'hardware': {
        'num_workers': 0,      # DataLoader进程数 (Windows建议0)
        'gpus': 1,             # GPU数量
    },
    
    # -------------------------------------------------------------------------
    # 恢复训练 (可选)
    # -------------------------------------------------------------------------
    'resume_from': None,       # checkpoint路径，如 'outputs/.../last.ckpt'
}

# =============================================================================
# 以下是代码实现，一般不需要修改
# =============================================================================

# Add src to path for uni2ts imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from uni2ts.data.dataset import TimeSeriesDataset, SampleTimeSeriesType
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.data.loader import DataLoader, PackCollate
from uni2ts.model.moirai import MoiraiPretrain, MoiraiModule, MoiraiFinetune
from uni2ts.distribution import (
    MixtureOutput,
    StudentTOutput,
    NormalFixedScaleOutput,
    NegativeBinomialOutput,
    LogNormalOutput,
)

# HuggingFace预训练模型映射
PRETRAINED_MODEL_MAP = {
    'small': 'Salesforce/moirai-1.0-R-small',
    'base': 'Salesforce/moirai-1.0-R-base', 
    'large': 'Salesforce/moirai-1.0-R-large',
}

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, Callback
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger


# =============================================================================
# Live Plotting Callback
# =============================================================================

class LivePlotCallback(Callback):
    """实时绘制训练曲线的Callback"""

    def __init__(self, output_dir: Path, update_interval: int = 5):
        super().__init__()
        self.output_dir = output_dir
        self.update_interval = update_interval
        self.train_losses = []
        self.val_losses = []
        self.epochs = []
        self.steps = []
        self.step_losses = []
        self.current_epoch_losses = []
        self.last_plot_time = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        """记录每个step的loss"""
        if outputs is not None and 'loss' in outputs:
            loss = outputs['loss'].item()
        elif hasattr(trainer, 'callback_metrics') and 'train/PackedNLLLoss' in trainer.callback_metrics:
            loss = trainer.callback_metrics['train/PackedNLLLoss'].item()
        else:
            return

        self.current_epoch_losses.append(loss)
        self.steps.append(trainer.global_step)
        self.step_losses.append(loss)

        # 定期更新图表
        current_time = time.time()
        if current_time - self.last_plot_time > self.update_interval:
            self._update_plot()
            self.last_plot_time = current_time

    def on_train_epoch_end(self, trainer, pl_module):
        """记录epoch级别的训练loss"""
        if self.current_epoch_losses:
            avg_loss = np.mean(self.current_epoch_losses)
            self.train_losses.append(avg_loss)
            self.epochs.append(trainer.current_epoch)
            self.current_epoch_losses = []

    def on_validation_epoch_end(self, trainer, pl_module):
        """记录验证loss"""
        if 'val/PackedNLLLoss' in trainer.callback_metrics:
            val_loss = trainer.callback_metrics['val/PackedNLLLoss'].item()
            while len(self.val_losses) < len(self.train_losses) - 1:
                self.val_losses.append(None)
            self.val_losses.append(val_loss)
            self._update_plot()
            self._print_status(trainer)

    def _print_status(self, trainer):
        """打印训练状态"""
        epoch = trainer.current_epoch
        train_loss = self.train_losses[-1] if self.train_losses else 0
        val_loss = self.val_losses[-1] if self.val_losses and self.val_losses[-1] is not None else 0

        # 检查是否收敛
        converged = False
        if len(self.val_losses) >= 10:
            recent = [v for v in self.val_losses[-10:] if v is not None]
            if len(recent) >= 5:
                improvement = (recent[0] - recent[-1]) / (recent[0] + 1e-8)
                if improvement < 0.01:
                    converged = True

        status = "CONVERGING" if converged else "TRAINING"
        print(f"\n[Epoch {epoch:3d}] Train: {train_loss:.4f} | Val: {val_loss:.4f} | {status}")

        if converged:
            print(f"  -> 模型接近收敛，可考虑停止训练")

    def _update_plot(self):
        """更新训练曲线图"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            # 图1: Step级别loss
            ax1 = axes[0]
            if len(self.step_losses) > 0:
                recent_steps = self.steps[-500:]
                recent_losses = self.step_losses[-500:]
                ax1.plot(recent_steps, recent_losses, 'b-', alpha=0.3, linewidth=0.5)
                if len(recent_losses) > 20:
                    window = min(50, len(recent_losses) // 4)
                    ma = np.convolve(recent_losses, np.ones(window)/window, mode='valid')
                    ma_steps = recent_steps[window-1:]
                    ax1.plot(ma_steps, ma, 'b-', linewidth=2, label=f'MA({window})')
                ax1.set_xlabel('Step')
                ax1.set_ylabel('Loss')
                ax1.set_title('Training Loss (Recent Steps)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

            # 图2: Epoch级别loss
            ax2 = axes[1]
            if len(self.epochs) > 0:
                ax2.plot(self.epochs, self.train_losses, 'b-o', label='Train', markersize=3)
                val_epochs = [e for e, v in zip(self.epochs, self.val_losses) if v is not None]
                val_vals = [v for v in self.val_losses if v is not None]
                if val_vals:
                    ax2.plot(val_epochs, val_vals, 'r-o', label='Validation', markersize=3)
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Training Progress')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                if len(val_vals) >= 2:
                    min_val = min(val_vals)
                    min_epoch = val_epochs[val_vals.index(min_val)]
                    ax2.annotate(f'Best: {min_val:.4f}',
                                xy=(min_epoch, min_val),
                                xytext=(min_epoch + 5, min_val + 0.1),
                                arrowprops=dict(arrowstyle='->', color='green'),
                                fontsize=9, color='green')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'training_progress.png', dpi=150)
            plt.close(fig)

            self._save_history()

        except Exception as e:
            print(f"Warning: Could not update plot: {e}")

    def _save_history(self):
        """保存训练历史到CSV"""
        history = {
            'epoch': self.epochs,
            'train_loss': self.train_losses,
            'val_loss': self.val_losses[:len(self.epochs)] if self.val_losses else [None] * len(self.epochs)
        }
        df = pd.DataFrame(history)
        df.to_csv(self.output_dir / 'training_history.csv', index=False)


# =============================================================================
# 数据集构建
# =============================================================================

def create_hf_dataset(samples: list, num_variates: int) -> datasets.Dataset:
    """转换为HuggingFace数据集格式"""

    def example_gen() -> Generator[dict, None, None]:
        for sample in samples:
            yield {
                'item_id': sample['item_id'],
                'start': sample['start'],
                'freq': sample['freq'],
                'target': sample['target'],
            }

    features = Features({
        'item_id': Value('string'),
        'start': Value('timestamp[s]'),
        'freq': Value('string'),
        'target': Sequence(Sequence(Value('float32')), length=num_variates),
    })

    return datasets.Dataset.from_generator(example_gen, features=features)


def build_dataset(data_dir: Path, output_dir: Path):
    """构建HuggingFace数据集"""
    from tqdm import tqdm

    print(f"Loading data from {data_dir}...")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    num_variates = metadata['num_variates']
    print(f"Number of variates: {num_variates}")

    for split in ['train', 'val', 'test']:
        jsonl_path = data_dir / 'jsonl' / f'{split}.jsonl'
        if not jsonl_path.exists():
            print(f"Skipping {split} (not found)")
            continue

        with open(jsonl_path, 'r') as f:
            total_lines = sum(1 for _ in f)

        samples = []
        with open(jsonl_path, 'r') as f:
            for line in tqdm(f, total=total_lines, desc=f"Loading {split}"):
                record = json.loads(line)
                target = np.array(record['target'], dtype=np.float32)
                samples.append({
                    'target': target,
                    'start': pd.Timestamp(record['start']),
                    'freq': record.get('freq', '1min'),
                    'item_id': record['item_id'],
                })

        print(f"Converting {split}: {len(samples)} samples")

        hf_dataset = create_hf_dataset(samples, num_variates)
        hf_dataset.info.dataset_name = f'buildingfm_{split}'

        save_path = output_dir / f'buildingfm_{split}'
        hf_dataset.save_to_disk(str(save_path))
        print(f"  Saved to {save_path}")

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Dataset build complete!")


# =============================================================================
# 模型创建
# =============================================================================

def create_model(
    num_variates: int,
    d_model: int = 384,
    num_layers: int = 6,
    lr: float = 1e-4,
    num_training_steps: int = 10000,
    num_warmup_steps: int = 1000,
) -> MoiraiPretrain:
    """创建MOIRAI预训练模型"""

    distr_output = MixtureOutput(
        components=[
            StudentTOutput(),
            NormalFixedScaleOutput(),
            NegativeBinomialOutput(),
            LogNormalOutput(),
        ]
    )

    model = MoiraiPretrain(
        module_kwargs={
            'distr_output': distr_output,
            'd_model': d_model,
            'num_layers': num_layers,
            'patch_sizes': (8, 16, 32, 64, 128),
            'max_seq_len': 512,
            'attn_dropout_p': 0.0,
            'dropout_p': 0.1,
            'scaling': True,
        },
        min_patches=2,
        min_mask_ratio=0.15,
        max_mask_ratio=0.5,
        max_dim=min(128, num_variates),
        lr=lr,
        weight_decay=0.1,
        beta1=0.9,
        beta2=0.98,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    return model


def save_baseline_model(model: MoiraiPretrain, output_dir: Path):
    """保存未训练的基线模型用于对比"""
    baseline_path = output_dir / 'baseline_untrained.pt'
    torch.save({
        'model_state_dict': model.module.state_dict(),
        'config': {
            'd_model': model.module.d_model,
            'num_layers': model.module.num_layers,
            'patch_sizes': model.module.patch_sizes,
            'max_seq_len': model.module.max_seq_len,
        }
    }, baseline_path)
    print(f"  Saved untrained baseline to: {baseline_path}")
    return baseline_path


# =============================================================================
# 训练函数
# =============================================================================

def train(
    data_dir: Path,
    output_dir: Path,
    model_name: str = 'moirai_small_scratch',
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-4,
    d_model: int = 384,
    num_layers: int = 6,
    num_workers: int = 4,
    gpus: int = 1,
    resume: Optional[str] = None,
    patience: int = 20,
):
    """训练BuildingFM模型"""

    # 设置GPU优化
    torch.set_float32_matmul_precision('high')

    # 加载元数据
    hf_data_dir = data_dir / 'hf'
    with open(hf_data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    num_variates = metadata['num_variates']

    print(f"\n{'='*60}")
    print(f"BuildingFM Training")
    print(f"{'='*60}")
    print(f"变量数: {num_variates}")

    # 加载数据集
    train_hf = datasets.load_from_disk(str(hf_data_dir / 'buildingfm_train'))
    val_hf = datasets.load_from_disk(str(hf_data_dir / 'buildingfm_val'))

    print(f"训练集: {len(train_hf)} samples")
    print(f"验证集: {len(val_hf)} samples")

    # 计算训练步数
    num_batches_per_epoch = max(10, len(train_hf) // batch_size)
    num_training_steps = epochs * num_batches_per_epoch
    num_warmup_steps = min(1000, num_training_steps // 10)

    print(f"\n训练配置:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches/epoch: {num_batches_per_epoch}")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Model: d_model={d_model}, layers={num_layers}")
    print(f"  Early stopping patience: {patience} epochs")

    # 创建模型
    model = create_model(
        num_variates=num_variates,
        d_model=d_model,
        num_layers=num_layers,
        lr=lr,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )

    # 创建输出目录 - 使用简洁命名，方便evaluate
    run_dir = output_dir / model_name
    if run_dir.exists():
        # 如果已存在，追加时间戳避免覆盖
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = output_dir / f'{model_name}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    # 保存未训练的基线模型
    print(f"\n保存基线模型 (未训练)...")
    save_baseline_model(model, run_dir)

    # 获取数据变换
    transform = model.train_transform_map['default']()

    # 创建数据集
    train_dataset = TimeSeriesDataset(
        HuggingFaceDatasetIndexer(train_hf),
        transform=transform,
        dataset_weight=1.0,
        sample_time_series=SampleTimeSeriesType.NONE,
    )

    val_dataset = TimeSeriesDataset(
        HuggingFaceDatasetIndexer(val_hf),
        transform=transform,
        dataset_weight=1.0,
        sample_time_series=SampleTimeSeriesType.NONE,
    )

    # 创建DataLoader - 优化GPU利用率
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        batch_size_factor=2.0,
        cycle=True,
        num_batches_per_epoch=num_batches_per_epoch,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=PackCollate(
            max_length=512,
            seq_fields=MoiraiPretrain.seq_fields,
            pad_func_map=MoiraiPretrain.pad_func_map,
        ),
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else 2,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        batch_size_factor=2.0,
        cycle=False,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=PackCollate(
            max_length=512,
            seq_fields=MoiraiPretrain.seq_fields,
            pad_func_map=MoiraiPretrain.pad_func_map,
        ),
        drop_last=False,
        pin_memory=True,
    )

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=run_dir / 'checkpoints',
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val/PackedNLLLoss',
            mode='min',
            save_top_k=3,
        ),
        ModelCheckpoint(
            dirpath=run_dir / 'checkpoints',
            filename='last',
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor='val/PackedNLLLoss',
            patience=patience,
            mode='min',
            verbose=True,
        ),
        # LivePlotCallback(output_dir=run_dir, update_interval=10),
    ]

    # Loggers
    tb_logger = TensorBoardLogger(save_dir=run_dir, name='tensorboard')
    csv_logger = CSVLogger(save_dir=run_dir, name='csv_logs')

    # 设备检测
    device_available = torch.cuda.is_available()
    accelerator = 'gpu' if device_available else 'cpu'
    devices_count = gpus if device_available else 1
    precision = 'bf16-mixed' if device_available else 32
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices_count,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        enable_progress_bar=True,
        precision=precision,
        accumulate_grad_batches=1,
        val_check_interval=1.0,
    )

    # 设备信息
    if device_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n设备: GPU - {gpu_name}")
        print(f"  显存: {gpu_mem:.1f} GB")
        print(f"  精度: bf16-mixed")
    else:
        print(f"\n设备: CPU")
        print(f"  精度: float32")
        print(f"  注意: CPU训练会比GPU慢很多，建议使用GPU")

    print(f"\n输出目录: {run_dir}")
    print(f"实时曲线: {run_dir / 'training_progress.png'}")
    print(f"\n{'='*60}")
    print("开始训练... (查看 training_progress.png 获取实时更新)")
    print(f"{'='*60}\n")

    # 训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume,
    )

    print(f"\n{'='*60}")
    print("训练完成!")
    print(f"{'='*60}")
    print(f"Checkpoints: {run_dir / 'checkpoints'}")
    print(f"训练曲线: {run_dir / 'training_progress.png'}")
    print(f"训练历史: {run_dir / 'training_history.csv'}")
    print(f"基线模型: {run_dir / 'baseline_untrained.pt'}")

    return run_dir


# =============================================================================
# 微调函数 (加载预训练权重)
# =============================================================================

def finetune(
    data_dir: Path,
    output_dir: Path,
    model_name: str = 'moirai_small',
    pretrained_model: str = 'small',
    finetune_pattern: str = 'full',
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 5e-6,
    weight_decay: float = 0.01,
    num_workers: int = 4,
    gpus: int = 1,
    resume: Optional[str] = None,
    patience: int = 5,
):
    """微调预训练MOIRAI模型"""
    
    # 设置GPU优化
    torch.set_float32_matmul_precision('high')
    
    # 解析预训练模型路径
    if pretrained_model in PRETRAINED_MODEL_MAP:
        model_path = PRETRAINED_MODEL_MAP[pretrained_model]
        model_size = pretrained_model
    else:
        model_path = pretrained_model
        model_size = 'custom'
    
    print(f"\n{'='*60}")
    print(f"BuildingFM Fine-tuning (微调预训练模型)")
    print(f"{'='*60}")
    print(f"预训练模型: {model_path}")
    print(f"微调策略: {finetune_pattern}")
    
    # 加载元数据
    hf_data_dir = data_dir / 'hf'
    with open(hf_data_dir / 'metadata.json') as f:
        metadata = json.load(f)
    
    num_variates = metadata['num_variates']
    print(f"变量数: {num_variates}")
    
    # 从HuggingFace加载预训练模型
    print(f"\n正在从HuggingFace加载预训练权重...")
    print(f"  -> {model_path}")
    pretrained_module = MoiraiModule.from_pretrained(model_path)
    
    d_model = pretrained_module.d_model
    num_layers = pretrained_module.num_layers
    print(f"  模型配置: d_model={d_model}, layers={num_layers}")
    print(f"  预训练权重加载成功!")
    
    # 加载数据集
    train_hf = datasets.load_from_disk(str(hf_data_dir / 'buildingfm_train'))
    val_hf = datasets.load_from_disk(str(hf_data_dir / 'buildingfm_val'))
    
    print(f"\n训练集: {len(train_hf)} samples")
    print(f"验证集: {len(val_hf)} samples")
    
    # 计算训练步数
    num_batches_per_epoch = max(10, len(train_hf) // batch_size)
    num_training_steps = epochs * num_batches_per_epoch
    num_warmup_steps = min(100, num_training_steps // 10)  # 微调warmup更少
    
    print(f"\n微调配置:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr:.2e}")
    print(f"  Finetune pattern: {finetune_pattern}")
    print(f"  Total steps: {num_training_steps}")
    print(f"  Warmup steps: {num_warmup_steps}")
    print(f"  Early stopping patience: {patience} epochs")
    
    # 创建模型 - 使用MoiraiPretrain + 预训练权重进行领域适配
    # 这样可以学习领域知识，同时支持多下游任务
    model = MoiraiPretrain(
        module=pretrained_module,  # 传入预训练权重!
        min_patches=2,
        min_mask_ratio=0.15,
        max_mask_ratio=0.5,
        max_dim=min(128, num_variates),
        lr=lr,
        weight_decay=weight_decay,
        beta1=0.9,
        beta2=0.98,
        num_training_steps=num_training_steps,
        num_warmup_steps=num_warmup_steps,
    )
    
    # 根据finetune_pattern冻结相应层
    if finetune_pattern == 'full':
        pass  # 全参数微调，不冻结任何层
    elif finetune_pattern == 'freeze_ffn':
        for name, param in model.named_parameters():
            if 'ffn' in name:
                param.requires_grad = False
    elif finetune_pattern == 'head_only':
        for name, param in model.named_parameters():
            if 'param_proj' not in name:
                param.requires_grad = False
    
    # 打印可训练参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数统计:")
    print(f"  总参数量: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # 创建输出目录 - 使用简洁命名，方便evaluate
    run_dir = output_dir / model_name
    if run_dir.exists():
        # 如果已存在，追加时间戳避免覆盖
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = output_dir / f'{model_name}_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存预训练模型作为 zero-shot 基线 (用于评估对比)
    print(f"\n保存预训练基线 (Zero-shot)...")
    save_baseline_model(model, run_dir)
    
    # 获取数据变换 - 使用MoiraiPretrain的transform (支持通用格式)
    transform = model.train_transform_map['default']()
    
    # 创建数据集
    train_dataset = TimeSeriesDataset(
        HuggingFaceDatasetIndexer(train_hf),
        transform=transform,
        dataset_weight=1.0,
        sample_time_series=SampleTimeSeriesType.NONE,
    )
    
    val_dataset = TimeSeriesDataset(
        HuggingFaceDatasetIndexer(val_hf),
        transform=transform,  # 验证集也用同样的transform
        dataset_weight=1.0,
        sample_time_series=SampleTimeSeriesType.NONE,
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        batch_size_factor=2.0,
        cycle=True,
        num_batches_per_epoch=num_batches_per_epoch,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=PackCollate(
            max_length=512,
            seq_fields=MoiraiPretrain.seq_fields,
            pad_func_map=MoiraiPretrain.pad_func_map,
        ),
        drop_last=True,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else 2,
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        batch_size_factor=2.0,
        cycle=False,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=PackCollate(
            max_length=512,
            seq_fields=MoiraiPretrain.seq_fields,
            pad_func_map=MoiraiPretrain.pad_func_map,
        ),
        drop_last=False,
        pin_memory=True,
    )
    
    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(
            dirpath=run_dir / 'checkpoints',
            filename='best-{epoch:02d}-{val_loss:.4f}',
            monitor='val/PackedNLLLoss',
            mode='min',
            save_top_k=3,
        ),
        ModelCheckpoint(
            dirpath=run_dir / 'checkpoints',
            filename='last',
            every_n_epochs=1,
        ),
        EarlyStopping(
            monitor='val/PackedNLLLoss',
            patience=patience,
            mode='min',
            verbose=True,
        ),
    ]
    
    # Loggers
    tb_logger = TensorBoardLogger(save_dir=run_dir, name='tensorboard')
    csv_logger = CSVLogger(save_dir=run_dir, name='csv_logs')
    
    # 设备检测
    device_available = torch.cuda.is_available()
    accelerator = 'gpu' if device_available else 'cpu'
    devices_count = gpus if device_available else 1
    precision = 'bf16-mixed' if device_available else 32
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        devices=devices_count,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        enable_progress_bar=True,
        precision=precision,
        accumulate_grad_batches=1,
        val_check_interval=1.0,
    )
    
    # 设备信息
    if device_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n设备: GPU - {gpu_name}")
        print(f"  显存: {gpu_mem:.1f} GB")
    else:
        print(f"\n设备: CPU")
    
    print(f"\n输出目录: {run_dir}")
    print(f"\n{'='*60}")
    print("开始微调...")
    print(f"{'='*60}\n")
    
    # 训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=resume,
    )
    
    print(f"\n{'='*60}")
    print("微调完成!")
    print(f"{'='*60}")
    print(f"Checkpoints: {run_dir / 'checkpoints'}")
    
    return run_dir


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == '__main__':
    
    # 解析配置
    mode = CONFIG['mode']
    data_dir = Path(CONFIG['data_dir'])
    output_dir = Path(CONFIG['output_dir'])
    hw = CONFIG['hardware']
    
    print(f"\n运行模式: {mode}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    if mode == 'build':
        build_dataset(data_dir, data_dir / 'hf')

    elif mode == 'train':
        cfg = CONFIG['train']
        train(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=cfg['model_name'],
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            lr=cfg['lr'],
            d_model=cfg['d_model'],
            num_layers=cfg['num_layers'],
            num_workers=hw['num_workers'],
            gpus=hw['gpus'],
            resume=CONFIG['resume_from'],
            patience=cfg['patience'],
        )
    
    elif mode == 'finetune':
        cfg = CONFIG['finetune']
        # 自动生成模型名: moirai_{size}_{pattern}
        # 例如: moirai_base_full, moirai_small_freeze_ffn
        model_name = cfg.get('model_name')
        if model_name is None:
            model_name = f"moirai_{cfg['pretrained']}_{cfg['pattern']}"
        
        finetune(
            data_dir=data_dir,
            output_dir=output_dir,
            model_name=model_name,
            pretrained_model=cfg['pretrained'],
            finetune_pattern=cfg['pattern'],
            epochs=cfg['epochs'],
            batch_size=cfg['batch_size'],
            lr=cfg['lr'],
            weight_decay=cfg['weight_decay'],
            num_workers=hw['num_workers'],
            gpus=hw['gpus'],
            resume=CONFIG['resume_from'],
            patience=cfg['patience'],
        )
    
    else:
        print(f"未知的运行模式: {mode}")
        print("请设置 CONFIG['mode'] = 'build' | 'train' | 'finetune'")
