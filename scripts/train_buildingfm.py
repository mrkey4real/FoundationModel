#!/usr/bin/env python
"""
BuildingFM Training Script - Spyder Friendly Version

直接修改下面的配置参数，然后在Spyder中运行整个脚本即可。

使用方法:
    1. 首次运行: 设置 RUN_MODE = 'build' 来构建数据集
    2. 训练: 设置 RUN_MODE = 'train' 来训练模型
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
# 配置参数 - 在这里直接修改！
# =============================================================================

# 运行模式: 'build' 构建数据集 (首次需要), 'train' 训练模型
RUN_MODE = 'train'

# 数据路径
DATA_DIR = '../data/buildingfm_processed_15min'
OUTPUT_DIR = '../outputs/buildingfm_15min'

# 训练参数
EPOCHS = 500          # 最大训练轮数 (early stopping可能提前停止)
BATCH_SIZE = 32       # 批大小 - RTX 5070 12GB 建议16-32
LEARNING_RATE = 1e-6  # 学习率
PATIENCE = 30         # Early stopping耐心值

# 模型参数
D_MODEL = 768         # 模型隐藏维度 (small=384, base=768, large=1024)
NUM_LAYERS = 6        # Transformer层数 (small=6, base=12, large=24)

# 数据加载
NUM_WORKERS = 0       # DataLoader工作进程数
GPUS = 0              # GPU数量

# 恢复训练 (设为checkpoint路径，如 'outputs/buildingfm/xxx/checkpoints/last.ckpt')
RESUME_FROM = None

# =============================================================================
# 以下是代码实现，一般不需要修改
# =============================================================================

# Add src to path for uni2ts imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from uni2ts.data.dataset import TimeSeriesDataset, SampleTimeSeriesType
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.data.loader import DataLoader, PackCollate
from uni2ts.model.moirai import MoiraiPretrain, MoiraiModule
from uni2ts.distribution import (
    MixtureOutput,
    StudentTOutput,
    NormalFixedScaleOutput,
    NegativeBinomialOutput,
    LogNormalOutput,
)

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

    # 创建输出目录
    run_dir = output_dir / datetime.now().strftime('%Y%m%d_%H%M%S')
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

    # Trainer
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=gpus,
        callbacks=callbacks,
        logger=[tb_logger, csv_logger],
        gradient_clip_val=1.0,
        log_every_n_steps=100,
        enable_progress_bar=True,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        accumulate_grad_batches=1,
        val_check_interval=1.0,
    )

    # GPU信息
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\nGPU: {gpu_name}")
        print(f"  显存: {gpu_mem:.1f} GB")
        print(f"  精度: bf16-mixed")

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
# 主程序入口
# =============================================================================

if __name__ == '__main__':

    data_dir = Path(DATA_DIR)
    output_dir = Path(OUTPUT_DIR)

    print(f"\n运行模式: {RUN_MODE}")
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")

    if RUN_MODE == 'build':
        # 构建数据集
        build_dataset(data_dir, data_dir / 'hf')

    elif RUN_MODE == 'train':
        # 训练模型
        train(
            data_dir=data_dir,
            output_dir=output_dir,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LEARNING_RATE,
            d_model=D_MODEL,
            num_layers=NUM_LAYERS,
            num_workers=NUM_WORKERS,
            gpus=GPUS,
            resume=RESUME_FROM,
            patience=PATIENCE,
        )
    else:
        print(f"未知的运行模式: {RUN_MODE}")
        print("请设置 RUN_MODE = 'build' 或 'train'")
