# MOIRAI 微调超参数调优计划

## 1. 问题诊断

根据你描述的现象和我从社区/论文收集的信息：

### 1.1 你遇到的问题
- **Small模型**: lr=1e-6时几乎不学习，lr=1e-4时训练震荡
- **Base模型**: 训练太慢，无法手动调参
- **Early Stopping**: epoch 1-2就触发，选择了epoch 0的模型

### 1.2 根本原因分析

**关键发现1**: 多篇研究论文（包括 [Multi-Scale Finetuning](https://arxiv.org/html/2506.14087)）明确指出：
> "Unlike pretraining which uses a learning rate of 1e-3, finetuning requires a **much smaller** learning rate. Based on validation performance, we select a learning rate of either **5e-6 or 5e-7** for finetuning."

**关键发现2**: [Less is More](https://arxiv.org/html/2505.23195v1) 论文指出：
> "We set the fine-tuning epoch to **1** for Time-MoE, Chronos, and Moirai; otherwise, these models after multi-epoch fine-tuning usually achieve lower validation MSE but **higher test MSE due to overfitting**."

**关键发现3**: 你的数据特点（100+ columns, 11个月15min数据）意味着每个epoch包含大量样本，**1-2个epoch可能已经足够**。

---

## 2. 超参数配置方案

### 2.1 核心参数对照表

| 参数 | Small (14M) | Base (91M) | 说明 |
|------|-------------|------------|------|
| **Learning Rate** | 5e-6 ~ 1e-5 | 1e-6 ~ 5e-6 | 模型越大lr越小 |
| **Batch Size** | 64 | 32 | 受限于12G显存 |
| **Epochs** | 3-10 | 1-5 | Foundation model易过拟合 |
| **Warmup Steps** | 50-100 | 100-200 | 占总steps的5-10% |
| **Weight Decay** | 0.1 | 0.1 | 保持与预训练一致 |
| **Patience** | 3-5 | 2-3 | 小epochs需要小patience |
| **Gradient Clip** | 1.0 | 1.0 | 防止梯度爆炸 |

### 2.2 不同微调策略的参数建议

#### Full (全参数微调)
```python
# Small-Full
'lr': 5e-6,           # 最保守
'epochs': 5,
'patience': 3,
'batch_size': 64,

# Base-Full  
'lr': 2e-6,           # 更保守
'epochs': 3,
'patience': 2,
'batch_size': 32,
```

#### Freeze FFN (冻结FFN层)
```python
# Small-Freeze
'lr': 1e-5,           # 可略高，因为冻结了部分参数
'epochs': 10,
'patience': 5,
'batch_size': 64,

# Base-Freeze
'lr': 5e-6,
'epochs': 5,
'patience': 3,
'batch_size': 32,
```

#### Head Only (只训练输出头)
```python
# Small-Head
'lr': 5e-5,           # 可以更高，只训练head
'epochs': 20,
'patience': 8,
'batch_size': 64,

# Base-Head
'lr': 2e-5,
'epochs': 15,
'patience': 6,
'batch_size': 32,
```

---

## 3. 自动化Grid Search脚本

### 3.1 创建实验配置文件

```python
# experiments_config.py
EXPERIMENTS = {
    # ========== Small 模型实验 ==========
    'small_full_lr5e6': {
        'pretrained': 'small',
        'pattern': 'full',
        'lr': 5e-6,
        'epochs': 5,
        'batch_size': 64,
        'patience': 3,
        'warmup_steps': 50,
    },
    'small_full_lr1e5': {
        'pretrained': 'small',
        'pattern': 'full',
        'lr': 1e-5,
        'epochs': 5,
        'batch_size': 64,
        'patience': 3,
        'warmup_steps': 50,
    },
    'small_freeze_lr1e5': {
        'pretrained': 'small',
        'pattern': 'freeze_ffn',
        'lr': 1e-5,
        'epochs': 10,
        'batch_size': 64,
        'patience': 5,
        'warmup_steps': 100,
    },
    'small_freeze_lr2e5': {
        'pretrained': 'small',
        'pattern': 'freeze_ffn',
        'lr': 2e-5,
        'epochs': 10,
        'batch_size': 64,
        'patience': 5,
        'warmup_steps': 100,
    },
    'small_head_lr5e5': {
        'pretrained': 'small',
        'pattern': 'head_only',
        'lr': 5e-5,
        'epochs': 20,
        'batch_size': 64,
        'patience': 8,
        'warmup_steps': 50,
    },
    'small_head_lr1e4': {
        'pretrained': 'small',
        'pattern': 'head_only',
        'lr': 1e-4,
        'epochs': 20,
        'batch_size': 64,
        'patience': 8,
        'warmup_steps': 50,
    },
    
    # ========== Base 模型实验 ==========
    'base_full_lr2e6': {
        'pretrained': 'base',
        'pattern': 'full',
        'lr': 2e-6,
        'epochs': 3,
        'batch_size': 32,
        'patience': 2,
        'warmup_steps': 100,
    },
    'base_full_lr5e6': {
        'pretrained': 'base',
        'pattern': 'full',
        'lr': 5e-6,
        'epochs': 3,
        'batch_size': 32,
        'patience': 2,
        'warmup_steps': 100,
    },
    'base_freeze_lr5e6': {
        'pretrained': 'base',
        'pattern': 'freeze_ffn',
        'lr': 5e-6,
        'epochs': 5,
        'batch_size': 32,
        'patience': 3,
        'warmup_steps': 150,
    },
    'base_freeze_lr1e5': {
        'pretrained': 'base',
        'pattern': 'freeze_ffn',
        'lr': 1e-5,
        'epochs': 5,
        'batch_size': 32,
        'patience': 3,
        'warmup_steps': 150,
    },
    'base_head_lr2e5': {
        'pretrained': 'base',
        'pattern': 'head_only',
        'lr': 2e-5,
        'epochs': 15,
        'batch_size': 32,
        'patience': 6,
        'warmup_steps': 100,
    },
    'base_head_lr5e5': {
        'pretrained': 'base',
        'pattern': 'head_only',
        'lr': 5e-5,
        'epochs': 15,
        'batch_size': 32,
        'patience': 6,
        'warmup_steps': 100,
    },
}

# 推荐的执行顺序（按训练速度排序）
EXECUTION_ORDER = [
    # 第一优先级：Small模型快速验证
    'small_head_lr5e5',      # ~10min, 验证head训练可行性
    'small_head_lr1e4',      # ~10min, 对比
    'small_freeze_lr1e5',    # ~15min
    'small_freeze_lr2e5',    # ~15min
    'small_full_lr5e6',      # ~20min
    'small_full_lr1e5',      # ~20min
    
    # 第二优先级：Base模型（基于Small最佳结果调整）
    'base_head_lr2e5',       # ~30min
    'base_head_lr5e5',       # ~30min
    'base_freeze_lr5e6',     # ~45min
    'base_freeze_lr1e5',     # ~45min
    'base_full_lr2e6',       # ~60min
    'base_full_lr5e6',       # ~60min
]
```

---

## 4. 修改建议：train_buildingfm.py

### 4.1 关键修改点

```python
# 修改 CONFIG['finetune'] 部分
'finetune': {
    'pretrained': 'small',          # 先从small开始
    'pattern': 'freeze_ffn',        # freeze_ffn是最稳定的起点
    
    # ===== 关键修改 =====
    'epochs': 10,                   # 减少epochs（原50太多）
    'lr': 1e-5,                     # 使用更小的学习率（原5e-5太大）
    'batch_size': 64,               # small可以用64
    'patience': 5,                  # 与epochs匹配的patience
    'weight_decay': 0.1,            # 保持与预训练一致（原0.01太小）
    'warmup_steps': 100,            # 适中的warmup
    
    # ===== 新增：学习率调度 =====
    'lr_scheduler': 'constant',     # 使用constant而非cosine（研究表明更稳定）
    'min_lr_ratio': 0.1,            # 如果用cosine，最小lr为初始lr的10%
}
```

### 4.2 建议添加的功能

```python
# 在 finetune() 函数中添加学习率调度选择
def get_lr_scheduler(optimizer, scheduler_type, num_training_steps, num_warmup_steps, min_lr_ratio=0.1):
    """
    创建学习率调度器
    
    scheduler_type: 'constant' | 'cosine' | 'linear'
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    if scheduler_type == 'constant':
        # Warmup then constant
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / num_warmup_steps
            return 1.0
    
    elif scheduler_type == 'cosine':
        # Warmup then cosine decay
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / num_warmup_steps
            progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    
    elif scheduler_type == 'linear':
        # Warmup then linear decay
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / num_warmup_steps
            progress = (step - num_warmup_steps) / (num_training_steps - num_warmup_steps)
            return max(min_lr_ratio, 1 - progress)
    
    return LambdaLR(optimizer, lr_lambda)
```

---

## 5. 自动化批量实验脚本

### 5.1 run_experiments.py

```python
#!/usr/bin/env python
"""
MOIRAI 批量微调实验脚本
自动运行所有配置并收集结果
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd

# 实验配置
EXPERIMENTS = {
    # ... (从上面的配置复制)
}

EXECUTION_ORDER = [
    # ... (从上面的顺序复制)
]

def modify_config_and_run(exp_name, exp_config, train_script_path):
    """修改配置文件并运行训练"""
    
    # 生成模型名称
    model_name = f"moirai_{exp_config['pretrained']}_{exp_config['pattern']}_{exp_name.split('_')[-1]}"
    
    # 创建临时配置
    config_str = f'''
CONFIG = {{
    'mode': 'finetune',
    'data_dir': '../data/buildingfm_processed_15min',
    'output_dir': '../outputs/buildingfm_15min',
    'finetune': {{
        'pretrained': '{exp_config["pretrained"]}',
        'pattern': '{exp_config["pattern"]}',
        'model_name': '{model_name}',
        'epochs': {exp_config["epochs"]},
        'lr': {exp_config["lr"]},
        'batch_size': {exp_config["batch_size"]},
        'patience': {exp_config["patience"]},
        'weight_decay': 0.1,
        'warmup_steps': {exp_config["warmup_steps"]},
    }},
    'hardware': {{
        'num_workers': 0,
        'gpus': 1,
    }},
    'resume_from': None,
}}
'''
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Model: {model_name}")
    print(f"Config: {exp_config}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # 运行训练
    result = subprocess.run(
        ['python', str(train_script_path)],
        capture_output=True,
        text=True,
        env={**os.environ, 'EXPERIMENT_CONFIG': json.dumps(exp_config)}
    )
    
    elapsed = time.time() - start_time
    
    return {
        'experiment': exp_name,
        'model_name': model_name,
        'elapsed_time': elapsed,
        'success': result.returncode == 0,
        'config': exp_config,
    }


def main():
    results = []
    train_script = Path(__file__).parent / 'train_buildingfm.py'
    
    for exp_name in EXECUTION_ORDER:
        if exp_name in EXPERIMENTS:
            result = modify_config_and_run(exp_name, EXPERIMENTS[exp_name], train_script)
            results.append(result)
            
            # 保存中间结果
            pd.DataFrame(results).to_csv('experiment_results.csv', index=False)
            
            print(f"\n[{exp_name}] Completed in {result['elapsed_time']/60:.1f} min")
            print(f"Success: {result['success']}")
    
    # 最终报告
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    df = pd.DataFrame(results)
    print(df.to_string())
    

if __name__ == '__main__':
    main()
```

---

## 6. 执行计划

### Phase 1: 快速验证 (约2小时)

```bash
# 先运行small模型的head_only，验证训练框架正常
python train_buildingfm.py  # 配置: small + head_only + lr=5e-5

# 预期：
# - 训练速度快（~10min/experiment）
# - 能看到loss下降
# - 验证early stopping工作正常
```

### Phase 2: Small模型完整实验 (约2小时)

```bash
# 按顺序运行所有small实验
# head_only → freeze_ffn → full
# 每种策略测试2个lr
```

### Phase 3: 分析Small结果，选择Base参数 (约30分钟)

```python
# analyze_small_results.py
import pandas as pd
import json
from pathlib import Path

def analyze_results():
    results = []
    output_dir = Path('../outputs/buildingfm_15min')
    
    for model_dir in output_dir.glob('moirai_small_*'):
        history_file = model_dir / 'training_history.csv'
        if history_file.exists():
            df = pd.read_csv(history_file)
            best_val_loss = df['val_loss'].min()
            best_epoch = df.loc[df['val_loss'].idxmin(), 'epoch']
            
            results.append({
                'model': model_dir.name,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'total_epochs': len(df),
            })
    
    results_df = pd.DataFrame(results)
    print(results_df.sort_values('best_val_loss'))
    return results_df

if __name__ == '__main__':
    analyze_results()
```

### Phase 4: Base模型实验 (约4小时)

```bash
# 基于Small最佳结果，运行Base实验
# 如果Small的freeze_ffn效果最好，Base优先测试freeze_ffn
```

### Phase 5: 评估对比 (约1小时)

```bash
python evaluate_models.py
```

---

## 7. 关键调参策略

### 7.1 如果遇到loss震荡

```python
# 降低学习率
'lr': current_lr / 2

# 增加warmup
'warmup_steps': current_warmup * 2

# 降低batch_size（但会增加训练时间）
'batch_size': current_batch_size // 2
```

### 7.2 如果loss不下降

```python
# 检查lr是否太小
# 如果lr < 1e-6，考虑略微增加
'lr': current_lr * 2

# 检查是否冻结了太多参数
# 尝试从head_only → freeze_ffn → full

# 增加epochs（但要同时增加patience）
'epochs': current_epochs + 5
'patience': current_patience + 2
```

### 7.3 如果早停过早

```python
# 增加patience
'patience': max(current_patience, epochs // 3)

# 使用更平滑的validation（多次验证取平均）
# 或者降低val_check_interval
```

---

## 8. 预期结果

基于社区经验和论文数据：

| 模型 | 策略 | 预期相对zero-shot提升 |
|------|------|----------------------|
| Small | head_only | 5-10% |
| Small | freeze_ffn | 10-20% |
| Small | full | 15-25% (有过拟合风险) |
| Base | head_only | 8-15% |
| Base | freeze_ffn | 15-25% |
| Base | full | 20-30% (有过拟合风险) |

**注意**: 如果你的数据与LOTSA预训练数据差异很大（HVAC数据可能是新领域），微调收益可能更大，但也更容易过拟合。

---

## 9. 参考来源

1. [Multi-Scale Finetuning for Encoder-based TSFMs](https://arxiv.org/html/2506.14087) - 推荐lr=5e-6或5e-7
2. [Less is More: Structured Pruning](https://arxiv.org/html/2505.23195v1) - MOIRAI微调1 epoch
3. [uni2ts GitHub Issues #188](https://github.com/SalesforceAIResearch/uni2ts/issues/188) - 社区微调问题讨论
4. [LoRA for TSFMs](https://arxiv.org/html/2405.10216v1) - 微调学习率范围1e-3到5e-5
5. [Moirai-MoE Paper](https://arxiv.org/html/2410.10469v1) - 预训练lr=1e-3, weight_decay=1e-1

---

## 10. 下一步行动

1. **立即修改** `CONFIG['finetune']` 中的参数，使用上述推荐值
2. **先跑small + head_only**，确认训练正常工作
3. **运行批量实验**，收集结果
4. **分析最佳配置**，应用到base模型
5. **运行evaluate_models.py** 对比所有模型
