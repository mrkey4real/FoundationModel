#!/usr/bin/env python
"""
train_buildingfm.py 的CONFIG修改建议

将下面的CONFIG复制粘贴到你的 train_buildingfm.py 中替换原有的CONFIG

关键修改点:
1. lr: 5e-5 -> 1e-5 (降低10倍)
2. epochs: 50 -> 10 (减少5倍)  
3. patience: 15 -> 5 (与epochs匹配)
4. weight_decay: 0.01 -> 0.1 (与预训练一致)
5. warmup_steps: 100 -> 根据数据量调整

基于论文研究:
- Multi-Scale Finetuning (arxiv.org/html/2506.14087): lr=5e-6 ~ 5e-7
- Less is More (arxiv.org/html/2505.23195v1): MOIRAI微调建议1 epoch
"""

# =============================================================================
# 推荐配置 - Small模型起步
# =============================================================================

CONFIG_SMALL_FREEZE = {
    'mode': 'finetune',
    'data_dir': '../data/buildingfm_processed_15min',
    'output_dir': '../outputs/buildingfm_15min',
    
    'finetune': {
        # 模型选择
        'pretrained': 'small',          # 先用small快速验证
        'pattern': 'freeze_ffn',        # freeze_ffn是最稳定的选择
        'model_name': None,             # 自动生成: moirai_small_freeze_ffn
        
        # ===== 关键超参数 =====
        'epochs': 10,                   # 减少! Foundation model容易过拟合
        'lr': 1e-5,                     # 降低! 比预训练低100-1000倍
        'batch_size': 64,               # small可以用64
        'patience': 5,                  # 与epochs匹配
        'weight_decay': 0.1,            # 与预训练保持一致!
        'warmup_steps': 100,            # 约占总steps的5-10%
    },
    
    'hardware': {
        'num_workers': 0,               # Windows建议0
        'gpus': 1,
    },
    
    'resume_from': None,
}


# =============================================================================
# 推荐配置 - Base模型
# =============================================================================

CONFIG_BASE_FREEZE = {
    'mode': 'finetune',
    'data_dir': '../data/buildingfm_processed_15min',
    'output_dir': '../outputs/buildingfm_15min',
    
    'finetune': {
        # 模型选择
        'pretrained': 'base',           # 91M参数
        'pattern': 'freeze_ffn',        # freeze_ffn推荐
        'model_name': None,
        
        # ===== 关键超参数 =====
        'epochs': 5,                    # base模型更少epoch
        'lr': 5e-6,                     # base模型lr更小!
        'batch_size': 32,               # 受限于12G显存
        'patience': 3,                  # 与epochs匹配
        'weight_decay': 0.1,
        'warmup_steps': 150,            # base需要更多warmup
    },
    
    'hardware': {
        'num_workers': 0,
        'gpus': 1,
    },
    
    'resume_from': None,
}


# =============================================================================
# 所有实验配置 - 按推荐顺序排列
# =============================================================================

ALL_CONFIGS = {
    # ========== Small 模型 (先跑这些) ==========
    
    # 1. Head Only - 最快，验证训练正常
    'small_head_fast': {
        'pretrained': 'small',
        'pattern': 'head_only',
        'epochs': 20,
        'lr': 5e-5,                     # head可以用较高lr
        'batch_size': 64,
        'patience': 8,
        'warmup_steps': 50,
        'weight_decay': 0.1,
    },
    
    # 2. Freeze FFN - 推荐起点
    'small_freeze_conservative': {
        'pretrained': 'small',
        'pattern': 'freeze_ffn',
        'epochs': 10,
        'lr': 1e-5,                     # 保守起步
        'batch_size': 64,
        'patience': 5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    
    # 3. Full - 全参数，有过拟合风险
    'small_full_conservative': {
        'pretrained': 'small',
        'pattern': 'full',
        'epochs': 5,                    # full需要更少epoch
        'lr': 5e-6,                     # full需要更小lr
        'batch_size': 64,
        'patience': 3,
        'warmup_steps': 50,
        'weight_decay': 0.1,
    },
    
    # ========== Base 模型 (基于Small结果调整) ==========
    
    # 4. Head Only
    'base_head': {
        'pretrained': 'base',
        'pattern': 'head_only',
        'epochs': 15,
        'lr': 2e-5,
        'batch_size': 32,
        'patience': 6,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
    
    # 5. Freeze FFN - Base推荐
    'base_freeze_conservative': {
        'pretrained': 'base',
        'pattern': 'freeze_ffn',
        'epochs': 5,
        'lr': 5e-6,                     # base需要更小lr
        'batch_size': 32,
        'patience': 3,
        'warmup_steps': 150,
        'weight_decay': 0.1,
    },
    
    # 6. Full
    'base_full_conservative': {
        'pretrained': 'base',
        'pattern': 'full',
        'epochs': 3,                    # base full只需要很少epoch
        'lr': 2e-6,                     # 非常保守的lr
        'batch_size': 32,
        'patience': 2,
        'warmup_steps': 100,
        'weight_decay': 0.1,
    },
}


# =============================================================================
# 快速测试配置 - 用于验证训练正常工作
# =============================================================================

CONFIG_QUICK_TEST = {
    'mode': 'finetune',
    'data_dir': '../data/buildingfm_processed_15min',
    'output_dir': '../outputs/buildingfm_15min',
    
    'finetune': {
        'pretrained': 'small',
        'pattern': 'head_only',         # 最快的训练方式
        'model_name': 'quick_test',
        
        'epochs': 3,                    # 只跑3个epoch测试
        'lr': 5e-5,
        'batch_size': 64,
        'patience': 10,                 # 大patience确保跑完
        'weight_decay': 0.1,
        'warmup_steps': 20,
    },
    
    'hardware': {
        'num_workers': 0,
        'gpus': 1,
    },
    
    'resume_from': None,
}


# =============================================================================
# 如何使用这些配置
# =============================================================================
"""
使用方法:

1. 快速测试 (验证训练正常):
   将 CONFIG_QUICK_TEST 复制到 train_buildingfm.py 替换 CONFIG
   运行训练，预计5-10分钟完成
   检查是否正常下降

2. Small模型实验:
   依次使用 ALL_CONFIGS 中的 small_* 配置
   每次修改 CONFIG['finetune'] 对应的参数
   或者直接使用 CONFIG_SMALL_FREEZE

3. Base模型实验:
   在Small实验完成后，使用 CONFIG_BASE_FREEZE
   根据Small的最佳结果调整参数

4. 批量实验:
   使用 run_finetune_experiments.py 自动化运行所有配置
"""


# =============================================================================
# 超参数调整指南
# =============================================================================
"""
遇到问题时的调整策略:

1. Loss震荡/不稳定:
   - lr 减半: 5e-6 -> 2.5e-6
   - warmup_steps 翻倍: 100 -> 200
   - batch_size 减半 (会增加训练时间)

2. Loss几乎不下降:
   - 检查lr是否太小，可以略微增加
   - 检查是否冻结了太多参数
   - 从 head_only -> freeze_ffn -> full 逐步放开

3. Early stopping过早触发:
   - 增加 patience: 3 -> 5
   - 或者干脆设置 patience = epochs (不使用early stopping)
   - 使用更平滑的validation

4. 过拟合 (train loss下降但val loss上升):
   - 减少 epochs
   - 增加 weight_decay: 0.1 -> 0.2
   - 使用 freeze_ffn 或 head_only 而不是 full
   - 减小 lr

5. 训练太慢:
   - 增加 batch_size (如果显存允许)
   - 使用 head_only 而不是 full
   - 减少 epochs
"""


if __name__ == '__main__':
    # 打印推荐配置
    import json
    
    print("=" * 60)
    print("推荐的Small起步配置:")
    print("=" * 60)
    print(json.dumps(CONFIG_SMALL_FREEZE, indent=2, default=str))
    
    print("\n" + "=" * 60)
    print("推荐的Base配置:")
    print("=" * 60)
    print(json.dumps(CONFIG_BASE_FREEZE, indent=2, default=str))
