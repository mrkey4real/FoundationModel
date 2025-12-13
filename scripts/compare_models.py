#!/usr/bin/env python
"""比较零样本基线和微调模型的权重和性能"""

import sys
sys.path.insert(0, 'E:/MOIRAI/src')

import torch
import numpy as np
import datasets
from pathlib import Path
from uni2ts.model.moirai import MoiraiModule

def main():
    # 配置
    data_dir = Path('E:/MOIRAI/data/buildingfm_processed_15min')
    model_dir = Path('E:/MOIRAI/outputs/buildingfm_15min/moirai_small_head_only_1e4')

    # 加载零样本基线
    print('加载零样本基线 (HuggingFace预训练)...')
    zeroshot_module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-small')

    # 加载微调后的模型
    print('加载微调模型...')
    ckpt_files = list(model_dir.glob('checkpoints/best-*.ckpt'))
    if not ckpt_files:
        print("没有找到checkpoint文件!")
        return
    ckpt_path = ckpt_files[-1]
    print(f'  Checkpoint: {ckpt_path.name}')

    # 使用weights_only=False加载
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    # 创建新模型并加载权重
    finetuned_module = MoiraiModule.from_pretrained('Salesforce/moirai-1.0-R-small')
    state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('module.')}
    finetuned_module.load_state_dict(state_dict)

    print(f'模型参数量: {sum(p.numel() for p in finetuned_module.parameters()):,}')

    # 检查权重变化
    print('\n' + '=' * 60)
    print('权重变化分析')
    print('=' * 60)

    zs_state = zeroshot_module.state_dict()
    ft_state = finetuned_module.state_dict()

    weight_changes = {}
    unchanged_layers = []

    for key in zs_state.keys():
        if key in ft_state:
            diff = (ft_state[key] - zs_state[key]).abs()
            mean_diff = diff.mean().item()
            max_diff = diff.max().item()
            if mean_diff > 1e-8:
                weight_changes[key] = {'mean': mean_diff, 'max': max_diff, 'numel': diff.numel()}
            else:
                unchanged_layers.append(key)

    print(f'\n有变化的参数层数: {len(weight_changes)} / {len(zs_state)}')
    print(f'未变化的参数层数: {len(unchanged_layers)}')

    # 按模块分组
    print('\n按模块统计变化:')
    module_stats = {}
    for name, stats in weight_changes.items():
        # 提取模块名
        if 'encoder.layers' in name:
            layer_idx = name.split('encoder.layers.')[1].split('.')[0]
            module = f'encoder.layer_{layer_idx}'
        elif 'in_proj' in name:
            module = 'in_proj (input projection)'
        elif 'out_proj' in name:
            module = 'out_proj (output projection)'
        elif 'param_proj' in name:
            module = 'param_proj (distribution params)'
        elif 'mask_encoding' in name:
            module = 'mask_encoding'
        else:
            module = name.split('.')[0]

        if module not in module_stats:
            module_stats[module] = {'count': 0, 'total_diff': 0, 'params': 0, 'layers': []}
        module_stats[module]['count'] += 1
        module_stats[module]['total_diff'] += stats['mean'] * stats['numel']
        module_stats[module]['params'] += stats['numel']
        module_stats[module]['layers'].append(name)

    for module, stats in sorted(module_stats.items(), key=lambda x: x[1]['total_diff'], reverse=True):
        avg_diff = stats['total_diff'] / stats['params'] if stats['params'] > 0 else 0
        print(f'  {module}:')
        print(f'    layers: {stats["count"]}, params: {stats["params"]:,}, avg_diff: {avg_diff:.6f}')

    # 显示变化最大的层
    print('\n变化最大的10层:')
    sorted_changes = sorted(weight_changes.items(), key=lambda x: x[1]['mean'], reverse=True)[:10]
    for name, stats in sorted_changes:
        print(f'  {name}')
        print(f'    mean_diff={stats["mean"]:.6f}, max_diff={stats["max"]:.6f}')

    # 检查head_only模式是否正确（应该只有output层变化）
    print('\n' + '=' * 60)
    print('Finetune Pattern检查 (head_only)')
    print('=' * 60)

    encoder_changed = any('encoder' in name for name in weight_changes.keys())
    proj_changed = any('proj' in name for name in weight_changes.keys())

    print(f'Encoder层是否变化: {encoder_changed}')
    print(f'Projection层是否变化: {proj_changed}')

    if encoder_changed:
        print('\n警告: head_only模式下encoder不应该变化!')
        encoder_layers = [name for name in weight_changes.keys() if 'encoder' in name]
        print(f'变化的encoder层: {encoder_layers[:5]}...')


if __name__ == '__main__':
    main()
