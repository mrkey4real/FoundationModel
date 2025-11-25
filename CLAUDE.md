# BuildingFM 技术实现指南 (v2)

## 项目目标

基于 Salesforce uni2ts/MOIRAI 架构，构建 HVAC 领域物理感知基座模型。

**核心假设**: MOIRAI 的 Masked Modeling + Any-variate Attention 机制天然适合我们的需求，不需要修改模型架构，只需要正确准备数据和配置训练。

---

## 1. 为什么选 MOIRAI

| 需求 | MOIRAI 如何满足 |
|------|----------------|
| 传感器配置不一致 | Any-variate Attention：变量数量可变，通过 Variate ID Embedding 区分 |
| 跨系统迁移 | Flatten + Patching：50个传感器和5个传感器在架构上无区别，仅序列长度不同 |
| 学习物理因果 | Masked Modeling：随机遮盖任意变量，迫使模型学习联合分布 P(V1,V2,...,Vn) |
| 处理真实数据缺失 | NaN 自动处理：DataLoader 将 NaN 识别为 mask 区域 |

---

## 2. 关键技术决策

### 2.1 All-in-Target 策略 (最重要)

**传统做法** (错误):
```python
# ❌ 把天气放 feat_dynamic_real
dataset = PandasDataset(
    target=['power', 'zone_temp'],
    feat_dynamic_real=['outdoor_temp', 'setpoint']  # 不参与 mask 训练
)
```

**我们的做法**:
```python
# ✅ 所有变量都放 target
dataset = PandasDataset(
    target=['power', 'zone_temp', 'outdoor_temp', 'setpoint', ...],  # 全部
    feat_dynamic_real=[]  # 留空
)
```

**原因**: 
- 只有 `target` 中的变量才会被 Tokenize 并拥有 Variate ID Embedding
- 只有 `target` 中的变量才会参与 Random Masking
- 这是实现"根据能耗反推天气"（双向物理推理）的唯一途径

**⚠️ 实现细节**: 当使用 `PandasDataset` 时，`target` 参数接受列名列表。列的顺序决定了 variate index（从0开始）。务必确保所有数据集中相同物理量的列顺序一致，否则 variate ID embedding 会混乱。

### 2.2 Super Schema (全集变量图谱)

定义一个 YAML/JSON 文件，锁定所有可能变量的 ID：

```yaml
# config/hvac_schema.yaml
# ID 映射必须全局一致，所有数据源遵守此规范

schema_version: "1.0"
frequency: "15T"  # 基准频率，pandas offset string

variables:
  # === 环境边界条件 ===
  0: 
    name: outdoor_drybulb_temp
    unit: "°C"
    source: ["energyplus", "weather_api", "sensor"]
    physical_range: [-40, 50]
  1: 
    name: outdoor_relative_humidity
    unit: "%"
    source: ["energyplus", "weather_api", "sensor"]
    physical_range: [0, 100]
  2:
    name: outdoor_wetbulb_temp
    unit: "°C"
    source: ["energyplus"]  # 通常真实系统不测量
    physical_range: [-40, 40]
    
  # === 区域状态 ===
  3:
    name: zone_mean_air_temp
    unit: "°C"
    source: ["energyplus", "thermostat", "sensor"]
    physical_range: [10, 40]
  4:
    name: zone_relative_humidity
    unit: "%"
    source: ["energyplus", "sensor"]
    physical_range: [0, 100]
    
  # === 控制设定点 ===
  5:
    name: cooling_setpoint
    unit: "°C"
    source: ["energyplus", "thermostat", "bas"]
    physical_range: [18, 30]
  6:
    name: heating_setpoint
    unit: "°C"
    source: ["energyplus", "thermostat", "bas"]
    physical_range: [15, 25]
    
  # === 设备运行状态 ===
  10:
    name: hvac_cooling_rate
    unit: "W"
    source: ["energyplus"]
    physical_range: [0, null]  # 无上限
  11:
    name: hvac_heating_rate
    unit: "W"
    source: ["energyplus"]
    physical_range: [0, null]
    
  # === 电力测量 ===
  20:
    name: total_hvac_power
    unit: "W"
    source: ["energyplus", "power_meter", "egauge"]
    physical_range: [0, null]
  21:
    name: cooling_electric_power
    unit: "W"
    source: ["energyplus", "submeter"]
    physical_range: [0, null]
    
  # ... 继续定义到 ID 49

# 系统类型编码 (用于 feat_static_cat)
system_types:
  0: "residential_split_ac"
  1: "residential_heat_pump"
  2: "commercial_chiller"
  3: "commercial_vav"
  4: "commercial_rtu"

# 气候区编码
climate_zones:
  0: "1A"  # Very Hot Humid
  1: "2A"  # Hot Humid
  2: "3A"  # Warm Humid
  # ... ASHRAE climate zones
```

**⚠️ 关键约束**:
- ID 一旦定义不可更改，后续只能追加新 ID
- EnergyPlus 仿真数据可能有 ID 0-49 全部
- 真实住宅数据可能只有 ID 0,3,5,20（稀疏子集）
- 模型通过 Variate ID Embedding 理解"ID=20 是总功率"这一语义
- `physical_range` 用于数据清洗，超出范围的值应标记为 NaN 而非截断

### 2.3 feat_static_cat 的正确使用

```python
# feat_static_cat 是静态分类特征，每条时间序列一个固定值
# 用于辅助 MoE 路由（如果用 MOIRAI-MoE）

# 正确做法：每条时间序列有一个类别向量
{
    'target': np.array(...),  # (num_variates, time_steps)
    'start': pd.Timestamp(...),
    'feat_static_cat': np.array([2, 1, 3]),  # [system_type, building_type, climate_zone]
}
```

**⚠️ 注意事项**:
- `feat_static_cat` 的维度必须在所有样本中一致
- 如果某些数据缺少类别信息，使用一个专门的 "unknown" 类别（如 -1 或最大ID+1）
- 这些特征不参与时序建模，仅作为辅助信息

---

## 3. 数据准备流程

### 3.1 理解 MOIRAI 的数据格式要求

MOIRAI 使用 GluonTS 的数据格式，但通过 uni2ts 进行了封装。核心数据结构：

```python
# 每条时间序列是一个 dict
sample = {
    'start': pd.Timestamp('2023-01-01 00:00:00'),  # 必须是 pd.Timestamp
    'target': np.array([[...], [...], ...]),       # shape: (num_variates, time_steps)
    'item_id': 'building_001_2023',                # 唯一标识字符串
    'freq': '15T',                                  # pandas offset string
    'feat_static_cat': np.array([2, 1, 3]),        # 可选，静态分类特征
}
```

**⚠️ target 的 shape 陷阱**:
- 必须是 `(num_variates, time_steps)`，不是 `(time_steps, num_variates)`
- 如果只有一个变量，必须是 `(1, time_steps)`，不能是 `(time_steps,)`
- NaN 表示该时间点该变量缺失，会被自动 mask

### 3.2 EnergyPlus 数据处理

#### 步骤1: 导出 EnergyPlus 变量

在 IDF 文件中配置输出变量：
```
Output:Variable,*,Site Outdoor Air Drybulb Temperature,Timestep;
Output:Variable,*,Zone Mean Air Temperature,Timestep;
Output:Variable,*,Zone Thermostat Cooling Setpoint Temperature,Timestep;
Output:Variable,*,Cooling Coil Total Cooling Rate,Timestep;
Output:Variable,*,Cooling Coil Electric Power,Timestep;
...
```

**⚠️ EnergyPlus 输出问题**:
- 变量名可能因 EnergyPlus 版本略有不同
- 需要先跑一次仿真，检查实际输出的 CSV 列名
- 某些变量只在特定条件下有值（如制冷功率在冬天可能全为0）

#### 步骤2: 列名映射脚本

```python
# scripts/map_energyplus_to_schema.py

import pandas as pd
import yaml

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        schema = yaml.safe_load(f)
    # 构建 name -> id 的映射
    return {v['name']: k for k, v in schema['variables'].items()}

def map_energyplus_csv(csv_path, schema_path, output_path):
    schema_map = load_schema(schema_path)
    df = pd.read_csv(csv_path)
    
    # EnergyPlus 列名到 schema name 的映射
    # 这部分需要根据实际 E+ 输出手动定义
    ep_to_schema = {
        'Environment:Site Outdoor Air Drybulb Temperature [C](TimeStep)': 'outdoor_drybulb_temp',
        'ZONE1:Zone Mean Air Temperature [C](TimeStep)': 'zone_mean_air_temp',
        # ... 完整映射
    }
    
    # 重命名列并只保留 schema 中定义的变量
    result_cols = {}
    for ep_col, schema_name in ep_to_schema.items():
        if ep_col in df.columns and schema_name in schema_map:
            var_id = schema_map[schema_name]
            result_cols[ep_col] = f"var_{var_id}"
    
    df_mapped = df.rename(columns=result_cols)[list(result_cols.values())]
    
    # 处理时间索引
    # E+ 的 Date/Time 列格式需要特殊处理
    df_mapped.index = parse_energyplus_datetime(df['Date/Time'])
    
    df_mapped.to_parquet(output_path)
    return df_mapped
```

**⚠️ EnergyPlus 时间戳问题**:
- E+ 的 `Date/Time` 列格式是 ` MM/DD  HH:MM:SS`，没有年份
- 需要根据仿真配置推断年份
- 注意 E+ 使用的是时段结束时间，而非开始时间

#### 步骤3: 重采样对齐

```python
def resample_to_target_freq(df, target_freq='15T'):
    """
    将数据重采样到目标频率
    
    ⚠️ 关键决策：
    - 状态量（温度、湿度）: 使用 mean 或 last
    - 瞬时功率: 使用 mean
    - 累积能耗: 使用 sum
    - 设定点: 使用 last（阶跃变化）
    - 占用状态: 使用 max（只要有人就算占用）
    """
    agg_rules = {
        'var_0': 'mean',   # outdoor_temp: 平均
        'var_3': 'mean',   # zone_temp: 平均
        'var_5': 'last',   # cooling_setpoint: 取最后值
        'var_20': 'mean',  # power: 平均
        # ...
    }
    
    # 检查所有列都有聚合规则
    for col in df.columns:
        if col not in agg_rules:
            raise ValueError(f"Column {col} missing aggregation rule")
    
    df_resampled = df.resample(target_freq).agg(agg_rules)
    return df_resampled
```

**⚠️ 重采样陷阱**:
- 不同变量需要不同的聚合方式
- 确保时间戳对齐后没有重复索引
- 检查重采样后是否产生了意外的 NaN（原数据空隙导致）

### 3.3 转换为 uni2ts Arrow 格式

uni2ts 使用 Apache Arrow 格式存储数据，提供高效的 I/O。

```python
# scripts/build_arrow_dataset.py

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import numpy as np
import pandas as pd

def build_arrow_dataset(
    processed_dir: Path,
    output_dir: Path,
    schema_path: Path,
    freq: str = '15T'
):
    """
    将处理后的 parquet 文件转换为 uni2ts Arrow 格式
    
    输出结构:
    output_dir/
    ├── train/
    │   ├── data-00000-of-00010.arrow
    │   ├── data-00001-of-00010.arrow
    │   └── ...
    └── metadata.json
    """
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    
    num_variates = len(schema['variables'])
    records = []
    
    for parquet_file in processed_dir.glob('*.parquet'):
        df = pd.read_parquet(parquet_file)
        
        # 构建 target array: (num_variates, time_steps)
        target = np.full((num_variates, len(df)), np.nan, dtype=np.float32)
        
        for col in df.columns:
            if col.startswith('var_'):
                var_id = int(col.split('_')[1])
                target[var_id, :] = df[col].values
        
        # 提取元数据
        item_id = parquet_file.stem
        start_time = df.index[0]
        
        # 从文件名或配置推断系统类型
        system_type = infer_system_type(item_id)
        climate_zone = infer_climate_zone(item_id)
        
        record = {
            'item_id': item_id,
            'start': start_time.isoformat(),
            'freq': freq,
            'target': target.tolist(),  # Arrow 需要 list
            'feat_static_cat': [system_type, climate_zone],
        }
        records.append(record)
    
    # 写入 Arrow 文件
    # 参考 uni2ts/data/builder/simple.py 的实现
    # ...
```

**⚠️ Arrow 格式细节**:
- `target` 在 Arrow 中存储为嵌套 list，不是 numpy array
- 需要保存 `freq` 字段让 MOIRAI 知道数据频率
- 大数据集应分片存储（每个 Arrow 文件不超过 1GB）

### 3.4 真实数据处理

真实数据与仿真数据的关键区别：

| 方面 | 仿真数据 | 真实数据 |
|------|----------|----------|
| 变量完整性 | 所有 50 个变量都有 | 可能只有 5-10 个 |
| 数据连续性 | 完美连续 | 存在空缺、异常 |
| 采样频率 | 可控制一致 | 可能不规则 |
| 物理一致性 | 完美遵守物理定律 | 存在噪声和偏差 |

```python
def process_real_data(raw_df, schema_path, target_freq='15T'):
    """
    处理真实建筑数据
    
    ⚠️ 关键原则：
    1. 不做插值填充 - 保留 NaN 让模型自己处理
    2. 只做物理范围清洗 - 超出范围的值设为 NaN
    3. 严格遵守 Schema ID - 相同物理量必须映射到相同 ID
    """
    with open(schema_path) as f:
        schema = yaml.safe_load(f)
    
    # 假设 raw_df 有原始列名，需要映射
    # 这部分需要根据实际数据源手动定义
    real_to_schema = {
        'outdoor_temp_sensor': 'outdoor_drybulb_temp',
        'room_temp': 'zone_mean_air_temp',
        'ac_power': 'total_hvac_power',
        # ...
    }
    
    num_variates = len(schema['variables'])
    target = np.full((num_variates, len(raw_df)), np.nan, dtype=np.float32)
    
    for real_col, schema_name in real_to_schema.items():
        if real_col not in raw_df.columns:
            continue  # 该传感器不存在，保持 NaN
        
        # 找到对应的 variable ID
        var_id = None
        for vid, vinfo in schema['variables'].items():
            if vinfo['name'] == schema_name:
                var_id = vid
                break
        
        if var_id is None:
            raise ValueError(f"Schema name '{schema_name}' not found in schema")
        
        values = raw_df[real_col].values.copy()
        
        # 物理范围清洗
        vinfo = schema['variables'][var_id]
        if 'physical_range' in vinfo:
            low, high = vinfo['physical_range']
            if low is not None:
                values[values < low] = np.nan
            if high is not None:
                values[values > high] = np.nan
        
        target[var_id, :] = values
    
    return target
```

**⚠️ 真实数据常见问题**:
- 传感器漂移：温度逐渐偏离真实值
- 时钟偏移：不同传感器的时间戳不同步
- 通信中断：连续多个时间点缺失
- 重复数据：同一时间戳多条记录

处理建议：
- 传感器漂移：不在预处理阶段处理，让模型学习鲁棒性
- 时钟偏移：重采样到统一时间网格时自然处理
- 通信中断：保留为 NaN，不插值
- 重复数据：保留最后一条或平均值

---

## 4. 训练配置

### 4.1 理解 MOIRAI 的训练机制

MOIRAI 的预训练有几个核心特点（来自论文）：

1. **Masked Encoder**: 使用完整的自注意力（不是 causal mask），通过 [mask] token 替换预测目标
2. **Multi Patch Size**: 不同频率使用不同 patch size，但共享 Transformer 权重
3. **任务分布采样**: 训练时随机采样 context length 和 prediction length
4. **Instance Normalization**: 对每个样本独立归一化

**⚠️ MOIRAI vs MOIRAI-MoE 的区别**:
| 方面 | MOIRAI (原版) | MOIRAI-MoE |
|------|--------------|------------|
| 架构 | Masked Encoder | Decoder-only |
| 投影层 | 多组 (per frequency) | 单组 + MoE |
| 训练目标 | Masked reconstruction | Autoregressive |
| 推理效率 | 一次前向 | 自回归多次 |

对于我们的 HVAC FDD 任务，**建议先使用原版 MOIRAI**:
- Masked Encoder 更适合重构任务（FDD 本质是重构）
- 实现更简单，调试更容易
- 数据量可能不足以支撑 MoE 专家分化

### 4.2 Hydra 配置文件详解

```yaml
# conf/pretrain/buildingfm.yaml

defaults:
  - _self_
  - /data: buildingfm_data  # 指向数据配置
  - /model: moirai_small    # 指向模型配置

run_name: buildingfm_v1
seed: 42

# 训练参数
trainer:
  max_epochs: 100
  accelerator: gpu
  devices: 1  # 或 [0, 1] 多卡
  precision: bf16-mixed  # A100 支持 bf16
  gradient_clip_val: 1.0
  accumulate_grad_batches: 1
  
# 优化器配置
optimizer:
  _target_: torch.optim.AdamW
  lr: 1e-3
  weight_decay: 0.1
  betas: [0.9, 0.98]

# 学习率调度
scheduler:
  _target_: transformers.get_cosine_schedule_with_warmup
  num_warmup_steps: 10000
  # num_training_steps 会自动计算
```

```yaml
# conf/model/moirai_small.yaml

_target_: uni2ts.model.moirai.MoiraiModule

# 架构参数
d_model: 384
nhead: 6
dim_feedforward: 1536
num_layers: 6
dropout: 0.1

# HVAC 特定参数
num_variates: 50              # ⚠️ 必须与 Schema 变量数一致
patch_sizes: [16, 32, 64]     # 支持的 patch size 列表
max_seq_len: 512              # 最大 token 序列长度

# 分布输出
distr_output:
  _target_: gluonts.torch.distributions.StudentTOutput
  # 或使用混合分布
  # _target_: uni2ts.distribution.MixtureOutput
```

```yaml
# conf/data/buildingfm_data.yaml

_target_: uni2ts.data.builder.lotsa.LOTSADatasetBuilder

# 数据路径
data_path: /path/to/arrow/data

# 采样配置
context_length: 512    # ⚠️ 这是 patches 数，不是时间步数
prediction_length: 64  # 预测的 patches 数

# 任务分布：训练时随机采样
min_context_length: 32
max_context_length: 512
min_prediction_length: 1
max_prediction_length: 128

# 批次配置
batch_size: 256
num_workers: 8

# 频率到 patch size 的映射
freq_to_patch_size:
  'T': 64       # 分钟级
  '5T': 64
  '15T': 32
  'H': 32
  'D': 16
```

**⚠️ context_length 的含义陷阱**:
- `context_length=512` 意味着 512 个 **patches**
- 如果 `patch_size=32`，实际时间步 = 512 × 32 = 16384
- 对于 15min 数据，这是 16384 × 15min ≈ 170 天

### 4.3 关键超参数选择

| 参数 | 建议值 | 理由 | 调整信号 |
|------|--------|------|----------|
| `num_variates` | Schema 大小 (50) | 必须固定 | 不可调整 |
| `patch_size` | 32 (15min 数据) | 1 patch = 8小时，覆盖日内周期 | 验证集 loss 平台期 |
| `context_length` | 512 patches | ~170天，覆盖季节性 | 内存限制 |
| `d_model` | 384/768/1024 | 小/中/大模型 | 过拟合程度 |
| `num_layers` | 6/12/24 | 小/中/大模型 | 计算资源 |
| `lr` | 1e-3 (阶段一), 1e-5 (阶段二) | 标准做法 | 训练不稳定时降低 |
| `batch_size` | 256 | 取决于 GPU 内存 | OOM 时减小 |
| `mask_ratio` | 0.3-0.5 | 高掩码率迫使跨变量推理 | 验证集 loss |

**⚠️ Patch Size 与 HVAC 动态的匹配**:
- HVAC 系统典型时间常数：15分钟 - 2小时
- 日周期：24小时
- 周周期：7天

对于 15min 采样：
- `patch_size=16`: 1 patch = 4小时，可能切断日内模式
- `patch_size=32`: 1 patch = 8小时，较好
- `patch_size=64`: 1 patch = 16小时，可能过粗

### 4.4 课程学习实现

**阶段一：物理逻辑注入**

```python
# 阶段一配置
stage1_config = {
    'data': 'energyplus_only',      # 100% 仿真数据
    'epochs': 50,
    'lr': 1e-3,
    'mask_ratio': 0.4,              # 较高掩码率
    'checkpoint_metric': 'val/loss',
}
```

阶段一的目标：
- 学会确定性物理关系：`P(Power | Temp, Flow, Setpoint)`
- 学会热力学延迟：温度变化滞后于功率变化
- 学会日周期：太阳辐射对温度的影响

**阶段一验证指标**:
- 在仿真测试集上，功率重构 MAE < 物理模型残差
- 从其他变量预测功率的 R² > 0.9

**阶段二：Sim-to-Real 域适应**

```python
# 阶段二配置
stage2_config = {
    'data': 'mixed',                # 80% 仿真 + 20% 真实
    'pretrained_checkpoint': 'stage1_best.ckpt',
    'epochs': 20,
    'lr': 1e-5,                     # 降低 1-2 个数量级
    'mask_ratio': 0.3,              # 稍低，因为真实数据已经有缺失
    'freeze_layers': None,          # 全部微调
}
```

**⚠️ 混合数据采样策略**:
```python
# 使用 uni2ts 的 sub-dataset 采样机制
# 限制每个子数据集的最大贡献比例
sub_dataset_cap = 0.2  # 单个数据集最多贡献 20%

# 这样真实数据即使量少，也能保证一定的采样频率
```

**阶段二验证指标**:
- 在真实数据测试集上的 FDD 准确率
- 跨建筑泛化：在未见过的建筑上性能不严重下降

---

## 5. 下游任务推理

### 5.1 理解 MOIRAI 的推理机制

MOIRAI 的推理与训练一致：将需要预测的部分替换为 [mask] token，然后模型输出该位置的概率分布。

```python
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from gluonts.dataset.pandas import PandasDataset
import torch

# 加载模型
module = MoiraiModule.from_pretrained("path/to/checkpoint")
model = MoiraiForecast(
    module=module,
    prediction_length=96,    # 预测 96 个时间步
    context_length=512,      # 使用 512 patches 的历史
    patch_size=32,           # 必须与训练一致
    num_samples=100,         # 采样次数（用于概率预测）
    target_dim=50,           # 变量数，必须与训练一致
    device_map="cuda:0",
)
```

**⚠️ 推理时的维度匹配**:
- `target_dim` 必须与训练时的 `num_variates` 一致
- 如果真实数据只有 5 个变量，其他 45 个必须填充 NaN

### 5.2 FDD (故障检测与诊断)

FDD 的核心思想：基于物理一致性的异常检测

```python
def fdd_inference(
    model: MoiraiForecast,
    data: pd.DataFrame,        # shape: (time_steps, num_variates)
    target_var_id: int,        # 要检测的变量 ID
    context_window: int = 512,
    stride: int = 96,
):
    """
    对指定变量进行故障检测
    
    算法：
    1. 滑动窗口遍历数据
    2. 对每个窗口，mask 掉目标变量的后半部分
    3. 模型根据其他变量重构目标变量的"应有值"
    4. 计算实际值与重构值的偏差
    5. 偏差超过阈值则报警
    """
    results = []
    num_variates = data.shape[1]
    
    for start_idx in range(0, len(data) - context_window, stride):
        window_data = data.iloc[start_idx:start_idx + context_window].copy()
        
        # 构造输入：mask 掉目标变量的后半部分（作为"预测目标"）
        prediction_length = context_window // 4  # 预测后 1/4
        context_length = context_window - prediction_length
        
        # 将目标变量的预测区间设为 NaN
        window_data.iloc[context_length:, target_var_id] = np.nan
        
        # 转换为 GluonTS 格式
        target_array = window_data.values.T  # (num_variates, time_steps)
        
        # 构造 dataset
        ds = [{
            'start': window_data.index[0],
            'target': target_array,
            'item_id': 'fdd_window',
        }]
        
        # 推理
        with torch.no_grad():
            forecasts = list(model.predict(ds, num_samples=100))
        
        # 提取预测分布
        forecast = forecasts[0]
        pred_mean = forecast.mean[target_var_id]      # 预测均值
        pred_std = forecast.std[target_var_id]        # 预测标准差
        
        # 获取实际值（从原始数据）
        actual = data.iloc[start_idx + context_length:start_idx + context_window, target_var_id].values
        
        # 计算 Z-score
        z_scores = np.abs(actual - pred_mean) / (pred_std + 1e-6)
        
        results.append({
            'start_time': window_data.index[context_length],
            'end_time': window_data.index[-1],
            'z_scores': z_scores,
            'max_z': np.max(z_scores),
            'mean_z': np.mean(z_scores),
            'anomaly': np.max(z_scores) > 3.0,  # 3-sigma 规则
        })
    
    return pd.DataFrame(results)
```

**⚠️ FDD 实现细节**:

1. **滑动窗口 vs 固定窗口**: 滑动窗口提供连续监控，但计算量大
2. **Z-score 阈值**: 3.0 是标准选择，但应根据误报率调整
3. **时间聚合**: 连续多个窗口报警才算真正故障，避免瞬时干扰

### 5.3 Forecasting (预测)

标准预测任务：给定历史，预测未来。

```python
def forecast(
    model: MoiraiForecast,
    history: pd.DataFrame,         # 历史数据
    prediction_length: int,        # 预测时长（时间步）
    known_future: pd.DataFrame = None,  # 已知的未来值（如天气预报）
):
    """
    时序预测
    
    ⚠️ 关键点：如何处理已知的未来协变量（如天气预报）
    - 天气预报应该保留其值，不设为 NaN
    - 需要预测的变量设为 NaN
    """
    num_variates = history.shape[1]
    context_length = len(history)
    
    # 构造完整的输入序列
    full_length = context_length + prediction_length
    target = np.full((num_variates, full_length), np.nan, dtype=np.float32)
    
    # 填充历史数据
    target[:, :context_length] = history.values.T
    
    # 填充已知的未来值
    if known_future is not None:
        for col in known_future.columns:
            var_id = int(col.split('_')[1]) if col.startswith('var_') else None
            if var_id is not None:
                target[var_id, context_length:] = known_future[col].values
    
    # 推理
    ds = [{
        'start': history.index[0],
        'target': target,
        'item_id': 'forecast',
    }]
    
    forecasts = list(model.predict(ds, num_samples=100))
    
    return forecasts[0]
```

**⚠️ 预测中的常见问题**:

1. **天气预报的处理**: 天气是已知的未来协变量，应该保留值而非 mask
2. **多步预测的误差累积**: MOIRAI 是一次性预测，不是自回归，误差不累积
3. **预测区间的物理合理性**: 检查预测值是否在物理范围内

### 5.4 Virtual Sensing (虚拟传感)

当真实建筑缺少某个传感器时，可以用模型"推断"该值。

```python
def virtual_sensing(
    model: MoiraiForecast,
    data: pd.DataFrame,
    missing_var_id: int,       # 缺失的变量 ID
):
    """
    虚拟传感器：推断缺失变量的值
    
    使用场景：
    - 真实建筑没有送风温度传感器
    - 但有室内温度、功率、设定点
    - 模型可以推断"物理一致"的送风温度
    """
    num_variates = 50  # Schema 定义的变量数
    
    # 确保缺失变量列全为 NaN
    target = np.full((num_variates, len(data)), np.nan, dtype=np.float32)
    
    # 填充有值的变量
    for col in data.columns:
        var_id = get_var_id_from_col(col)
        if var_id is not None and var_id != missing_var_id:
            target[var_id, :] = data[col].values
    
    # 推理
    ds = [{
        'start': data.index[0],
        'target': target,
        'item_id': 'virtual_sensing',
    }]
    
    forecasts = list(model.predict(ds, num_samples=100))
    
    # 提取虚拟传感器值
    virtual_values = forecasts[0].mean[missing_var_id]
    confidence = forecasts[0].std[missing_var_id]
    
    return virtual_values, confidence
```

**⚠️ 虚拟传感的局限性**:
- 推断值的置信度取决于相关变量的信息量
- 如果缺失变量与其他变量相关性弱，推断会不准确
- 应该输出置信区间，而非点估计

---

## 6. 代码复用与扩展

### 6.1 uni2ts 代码结构

```
uni2ts/
├── data/
│   ├── builder/           # 数据构建器
│   │   ├── simple.py      # 简单格式构建
│   │   └── lotsa.py       # LOTSA 格式构建
│   └── dataset.py         # Dataset 类
├── model/
│   ├── moirai.py          # MoiraiModule, MoiraiForecast
│   └── moirai_moe.py      # MoE 版本
├── transform/             # 数据变换
└── distribution/          # 概率分布输出
```

### 6.2 直接复用的模块

| 模块 | 路径 | 用途 | 备注 |
|------|------|------|------|
| MoiraiModule | `uni2ts/model/moirai.py` | 核心 Transformer | 不需修改 |
| MoiraiForecast | `uni2ts/model/moirai.py` | 推理封装 | 不需修改 |
| 训练 CLI | `cli/train.py` | 训练入口 | 通过配置文件定制 |
| 数据加载 | `uni2ts/data/` | DataLoader | 可能需要适配 |
| Instance Norm | `uni2ts/transform/` | 归一化 | 不需修改 |

### 6.3 需要编写的模块

| 任务 | 说明 | 复杂度 |
|------|------|--------|
| Schema 定义 | YAML 文件 | 低 |
| EnergyPlus 导出脚本 | CSV → Parquet | 中 |
| 真实数据预处理 | 各种来源 → 统一格式 | 高 |
| Arrow 数据构建 | Parquet → Arrow | 中 |
| Hydra 配置文件 | 训练配置 | 低 |
| FDD 推理脚本 | 基于 MoiraiForecast | 中 |
| 评估脚本 | 计算各种指标 | 中 |

### 6.4 可能需要的代码修改

**情况1：自定义 DataLoader**

如果 uni2ts 的 DataLoader 不能满足需求（如特殊的采样策略），可能需要：

```python
# 继承 uni2ts 的 Dataset 类
from uni2ts.data.dataset import TimeSeriesDataset

class HVACDataset(TimeSeriesDataset):
    def __init__(self, ...):
        super().__init__(...)
        # 添加 HVAC 特定的逻辑
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        # 自定义处理
        return sample
```

**情况2：自定义分布输出**

如果需要特定的分布（如截断正态分布，确保功率非负）：

```python
from gluonts.torch.distributions import DistributionOutput
import torch.distributions as td

class TruncatedNormalOutput(DistributionOutput):
    """截断正态分布，用于非负变量"""
    
    def __init__(self, lower=0.0, upper=float('inf')):
        self.lower = lower
        self.upper = upper
    
    def distribution(self, distr_args):
        loc, scale = distr_args
        base_dist = td.Normal(loc, scale)
        return td.TruncatedNormal(base_dist, self.lower, self.upper)
```

---

## 7. 验证实验设计

### 7.1 Baseline 对比

**Baseline 1: 传统迁移学习 (LSTM/Transformer)**

```
训练: EnergyPlus 全量数据 (50 变量)
测试: 真实建筑数据 (5 变量)
问题: 输入维度不匹配

解决尝试:
- 重训输入层 → 需要大量真实数据
- 只用共有变量 → 丢失物理信息
- 填充缺失变量为 0 → 物理不合理
```

**Baseline 2: 独立训练**

```
训练: 只用真实建筑数据
问题: 数据量不足，无法学习复杂物理关系
表现: 过拟合，泛化差
```

**Ours: BuildingFM**

```
训练: 
- 阶段一: EnergyPlus (学习物理)
- 阶段二: 混合数据 (域适应)

测试: 真实建筑数据 (稀疏变量)
优势:
- 变量数量可变 (Any-variate)
- 已学习物理关系可迁移
- 缺失变量自动处理
```

### 7.2 评估指标

**FDD 评估**:
- Precision: 报警中真正故障的比例
- Recall: 真实故障被检测出的比例
- F1 Score: 综合指标
- Detection Delay: 故障发生到检测到的时间

**预测评估**:
- MAE, RMSE: 点预测精度
- CRPS: 概率预测质量
- Coverage: 预测区间覆盖率

**物理一致性评估**:
- 能量守恒偏差: 输入能量 - 输出能量
- 温度响应合理性: 功率变化后温度应该变化

### 7.3 消融实验

| 实验 | 变量 | 预期结果 |
|------|------|----------|
| All-in-Target vs Weather-as-Covariate | 天气变量位置 | All-in-Target 更好 |
| 有/无阶段二 | 域适应 | 有阶段二在真实数据上更好 |
| 不同 mask_ratio | 0.2 vs 0.4 vs 0.6 | 0.4 可能最佳 |
| 不同 patch_size | 16 vs 32 vs 64 | 取决于数据频率 |

---

## 8. 开发顺序与检查点

### Week 1-2: 数据准备

**任务**:
1. 运行 EnergyPlus 仿真，获取输出变量列表
2. 定义 Schema YAML
3. 编写 EnergyPlus CSV 转换脚本
4. 生成 Arrow 格式数据

**检查点**:
- [ ] Schema 包含所有需要的变量
- [ ] 转换脚本能处理所有 E+ 输出文件
- [ ] Arrow 数据可被 uni2ts 正确加载
- [ ] 验证数据的 shape 和 dtype

### Week 3-4: 预训练

**任务**:
1. 配置 Hydra 文件
2. 小规模验证（1 GPU，小数据子集）
3. 修复问题
4. 全量预训练

**检查点**:
- [ ] 训练 loss 正常下降
- [ ] 验证 loss 不过拟合
- [ ] 能保存和加载 checkpoint
- [ ] 推理时能输出合理预测

### Week 5-6: 下游任务与评估

**任务**:
1. 实现 FDD 推理脚本
2. 准备测试数据集
3. 运行 baseline 对比
4. 整理结果

**检查点**:
- [ ] FDD 脚本能处理真实数据
- [ ] 评估指标计算正确
- [ ] Baseline 结果合理
- [ ] 可视化对比清晰

---

## 9. 常见问题与解决方案

### 9.1 数据相关

**Q: EnergyPlus 不同建筑的变量列表不一致怎么办？**

A: 使用 Schema 作为超集。缺失的变量填充 NaN。模型会自动处理。

**Q: 真实数据采样频率不规则怎么办？**

A: 重采样到统一频率。对于状态量用插值，对于累积量用求和。重采样产生的新 NaN 保留，不要再插值。

**Q: 数据量太大，Arrow 文件太大怎么办？**

A: 分片存储。每个 Arrow 文件 ~500MB。使用 uni2ts 的分片加载机制。

### 9.2 训练相关

**Q: 训练 loss 不下降怎么办？**

A: 
1. 检查数据是否正确加载（打印几个样本看看）
2. 降低学习率
3. 检查 patch_size 与 context_length 的配合
4. 检查 num_variates 是否正确

**Q: 训练速度太慢怎么办？**

A:
1. 使用 sequence packing（uni2ts 已支持）
2. 减小 batch_size，增加 gradient_accumulation
3. 使用混合精度训练（bf16）
4. 多卡并行

**Q: GPU 内存不够怎么办？**

A:
1. 减小 batch_size
2. 减小 max_seq_len
3. 使用 gradient checkpointing
4. 使用更小的模型

### 9.3 推理相关

**Q: 推理时维度不匹配怎么办？**

A: 确保 `target_dim` 与训练时的 `num_variates` 完全一致。真实数据缺少的变量必须填充 NaN，而不是减少维度。

**Q: 推理结果不合理（如功率为负）怎么办？**

A:
1. 后处理：截断到物理范围
2. 或者使用截断分布作为输出

**Q: 推理速度太慢怎么办？**

A:
1. 使用 ONNX 或 TensorRT 加速
2. 减少采样次数（num_samples）
3. 批量推理而非单条

---

## 10. 参考资源

### 核心资源
- uni2ts 仓库: https://github.com/SalesforceAIResearch/uni2ts
- MOIRAI 论文: https://arxiv.org/abs/2402.02592
- MOIRAI-MoE 论文: https://arxiv.org/abs/2410.10469

### 依赖库文档
- GluonTS: https://ts.gluon.ai/
- Hydra: https://hydra.cc/
- PyTorch Lightning: https://lightning.ai/docs/pytorch/stable/
- Apache Arrow: https://arrow.apache.org/docs/python/

### HVAC 仿真
- EnergyPlus: https://energyplus.net/
- eppy (EnergyPlus Python 接口): https://eppy.readthedocs.io/

### 相关研究
- BuildingsBench: https://github.com/NREL/BuildingsBench
- ASHRAE 1478-RP (FDD 基准故障)
