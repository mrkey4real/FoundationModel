# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 12:59:28 2025

@author: qizixuan
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from huggingface_hub import hf_hub_download

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# ========== 公平对比配置 ==========
# 为了公平对比不同模型，所有模型应使用相同的参数：
# - prediction_length (PDT): 预测的时间步数
# - context_length (CTX): 用于预测的历史长度
# - patch_size (PSZ): 每个patch包含的时间步数
# - batch_size (BSZ): 批次大小
# - num_samples: 概率预测的采样次数（已统一为100）
# ===================================

MODEL = "moirai2"  # model name: choose from {'moirai', 'moirai-moe', 'moirai2'}
SIZE = "small"  # model size: choose from {'small', 'base', 'large'}
PDT = 200  # prediction length: any positive integer
CTX = 400  # context length: any positive integer
PSZ = "auto"  # patch size: choose from {"auto", 8, 16, 32, 64, 128}
BSZ = 32  # batch size: any positive integer
TEST = 200  # test set length: any positive integer

# Read data into pandas DataFrame
df = pd.read_csv("./data/merged_labview_egauge.csv", index_col="timestamp", parse_dates=True)

# 选择要预测的目标列（可以改成你想预测的任意列名）
TARGET_COLUMN = "labview_Thermostat"

# 只保留目标列，并去除NaN值
df = df[[TARGET_COLUMN]].dropna()

# 确保时间索引排序，并设置频率为15分钟
df = df.sort_index()
df = df.asfreq("15min")

# Convert into GluonTS dataset
ds = PandasDataset(dict(df), freq="15min")

# Split into train/test set
train, test_template = split(
    ds, offset=-TEST
)  # assign last TEST time steps as test set

# Construct rolling window evaluation
test_data = test_template.generate_instances(
    prediction_length=PDT,  # number of time steps for each prediction
    windows=TEST // PDT,  # number of windows in rolling window evaluation
    distance=PDT,  # number of time steps between each window - distance=PDT for non-overlapping windows
)

# Prepare pre-trained model by downloading model weights from huggingface hub
if MODEL == "moirai":
    model = MoiraiForecast(
        module=MoiraiModule.from_pretrained(f"Salesforce/moirai-1.1-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )
elif MODEL == "moirai-moe":
    model = MoiraiMoEForecast(
        module=MoiraiMoEModule.from_pretrained(f"Salesforce/moirai-moe-1.0-R-{SIZE}"),
        prediction_length=PDT,
        context_length=CTX,
        patch_size=PSZ,  # 使用统一的 PSZ 参数
        num_samples=100,
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

elif MODEL == "moirai2":
    model = Moirai2Forecast(
        module=Moirai2Module.from_pretrained(
            f"Salesforce/moirai-2.0-R-{SIZE}",  # 使用 SIZE 变量
        ),
        prediction_length=PDT,  # 使用统一的 PDT 参数
        context_length=CTX,     # 使用统一的 CTX 参数
        target_dim=1,
        feat_dynamic_real_dim=ds.num_feat_dynamic_real,
        past_feat_dynamic_real_dim=ds.num_past_feat_dynamic_real,
    )

predictor = model.create_predictor(batch_size=BSZ)
forecasts = predictor.predict(test_data.input)

input_it = iter(test_data.input)
label_it = iter(test_data.label)
forecast_it = iter(forecasts)

inp = next(input_it)
label = next(label_it)
forecast = next(forecast_it)

plot_single(
    inp, 
    label, 
    forecast, 
    context_length=CTX,
    name="pred",
    show_label=True,
)
plt.show()