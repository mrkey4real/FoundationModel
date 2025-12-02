# -*- coding: utf-8 -*-
"""
模型公平对比脚本
对比 moirai, moirai-moe, moirai2 的预测性能

作者: Claude Code
日期: 2025-12-01
"""

import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MAE, RMSE, MAPE, MASE
import time

from uni2ts.eval_util.plot import plot_single
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai_moe import MoiraiMoEForecast, MoiraiMoEModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

# ========== 统一对比参数 ==========
SIZE = "small"      # 模型大小: {'small', 'base', 'large'}
PDT = 20            # 预测长度: 预测多少个时间步
CTX = 200           # 上下文长度: 使用多少历史数据
PSZ = 32            # patch size: {8, 16, 32, 64, 128} (moirai-moe不支持"auto")
BSZ = 32            # batch size
TEST = 100          # 测试集长度
NUM_SAMPLES = 100   # 概率预测采样次数
# ==================================

print("=" * 60)
print("MOIRAI 模型对比实验")
print("=" * 60)
print(f"配置:")
print(f"  模型大小: {SIZE}")
print(f"  预测长度: {PDT}")
print(f"  上下文长度: {CTX}")
print(f"  Patch Size: {PSZ}")
print(f"  Batch Size: {BSZ}")
print(f"  测试集长度: {TEST}")
print("=" * 60)

# 读取数据
print("\n加载数据...")
url = (
    "https://gist.githubusercontent.com/rsnirwan/c8c8654a98350fadd229b00167174ec4"
    "/raw/a42101c7786d4bc7695228a0f2c8cea41340e18f/ts_wide.csv"
)
df = pd.read_csv(url, index_col=0, parse_dates=True)
print(f"数据形状: {df.shape}")

# 转换为 GluonTS dataset
ds = PandasDataset(dict(df))

# 划分训练/测试集
train, test_template = split(ds, offset=-TEST)

# 构造滚动窗口评估
test_data = test_template.generate_instances(
    prediction_length=PDT,
    windows=TEST // PDT,
    distance=PDT,
)

# 定义要对比的模型
MODELS = {
    "moirai": {
        "class": MoiraiForecast,
        "module_class": MoiraiModule,
        "repo": f"Salesforce/moirai-1.1-R-{SIZE}",
        "supports_patch_size": True,
    },
    "moirai-moe": {
        "class": MoiraiMoEForecast,
        "module_class": MoiraiMoEModule,
        "repo": f"Salesforce/moirai-moe-1.0-R-{SIZE}",
        "supports_patch_size": True,
    },
    "moirai2": {
        "class": Moirai2Forecast,
        "module_class": Moirai2Module,
        "repo": f"Salesforce/moirai-2.0-R-{SIZE}",
        "supports_patch_size": False,  # moirai2 不使用 patch_size
    },
}

# 存储结果
results = {}

# 对每个模型进行评估
for model_name, model_config in MODELS.items():
    print(f"\n{'='*60}")
    print(f"评估模型: {model_name.upper()}")
    print(f"{'='*60}")

    try:
        # 加载模型
        print(f"  加载预训练权重: {model_config['repo']}")
        module = model_config["module_class"].from_pretrained(model_config["repo"])

        # 构建模型参数
        model_params = {
            "module": module,
            "prediction_length": PDT,
            "context_length": CTX,
            "target_dim": 1,
            "feat_dynamic_real_dim": ds.num_feat_dynamic_real,
            "past_feat_dynamic_real_dim": ds.num_past_feat_dynamic_real,
        }

        # moirai2 不支持 num_samples 参数
        if model_name != "moirai2":
            model_params["num_samples"] = NUM_SAMPLES

        # 如果模型支持 patch_size，添加该参数
        if model_config["supports_patch_size"]:
            model_params["patch_size"] = PSZ

        model = model_config["class"](**model_params)

        # 创建 predictor
        predictor = model.create_predictor(batch_size=BSZ)

        # 进行预测
        print(f"  开始预测...")
        start_time = time.time()
        forecasts = list(predictor.predict(test_data.input))
        inference_time = time.time() - start_time
        print(f"  预测完成，耗时: {inference_time:.2f}秒")

        # 计算评估指标
        print(f"  计算评估指标...")

        # 提取真实值和预测值
        all_actuals = []
        all_predictions = []

        label_it = iter(test_data.label)
        for forecast in forecasts:
            label = next(label_it)
            actual = label['target']
            pred = forecast.mean

            # 处理预测长度不一致的情况（如moirai-moe可能调整预测长度）
            min_len = min(len(actual), len(pred))
            if len(actual) != len(pred):
                print(f"  注意: 预测长度 ({len(pred)}) 与真实长度 ({len(actual)}) 不匹配，截取前 {min_len} 个")

            all_actuals.append(actual[:min_len])
            all_predictions.append(pred[:min_len])

        # 计算指标
        all_actuals = np.array(all_actuals)
        all_predictions = np.array(all_predictions)

        mae = np.mean(np.abs(all_actuals - all_predictions))
        rmse = np.sqrt(np.mean((all_actuals - all_predictions) ** 2))
        mape = np.mean(np.abs((all_actuals - all_predictions) / (all_actuals + 1e-8))) * 100

        # 存储结果
        results[model_name] = {
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape,
            "inference_time": inference_time,
            "forecasts": forecasts,
        }

        print(f"  结果:")
        print(f"    MAE:  {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAPE: {mape:.2f}%")
        print(f"    推理时间: {inference_time:.2f}秒")

    except Exception as e:
        print(f"  [ERROR] Model {model_name} evaluation failed: {str(e)}")
        results[model_name] = None

# 打印对比结果
print(f"\n{'='*60}")
print("对比结果汇总")
print(f"{'='*60}")

# 创建对比表格
comparison_df = pd.DataFrame({
    model_name: {
        "MAE": res["MAE"],
        "RMSE": res["RMSE"],
        "MAPE (%)": res["MAPE"],
        "推理时间 (秒)": res["inference_time"],
    }
    for model_name, res in results.items()
    if res is not None
}).T

print(comparison_df.to_string())

# 找出最佳模型
if len(comparison_df) > 0:
    best_mae_model = comparison_df["MAE"].idxmin()
    best_rmse_model = comparison_df["RMSE"].idxmin()
    fastest_model = comparison_df["推理时间 (秒)"].idxmin()

    print(f"\n最佳模型 (MAE): {best_mae_model}")
    print(f"最佳模型 (RMSE): {best_rmse_model}")
    print(f"最快模型: {fastest_model}")

# 可视化对比
print(f"\n生成可视化对比图...")

# 1. 指标对比柱状图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('MOIRAI Model Comparison', fontsize=16, fontweight='bold')

metrics = ["MAE", "RMSE", "MAPE (%)", "推理时间 (秒)"]
metric_labels = ["MAE", "RMSE", "MAPE (%)", "Inference Time (s)"]
for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
    ax = axes[idx // 2, idx % 2]
    comparison_df[metric].plot(kind='bar', ax=ax, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax.set_title(label, fontsize=12, fontweight='bold')
    ax.set_ylabel(label)
    ax.set_xlabel('Model')
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print(f"  对比图已保存: model_comparison.png")

# 2. 预测结果可视化（展示第一个窗口）
if len(results) > 0:
    fig, axes = plt.subplots(1, len(results), figsize=(6*len(results), 4))
    if len(results) == 1:
        axes = [axes]

    fig.suptitle('Prediction Comparison (First Window)', fontsize=16, fontweight='bold')

    input_it = iter(test_data.input)
    label_it = iter(test_data.label)

    inp = next(input_it)
    label = next(label_it)

    for idx, (model_name, res) in enumerate(results.items()):
        if res is not None:
            forecast = res["forecasts"][0]

            ax = axes[idx]

            # 绘制历史数据
            history = inp['target']
            history_len = len(history)
            ax.plot(range(-history_len, 0), history, label='History', color='black', linewidth=1.5)

            # 绘制真实值和预测值
            actual = label['target']
            pred_mean = forecast.mean

            # 处理预测长度可能不一致的情况
            pred_len = len(pred_mean)
            actual_len = len(actual)
            plot_len = min(pred_len, actual_len)

            ax.plot(range(0, actual_len), actual, label='Actual', color='green', linewidth=2, marker='o', markersize=4)
            ax.plot(range(0, pred_len), pred_mean, label='Prediction', color='red', linewidth=2, linestyle='--')

            # 绘制预测区间（如果有）
            if hasattr(forecast, 'std'):
                pred_std = forecast.std
                ax.fill_between(
                    range(0, pred_len),
                    pred_mean - 1.96 * pred_std,
                    pred_mean + 1.96 * pred_std,
                    alpha=0.2,
                    color='red',
                    label='95% CI'
                )
            else:
                # 使用quantile计算置信区间
                try:
                    q_low = forecast.quantile(0.025)
                    q_high = forecast.quantile(0.975)
                    ax.fill_between(
                        range(0, pred_len),
                        q_low,
                        q_high,
                        alpha=0.2,
                        color='red',
                        label='95% CI'
                    )
                except:
                    pass  # 无法获取置信区间

            ax.set_title(f'{model_name}\nMAE: {res["MAE"]:.4f}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Steps')
            ax.set_ylabel('Value')
            ax.legend(loc='best', fontsize=8)
            ax.grid(alpha=0.3)
            ax.axvline(x=0, color='gray', linestyle=':', linewidth=1)

    plt.tight_layout()
    plt.savefig('prediction_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  预测对比图已保存: prediction_comparison.png")

plt.show()

print(f"\n{'='*60}")
print("对比实验完成！")
print(f"{'='*60}")
