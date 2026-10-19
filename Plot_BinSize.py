# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 22:09:34 2025

@author: zsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取CSV数据
csv_file = r"D:\My_WorkSpace\毕业课题\学位论文\结果\human\不同bin模型-5folds.csv"
df = pd.read_csv(csv_file)

df["mean"] = df[["fold1", "fold2", "fold3", "fold4", "fold5"]].mean(axis=1)
df["std"] = df[["fold1", "fold2", "fold3", "fold4", "fold5"]].std(axis=1)

# 获取binsize的唯一值（字符串），并按照原始顺序排序（也可以自定义顺序）
bin_labels = sorted(df["binsize"].unique())
# 使用等距索引作为x轴数据
x_axis = np.arange(len(bin_labels))

# 创建结果存储字典
results = {}
for cnv_type in ["DEL", "DUP"]:
    results[cnv_type] = {}
    for rate_type in ["Precision", "Recall"]:
        subset = df[(df["cnvtype"] == cnv_type) & (df["ratetype"] == rate_type)]
        # 按binsize排序（排序依据需要保证和bin_labels一致）
        subset = subset.sort_values("binsize")
        results[cnv_type][rate_type] = {
            "values": subset["mean"].values,
            "std": subset["std"].values
        }

# 计算F1分数
del_f1 = []
del_f1_std = []
dup_f1 = []
dup_f1_std = []
for i in range(len(bin_labels)):
    del_precision = results["DEL"]["Precision"]["values"][i]
    del_recall = results["DEL"]["Recall"]["values"][i]
    dup_precision = results["DUP"]["Precision"]["values"][i]
    dup_recall = results["DUP"]["Recall"]["values"][i]
    
    # 计算F1分数
    del_f1.append(2 * (del_precision * del_recall) / (del_precision + del_recall))
    dup_f1.append(2 * (dup_precision * dup_recall) / (dup_precision + dup_recall))
    
    # 计算F1标准差（使用误差传播公式的简化版）
    del_p_std = results["DEL"]["Precision"]["std"][i]
    del_r_std = results["DEL"]["Recall"]["std"][i]
    dup_p_std = results["DUP"]["Precision"]["std"][i]
    dup_r_std = results["DUP"]["Recall"]["std"][i]
    
    del_f1_std.append(np.sqrt((del_p_std/del_precision)**2 + (del_r_std/del_recall)**2) * del_f1[-1] * 0.5)
    dup_f1_std.append(np.sqrt((dup_p_std/dup_precision)**2 + (dup_r_std/dup_recall)**2) * dup_f1[-1] * 0.5)

# =============================================================================
# 画图
# =============================================================================
# 设置全局字体样式
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'font.weight': 'bold'
})

# 图1: 准确率图表
plt.figure(figsize=(10, 7))
plt.errorbar(x_axis, results["DEL"]["Precision"]["values"], yerr=results["DEL"]["Precision"]["std"], 
             fmt='o-', color='navy', capsize=5, label='DEL', linewidth=2, markersize=10, linestyle='-.')
plt.errorbar(x_axis, results["DUP"]["Precision"]["values"], yerr=results["DUP"]["Precision"]["std"], 
             fmt='v-', color='darkred', capsize=5, label='DUP', linewidth=2, markersize=10, linestyle='-.')
# plt.xlabel('binsize', fontsize=14)
# plt.ylabel('Precision', fontsize=14)
# plt.title('Precision at Different Binsizes', fontsize=15)
plt.grid(True, linestyle='--', alpha=0.7)
# 设置x轴刻度为等距的索引，并将标签设为bin_labels，同时加粗刻度
plt.xticks(x_axis, bin_labels, fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('D:/My_WorkSpace/毕业课题/学位论文/图片/不同bin性能评估/human/precision_comparison.tiff', dpi=300)
plt.show()

# 图2: 召回率图表
plt.figure(figsize=(10, 7))
plt.errorbar(x_axis, results["DEL"]["Recall"]["values"], yerr=results["DEL"]["Recall"]["std"], 
             fmt='o-', color='navy', capsize=5, label='DEL', linewidth=2, markersize=10, linestyle='-.')
plt.errorbar(x_axis, results["DUP"]["Recall"]["values"], yerr=results["DUP"]["Recall"]["std"], 
             fmt='v-', color='darkred', capsize=5, label='DUP', linewidth=2, markersize=10, linestyle='-.')
# plt.xlabel('binsize', fontsize=14)
# plt.ylabel('Recall', fontsize=14)
# plt.title('Recall at Different Binsizes', fontsize=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(x_axis, bin_labels, fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('D:/My_WorkSpace/毕业课题/学位论文/图片/不同bin性能评估/human/recall_comparison.tiff', dpi=300)
plt.show()

# 图3: F1分数图表
plt.figure(figsize=(10, 7))
plt.errorbar(x_axis, del_f1, yerr=del_f1_std, fmt='o-', color='navy', 
             capsize=5, label='DEL', linewidth=2, markersize=10, linestyle='-.')
plt.errorbar(x_axis, dup_f1, yerr=dup_f1_std, fmt='v-', color='darkred', 
             capsize=5, label='DUP', linewidth=2, markersize=10, linestyle='-.')
# plt.xlabel('binsize', fontsize=14)
# plt.ylabel('F1 score', fontsize=14)
# plt.title('F1 Score at Different Binsizes', fontsize=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(x_axis, bin_labels, fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('D:/My_WorkSpace/毕业课题/学位论文/图片/不同bin性能评估/human/f1_score_comparison.tiff', dpi=300)
plt.show()