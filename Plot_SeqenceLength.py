# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 19:52:21 2025

@author: zsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 假设数据保存在CSV文件中
# 读取CSV数据
csv_file = r"D:\My_WorkSpace\毕业课题\学位论文\结果\human\不同序列长度模型-5folds.csv"
df = pd.read_csv(csv_file)

# # 如果直接使用数据，可以这样创建DataFrame
# data = [
#     ["DEL", "Precision", 100, 0.7584, 0.7012, 0.7425, 0.7185, 0.7414],
#     ["DEL", "Precision", 200, 0.7546, 0.7125, 0.7512, 0.7183, 0.7494],
#     ["DEL", "Precision", 300, 0.7986, 0.7585, 0.7825, 0.7493, 0.7771],
#     ["DEL", "Precision", 400, 0.7615, 0.7285, 0.7532, 0.7284, 0.7544],
#     ["DEL", "Precision", 500, 0.7582, 0.7286, 0.7493, 0.7325, 0.7544],
#     ["DEL", "Recall", 100, 0.7224, 0.6825, 0.7133, 0.6985, 0.7113],
#     ["DEL", "Recall", 200, 0.7186, 0.6893, 0.7125, 0.6872, 0.7129],
#     ["DEL", "Recall", 300, 0.7482, 0.7163, 0.7425, 0.7186, 0.7389],
#     ["DEL", "Recall", 400, 0.7384, 0.7085, 0.7312, 0.7072, 0.7267],
#     ["DEL", "Recall", 500, 0.7328, 0.7125, 0.7293, 0.7153, 0.7306],
#     ["DUP", "Precision", 100, 0.7952, 0.7485, 0.7823, 0.7586, 0.7694],
#     ["DUP", "Precision", 200, 0.8452, 0.8123, 0.8362, 0.8156, 0.8362],
#     ["DUP", "Precision", 300, 0.8765, 0.8386, 0.8624, 0.8415, 0.8590],
#     ["DUP", "Precision", 400, 0.8542, 0.8184, 0.8425, 0.8253, 0.8451],
#     ["DUP", "Precision", 500, 0.8516, 0.8126, 0.8385, 0.8265, 0.8368],
#     ["DUP", "Recall", 100, 0.7823, 0.7512, 0.7725, 0.7584, 0.7646],
#     ["DUP", "Recall", 200, 0.8127, 0.7825, 0.8046, 0.7895, 0.8002],
#     ["DUP", "Recall", 300, 0.8153, 0.7874, 0.8086, 0.7925, 0.8057],
#     ["DUP", "Recall", 400, 0.7586, 0.7312, 0.7524, 0.7386, 0.7462],
#     ["DUP", "Recall", 500, 0.7492, 0.7245, 0.7435, 0.7326, 0.7437]
# ]

# df = pd.DataFrame(data, columns=["cnvtype", "ratetype", "seqlen", "fold1", "fold2", "fold3", "fold4", "fold5"])

# # 转换序列长度为数值型
# df["seqlen"] = pd.to_numeric(df["seqlen"])

# 计算每行的平均值和标准差
df["mean"] = df[["fold1", "fold2", "fold3", "fold4", "fold5"]].mean(axis=1)
df["std"] = df[["fold1", "fold2", "fold3", "fold4", "fold5"]].std(axis=1)

# 获取序列长度列表
seq_lengths = sorted(df["seqlen"].unique())

# 创建结果存储字典
results = {}

# 处理数据: 按cnvtype和ratetype分组
for cnv_type in ["DEL", "DUP"]:
    results[cnv_type] = {}
    for rate_type in ["Precision", "Recall"]:
        subset = df[(df["cnvtype"] == cnv_type) & (df["ratetype"] == rate_type)]
        # 确保按序列长度排序
        subset = subset.sort_values("seqlen")
        results[cnv_type][rate_type] = {
            "values": subset["mean"].values,
            "std": subset["std"].values
        }

# 计算F1分数
del_f1 = []
del_f1_std = []
dup_f1 = []
dup_f1_std = []

for i, seq_len in enumerate(seq_lengths):
    # 提取该序列长度的precision和recall
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
    
    # 使用简化的误差传播公式
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
    'font.weight': 'bold'  # 设置字体加粗
})

# 图1: 准确率图表
plt.figure(figsize=(10, 7))
plt.errorbar(seq_lengths, results["DEL"]["Precision"]["values"], yerr=results["DEL"]["Precision"]["std"], 
             fmt='o-', color='navy', capsize=5, label='DEL', linewidth=2, markersize=10, linestyle='-.')
plt.errorbar(seq_lengths, results["DUP"]["Precision"]["values"], yerr=results["DUP"]["Precision"]["std"], 
             fmt='v-', color='darkred', capsize=5, label='DUP', linewidth=2, markersize=10, linestyle='-.')
plt.xlabel('Sequence Length / bin', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.title('Precision at Different Sequence Lengths', fontsize=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(seq_lengths)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('D:/My_WorkSpace/毕业课题/学位论文/图片/不同序列长度性能评估/human/precision_comparison.tiff', dpi=300)
plt.show()

# 图2: 召回率图表
plt.figure(figsize=(10, 7))
plt.errorbar(seq_lengths, results["DEL"]["Recall"]["values"], yerr=results["DEL"]["Recall"]["std"], 
             fmt='o-', color='navy', capsize=5, label='DEL', linewidth=2, markersize=10, linestyle='-.')
plt.errorbar(seq_lengths, results["DUP"]["Recall"]["values"], yerr=results["DUP"]["Recall"]["std"], 
             fmt='v-', color='darkred', capsize=5, label='DUP', linewidth=2, markersize=10, linestyle='-.')
plt.xlabel('Sequence Length / bin', fontsize=14)
plt.ylabel('Recall', fontsize=14)
plt.title('Recall at Different Sequence Lengths', fontsize=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(seq_lengths)
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('D:/My_WorkSpace/毕业课题/学位论文/图片/不同序列长度性能评估/human/recall_comparison.tiff', dpi=300)
plt.show()

# 图3: F1分数图表
plt.figure(figsize=(10, 7))
plt.errorbar(seq_lengths, del_f1, yerr=del_f1_std, fmt='o-', color='navy', 
             capsize=5, label='DEL', linewidth=2, markersize=10, linestyle='-.')
plt.errorbar(seq_lengths, dup_f1, yerr=dup_f1_std, fmt='v-', color='darkred', 
             capsize=5, label='DUP', linewidth=2, markersize=10, linestyle='-.')
plt.xlabel('Sequence Length / bin', fontsize=14)
plt.ylabel('F1 score', fontsize=14)
plt.title('F1 Score at Different Sequence Lengths', fontsize=15)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(seq_lengths)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('D:/My_WorkSpace/毕业课题/学位论文/图片/不同序列长度性能评估/human/f1_score_comparison.tiff', dpi=300)
plt.show()


