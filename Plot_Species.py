# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 19:25:16 2025

@author: zsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 创建包含所有数据的DataFrame
data = {
    'Species': ['Human', 'Human', 'Pig', 'Pig', 'Cattle', 'Cattle'],
    'CNV_Type': ['DEL', 'DUP', 'DEL', 'DUP', 'DEL', 'DUP'],
    'Precision': [0.7732, 0.8556, 0.7046, 0.7656, 0.7245, 0.7186],
    'Precision_std': [0.0195, 0.0157, 0.0143, 0.0141, 0.0158, 0.0165],
    'Recall': [0.7329, 0.8019, 0.6775, 0.7194, 0.7032, 0.6974],
    'Recall_std': [0.0143, 0.0112, 0.0162, 0.0108, 0.0146, 0.0152],
    'F1': [0.7525, 0.8279, 0.6907, 0.7417, 0.7137, 0.7078],
    # 通过误差传播计算F1分数的标准差
    'F1_std': [0.0169, 0.0134, 0.0153, 0.0124, 0.0152, 0.0158]
}

df = pd.DataFrame(data)
print(df)

# 设置图表样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
    'legend.fontsize': 12
})

# 设置颜色
colors = {'DEL': 'navy', 'DUP': 'darkred'}
bar_width = 0.15
opacity = 0.8

# 图1: 精确率(Precision)柱状图
plt.figure(figsize=(6, 6))
x = np.arange(len(df['Species'].unique()))

# 绘制DEL精确率柱状图
del_precision = df[df['CNV_Type'] == 'DEL']['Precision'].values
del_precision_std = df[df['CNV_Type'] == 'DEL']['Precision_std'].values
plt.bar(x - bar_width/2, del_precision, bar_width, 
        alpha=opacity, color=colors['DEL'], label='DEL')
plt.errorbar(x - bar_width/2, del_precision, yerr=del_precision_std,
             fmt='none', ecolor='black', capsize=5)

# 绘制DUP精确率柱状图
dup_precision = df[df['CNV_Type'] == 'DUP']['Precision'].values
dup_precision_std = df[df['CNV_Type'] == 'DUP']['Precision_std'].values
plt.bar(x + bar_width/2, dup_precision, bar_width,
        alpha=opacity, color=colors['DUP'], label='DUP')
plt.errorbar(x + bar_width/2, dup_precision, yerr=dup_precision_std,
             fmt='none', ecolor='black', capsize=5)

# 添加标签和标题
plt.xlabel('Species', fontsize=16, fontweight='bold')
plt.ylabel('Precision', fontsize=16, fontweight='bold')
# plt.title('CNV Detection Precision Across Species', fontweight='bold')
plt.xticks(x, df['Species'].unique(), fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend()

# 设置y轴范围
plt.ylim(0.6, 0.9)

# 在柱子上添加数值标签
# for i, value in enumerate(del_precision):
#     plt.text(i - bar_width/2, value + 0.01, f'{value:.3f}', 
#              ha='center', va='bottom', fontsize=10)
# for i, value in enumerate(dup_precision):
#     plt.text(i + bar_width/2, value + 0.01, f'{value:.3f}', 
#              ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(r'D:\My_WorkSpace\毕业课题\学位论文\图片\不同物种间比较\precision_comparison.tiff', dpi=300, bbox_inches='tight')
plt.show()


##=======================================================================================
# 图2: 召回率(Recall)柱状图
plt.figure(figsize=(6, 6))

# 绘制DEL召回率柱状图
del_recall = df[df['CNV_Type'] == 'DEL']['Recall'].values
del_recall_std = df[df['CNV_Type'] == 'DEL']['Recall_std'].values
plt.bar(x - bar_width/2, del_recall, bar_width, 
        alpha=opacity, color=colors['DEL'], label='DEL')
plt.errorbar(x - bar_width/2, del_recall, yerr=del_recall_std,
             fmt='none', ecolor='black', capsize=5)

# 绘制DUP召回率柱状图
dup_recall = df[df['CNV_Type'] == 'DUP']['Recall'].values
dup_recall_std = df[df['CNV_Type'] == 'DUP']['Recall_std'].values
plt.bar(x + bar_width/2, dup_recall, bar_width,
        alpha=opacity, color=colors['DUP'], label='DUP')
plt.errorbar(x + bar_width/2, dup_recall, yerr=dup_recall_std,
             fmt='none', ecolor='black', capsize=5)

# 添加标签和标题
plt.xlabel('Species', fontsize=16, fontweight='bold')
plt.ylabel('Recall', fontsize=16, fontweight='bold')
# plt.title('CNV Detection Recall Across Species', fontweight='bold')
plt.xticks(x, df['Species'].unique(), fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend()

# 设置y轴范围
plt.ylim(0.6, 0.9)

# 在柱子上添加数值标签
# for i, value in enumerate(del_recall):
#     plt.text(i - bar_width/2, value + 0.01, f'{value:.3f}', 
#              ha='center', va='bottom', fontsize=10)
# for i, value in enumerate(dup_recall):
#     plt.text(i + bar_width/2, value + 0.01, f'{value:.3f}', 
#              ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(r'D:\My_WorkSpace\毕业课题\学位论文\图片\不同物种间比较\recall_comparison.tiff', dpi=300, bbox_inches='tight')
plt.show()


## ====================================================================================================
# 图3: F1分数柱状图
plt.figure(figsize=(6, 6))

# 绘制DEL F1分数柱状图
del_f1 = df[df['CNV_Type'] == 'DEL']['F1'].values
del_f1_std = df[df['CNV_Type'] == 'DEL']['F1_std'].values
plt.bar(x - bar_width/2, del_f1, bar_width, 
        alpha=opacity, color=colors['DEL'], label='DEL')
plt.errorbar(x - bar_width/2, del_f1, yerr=del_f1_std,
             fmt='none', ecolor='black', capsize=5)

# 绘制DUP F1分数柱状图
dup_f1 = df[df['CNV_Type'] == 'DUP']['F1'].values
dup_f1_std = df[df['CNV_Type'] == 'DUP']['F1_std'].values
plt.bar(x + bar_width/2, dup_f1, bar_width,
        alpha=opacity, color=colors['DUP'], label='DUP')
plt.errorbar(x + bar_width/2, dup_f1, yerr=dup_f1_std,
             fmt='none', ecolor='black', capsize=5)

# 添加标签和标题
plt.xlabel('Species', fontsize=16, fontweight='bold')
plt.ylabel('F1 score', fontsize=16, fontweight='bold')
# plt.title('CNV Detection F1 Score Across Species', fontweight='bold')
plt.xticks(x, df['Species'].unique(), fontsize=14, fontweight='bold')
plt.yticks(fontsize=14, fontweight='bold')
plt.legend()

# 设置y轴范围
plt.ylim(0.6, 0.9)

# 在柱子上添加数值标签
# for i, value in enumerate(del_f1):
#     plt.text(i - bar_width/2, value + 0.01, f'{value:.3f}', 
#              ha='center', va='bottom', fontsize=10)
# for i, value in enumerate(dup_f1):
#     plt.text(i + bar_width/2, value + 0.01, f'{value:.3f}', 
#              ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(r'D:\My_WorkSpace\毕业课题\学位论文\图片\不同物种间比较\f1_comparison.tiff', dpi=300, bbox_inches='tight')
plt.show()