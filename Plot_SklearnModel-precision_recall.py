# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 20:31:37 2025

@author: zsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as mpatches


csv_file = r"D:\My_WorkSpace\毕业课题\学位论文\结果\human\不同sklearn模型-5folds.csv"
df = pd.read_csv(csv_file)


df["mean"] = df[["fold1", "fold2", "fold3", "fold4", "fold5"]].mean(axis=1)

del_data = df[df['cnvtype'] == 'DEL']
dup_data = df[df['cnvtype'] == 'DUP']

def F1_values(grid_x, grid_y):
    return 2 * (grid_x * grid_y) / (grid_x + grid_y + 1e-6)

grid_x, grid_y = np.meshgrid(np.linspace(0, 1.05, 100), 
                              np.linspace(0, 1.05, 100))

colors = {'LSTM': '#AA0000', 'RandomForest': '#CC6600', 'XGBoost': '#88AA00', 'KNN': '#0088A8', 'Decision Tree': '#990099', 'AdaBoost':'#330066'}

fig, ax = plt.subplots(figsize=(6, 6))
for model in df['model'].unique():
    model_data = del_data[del_data['model'] == model]
    ax.scatter(model_data[model_data['ratetype'] == 'Recall']['meanvalue'].values, 
               model_data[model_data['ratetype'] == 'Precision']['meanvalue'].values, 
               color=colors[model], 
               marker='v', 
               s=250,
               edgecolor='grey',
               linewidth=1.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.05)

for model in df['model'].unique():
    model_data = dup_data[dup_data['model'] == model]
    ax.scatter(model_data[model_data['ratetype'] == 'Recall']['meanvalue'].values, 
               model_data[model_data['ratetype'] == 'Precision']['meanvalue'].values, 
               color=colors[model], 
               marker='o', 
               s=250,
               edgecolor='grey',
               linewidth=1.5)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 1.05)
    
    contour = ax.contour(grid_x, grid_y, F1_values(grid_x,grid_y), 10, linestyles='--', colors='grey')
    # plt.clabel(contour, inline=True, fontsize=8, colors='black')

ax.set_xticks(np.linspace(0, 1, 6))
ax.set_yticks(np.linspace(0, 1, 6))  
ax.set_xlabel("Recall", fontsize=18, fontweight='bold')
ax.set_ylabel("Precision", fontsize=18, fontweight='bold')

line1 = lines.Line2D([0],[0], label='DUP', marker='v', linestyle='None', markerfacecolor='white', markeredgecolor='black', markersize=14)
line2 = lines.Line2D([0],[0], label='DEL', marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='black', markersize=14)

patch1 = mpatches.Patch(color='#AA0000', label='LSTM')
patch2 = mpatches.Patch(color='#CC6600', label='RandomForest')
patch3 = mpatches.Patch(color='#88AA00', label='XGBoost')
patch4 = mpatches.Patch(color='#0088A8', label='KNN')
patch5 = mpatches.Patch(color='#990099', label='DT')
patch6 = mpatches.Patch(color='#330066', label='AdaBoost')


handles1 = [patch1, patch2, patch3, patch4, patch5, patch6]
handles2 = [line1, line2]

fig.legend(handles=handles1, ncol=7, frameon=False, borderaxespad=1.5, fontsize=12)
fig.legend(handles=handles2, ncol=2, frameon=True, borderaxespad=1, loc='upper left', fontsize=13)

plt.grid(False)

# plt.tight_layout()
plt.show()
plt.savefig(r'D:\My_WorkSpace\毕业课题\学位论文\图片\与传统机器学习模型比较\human\f1_sklearn_comparison_new2.tiff', dpi=300)


    