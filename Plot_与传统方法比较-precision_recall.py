# -*- coding: utf-8 -*-
"""
Created on Mon Mar 24 10:42:12 2025

@author: zsh
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as mpatches

csv_file = r"D:\My_WorkSpace\毕业课题\学位论文\结果\与传统方法比较.csv"
df = pd.read_csv(csv_file)

human_data = df[df['species'] == 'human']
pig_data = df[df['species'] == 'pig']
cattle_data = df[df['species'] == 'cattle']

# =============================================================================
# human
# =============================================================================
del_data = human_data[human_data['cnvtype'] == 'DEL']
dup_data = human_data[human_data['cnvtype'] == 'DUP']

def F1_values(grid_x, grid_y):
    return 2 * (grid_x * grid_y) / (grid_x + grid_y + 1e-6)

grid_x, grid_y = np.meshgrid(np.linspace(0, 1.05, 100), 
                              np.linspace(0, 1.05, 100))

colors = {'MethylCNV': '#D32F2F', 'BreakDancer': '#FF9800', 'Pindel': '#4CAF50', 'cn.mops': '#2196F3', 'CNVnator': '#9C27B0', 'DELLY':'#3F51B5', 'CNVkit': '#FF5722'}

fig, ax = plt.subplots(figsize=(6, 5))
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

# line1 = lines.Line2D([0],[0], label='DUP', marker='v', linestyle='None', markerfacecolor='white', markeredgecolor='black', markersize=14)
# line2 = lines.Line2D([0],[0], label='DEL', marker='o', linestyle='None', markerfacecolor='white', markeredgecolor='black', markersize=14)

# patch1 = mpatches.Patch(color='#D32F2F', label='MethylCNV')
# patch2 = mpatches.Patch(color='#FF9800', label='BreakDancer')
# patch3 = mpatches.Patch(color='#4CAF50', label='Pindel')
# patch4 = mpatches.Patch(color='#2196F3', label='cn.mops')
# patch5 = mpatches.Patch(color='#9C27B0', label='CNVnator')
# patch6 = mpatches.Patch(color='#3F51B5', label='DELLY')
# patch7 = mpatches.Patch(color='#FF5722', label='CNVkit')


# handles1 = [patch1, patch2, patch3, patch4, patch5, patch6, patch7]
# handles2 = [line1, line2]

# fig.legend(handles=handles1, ncol=7, frameon=False, borderaxespad=1.5, fontsize=12)
# fig.legend(handles=handles2, ncol=2, frameon=True, borderaxespad=1, loc='upper left', fontsize=13)

plt.grid(False)

plt.tight_layout()
plt.show()
plt.savefig(r'D:\My_WorkSpace\毕业课题\学位论文\图片\与传统方法比较\comparison_human.tiff', dpi=300)


# =============================================================================
# pig
# =============================================================================
del_data = pig_data[pig_data['cnvtype'] == 'DEL']
dup_data = pig_data[pig_data['cnvtype'] == 'DUP']

def F1_values(grid_x, grid_y):
    return 2 * (grid_x * grid_y) / (grid_x + grid_y + 1e-6)

grid_x, grid_y = np.meshgrid(np.linspace(0, 1.05, 100), 
                              np.linspace(0, 1.05, 100))

colors = {'MethylCNV': '#D32F2F', 'BreakDancer': '#FF9800', 'Pindel': '#4CAF50', 'cn.mops': '#2196F3', 'CNVnator': '#9C27B0', 'DELLY':'#3F51B5', 'CNVkit': '#FF5722'}

fig, ax = plt.subplots(figsize=(6, 5))
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

plt.grid(False)

plt.tight_layout()
plt.show()
plt.savefig(r'D:\My_WorkSpace\毕业课题\学位论文\图片\与传统方法比较\comparison_pig.tiff', dpi=300)

# =============================================================================
# cattle
# =============================================================================
del_data = cattle_data[cattle_data['cnvtype'] == 'DEL']
dup_data = cattle_data[cattle_data['cnvtype'] == 'DUP']

def F1_values(grid_x, grid_y):
    return 2 * (grid_x * grid_y) / (grid_x + grid_y + 1e-6)

grid_x, grid_y = np.meshgrid(np.linspace(0, 1.05, 100), 
                              np.linspace(0, 1.05, 100))

colors = {'MethylCNV': '#D32F2F', 'BreakDancer': '#FF9800', 'Pindel': '#4CAF50', 'cn.mops': '#2196F3', 'CNVnator': '#9C27B0', 'DELLY':'#3F51B5', 'CNVkit': '#FF5722'}

fig, ax = plt.subplots(figsize=(6, 5))
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

plt.grid(False)

plt.tight_layout()
plt.show()
plt.savefig(r'D:\My_WorkSpace\毕业课题\学位论文\图片\与传统方法比较\comparison_cattle.tiff', dpi=300)
























