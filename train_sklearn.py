#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train five classifiers using the same features and region split as MethylCNV.
Output one region-level prediction TSV per model for subsequent evaluation.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from collections import Counter

NPZ = '/public/home/wangruiqi/bam/cnv_benchmark/train_data_human_bin100_50only.npz'
REGIONS_TSV = '/public/home/wangruiqi/bam/cnv_benchmark/regions.tsv'
OUT = '/public/home/wangruiqi/bam/cnv_benchmark/sklearn_result'
os.makedirs(OUT, exist_ok=True)

FEAT_COLS = ['rd_u', 'gc_content', 'methyl_penalty',
             'meth_density', 'meth_variance']
LABEL_MAP = {0: 'Normal', 1: 'DEL', 2: 'DUP'}

# ========== Data preparation (same procedure as the MethylCNV script) ==========
print("[1/3] Load & prepare ...")
data = np.load(NPZ, allow_pickle=True)
X = pd.DataFrame(data['features'], columns=data['feature_names'])
y = np.asarray(data['labels'])
X['chrom'] = X['chrom'].astype(str).apply(
    lambda c: c if c.startswith('chr') else f"chr{c}")
X['bin_start'] = X['bin_start'].astype(int)
X['label'] = y

df = X[['chrom', 'bin_start', 'label'] + FEAT_COLS].copy()
df = df.groupby(['chrom', 'bin_start'], as_index=False).agg(
    {**{c: 'mean' for c in FEAT_COLS},
     'label': lambda s: s.value_counts().idxmax()}
)

regions = pd.read_csv(REGIONS_TSV, sep='\t')
df['region_id'] = None
for chrom, sub_x in df.groupby('chrom'):
    sub_r = regions[regions['chrom'] == chrom].sort_values('start').reset_index(drop=True)
    if len(sub_r) == 0:
        continue
    starts = sub_r['start'].values
    ends = sub_r['end'].values
    rids = sub_r['region_id'].values
    positions = sub_x['bin_start'].values
    idx = np.searchsorted(starts, positions, side='right') - 1
    idx_clip = np.maximum(idx, 0)
    valid = (idx >= 0) & (positions < ends[idx_clip])
    df.loc[sub_x.index, 'region_id'] = np.where(valid, rids[idx_clip], None)
df = df.dropna(subset=['region_id']).reset_index(drop=True)

# ========== Region split (same seed and proportion as MethylCNV) ==========
region_ids = df['region_id'].unique()
np.random.seed(42)
np.random.shuffle(region_ids)
n_test = int(len(region_ids) * 0.2)
test_regions = set(region_ids[:n_test])
train_regions = set(region_ids[n_test:])

train_mask = df['region_id'].isin(train_regions).values
test_mask = df['region_id'].isin(test_regions).values

scaler = StandardScaler()
X_train = scaler.fit_transform(df.loc[train_mask, FEAT_COLS].values)
X_test  = scaler.transform(df.loc[test_mask, FEAT_COLS].values)
y_train = df.loc[train_mask, 'label'].values
y_test  = df.loc[test_mask, 'label'].values
region_test = df.loc[test_mask, 'region_id'].values

print(f"  train bins: {train_mask.sum()}, test bins: {test_mask.sum()}")

# ========== Five classifiers ==========
MODELS = {
    'KNN':      KNeighborsClassifier(n_neighbors=5, algorithm='kd_tree', leaf_size=40, n_jobs=-1),
    'RF':       RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    'XGBoost':  XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                          n_jobs=-1, random_state=42),
    'DT':       DecisionTreeClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42),
}

print("[2/3] Train & evaluate each model ...")
summary = []
for name, clf in MODELS.items():
    print(f"\n--- {name} ---")
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    bin_f1 = f1_score(y_test, preds, average='weighted')
    print(f"  bin-level weighted F1: {bin_f1:.4f}")

    # Aggregate bin predictions into region-level predictions.
    df_pred = pd.DataFrame({'region_id': region_test, 'pred': preds})
    region_pred = df_pred.groupby('region_id')['pred'].agg(
        lambda x: Counter(x).most_common(1)[0][0]
    ).reset_index()
    region_pred[f'{name}_pred'] = region_pred['pred'].map(LABEL_MAP)
    region_pred = region_pred[['region_id', f'{name}_pred']]
    out_tsv = os.path.join(OUT, f'sklearn_{name}_predictions.tsv')
    region_pred.to_csv(out_tsv, sep='\t', index=False)
    print(f"  -> {out_tsv} ({len(region_pred)} regions)")

    summary.append((name, bin_f1))

# ========== Save test region IDs for comparison with the MethylCNV split ==========
with open(os.path.join(OUT, 'test_regions_sklearn.txt'), 'w') as f:
    for r in sorted(test_regions):
        f.write(f"{r}\n")

print("\n[3/3] Summary")
for name, f1 in summary:
    print(f"  {name:10s} bin-F1={f1:.4f}")
