# -*- coding: utf-8 -*-
"""Region-grouped 90/10 holdout and five-fold development CV.

Metrics here describe sequence-center classification, not BED event detection.
Requires train_cnv_lstm.py in the same directory.
"""
import argparse
import importlib
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


def canonical(value):
    value = str(value)
    return value[3:] if value.startswith('chr') else value


def reconstruct_regions(frame, bed_path, context=10000, bin_size=100):
    """Reconstruct merged near-CNV extraction intervals, fail on unmatched bins."""
    bed = pd.read_csv(bed_path, sep='\t', dtype={'Chr': str})
    for name in ['Chr', 'Start', 'End', 'Type']:
        if name not in bed:
            raise ValueError(f'CNV BED lacks column: {name}')
    bed['Chr'] = bed['Chr'].map(canonical)
    groups = np.full(len(frame), -1, dtype=np.int64)
    records = []
    for chrom in sorted(frame['chrom'].unique()):
        intervals = sorted((max(0, int(r.Start) - context), int(r.End) + context)
                           for r in bed.loc[bed.Chr == chrom].itertuples())
        merged = []
        for left, right in intervals:
            if right <= left:
                raise ValueError('Invalid CNV interval')
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(right, merged[-1][1])
            else:
                merged.append([left, right])
        rows = np.flatnonzero(frame.chrom.to_numpy() == chrom)
        positions = frame.iloc[rows].bin_start.to_numpy(dtype=np.int64)
        for left, right in merged:
            selected = rows[(positions >= left) & (positions < right)]
            if not len(selected):
                continue
            gid = len(records)
            if (groups[selected] != -1).any():
                raise ValueError('A bin belongs to multiple regions')
            # Original extraction starts each bin grid at the interval start.
            if ((frame.iloc[selected].bin_start.to_numpy() - left) % bin_size != 0).any():
                raise ValueError('Bin grid does not match reconstructed extraction interval')
            groups[selected] = gid
            records.append({'region_id': gid, 'chrom': chrom, 'start': left,
                            'end': right, 'n_bins': len(selected)})
    if (groups < 0).any():
        raise ValueError(f'{int((groups < 0).sum())} bins are outside reconstructed near-CNV regions; '
                         'check context/BED or provide explicit region_id in NPZ')
    return groups, records


def make_windows(frame, groups, labels, length, stride, bin_size):
    windows, targets, window_groups = [], [], []
    for gid in np.unique(groups):
        indices = np.flatnonzero(groups == gid)
        indices = indices[np.argsort(frame.iloc[indices].bin_start.to_numpy(), kind='stable')]
        positions = frame.iloc[indices].bin_start.to_numpy(dtype=np.int64)
        if len(np.unique(positions)) != len(positions):
            raise ValueError(f'Duplicate coordinates in region {gid}')
        edges = np.r_[0, np.flatnonzero(np.diff(positions) != bin_size) + 1, len(indices)]
        for left, right in zip(edges[:-1], edges[1:]):
            for offset in range(int(left), int(right) - length + 1, stride):
                window = indices[offset:offset + length]
                windows.append(window)
                targets.append(labels[window[length // 2]])
                window_groups.append(gid)
    if not windows:
        raise ValueError('No continuous regions long enough for the requested sequence length')
    return np.asarray(windows), np.asarray(targets), np.asarray(window_groups)


class WindowDataset(Dataset):
    def __init__(self, values, windows, labels, scaler, augment=False):
        self.values, self.windows, self.labels = values, windows, labels
        self.mean = scaler.mean_.astype(np.float32)
        self.scale = scaler.scale_.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = (self.values[self.windows[index]] - self.mean) / self.scale
        label = int(self.labels[index])
        if self.augment and label in (1, 2) and np.random.random() < 0.5:
            x = x + np.random.normal(0, 0.05, x.shape).astype(np.float32)
        return {'features': torch.from_numpy(x),
                'label': torch.tensor([label], dtype=torch.long)}


def assert_disjoint(groups, a, b):
    if set(groups[a]) & set(groups[b]):
        raise AssertionError('Region leakage between subsets')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', default='/public/home/wangruiqi/bam/new/train_data_sample_bin100.npz')
    parser.add_argument('--cnv_bed', default='/public/home/wangruiqi/bam/reference_cnvs_human.bed')
    parser.add_argument('--save_root', default='/public/home/wangruiqi/bam/new/region_cv_output')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--seq_length', type=int, default=300)
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--bin_size', type=int, default=100)
    parser.add_argument('--context', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=192)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.seq_length, args.stride, args.bin_size, args.patience) < 1:
        parser.error('Epochs, lengths, stride and patience must be positive')
    output = Path(args.save_root)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'split_manifest.npz').exists():
        raise FileExistsError('Output already contains a split manifest; use a new save_root')
    # Existing training module parses CLI on import; isolate its arguments.
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0], '--save_root', str(output)]
        model_code = importlib.import_module('train_cnv_lstm')
    finally:
        sys.argv = saved_argv
    log = logging.getLogger('region_cv')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    with np.load(args.data_path, allow_pickle=True) as data:
        frame = pd.DataFrame(data['features'], columns=data['feature_names'])
        labels = np.asarray(data['labels'], dtype=np.int64)
        explicit_groups = np.asarray(data['region_id']) if 'region_id' in data else None
    frame['chrom'] = frame.chrom.map(canonical)
    frame['bin_start'] = pd.to_numeric(frame.bin_start).astype(np.int64)
    columns = [c for c in frame if c not in ['chrom', 'bin_start', 'sample_id', 'region_id']]
    values = frame[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    if len(frame) != len(labels) or not np.isin(labels, [0, 1, 2]).all() or not np.isfinite(values).all():
        raise ValueError('Invalid labels, row counts or non-finite features')
    if explicit_groups is None:
        groups, regions = reconstruct_regions(frame, args.cnv_bed, args.context, args.bin_size)
    else:
        groups = explicit_groups
        if len(groups) != len(frame):
            raise ValueError('region_id length does not match features')
        regions = []
    windows, targets, wg = make_windows(frame, groups, labels, args.seq_length, args.stride, args.bin_size)
    # Fixed first fold of a 10-fold stratified grouped split gives ~10% holdout.
    outer = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=args.seed)
    development, holdout = next(outer.split(np.zeros(len(targets)), targets, wg))
    assert_disjoint(wg, development, holdout)
    inner = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + 1)
    folds = list(inner.split(np.zeros(len(development)), targets[development], wg[development]))
    manifest = {'windows': windows, 'targets': targets, 'window_groups': wg,
                'development': development, 'test': holdout}
    for i, (tr, va) in enumerate(folds, 1):
        manifest[f'fold{i}_train'] = development[tr]
        manifest[f'fold{i}_val'] = development[va]
    np.savez_compressed(output / 'split_manifest.npz', **manifest)
    (output / 'protocol.json').write_text(json.dumps({**vars(args), 'features': columns,
        'region_definition': 'merged CNV +/- context intervals including normal flanks',
        'metric_unit': 'sequence-center classification', 'regions': regions,
        'test_fraction_actual': len(holdout) / len(targets)}, indent=2), encoding='utf-8')
    log.info('Device=%s windows=%d development=%d holdout=%d (%.2f%%)',
             device, len(targets), len(development), len(holdout), 100 * len(holdout)/len(targets))
    model_args = argparse.Namespace(n_features=len(columns), seq_length=args.seq_length,
        bin_rnn_size=128, num_layers=1, bidirectional=True, dropout=0.5)

    def loader(indices, scaler, training=False):
        dataset = WindowDataset(values, windows[indices], targets[indices], scaler, training)
        return DataLoader(dataset, batch_size=args.batch_size, shuffle=training,
                          num_workers=0, pin_memory=device.type == 'cuda')

    def fit_scaler(indices):
        return StandardScaler().fit(values[np.unique(windows[indices])])

    def setup():
        model = model_code.CNVDetector(model_args).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        criterion = model_code.FocalLoss(gamma=2, weight=torch.tensor([1., 3., 1.5], device=device))
        return model, optimizer, criterion

    summaries = []
    best_epochs = []
    oof = np.full(len(targets), -1, dtype=np.int64)
    for fold, (tr, va) in enumerate(folds, 1):
        train_ids, val_ids = development[tr], development[va]
        assert_disjoint(wg, train_ids, val_ids)
        assert_disjoint(wg, train_ids, holdout)
        if (np.bincount(targets[train_ids], minlength=3) == 0).any():
            raise ValueError(f'Fold {fold} training data lacks a class')
        fold_dir = output / f'fold_{fold}'
        fold_dir.mkdir()
        seed_all(args.seed + fold)
        scaler = fit_scaler(train_ids)
        train_loader, val_loader = loader(train_ids, scaler, True), loader(val_ids, scaler)
        model, optimizer, criterion = setup()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
        best, stale, best_epoch, best_summary = -np.inf, 0, 0, None
        for epoch in range(1, args.epochs + 1):
            train_loss, _, _, _, _ = model_code.train(model, train_loader, optimizer, criterion, device)
            val_loss, accuracy, score, _, predictions, truth = model_code.validate(model, val_loader, criterion, device)
            log.info('Fold %d epoch %d/%d train_loss=%.5f val_loss=%.5f val_F1=%.5f',
                     fold, epoch, args.epochs, train_loss, val_loss, score)
            if not np.isfinite(val_loss) or not np.isfinite(train_loss):
                raise ValueError('Non-finite loss')
            if score > best:
                best, stale, best_epoch = score, 0, epoch
                oof[val_ids] = predictions
                best_summary = classification_report(truth, predictions, labels=[0,1,2],
                    target_names=['Normal','DEL','DUP'], output_dict=True, zero_division=0)
                torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler,
                    'features': columns, 'model_args': vars(model_args), 'epoch': epoch}, fold_dir / 'best.pth')
                np.savez_compressed(fold_dir / 'val_predictions.npz', window_ids=val_ids,
                                    predictions=predictions, labels=truth)
            else:
                stale += 1
            scheduler.step(score)
            if stale >= args.patience:
                break
        summaries.append({'fold': fold, 'best_epoch': best_epoch, 'best_val_weighted_f1': float(best),
                          'classification': best_summary})
        best_epochs.append(best_epoch)
        (fold_dir / 'metrics.json').write_text(json.dumps(summaries[-1], indent=2), encoding='utf-8')
        del model, optimizer, train_loader, val_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    final_epochs = max(1, int(np.median(best_epochs)))
    scores = [item['best_val_weighted_f1'] for item in summaries]
    summary = {'folds': summaries, 'cv_weighted_f1_mean': float(np.mean(scores)),
        'cv_weighted_f1_std': float(np.std(scores, ddof=1)), 'final_epochs': final_epochs,
        'note': 'Validation scores are used for checkpoint selection; independent holdout is evaluated only after final training.'}
    np.savez_compressed(output / 'oof_predictions.npz', window_ids=development,
                        predictions=oof[development], labels=targets[development])
    # Refit on all development data using fixed CV-derived epoch count.
    seed_all(args.seed + 100)
    scaler = fit_scaler(development)
    model, optimizer, criterion = setup()
    train_loader = loader(development, scaler, True)
    for epoch in range(1, final_epochs + 1):
        log.info('Final refit epoch %d/%d', epoch, final_epochs)
        model_code.train(model, train_loader, optimizer, criterion, device)
    torch.save({'model_state_dict': model.state_dict(), 'scaler': scaler, 'features': columns,
        'model_args': vars(model_args), 'epoch': final_epochs}, output / 'final_model.pth')
    _, _, _, _, predictions, truth = model_code.validate(model, loader(holdout, scaler), criterion, device)
    summary['independent_test'] = classification_report(truth, predictions, labels=[0,1,2],
        target_names=['Normal','DEL','DUP'], output_dict=True, zero_division=0)
    np.savez_compressed(output / 'test_predictions.npz', window_ids=holdout,
                        predictions=predictions, labels=truth,
                        chrom=frame.iloc[windows[holdout, args.seq_length//2]].chrom.to_numpy(dtype=str),
                        bin_start=frame.iloc[windows[holdout, args.seq_length//2]].bin_start.to_numpy())
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    log.info('Completed five folds and independent holdout evaluation: %s', output)


if __name__ == '__main__':
    main()
