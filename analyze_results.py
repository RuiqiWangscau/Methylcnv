#!/usr/bin/env python
"""Analyze saved sequence-label predictions without importing or training a model.

Metrics are classification metrics, not CNV event overlap metrics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

CLASSES = ['Normal', 'DEL', 'DUP']


def analyze_file(source, output, plots=True):
    with np.load(source, allow_pickle=False) as data:
        truth = np.asarray(data['labels'])
        pred = np.asarray(data['predictions'])
        if truth.ndim != 1 or pred.ndim != 1 or truth.shape != pred.shape or not len(truth):
            raise ValueError(f'{source}: expected nonempty matching 1-D labels and predictions')
        if not np.isin(truth, [0, 1, 2]).all() or not np.isin(pred, [0, 1, 2]).all():
            raise ValueError(f'{source}: labels must be 0, 1 or 2')
        cm = np.zeros((3, 3), dtype=np.int64)
        np.add.at(cm, (truth.astype(int), pred.astype(int)), 1)
        attention = np.asarray(data['feature_attention']) if 'feature_attention' in data else None
        names = np.asarray(data['feature_names']).astype(str) if 'feature_names' in data else None
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, name in enumerate(CLASSES):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        rows.append(dict(class_name=name, TP=tp, FP=fp, FN=fn, support=int(cm[i].sum()),
                         precision=precision, recall=recall, f1=f1,
                         precision_defined=bool(tp + fp), recall_defined=bool(tp + fn)))
    table = pd.DataFrame(rows)
    summary = {'source': str(source.resolve()), 'n_sequences': len(truth),
               'evaluation_unit': 'sequence_label', 'accuracy': float(np.trace(cm) / cm.sum()),
               'undefined_metric_policy': 'zero, with definition flags in per_class_metrics.tsv'}
    for metric in ['precision', 'recall', 'f1']:
        summary['macro_' + metric] = float(table[metric].mean())
        summary['weighted_' + metric] = float(np.average(table[metric], weights=table.support))
    table.to_csv(output / 'per_class_metrics.tsv', sep='\t', index=False)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(output / 'confusion_matrix.tsv', sep='\t', index_label='true_label')
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if not plots:
        print(f'{source}: accuracy={summary["accuracy"]:.4f}, weighted F1={summary["weighted_f1"]:.4f}')
        return summary
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted label', ylabel='True label', title='Confusion Matrix')
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.tight_layout()
    fig.savefig(output / 'confusion_matrix.png', dpi=200)
    plt.close(fig)
    if attention is not None:
        if attention.ndim != 2 or attention.shape[0] != len(truth) or not np.isfinite(attention).all():
            raise ValueError('feature_attention must have shape (n_sequences, n_features) with finite values')
        mean = attention.mean(axis=0)
        if names is None:
            names = np.array([f'Feature {i}' for i in range(len(mean))])
        if names.ndim != 1 or len(names) != len(mean):
            raise ValueError('feature_names does not match feature_attention')
        pd.DataFrame({'feature': names, 'mean_attention': mean}).to_csv(output / 'feature_attention.tsv', sep='\t', index=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(range(len(mean)), mean)
        ax.set(xticks=range(len(mean)), xticklabels=names, ylabel='Mean attention weight', title='Feature Attention')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        fig.savefig(output / 'feature_attention.png', dpi=200)
        plt.close(fig)
    print(f'{source}: accuracy={summary["accuracy"]:.4f}, weighted F1={summary["weighted_f1"]:.4f}')
    return summary


def plot_history(path, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    history = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['loss', 'f1']):
        for split in ['train', 'val']:
            column = f'{split}_{metric}'
            ax.plot(history['epoch'], history[column], label=split)
        ax.set(xlabel='Epoch', ylabel=metric.upper(), title=f'Training and validation {metric}')
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / 'training_curves.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--results', type=Path, help='Saved test_results.npz, test_predictions.npz or val_predictions.npz')
    group.add_argument('--cv_dir', type=Path, help='Completed five-fold output directory')
    parser.add_argument('--history', type=Path, help='Optional training_history.csv for --results')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--no_plots', action='store_true', help='Calculate metrics without requiring matplotlib')
    args = parser.parse_args()
    if args.cv_dir and args.history:
        parser.error('--history is only supported with --results')
    if args.results:
        analyze_file(args.results, args.output, plots=not args.no_plots)
        if args.history and not args.no_plots:
            plot_history(args.history, args.output)
    else:
        files = [args.cv_dir / f'fold_{i}' / 'val_predictions.npz' for i in range(1, 6)]
        missing = [str(p) for p in files if not p.is_file()]
        if missing:
            raise FileNotFoundError('All five completed fold prediction files are required: ' + ', '.join(missing))
        rows = []
        for i, path in enumerate(files, 1):
            summary = analyze_file(path, args.output / f'fold_{i}', plots=not args.no_plots)
            rows.append({'fold': i, **{k: v for k, v in summary.items() if k in ['accuracy', 'macro_f1', 'weighted_f1', 'macro_precision', 'macro_recall', 'weighted_precision', 'weighted_recall']}})
        table = pd.DataFrame(rows).set_index('fold')
        table.to_csv(args.output / 'fold_metrics.tsv', sep='\t')
        table.agg(['mean', 'std']).to_csv(args.output / 'cv_mean_std.tsv', sep='\t', index_label='statistic')
        (args.output / 'cv_note.txt').write_text('Fold predictions use validation-selected checkpoints. CV scores are model-selection results; use the independent holdout for final evaluation. Standard deviation uses ddof=1.\n', encoding='utf-8')
        for filename, folder in [('test_predictions.npz', 'independent_test'), ('oof_predictions.npz', 'oof')]:
            path = args.cv_dir / filename
            if path.is_file():
                analyze_file(path, args.output / folder, plots=not args.no_plots)
            else:
                print(f'Optional result not found; skipped: {path}')


if __name__ == '__main__':
    main()
