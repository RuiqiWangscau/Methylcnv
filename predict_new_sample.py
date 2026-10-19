#!/usr/bin/env python
"""Export fixed legacy preprocessing, or predict a new unlabeled 29-feature sample.

Requires predict_epoch80.py for the exact legacy CNNLSTM architecture.
All config, scaler and checkpoint files must be trusted.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_features(path):
    with np.load(path, allow_pickle=True) as data:
        frame = pd.DataFrame(data['features'], columns=data['feature_names'])
    if frame.columns.duplicated().any() or frame.empty:
        raise ValueError('Duplicate feature names or empty input')
    return frame


def load_config(path):
    with np.load(path, allow_pickle=True) as c:
        config = dict(features=np.asarray(c['features_to_use']).astype(str).tolist(),
                      length=int(c['sequence_length'].item()), overlap=int(c['overlap'].item()),
                      input_dim=int(c['input_dim'].item()), classes=int(c['num_classes'].item()))
    if config['input_dim'] != 29 or len(config['features']) != 29 or len(set(config['features'])) != 29 or config['classes'] != 3:
        raise ValueError('This checkpoint requires 29 ordered features and 3 classes')
    if not 0 <= config['overlap'] < config['length']:
        raise ValueError('Invalid sequence settings')
    return config


def export_preprocessing(args):
    import joblib
    config = load_config(args.config)
    frame = read_features(args.training_data)
    names = config['features']
    actual = [c for c in frame if c not in ['chrom', 'bin_start', 'sample_id']]
    if actual != names:
        raise ValueError('Original training feature order differs from configuration')
    x = frame[names].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    rules = {}
    for name in names:
        col = x[name]
        if col.isna().all():
            # Original pipeline replaces an entirely missing training column by zero.
            rules[name] = {'lower': None, 'upper': None, 'fill': 0.0, 'training_all_missing': True}
            x[name] = 0.0
            continue
        q1, q99 = col.quantile(.01), col.quantile(.99)
        lower, upper = float(q1-3*(q99-q1)), float(q99+3*(q99-q1))
        clipped = col.clip(lower, upper)
        fill = float(clipped.mean())
        rules[name] = {'lower': lower, 'upper': upper, 'fill': fill, 'training_all_missing': False}
        x[name] = clipped.fillna(fill)
    scaler = joblib.load(args.scaler)
    if scaler.n_features_in_ != 29 or (hasattr(scaler, 'feature_names_in_') and list(scaler.feature_names_in_) != names):
        raise ValueError('Scaler feature metadata mismatch')
    if not np.allclose(x.mean().to_numpy(), scaler.mean_, rtol=1e-5, atol=1e-7) or not np.allclose(x.var(ddof=0).to_numpy(), scaler.var_, rtol=1e-5, atol=1e-7):
        raise ValueError('Reconstructed cleaning statistics do not match the saved scaler; verify original data and preprocessing')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'features': names, 'rules': rules,
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'scaler_sha256': hashlib.sha256(args.scaler.read_bytes()).hexdigest(),
        'source': 'Recovered from original full training input, checked against saved scaler; no new model or scaler fitted.'}, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Exported fixed training preprocessing: {args.output}')


def apply_rules(frame, config, preprocessing):
    names = config['features']
    if preprocessing['features'] != names:
        raise ValueError('Preprocessing feature order mismatch')
    missing = set(names)-set(frame.columns)
    if missing:
        raise ValueError(f'Missing features: {sorted(missing)}. Do not pad or substitute features.')
    x = frame[names].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    for name in names:
        rule = preprocessing['rules'][name]
        if rule['training_all_missing']:
            # Fixed constant transform for columns that had no information in training.
            x[name] = 0.0
        else:
            x[name] = x[name].clip(rule['lower'], rule['upper']).fillna(rule['fill'])
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Non-finite values after fixed preprocessing')
    return x


def create_windows(frame, length, overlap, bin_size):
    required = {'chrom', 'bin_start'}
    if not required <= set(frame):
        raise ValueError('Input must contain chrom and bin_start')
    pos = pd.to_numeric(frame.bin_start, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(pos).all() or (pos < 0).any() or (pos % 1 != 0).any():
        raise ValueError('Invalid bin_start coordinates')
    keys = pd.DataFrame({'chrom': frame.chrom.astype(str),
                         'sample': frame.sample_id.astype(str) if 'sample_id' in frame else 'sample'})
    if 'region_id' in frame:
        keys['region'] = frame.region_id.astype(str)
    windows = []
    for _, rows in keys.groupby(list(keys.columns), sort=False, dropna=False):
        indices = rows.index.to_numpy()
        indices = indices[np.argsort(pos[indices], kind='stable')]
        if (np.diff(pos[indices]) == 0).any():
            raise ValueError('Duplicate bin coordinates within a sample/chromosome/region group')
        edges = np.r_[0, np.flatnonzero(np.diff(pos[indices]) != bin_size)+1, len(indices)]
        for left, right in zip(edges[:-1], edges[1:]):
            for start in range(int(left), int(right)-length+1, length-overlap):
                windows.append(indices[start:start+length])
    if not windows:
        raise ValueError('No continuous region long enough for the saved sequence length')
    return np.asarray(windows, dtype=np.int64), pos.astype(np.int64)


def predict(args):
    import joblib
    import torch
    from predict_epoch80 import CNNLSTM
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    config = load_config(args.config)
    preprocessing = json.loads(args.preprocessing.read_text(encoding='utf-8'))
    for path, key in [(args.config, 'config_sha256'), (args.scaler, 'scaler_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != preprocessing[key]:
            raise ValueError(f'{path} differs from the exported preprocessing bundle')
    frame = read_features(args.data)
    numeric = apply_rules(frame, config, preprocessing)
    scaler = joblib.load(args.scaler)
    if scaler.n_features_in_ != 29 or (hasattr(scaler, 'feature_names_in_') and list(scaler.feature_names_in_) != config['features']):
        raise ValueError('Scaler feature order mismatch')
    values = scaler.transform(numeric if hasattr(scaler, 'feature_names_in_') else numeric.to_numpy()).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite standardized values')
    windows, positions = create_windows(frame, config['length'], config['overlap'], args.bin_size)
    torch.set_num_threads(args.threads)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = CNNLSTM(29, 3)
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    batches = []
    with torch.inference_mode():
        for start in range(0, len(windows), args.batch_size):
            batch = torch.from_numpy(values[windows[start:start+args.batch_size]]).to(device)
            batches.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    probabilities = np.concatenate(batches)
    predictions = probabilities.argmax(axis=1)
    last = windows[:, -1]
    result = pd.DataFrame({'window_id': np.arange(len(windows)),
        'sample_id': frame.iloc[last].sample_id.to_numpy() if 'sample_id' in frame else 'sample',
        'chrom': frame.iloc[last].chrom.to_numpy(),
        'window_start': positions[windows[:, 0]], 'window_end': positions[last]+args.bin_size,
        'target_bin_start': positions[last], 'target_bin_end': positions[last]+args.bin_size,
        'prediction_id': predictions, 'prob_class0': probabilities[:, 0],
        'prob_class1': probabilities[:, 1], 'prob_class2': probabilities[:, 2]})
    if args.class_names:
        result['prediction'] = np.asarray(args.class_names)[predictions]
    args.output.mkdir(parents=True)
    result.to_csv(args.output/'predictions.tsv', sep='\t', index=False)
    np.savez_compressed(args.output/'predictions.npz', predictions=predictions, probabilities=probabilities)
    (args.output/'protocol.json').write_text(json.dumps({
        'features': config['features'], 'sequence_length': config['length'], 'overlap': config['overlap'],
        'bin_size': args.bin_size, 'class_names': args.class_names,
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'data': str(args.data), 'n_windows': len(windows),
        'note': 'Unlabeled new-sample inference, no training or scaler fitting. Last-bin predictions, not inferred CNV event boundaries. Feature engineering must be training-compatible and independent of reference truth.'
    }, indent=2), encoding='utf-8')
    print(f'Predicted {len(windows)} windows without training: {args.output}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export-preprocessing', help='Publisher-only step using exact original training data')
    export.add_argument('--training_data', type=Path, required=True)
    infer = sub.add_parser('predict', help='New user: supply unlabeled, raw training-compatible features')
    infer.add_argument('--data', type=Path, required=True)
    infer.add_argument('--checkpoint', type=Path, required=True)
    infer.add_argument('--preprocessing', type=Path, required=True)
    infer.add_argument('--bin_size', type=int, required=True, help='Original training feature bin size in bp; verify from feature-generation code')
    infer.add_argument('--class_names', nargs=3, help='Verified class names in numeric ID order')
    infer.add_argument('--batch_size', type=int, default=128)
    infer.add_argument('--threads', type=int, default=16)
    infer.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    for command in [export, infer]:
        command.add_argument('--config', type=Path, required=True)
        command.add_argument('--scaler', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'predict':
        if min(args.bin_size, args.batch_size, args.threads) < 1:
            p.error('Sizes and thread count must be positive')
        predict(args)
    else:
        export_preprocessing(args)


if __name__ == '__main__':
    main()
