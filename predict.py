#!/usr/bin/env python
"""Smoke-test or predict with the supplied feature BiLSTM, without training."""
import argparse
import json
import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


def canonical(name):
    name = name[3:] if name.startswith('chr') else name
    return 'MT' if name == 'M' else name


class ReferenceAdapter:
    """Resolve canonical names while preserving the underlying FASTA sequences."""
    def __init__(self, source):
        self.source = source
        self.names = {}
        for name in source.references:
            key = canonical(name)
            if key in self.names:
                raise ValueError('Duplicate canonical reference name')
            self.names[key] = name

    def fetch(self, name, *args, **kwargs):
        return self.source.fetch(self.names[canonical(name)], *args, **kwargs)

    def get_reference_length(self, name):
        return self.source.get_reference_length(self.names[canonical(name)])

    def close(self):
        self.source.close()


def window_starts(n, length, stride):
    if n < length:
        raise ValueError('Region has fewer bins than the sequence length')
    starts = list(range(0, n-length+1, stride))
    if starts[-1] != n-length:
        starts.append(n-length)
    return starts


def average_window_probabilities(n, length, starts, batches):
    total = np.zeros((n, 3), dtype=np.float64)
    counts = np.zeros(n, dtype=np.int64)
    for start, probabilities in zip(starts, batches):
        total[start:start+length] += probabilities
        counts[start:start+length] += 1
    if (counts == 0).any():
        raise ValueError('Uncovered bins; use stride <= sequence length')
    return total/counts[:, None], counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bam', type=Path, required=True)
    parser.add_argument('--meth', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--model', type=Path, default=Path(__file__).parent/'final_model.pt')
    parser.add_argument('--scaler', type=Path, default=Path(__file__).parent/'feature_scaler.pkl')
    parser.add_argument('--feature_names', type=Path, default=Path(__file__).parent/'feature_names.npy')
    parser.add_argument('--region', required=True, help='One 0-based half-open region, e.g. chr1:0-1000000')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bin_size', type=int, default=100)
    parser.add_argument('--seq_length', type=int, default=100)
    parser.add_argument('--stride', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if min(args.bin_size, args.seq_length, args.stride, args.batch_size, args.threads) < 1 or args.stride > args.seq_length:
        parser.error('Sizes must be positive, with stride <= seq_length')
    if args.output.exists():
        raise FileExistsError('Use a new output directory')
    import pysam
    import torch
    from model import BiLSTM_CNV
    from Depth_Methy_Features import DepthCalculator, MethylationProcessor

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    match = re.fullmatch(r'([^:]+):(\d+)-(\d+)', args.region)
    if not match:
        parser.error('--region must be chrom:start-end')
    chrom, left, right = match.groups()
    chrom, left, right = canonical(chrom), int(left), int(right)
    if left >= right:
        raise ValueError('Invalid region')
    names = np.load(args.feature_names, allow_pickle=False).astype(str).tolist()
    expected = ['rd_u', 'gc_content', 'methyl_penalty_norm', 'meth_density', 'meth_variance']
    if names != expected:
        raise ValueError('Saved feature order does not match this feature extractor')
    # Load only artifacts you trust; the scaler is a pickled Python object.
    with args.scaler.open('rb') as handle:
        scaler = pickle.load(handle)
    if scaler.n_features_in_ != len(names):
        raise ValueError('Scaler feature count mismatch')
    if hasattr(scaler, 'feature_names_in_') and list(scaler.feature_names_in_) != names:
        raise ValueError('Scaler feature order mismatch')
    state = torch.load(args.model, map_location='cpu', weights_only=True)
    if tuple(state['lstm.weight_ih_l0'].shape) != (512, 5):
        raise ValueError('Expected the supplied feature BiLSTM checkpoint')
    model = BiLSTM_CNV(5, 128, 2, 3, 0.3)
    model.load_state_dict(state, strict=True)
    torch.set_num_threads(args.threads)
    model.eval()
    with pysam.AlignmentFile(str(args.bam), 'rb') as bam, pysam.FastaFile(str(args.reference)) as fasta:
        if not bam.has_index():
            raise ValueError('BAM index is required')
        bam_names = {canonical(name): name for name in bam.references}
        ref_names = {canonical(name): name for name in fasta.references}
        if chrom not in bam_names or chrom not in ref_names:
            raise ValueError('Target chromosome is absent from BAM or FASTA')
        bam_length = bam.get_reference_length(bam_names[chrom])
        ref_length = fasta.get_reference_length(ref_names[chrom])
        if bam_length != ref_length or right > ref_length:
            raise ValueError('BAM/FASTA lengths or target coordinates are incompatible')
    # Original processor requires six named header columns and percentages in 0-100.
    processor = MethylationProcessor(str(args.meth), str(args.reference), args.bin_size, (left, right))
    required = ['Chr', 'Start', 'End', 'MethPercent', 'MethCount', 'UnmethCount']
    if not set(required) <= set(processor.meth_data.columns):
        raise ValueError('Methylation BED requires header: ' + ' '.join(required))
    processor.meth_data['Chr'] = processor.meth_data['Chr'].astype(str).map(canonical)
    for col in required[1:]:
        processor.meth_data[col] = pd.to_numeric(processor.meth_data[col], errors='raise')
    if not processor.meth_data.MethPercent.between(0, 100).all():
        raise ValueError('MethPercent must be 0-100')
    processor.meth_data = processor.meth_data.loc[
        processor.meth_data.Chr.eq(chrom) & processor.meth_data.Start.ge(left)
        & processor.meth_data.End.le(right)].copy()
    if processor.meth_data.empty:
        raise ValueError('No methylation data in this region; check names, coordinates and sample')
    if processor.fasta:
        processor.fasta = ReferenceAdapter(processor.fasta)
    depth = None
    try:
        depth = DepthCalculator(str(args.bam), str(args.reference), bin_size=args.bin_size,
                                region=(left, right), methylation_processor=processor)
        depth.bam_chrom_names = {canonical(name): name for name in depth.file.references}
        if len(depth.bam_chrom_names) != len(depth.file.references):
            raise ValueError('Duplicate canonical BAM names')
        # Adapt fetch names without changing original depth/penalty calculations.
        class AlignmentAdapter:
            def __init__(self, source, names):
                self.source, self.names = source, names

            def fetch(self, name, *a, **kw):
                return self.source.fetch(self.names[canonical(name)], *a, **kw)

            def close(self):
                self.source.close()

        depth.references = tuple(depth.bam_chrom_names)
        depth.chrom_lengths = dict(zip(depth.references, depth.lengths))
        depth.file = AlignmentAdapter(depth.file, depth.bam_chrom_names)
        if depth.fasta:
            depth.fasta = ReferenceAdapter(depth.fasta)
        d = depth.process_region(chrom)
        m = processor.process_chromosome(chrom)
        if d is None or m is None:
            raise ValueError('Feature extraction failed')
        features = pd.DataFrame({name: d[name] if name in d else m[name] for name in names})
    finally:
        if depth is not None:
            depth.close()
        if processor.fasta:
            processor.fasta.close()
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError('Non-finite extracted features')
    values = scaler.transform(features if hasattr(scaler, 'feature_names_in_') else features.to_numpy()).astype(np.float32)
    starts = window_starts(len(values), args.seq_length, args.stride)
    sums = np.zeros((len(values), 3), dtype=np.float64)
    counts = np.zeros(len(values), dtype=np.int64)
    with torch.inference_mode():
        for offset in range(0, len(starts), args.batch_size):
            batch_starts = starts[offset:offset+args.batch_size]
            x = torch.from_numpy(np.stack([values[start:start+args.seq_length] for start in batch_starts]))
            probs = torch.softmax(model(x), dim=2).numpy()
            for start, probabilities in zip(batch_starts, probs):
                sums[start:start+args.seq_length] += probabilities
                counts[start:start+args.seq_length] += 1
    if (counts == 0).any():
        raise ValueError('Unexpected uncovered bins')
    probs = sums/counts[:, None]
    labels = probs.argmax(axis=1)
    positions = left + np.arange(len(values))*args.bin_size
    output = pd.DataFrame({'chrom': bam_names[chrom], 'start': positions,
                           'end': np.minimum(positions+args.bin_size, right), 'prediction_id': labels,
                           'prediction': np.asarray(['Normal', 'DEL', 'DUP'])[labels],
                           'prob_Normal': probs[:, 0], 'prob_DEL': probs[:, 1], 'prob_DUP': probs[:, 2],
                           'n_windows': counts})
    args.output.mkdir(parents=True)
    output.to_csv(args.output/'bin_predictions.tsv', sep='\t', index=False)
    features.to_csv(args.output/'extracted_features.tsv', sep='\t', index=False)
    (args.output/'protocol.json').write_text(json.dumps({
        'bam': str(args.bam), 'methylation': str(args.meth), 'reference': str(args.reference),
        'model': str(args.model), 'features': names, 'region': args.region, 'bin_size': args.bin_size,
        'sequence_length': args.seq_length, 'stride': args.stride,
        'class_mapping': {'0': 'Normal', '1': 'DEL', '2': 'DUP'},
        'aggregation': 'mean softmax probabilities across overlapping windows; final tail window included',
        'note': 'feature archive model; not the 29-feature epoch80 checkpoint. Aggregation differs from the original overwrite/smoothing logic; deployment behavior requires validation.'
    }, indent=2), encoding='utf-8')
    print(f'Completed without training: {args.output}')


if __name__ == '__main__':
    main()
