#!/usr/bin/env python3
"""Predict from BAM, methylation table and FASTA using a saved CNVDetector."""
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):

    def __init__(self, hidden_size, level_name='bin'):
        super(Attention, self).__init__()
        self.level_name = level_name
        self.context_vector = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.softmax = nn.Softmax(dim=1)
        nn.init.uniform_(self.context_vector, -0.1, 0.1)

    def forward(self, x):
        attn_scores = torch.matmul(x, self.context_vector)
        attn_weights = self.softmax(attn_scores.squeeze(2))
        weighted_output = torch.bmm(attn_weights.unsqueeze(1), x)
        return (weighted_output.squeeze(1), attn_weights)

class FeatureEncoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bidirectional=True, dropout=0.5):
        super(FeatureEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_directions = 2 if bidirectional else 1
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0)
        self.attention = Attention(hidden_size * self.num_directions, 'bin')

    def forward(self, x):
        outputs, _ = self.lstm(x)
        context, attention = self.attention(outputs)
        global_avg = torch.mean(outputs, dim=1)
        context = context + global_avg
        return (context, attention)

class CNVDetector(nn.Module):

    def __init__(self, args):
        super(CNVDetector, self).__init__()
        self.n_features = args.n_features
        self.seq_length = args.seq_length
        self.hidden_size = args.bin_rnn_size
        self.num_directions = 2 if args.bidirectional else 1
        self.feature_encoders = nn.ModuleList()
        for i in range(self.n_features):
            self.feature_encoders.append(FeatureEncoder(input_size=1, hidden_size=self.hidden_size, num_layers=args.num_layers, bidirectional=args.bidirectional, dropout=args.dropout))
        feature_hidden_size = self.hidden_size * self.num_directions
        self.feature_fusion = nn.LSTM(input_size=feature_hidden_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True, bidirectional=args.bidirectional)
        self.feature_attention = Attention(self.hidden_size * self.num_directions, 'feature')
        classifier_input_size = self.hidden_size * self.num_directions
        self.classifier = nn.Sequential(nn.Linear(classifier_input_size, 128), nn.ReLU(), nn.Dropout(args.dropout), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(args.dropout / 2), nn.Linear(64, 3))

    def forward(self, x):
        batch_size = x.size(0)
        actual_features = x.size(2)
        feature_outputs = []
        bin_attentions = []
        for i in range(min(self.n_features, actual_features)):
            feature = x[:, :, i:i + 1]
            feature_output, bin_attn = self.feature_encoders[i](feature)
            feature_outputs.append(feature_output.unsqueeze(1))
            bin_attentions.append(bin_attn)
        feature_repr = torch.cat(feature_outputs, dim=1)
        feature_interact = torch.bmm(feature_repr.transpose(1, 2), feature_repr)
        feature_interact = F.relu(feature_interact)
        feature_interact = torch.bmm(feature_repr, feature_interact)
        feature_repr = feature_repr + 0.15 * feature_interact
        fused_output, _ = self.feature_fusion(feature_repr)
        context_vector, feature_attn = self.feature_attention(fused_output)
        logits = self.classifier(context_vector)
        bin_attentions_tensor = torch.stack(bin_attentions, dim=1)
        return (logits, feature_attn, bin_attentions_tensor)
import logging
import numpy as np
import pandas as pd
from scipy import stats
import os
import time

class DepthCalculator:

    def __init__(self, filename, reference_filename=None, cpg_island_file=None, bin_size=100, region=None, mismatch_penalty=4, go_penalty=6, ge_penalty=1, unaligned_penalty=999999999, methylation_processor=None):
        import pysam
        self.logger = logging.getLogger('methylcnv.depth')
        self.filename = filename
        self.reference_filename = reference_filename
        self.cpg_island_file = cpg_island_file
        self.bin_size = bin_size
        self.region = region
        if region is None:
            raise ValueError('region cannot be None. Please specify a region to process.')
        self.start, self.end = region
        self.mismatch_penalty = mismatch_penalty
        self.go_penalty = go_penalty
        self.ge_penalty = ge_penalty
        self.unaligned_penalty = unaligned_penalty
        self.methylation_processor = methylation_processor
        self.cpg_sites = {}
        self.cpg_meth_levels = {}
        if self.methylation_processor:
            self._prepare_region_methylation_data()
        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == '.bam':
                self.file = pysam.AlignmentFile(filename, 'rb', reference_filename=reference_filename if reference_filename else None)
            else:
                self.file = pysam.AlignmentFile(filename, 'r', reference_filename=reference_filename if reference_filename else None)
        except Exception as e:
            self.logger.error(f'Unable to open alignment file {filename}: {str(e)}')
            raise
        self.bam_chrom_names = {}
        for bam_chrom in self.file.references:
            canonical = bam_chrom[3:] if bam_chrom.startswith('chr') else bam_chrom
            if canonical == 'M':
                canonical = 'MT'
            if canonical in self.bam_chrom_names:
                raise ValueError(f'Duplicate BAM chromosome name after normalization: {canonical}')
            self.bam_chrom_names[canonical] = bam_chrom
        self.references = tuple(self.bam_chrom_names)
        self.lengths = self.file.lengths
        self.chrom_lengths = dict(zip(self.references, self.lengths))
        if reference_filename and os.path.isfile(reference_filename):
            try:
                self.fasta = pysam.FastaFile(reference_filename)
            except Exception as e:
                self.logger.error(f'Unable to open reference genome: {str(e)}')
                self.fasta = None
        else:
            self.fasta = None
        self.cpg_ids = set()
        if self.fasta and self.cpg_island_file:
            self.load_reference_and_cpg()

    def _prepare_region_methylation_data(self):
        if not self.methylation_processor or not self.region:
            return
        self.logger.info('Preparing methylation data for the current region from MethylationProcessor...')
        start_time = time.time()
        try:
            region_start, region_end = self.region
            chrom = None
            self.cpg_sites = {}
            self.cpg_meth_levels = {}
            self.logger.info(f'Regional methylation data preparation completed in {time.time() - start_time:.2f} seconds')
        except Exception as e:
            self.logger.error(f'Error preparing regional methylation data: {str(e)}')

    def extract_region_cpg_data(self, chr_name):
        if not self.methylation_processor:
            return
        start_time = time.time()
        region_start, region_end = self.region
        filtered_data = self.methylation_processor.meth_data[(self.methylation_processor.meth_data['Chr'] == chr_name) & (self.methylation_processor.meth_data['Start'] >= region_start) & (self.methylation_processor.meth_data['End'] <= region_end)]
        if chr_name not in self.cpg_sites:
            self.cpg_sites[chr_name] = []
            self.cpg_meth_levels[chr_name] = {}
        for _, row in filtered_data.iterrows():
            pos = row['Start']
            meth_percent = row['MethPercent'] / 100.0
            self.cpg_sites[chr_name].append(pos)
            self.cpg_meth_levels[chr_name][pos] = meth_percent
        if chr_name in self.cpg_sites:
            self.cpg_sites[chr_name].sort()
        self.logger.info(f'Extracted methylation data for {chr_name}:{region_start}-{region_end}: {len(self.cpg_sites.get(chr_name, []))} CpG sites in {time.time() - start_time:.2f} seconds')

    def is_cpg_site(self, chrom, position):
        if self.methylation_processor:
            if chrom not in self.cpg_meth_levels:
                return False
            return position in self.cpg_meth_levels[chrom]
        else:
            return (chrom, position) in self.cpg_ids

    def get_methylation_level(self, chrom, position):
        if self.methylation_processor and chrom in self.cpg_meth_levels and (position in self.cpg_meth_levels[chrom]):
            return self.cpg_meth_levels[chrom][position]
        return 0.5

    def load_reference_and_cpg(self):
        self.logger.info('Loading CpG island data and scanning the reference sequence for CG dinucleotides...')
        cpg_island_regions = []
        try:
            with open(self.cpg_island_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    fields = line.strip().split()
                    if len(fields) < 3:
                        continue
                    c_chr = fields[0].replace('chr', '')
                    c_start = int(fields[1])
                    c_end = int(fields[2])
                    cpg_island_regions.append((c_chr, c_start, c_end))
        except Exception as e:
            self.logger.warning(f'Failed to read cpg_island_file: {str(e)}')
        for chrom_name in self.references:
            try:
                seq_length = self.chrom_lengths[chrom_name]
                refseq = self.fasta.fetch(chrom_name, 0, seq_length).upper()
            except Exception as e:
                self.logger.warning(f'Unable to fetch reference sequence {chrom_name}: {str(e)}')
                continue
            for i in range(len(refseq) - 1):
                if refseq[i] == 'C' and refseq[i + 1] == 'G':
                    self.cpg_ids.add((chrom_name, i))
                    self.cpg_ids.add((chrom_name, i + 1))

    def get_chromosome_length(self, chr_name):
        idx = self.references.index(chr_name)
        return self.lengths[idx]

    def read_chromosome(self, chr_name):
        if chr_name not in self.chrom_lengths:
            self.logger.warning(f"Chromosome '{chr_name}' was not found in the alignment file.")
            return (None, None, None)
        if self.region:
            start, end = self.region
            length = end - start
            offset = start
        else:
            length = self.chrom_lengths[chr_name]
            offset = 0
        n = (length + self.bin_size - 1) // self.bin_size
        rd_p = np.zeros(n)
        rd_u = np.zeros(n)
        if self.region:
            fetch_start = max(0, start - 300)
            fetch_end = min(self.chrom_lengths[chr_name], end + 300)
            reads_iter = self.file.fetch(self.bam_chrom_names[chr_name], fetch_start, fetch_end)
        else:
            reads_iter = self.file.fetch(self.bam_chrom_names[chr_name])
        for r in reads_iter:
            if r.is_unmapped or r.is_secondary or r.is_duplicate:
                continue
            if self.region:
                overlap_start = max(start, r.reference_start)
                overlap_end = min(end, r.reference_end)
            else:
                overlap_start = r.reference_start
                overlap_end = r.reference_end
            if overlap_end is None:
                continue
            if overlap_start < overlap_end:
                start_bin = (overlap_start - offset) // self.bin_size
                end_bin = (overlap_end - offset - 1) // self.bin_size + 1
                start_bin = max(0, min(start_bin, n - 1))
                end_bin = max(0, min(end_bin, n))
                for bin_idx in range(start_bin, end_bin):
                    rd_p[bin_idx] += 1
                    if r.mapping_quality > 0:
                        rd_u[bin_idx] += 1
        return (rd_p, rd_u, None)

    def calculate_gc_content(self, chr_name, start, end):
        if not self.fasta:
            return None
        try:
            sequence = self.fasta.fetch(chr_name, start, end).upper()
            n_bins = (end - start + self.bin_size - 1) // self.bin_size
            gc = np.zeros(n_bins)
            for i in range(n_bins):
                bin_start = i * self.bin_size
                bin_end = min(bin_start + self.bin_size, len(sequence))
                if bin_end <= bin_start:
                    break
                bin_seq = sequence[bin_start:bin_end]
                gc_count = bin_seq.count('G') + bin_seq.count('C')
                total_count = gc_count + bin_seq.count('A') + bin_seq.count('T')
                if total_count > 0:
                    gc[i] = gc_count * 100.0 / total_count
            return gc
        except Exception as e:
            self.logger.error(f'Error calculating GC content for {chr_name}: {str(e)}')
            return None

    def calc_base_nuc_penalty(self, chr_name, ref_pos, read_base):
        if not self.fasta:
            return self.unaligned_penalty
        try:
            ref_base_str = self.fasta.fetch(chr_name, ref_pos, ref_pos + 1).upper()
        except:
            return self.unaligned_penalty
        if len(ref_base_str) == 0:
            return self.unaligned_penalty
        ref_b = ref_base_str[0]
        read_b = read_base.upper()
        if ref_b == read_b:
            return 0
        is_cpg_site = self.is_cpg_site(chr_name, ref_pos)
        if is_cpg_site:
            is_ct_conversion = ref_b == 'C' and read_b == 'T'
            is_ga_conversion = ref_b == 'G' and read_b == 'A'
            is_methylated = not (is_ct_conversion or is_ga_conversion)
            if ref_b == 'C' or ref_b == 'G':
                methyl_ratio = self.get_methylation_level(chr_name, ref_pos)
                if is_methylated:
                    return (1 - methyl_ratio) * self.mismatch_penalty
                else:
                    return methyl_ratio * self.mismatch_penalty
        return self.mismatch_penalty

    def calculate_methylation_penalty(self, chr_name):
        if self.region:
            start, end = self.region
            length = end - start
            offset = start
            n_bins = (length + self.bin_size - 1) // self.bin_size
            penalty_by_bin = np.zeros(n_bins)
            fetch_start = max(0, start)
            fetch_end = min(self.chrom_lengths[chr_name], end)
            read_iter = self.file.fetch(self.bam_chrom_names[chr_name], fetch_start, fetch_end)
        else:
            length = self.chrom_lengths[chr_name]
            offset = 0
            n_bins = (length + self.bin_size - 1) // self.bin_size
            penalty_by_bin = np.zeros(n_bins)
            read_iter = self.file.fetch(self.bam_chrom_names[chr_name])
        self.logger.info('Calculating methylation penalties...')
        start_time = time.time()

        def calc_read_penalty(aln):
            if aln.is_unmapped:
                return self.unaligned_penalty
            penalty_sum = 0
            ref_idx = aln.reference_start
            seq = aln.query_sequence
            read_idx = 0
            ctups = aln.cigartuples
            if not ctups:
                return self.unaligned_penalty
            for op, length_op in ctups:
                if op == 0:
                    for _ in range(length_op):
                        if read_idx >= len(seq):
                            penalty_sum += self.unaligned_penalty
                            return penalty_sum
                        penalty_sum += self.calc_base_nuc_penalty(chr_name, ref_idx, seq[read_idx])
                        read_idx += 1
                        ref_idx += 1
                elif op == 1:
                    penalty_sum += self.go_penalty + self.ge_penalty * (length_op - 1)
                    read_idx += length_op
                elif op == 2:
                    penalty_sum += self.go_penalty + self.ge_penalty * (length_op - 1)
                    ref_idx += length_op
                elif op in [4, 5]:
                    read_idx += length_op
                elif op == 3:
                    ref_idx += length_op
            return penalty_sum
        count = 0
        for aln in read_iter:
            if count % 10000 == 0:
                self.logger.debug(f'Processed {count} reads')
            count += 1
            if aln.is_unmapped or aln.is_secondary or aln.is_duplicate:
                continue
            penalty_val = calc_read_penalty(aln)
            overlap_start = aln.reference_start
            overlap_end = aln.reference_end
            if overlap_start is None or overlap_end is None:
                continue
            if self.region:
                start, end = self.region
                overlap_start = max(start, overlap_start)
                overlap_end = min(end, overlap_end)
            if overlap_start < overlap_end:
                start_bin = (overlap_start - offset) // self.bin_size
                end_bin = (overlap_end - offset - 1) // self.bin_size + 1
                start_bin = max(0, min(start_bin, n_bins - 1))
                end_bin = max(0, min(end_bin, n_bins))
                for b in range(start_bin, end_bin):
                    penalty_by_bin[b] += penalty_val
        self.logger.info(f'Methylation penalty calculation completed: {count} reads processed in {time.time() - start_time:.2f} seconds')
        return penalty_by_bin

    def process_region(self, chr_name):
        if self.methylation_processor:
            self.extract_region_cpg_data(chr_name)
        self.logger.info(f'Processing region {chr_name}:{self.region[0]}-{self.region[1]}')
        start_time = time.time()
        rd_p, rd_u, _ = self.read_chromosome(chr_name)
        if rd_p is None:
            return None
        results = {'rd_p': rd_p, 'rd_u': rd_u}
        if self.region:
            start, end = self.region
            gc_vals = self.calculate_gc_content(chr_name, start, end)
        else:
            length = self.chrom_lengths[chr_name]
            gc_vals = self.calculate_gc_content(chr_name, 0, length)
        results['gc_content'] = gc_vals
        penalty_by_bin = self.calculate_methylation_penalty(chr_name)
        if penalty_by_bin is not None:
            results['methyl_penalty'] = penalty_by_bin
            norm_penalty = penalty_by_bin / (rd_p + 1)
            results['methyl_penalty_norm'] = norm_penalty
            if 'meth_density' in results and 'meth_variance' in results:
                epsilon = 1e-06
                results['depth_var_ratio'] = rd_p / (results['meth_variance'] + epsilon)
                results['depth_density_ratio'] = rd_p / (results['meth_density'] + epsilon)
                results['penalty_var_ratio'] = penalty_by_bin / (results['meth_variance'] + epsilon)
            depth_gradient = np.zeros_like(rd_p)
            depth_gradient[1:] = np.diff(rd_p)
            window_size = 5
            depth_smooth = np.zeros_like(rd_p)
            depth_local_std = np.zeros_like(rd_p)
            for i in range(len(rd_p)):
                window_start = max(0, i - window_size // 2)
                window_end = min(len(rd_p), i + window_size // 2 + 1)
                window_data = rd_p[window_start:window_end]
                depth_smooth[i] = np.mean(window_data)
                depth_local_std[i] = np.std(window_data)
            results['depth_gradient'] = depth_gradient
            results['depth_smooth'] = depth_smooth
            results['depth_local_std'] = depth_local_std
            depth_change_score = np.abs(depth_gradient) / (depth_smooth + 1e-06)
            results['depth_change_score'] = depth_change_score
            if self.methylation_processor:
                import copy
                region_meth_processor = copy.copy(self.methylation_processor)
                region_meth_processor.region = self.region
                meth_features = region_meth_processor.process_chromosome(chr_name)
                if meth_features:
                    for key, value in meth_features.items():
                        if key not in ['bin_coords', 'bin_start', 'chrom']:
                            results[key] = value
            if 'meth_variance' in results:
                meth_var = results['meth_variance']
                meth_var_gradient = np.zeros_like(meth_var)
                meth_var_gradient[1:] = np.diff(meth_var)
                meth_var_smooth = np.zeros_like(meth_var)
                for i in range(len(meth_var)):
                    window_start = max(0, i - window_size // 2)
                    window_end = min(len(meth_var), i + window_size // 2 + 1)
                    window_data = meth_var[window_start:window_end]
                    meth_var_smooth[i] = np.mean(window_data)
                if len(meth_var_gradient) != len(depth_gradient):
                    self.logger.warning(f'Length mismatch: depth_gradient({len(depth_gradient)}) versus meth_var_gradient({len(meth_var_gradient)})')
                    if len(meth_var_gradient) > len(depth_gradient):
                        meth_var_gradient = meth_var_gradient[:len(depth_gradient)]
                    else:
                        depth_gradient = depth_gradient[:len(meth_var_gradient)]
                coherence_score = depth_gradient * meth_var_gradient
                results['meth_var_gradient'] = meth_var_gradient
                results['meth_var_smooth'] = meth_var_smooth
                results['depth_meth_coherence'] = coherence_score
        if self.region:
            bin_count = len(rd_p)
            results['bin_start'] = [self.region[0] + i * self.bin_size for i in range(bin_count)]
            results['chrom'] = [chr_name] * bin_count
        self.logger.info(f'Region processing completed in {time.time() - start_time:.2f} seconds')
        return results

    def close(self):
        try:
            if self.file:
                self.file.close()
        except:
            pass
        try:
            if self.fasta:
                self.fasta.close()
        except:
            pass

class MethylationProcessor:

    def __init__(self, meth_file, reference_filename=None, bin_size=100, region=None):
        import pysam
        self.meth_file = meth_file
        self.reference_file = reference_filename
        self.bin_size = bin_size
        self.region = region
        self.logger = logging.getLogger('methylcnv.methylation')
        if reference_filename and os.path.isfile(reference_filename):
            self.fasta = pysam.FastaFile(reference_filename)
        else:
            self.fasta = None
        self.meth_data = pd.read_csv(meth_file, sep='\t', dtype={'Chr': str, 'Start': int, 'End': int, 'MethPercent': float, 'MethCount': int, 'UnmethCount': int})
        self.meth_data['Chr'] = self.meth_data['Chr'].str.replace('chr', '')

    def process_chromosome(self, chrom):
        if self.fasta:
            chrom_length = self.fasta.get_reference_length(chrom)
        else:
            chrom_length = None
        chr_data = self.meth_data[self.meth_data['Chr'] == chrom].copy()
        if len(chr_data) == 0:
            self.logger.warning(f'No methylation data found for {chrom}.')
            return None
        if self.region and chrom_length is not None:
            start, end = self.region
            chr_data = chr_data[(chr_data['Start'] >= start) & (chr_data['End'] <= end)]
            region_length = end - start
        elif chrom_length is not None:
            start = 0
            end = chrom_length
            region_length = end - start
        else:
            start = chr_data['Start'].min()
            end = chr_data['End'].max()
            region_length = end - start
        num_bins = (region_length + self.bin_size - 1) // self.bin_size
        features = {'meth_level': np.zeros(num_bins), 'meth_density': np.zeros(num_bins), 'meth_variance': np.zeros(num_bins), 'coverage': np.zeros(num_bins), 'meth_entropy': np.zeros(num_bins), 'bin_coords': []}
        chr_data['rel_start'] = chr_data['Start'] - start
        chr_data['bin_idx'] = chr_data['rel_start'] // self.bin_size
        for bin_idx in range(num_bins):
            bin_start = start + bin_idx * self.bin_size
            bin_end = min(bin_start + self.bin_size, end)
            features['bin_coords'].append((bin_start, bin_end))
            bin_data = chr_data[chr_data['bin_idx'] == bin_idx]
            if len(bin_data) > 0:
                coverage_arr = bin_data['MethCount'] + bin_data['UnmethCount']
                if coverage_arr.sum() > 0:
                    features['meth_level'][bin_idx] = np.average(bin_data['MethPercent'], weights=coverage_arr)
                features['meth_density'][bin_idx] = len(bin_data) / self.bin_size
                features['meth_variance'][bin_idx] = np.var(bin_data['MethPercent'])
                features['coverage'][bin_idx] = np.mean(coverage_arr)
                total_meth = bin_data['MethCount'].sum()
                total_unmeth = bin_data['UnmethCount'].sum()
                total_sum = total_meth + total_unmeth
                if total_sum > 0:
                    from scipy.stats import entropy
                    dist_norm = np.array([total_meth, total_unmeth]) / total_sum
                    features['meth_entropy'][bin_idx] = entropy(dist_norm)
        bin_starts = [start + i * self.bin_size for i in range(num_bins)]
        features['bin_start'] = bin_starts
        features['chrom'] = [chrom] * num_bins
        del features['bin_coords']
        return features
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
METH_COLUMNS = ['Chr', 'Start', 'End', 'MethPercent', 'MethCount', 'UnmethCount']
FEATURES = ['rd_u', 'gc_content', 'methyl_penalty', 'depth_gradient', 'depth_smooth', 'depth_local_std', 'depth_change_score', 'meth_var_gradient', 'meth_var_smooth', 'depth_meth_coherence']

def canonical(name):
    name = str(name)
    name = name[3:] if name.startswith('chr') else name
    return 'MT' if name == 'M' else name

def name_map(names):
    result = {}
    for name in names:
        key = canonical(name)
        if key in result:
            raise ValueError(f'Duplicate canonical chromosome name: {key}')
        result[key] = name
    return result

class ReferenceAdapter:

    def __init__(self, source):
        self.source, self.names = (source, name_map(source.references))

    def fetch(self, name, *args, **kwargs):
        return self.source.fetch(self.names[canonical(name)], *args, **kwargs)

    def get_reference_length(self, name):
        return self.source.get_reference_length(self.names[canonical(name)])

    def close(self):
        self.source.close()

def load_methylation(path, chrom):
    parts = []
    for block in pd.read_csv(path, sep='\t', usecols=METH_COLUMNS, dtype={'Chr': str}, chunksize=500000):
        block['Chr'] = block.Chr.map(canonical)
        block = block.loc[block.Chr.eq(chrom)].copy()
        if block.empty:
            continue
        for name in METH_COLUMNS[1:]:
            block[name] = pd.to_numeric(block[name], errors='raise')
        if not np.isfinite(block[METH_COLUMNS[1:]].to_numpy()).all():
            raise ValueError('Non-finite methylation data')
        if not block.MethPercent.between(0, 100).all():
            raise ValueError('MethPercent must be in 0-100')
        for name in ['Start', 'End', 'MethCount', 'UnmethCount']:
            if (block[name] < 0).any() or (block[name] % 1 != 0).any():
                raise ValueError(f'{name} must contain nonnegative integers')
        if (block.End <= block.Start).any():
            raise ValueError('Methylation intervals must have End > Start')
        parts.append(block)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=METH_COLUMNS)

def extract_block(bam, reference, processor, chrom, left, right, bin_size):
    depth = DepthCalculator(str(bam), str(reference), bin_size=bin_size, region=(left, right), methylation_processor=processor)
    try:
        if depth.fasta is None:
            raise ValueError('Reference FASTA could not be opened')
        depth.fasta = ReferenceAdapter(depth.fasta)
        result = depth.process_region(chrom)
        if result is None or any((name not in result for name in FEATURES)):
            raise ValueError(f'Incomplete extracted features: {chrom}:{left}-{right}')
        values = np.column_stack([result[name] for name in FEATURES]).astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite extracted features')
        return values
    finally:
        depth.close()

def predict_raw_main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bam', type=Path, required=True)
    p.add_argument('--meth', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=Path(__file__).parent / 'final_model.pth')
    p.add_argument('--chromosomes', nargs='+', help='Default: shared 1-22, X, Y, MT')
    p.add_argument('--bin_size', type=int, default=100)
    p.add_argument('--stride', type=int, default=10)
    p.add_argument('--chunk_bins', type=int, default=10000)
    p.add_argument('--batch_size', type=int, default=192)
    p.add_argument('--threads', type=int, default=16)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    a = p.parse_args()
    if min(a.bin_size, a.stride, a.chunk_bins, a.batch_size, a.threads) < 1:
        p.error('Sizes must be positive')
    if a.output.exists():
        raise FileExistsError('Use a new output directory')
    for path in [a.bam, a.meth, a.reference, a.checkpoint]:
        if not path.is_file():
            raise FileNotFoundError(path)
    import pysam
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    cfg, names, scaler = (saved['model_args'], saved['features'], saved['scaler'])
    if list(names) != FEATURES or cfg['n_features'] != len(FEATURES):
        raise ValueError('Extractor requires the supplied 10-feature CNVDetector checkpoint')
    if scaler.n_features_in_ != len(FEATURES):
        raise ValueError('Scaler dimension mismatch')
    length = int(cfg['seq_length'])
    torch.set_num_threads(a.threads)
    device = torch.device(a.device)
    model = CNVDetector(SimpleNamespace(**cfg)).to(device).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    mean, scale = (scaler.mean_.astype(np.float32), scaler.scale_.astype(np.float32))
    statuses, total = ([], 0)
    with pysam.AlignmentFile(str(a.bam), 'rb') as bam, pysam.FastaFile(str(a.reference)) as fasta:
        if not bam.has_index():
            raise ValueError('BAM index required: samtools index sample.sorted.bam')
        if bam.header.to_dict().get('HD', {}).get('SO') != 'coordinate':
            raise ValueError('BAM header must specify coordinate sorting')
        bm, fm = (name_map(bam.references), name_map(fasta.references))
        candidates = [canonical(c) for c in a.chromosomes] if a.chromosomes else [str(i) for i in range(1, 23)] + ['X', 'Y', 'MT']
        selected = []
        for chrom in dict.fromkeys(candidates):
            if chrom not in bm or chrom not in fm:
                if a.chromosomes:
                    raise ValueError(f'Chromosome absent from BAM or FASTA: {chrom}')
                continue
            size = bam.get_reference_length(bm[chrom])
            if size != fasta.get_reference_length(fm[chrom]):
                raise ValueError(f'BAM/FASTA length mismatch: {chrom}')
            selected.append((chrom, size))
        if not selected:
            raise ValueError('No shared target chromosomes')
        a.output.mkdir(parents=True)
        config = {'status': 'running', 'checkpoint_sha256': hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(), 'model_args': cfg, 'features': FEATURES, 'inputs': {k: str(getattr(a, k).resolve()) for k in ['bam', 'meth', 'reference', 'checkpoint']}, 'bin_size': a.bin_size, 'stride': a.stride, 'chunk_bins': a.chunk_bins, 'metric_unit': 'window-center classification', 'note': 'Genome-wide accuracy is not established by region-guided training.', 'chromosomes': statuses}
        config_path = a.output / 'inference_config.json'
        config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
        try:
            with (a.output / 'window_predictions.tsv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.writer(handle, delimiter='\t')
                writer.writerow(['chrom', 'window_start', 'window_end', 'bin_start', 'bin_end', 'prediction', 'prob_Normal', 'prob_DEL', 'prob_DUP'])
                for chrom, size in selected:
                    n = size // a.bin_size
                    if n < length:
                        statuses.append({'chrom': chrom, 'status': 'skipped_short'})
                        continue
                    logging.info('Reading methylation for chromosome %s', chrom)
                    meth = load_methylation(a.meth, chrom)
                    if meth.empty:
                        logging.warning('Skipping %s: no methylation observations', chrom)
                        statuses.append({'chrom': chrom, 'status': 'skipped_no_methylation'})
                        continue
                    if meth.End.max() > size:
                        raise ValueError(f'Methylation coordinates exceed chromosome length: {chrom}')
                    processor = MethylationProcessor.__new__(MethylationProcessor)
                    processor.meth_file, processor.reference_file = (str(a.meth), str(a.reference))
                    processor.bin_size, processor.region = (a.bin_size, None)
                    processor.logger = logging.getLogger('methylcnv.methylation')
                    processor.meth_data = meth
                    processor.fasta = ReferenceAdapter(fasta)
                    count = 0
                    for core in range(0, n - length + 1, a.chunk_bins):
                        stop = min(core + a.chunk_bins, n - length + 1)
                        first = (core + a.stride - 1) // a.stride * a.stride
                        starts = np.arange(first, stop, a.stride, dtype=np.int64)
                        if not len(starts):
                            continue
                        left = max(0, int(starts[0]) - 3)
                        right = min(n, int(starts[-1]) + length + 3)
                        values = extract_block(a.bam, a.reference, processor, chrom, left * a.bin_size, right * a.bin_size, a.bin_size)
                        values = (values - mean) / scale
                        for offset in range(0, len(starts), a.batch_size):
                            batch = starts[offset:offset + a.batch_size]
                            x = np.stack([values[s - left:s - left + length] for s in batch])
                            with torch.inference_mode():
                                probs = torch.softmax(model(torch.from_numpy(x).to(device))[0], dim=1).cpu().numpy()
                            for start, prob in zip(batch, probs):
                                center = int(start + length // 2) * a.bin_size
                                writer.writerow([bm[chrom], int(start) * a.bin_size, int(start + length) * a.bin_size, center, center + a.bin_size, ['Normal', 'DEL', 'DUP'][int(prob.argmax())], *map(float, prob)])
                            count += len(batch)
                        handle.flush()
                        logging.info('%s: %d windows predicted', chrom, count)
                    total += count
                    statuses.append({'chrom': chrom, 'status': 'completed', 'n_windows': count})
                    del processor, meth
            if total == 0:
                raise ValueError('No windows predicted; check chromosome methylation coverage')
            config.update(status='completed', n_windows=total)
        except Exception as exc:
            config.update(status='failed', error=str(exc), n_windows=total)
            raise
        finally:
            config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    print(f'Prediction completed without training: {total} windows -> {a.output}')

if __name__ == '__main__':
    predict_raw_main()
