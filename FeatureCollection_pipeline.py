import numpy as np
import pandas as pd
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import logging
import random
import pysam
import time

from Depth_Methy_Features import DepthCalculator, MethylationProcessor

def get_chrom_lengths_from_fasta(reference_file):
    """
    Read all chromosome lengths from reference_file (FASTA) and return a dictionary:
    {
        '1': 248956422,
        '2': 242193529,
        ...
    }
    """
    with pysam.FastaFile(reference_file) as ref_f:
        chrom_lengths = {}
        for chrom in ref_f.references:
            length = ref_f.get_reference_length(chrom)
            stripped_chrom = chrom.replace("chr", "")
            chrom_lengths[stripped_chrom] = length
    return chrom_lengths

class TrainingDataBuilderRegion:
    def __init__(self, bin_size=100, context_size=10000, normal_keep_fraction=0.0, n_processes=None):
        """
        Parameters
        ----------
        bin_size : int
            Width of each genomic bin.
        context_size : int
            Context length added before and after each CNV region.
        normal_keep_fraction : float
            Fraction of normal intervals far from CNVs to retain at random.
        n_processes : int or None
            Number of worker processes.
        """
        self.bin_size = bin_size
        self.context_size = context_size
        self.normal_keep_fraction = normal_keep_fraction
        self.n_processes = n_processes
        self.logger = logging.getLogger("lstm_training")

    def process_chromosome(self, chrom, bam_file, meth_file, reference_file, chrom_cnv_df, chrom_length=None):
        """
        Extract features and assign labels (0/1/2) for CNVs with context and sampled distant intervals.
        chrom_cnv_df : CNV records for this chromosome only.
        chrom_length : Chromosome length used to determine intervals far from CNVs.
        """
        start_time = time.time()
        self.logger.info(f"Processing chromosome {chrom}")
        
        if len(chrom_cnv_df) == 0:
            self.logger.info(f"[{chrom}] No CNVs found, skipping.")
            return None

        # 1) Build near-CNV intervals (CNV plus or minus context_size).
        intervals = []
        for _, row in chrom_cnv_df.iterrows():
            start_ctx = max(0, row['Start'] - self.context_size)
            end_ctx   = row['End'] + self.context_size
            if chrom_length is not None:
                end_ctx = min(end_ctx, chrom_length)
            intervals.append((start_ctx, end_ctx))

        # Merge overlapping intervals.
        intervals.sort(key=lambda x: x[0])
        merged = []
        cur_start, cur_end = intervals[0]
        for i in range(1, len(intervals)):
            st, ed = intervals[i]
            if st <= cur_end:
                cur_end = max(cur_end, ed)
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = st, ed
        merged.append((cur_start, cur_end))

        # Log the number of merged near-CNV intervals.
        self.logger.info(f"[{chrom}] Merged nearCNV intervals count = {len(merged)}")

        # 2) Build normal intervals far from CNVs (example).
        far_regions = []
        if chrom_length is None:
            # Skip distant intervals if the chromosome length is unavailable.
            self.logger.warning(f"[{chrom}] chrom_length not provided, skip far-region.")
        else:
            # Subtract merged intervals from [0, chrom_length) to obtain the remaining gaps.
            prev_end = 0
            for (st, ed) in merged:
                if st > prev_end:
                    far_regions.append((prev_end, st))
                prev_end = ed
            if prev_end < chrom_length:
                far_regions.append((prev_end, chrom_length))

            # Randomly retain the fraction of distant intervals specified by normal_keep_fraction.
            keep_num = int(len(far_regions) * self.normal_keep_fraction)
            keep_regions = random.sample(far_regions, keep_num) if keep_num > 0 else []
            far_regions = sorted(keep_regions, key=lambda x: x[0])
            
            self.logger.info(f"[{chrom}] far_regions raw: {len(far_regions)} -> downsampled to {len(keep_regions)}")

        # Combine near-CNV and distant intervals into final_intervals.
        final_intervals = merged + far_regions
        self.logger.info(f"[{chrom}] final intervals count = {len(final_intervals)}")

        # 3) Create a shared MethylationProcessor for this chromosome and load the data once.
        try:
            self.logger.info(f"[{chrom}] Creating the chromosome-level MethylationProcessor")
            global_meth_start = time.time()
            global_meth_processor = MethylationProcessor(
                meth_file=meth_file,
                reference_filename=reference_file,
                bin_size=self.bin_size,
                region=None  # Load all data for subsequent filtering.
            )
            global_meth_processor.meth_data = global_meth_processor.meth_data[
                global_meth_processor.meth_data['Chr'] == chrom
            ].copy()
            self.logger.info(f"[{chrom}] Chromosome-level MethylationProcessor created in {time.time() - global_meth_start:.2f} seconds")
        except Exception as e:
            self.logger.error(f"[{chrom}] Failed to create MethylationProcessor: {str(e)}")
            return None
            
        # 4) Call DepthCalculator for each interval and assign labels.
        all_sub_dfs = []
        all_sub_labels = []

        # Process each interval.
        for idx, (int_start, int_end) in enumerate(final_intervals):
            interval_start_time = time.time()
            self.logger.info(f"[{chrom}] Processing interval {idx+1}/{len(final_intervals)}: {int_start}-{int_end}")
            
            if int_end <= int_start:
                self.logger.warning(f"[{chrom}] Skipping invalid interval: {int_start}-{int_end}")
                continue  # Skip invalid intervals.
                
            try:
                depth_calculator = DepthCalculator(
                    filename=bam_file,
                    reference_filename=reference_file,
                    bin_size=self.bin_size,
                    region=(int_start, int_end),  # Process only this interval.
                    methylation_processor=global_meth_processor  # Pass the shared processor; only current-region data will be extracted.
                )
                depth_feats = depth_calculator.process_region(chrom)
                
                depth_calculator.close()

                # Reuse methylation features already calculated by DepthCalculator for this interval.
                if depth_feats is None or 'meth_level' not in depth_feats:
                    raise ValueError(f'[{chrom}] Failed to obtain features for interval {int_start}-{int_end}.')

                # Create the feature DataFrame.
                sub_df = pd.DataFrame({
                    'rd_u': depth_feats['rd_u'],
                    'gc_content': depth_feats['gc_content'],
                    'methyl_penalty': depth_feats['methyl_penalty'],
                    # 'meth_density': depth_feats['meth_density'],
                    # 'meth_variance': depth_feats['meth_variance'],
                    'chrom': chrom
                })
                
                # Add depth-to-methylation ratio features.
                if 'depth_var_ratio' in depth_feats:
                    sub_df['depth_var_ratio'] = depth_feats['depth_var_ratio']
                if 'depth_density_ratio' in depth_feats:
                    sub_df['depth_density_ratio'] = depth_feats['depth_density_ratio']
                if 'penalty_var_ratio' in depth_feats:
                    sub_df['penalty_var_ratio'] = depth_feats['penalty_var_ratio']

                # Add local context features.
                if 'depth_gradient' in depth_feats:
                    sub_df['depth_gradient'] = depth_feats['depth_gradient']
                if 'depth_smooth' in depth_feats:
                    sub_df['depth_smooth'] = depth_feats['depth_smooth']
                if 'depth_local_std' in depth_feats:
                    sub_df['depth_local_std'] = depth_feats['depth_local_std']
                if 'depth_change_score' in depth_feats:
                    sub_df['depth_change_score'] = depth_feats['depth_change_score']

                # Add methylation-related local context features.
                if 'meth_var_gradient' in depth_feats:
                    sub_df['meth_var_gradient'] = depth_feats['meth_var_gradient']
                if 'meth_var_smooth' in depth_feats:
                    sub_df['meth_var_smooth'] = depth_feats['meth_var_smooth']
                if 'depth_meth_coherence' in depth_feats:
                    sub_df['depth_meth_coherence'] = depth_feats['depth_meth_coherence']

                # Assign a CNV type label to each bin.
                N_bins = len(sub_df)
                labels = np.zeros(N_bins, dtype=int)
                for bin_idx in range(N_bins):
                    bin_global_start = int_start + bin_idx*self.bin_size
                    bin_global_end   = bin_global_start + self.bin_size
                    overlap_cnv = chrom_cnv_df[
                        (chrom_cnv_df['Start'] < bin_global_end) &
                        (chrom_cnv_df['End'] > bin_global_start)
                    ]
                    if not overlap_cnv.empty:
                        cnv_type = overlap_cnv.iloc[0]['Type']
                        if cnv_type == 'DEL':
                            labels[bin_idx] = 1
                        elif cnv_type == 'DUP':
                            labels[bin_idx] = 2

                # Add bin_start for subsequent sorting or analysis.
                sub_df['bin_start'] = [int_start + i*self.bin_size for i in range(N_bins)]

                all_sub_dfs.append(sub_df)
                all_sub_labels.append(labels)
                
                self.logger.info(f"[{chrom}] Interval {idx+1}/{len(final_intervals)} completed: {N_bins} bins generated in {time.time() - interval_start_time:.2f} seconds")
            
            except Exception as e:
                self.logger.error(f"[{chrom}] Failed to process interval {int_start}-{int_end}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
                continue

        if not all_sub_dfs:
            self.logger.info(f"[{chrom}] 0 bins in final intervals, skipping.")
            return None

        features_df = pd.concat(all_sub_dfs, ignore_index=True)
        labels_all = np.concatenate(all_sub_labels)
        
        self.logger.info(
            f"[{chrom}] process_chromosome done. Bins: {len(features_df)}, near/far/total intervals: {len(merged)}/{len(far_regions)}/{len(final_intervals)}, total elapsed time: {time.time() - start_time:.2f} seconds"
        )

        return (features_df, labels_all)

    def build_dataset(self, bam_file, meth_file, cnv_bed_file, reference_file, chrom_lengths_dict=None):
        """
        Build a training dataset by extracting features and assigning labels within selected intervals.
        Parameters:
        -----------
        bam_file: str
            Path to the BAM file.
        meth_file: str
            Path to the methylation data file.
        cnv_bed_file: str
            Path to the reference CNV BED file.
        reference_file: str
            Path to the reference genome FASTA file.
        chrom_lengths_dict: dict
            Optional dictionary of chromosome lengths.
        
        Returns:
        --------
        X: pd.DataFrame
            Feature matrix.
        y: np.ndarray
            Label vector.
        """
        # Read the CNV BED file.
        try:
            cnv_df = pd.read_csv(cnv_bed_file, sep='\t')
            cnv_df['Chr'] = cnv_df['Chr'].str.replace('chr', '')
            cnv_df['Start'] = pd.to_numeric(cnv_df['Start'])
            cnv_df['End'] = pd.to_numeric(cnv_df['End'])
            
            
            self.logger.info(f"Loaded {len(cnv_df)} CNV records")
        except Exception as e:
            self.logger.error(f"Failed to read the CNV file: {str(e)}")
            raise

        # Create a pool with the requested number of worker processes.
        if self.n_processes is None:
            self.n_processes = multiprocessing.cpu_count() - 2
        
        self.logger.info(f"Using {self.n_processes} parallel worker processes")
        process_pool = ProcessPoolExecutor(max_workers=self.n_processes)
        
        # Process chromosomes in parallel.
        all_chromosomes = sorted(cnv_df['Chr'].unique())
        self.logger.info(f"Found {len(all_chromosomes)} chromosomes: {all_chromosomes}")
        
        chromosomes = []
        pool_rets = []
        
        for chrom in all_chromosomes:
            chrom_cnv_df = cnv_df[cnv_df['Chr'] == chrom]
            
            # Skip the chromosome if it has no CNV records.
            if len(chrom_cnv_df) == 0:
                self.logger.info(f"Chromosome {chrom} has no CNV records; skipping")
                continue
                
            chromosomes.append(chrom)
            
            # Get the chromosome length.
            chr_len = None
            if chrom_lengths_dict and chrom in chrom_lengths_dict:
                chr_len = chrom_lengths_dict[chrom]
            
            # Log submission of the chromosome processing task.
            self.logger.info(f"Submitting a task for chromosome {chrom}, length={chr_len}")

            pool_rets.append(
                process_pool.submit(
                    self.process_chromosome,
                    chrom, bam_file, meth_file, reference_file, chrom_cnv_df, chr_len
                )
            )

        # Shut down ProcessPoolExecutor after collecting the results.
        
        # Collect results.
        all_features = []
        all_labels = []
        
        for chrom, ret in zip(chromosomes, pool_rets):
            self.logger.info(f"Waiting for results for chromosome {chrom}...")
            try:
                result = ret.result()
                if result is not None:
                    feat_df, labs = result
                    all_features.append(feat_df)
                    all_labels.append(labs)
                    self.logger.info(f"Chromosome {chrom} completed. Number of output bins: {len(feat_df)}")
                else:
                    self.logger.info(f"Chromosome {chrom} returned None.")
            except Exception as e:
                self.logger.error(f"Error processing chromosome {chrom}: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
                raise
        
        process_pool.shutdown(wait=True)

        if not all_features:
            raise ValueError("No valid data remain after regional feature extraction")

        X = pd.concat(all_features, ignore_index=True)
        y = np.concatenate(all_labels)
        self.logger.info(f"Final dataset dimensions => X:{X.shape}, y:{y.shape}")
        return X, y
# =========================================
# Main function
# =========================================
def main():
    logging.basicConfig(level=logging.INFO, 
                      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    reference_file = '/public/home/wangruiqi/bam/Homo_sapiens.GRCh38.dna.primary_assembly.fa'
    chrom_lengths_dict = get_chrom_lengths_from_fasta(reference_file)
    print(chrom_lengths_dict)

    builder = TrainingDataBuilderRegion(
        bin_size=100,
        context_size=10000,
        normal_keep_fraction=0.0,
        n_processes=4
    )
    
    start_time = time.time()
    X, y = builder.build_dataset(
        bam_file='/public/home/wangruiqi/bam/sample.sorted.bam',
        meth_file='/public/home/wangruiqi/bam/sample_CpG.bed',
        cnv_bed_file='/public/home/wangruiqi/bam/reference_cnvs_human.bed',
        reference_file=reference_file,
        chrom_lengths_dict=chrom_lengths_dict
    )
    logging.info(f"Dataset construction completed; total elapsed time: {time.time() - start_time:.2f} seconds")
    
    # Calculate the number of samples in each class.
    class_counts = np.bincount(y, minlength=3)
    logging.info("=== Class sample counts ===")
    logging.info(f"Label 0 (Normal): {class_counts[0]} samples, fraction: {class_counts[0]/len(y):.2%}")
    logging.info(f"Label 1 (DEL): {class_counts[1]} samples, fraction: {class_counts[1]/len(y):.2%}")
    logging.info(f"Label 2 (DUP): {class_counts[2]} samples, fraction: {class_counts[2]/len(y):.2%}")
    logging.info("===================")
    
    # Create the output directory if needed.
    from pathlib import Path
    Path('/public/home/wangruiqi/bam/new').mkdir(parents=True, exist_ok=True)

    # Save features and labels to the NPZ file.
    np.savez('/public/home/wangruiqi/bam/new/train_data_sample_bin100.npz', 
             features=X, 
             labels=y,
             feature_names=X.columns)
    logging.info("Data saved to /public/home/wangruiqi/bam/new/train_data_sample_bin100.npz")

if __name__ == "__main__":
    main() 
