# -*- coding: utf-8 -*-
import logging
import numpy as np
import pandas as pd
import pysam
from scipy import stats
import os
import time


class DepthCalculator:
    """Calculate coverage and GC content for a chromosome or region.
    Apply an align_bs-like methylation penalty using CIGAR operations
    and aligned bases to score individual reads."""

    def __init__(self,
                 filename,
                 reference_filename=None,
                 cpg_island_file=None,
                 bin_size=100,
                 region=None,
                 mismatch_penalty=4,
                 go_penalty=6,
                 ge_penalty=1,
                 unaligned_penalty=999999999,
                 methylation_processor=None):
        """Parameters
        ----------
        filename : str
            Path to the BAM/SAM file.
        reference_filename : str, optional
            Path to the reference genome FASTA file.
        cpg_island_file : str, optional
            Path to a CpG island file, such as a BED file.
        bin_size : int
            Width of each genomic bin.
        region : tuple or None
            Target (start, end) interval. A region must be provided;
            this implementation raises ValueError when it is None.
        mismatch_penalty : int
            Penalty for a standard base mismatch.
        go_penalty : int
            Gap-opening penalty.
        ge_penalty : int
            Gap-extension penalty.
        unaligned_penalty : int
            Penalty for unaligned or invalid reads.
        methylation_processor : MethylationProcessor, optional
            Source of methylation site information for the current region."""
        self.logger = logging.getLogger("methylcnv.depth")
        self.filename = filename
        self.reference_filename = reference_filename
        self.cpg_island_file = cpg_island_file
        self.bin_size = bin_size
        self.region = region
        if region is None:
            raise ValueError("region cannot be None. Please specify a region to process.")
        self.start, self.end = region

        self.mismatch_penalty = mismatch_penalty
        self.go_penalty = go_penalty
        self.ge_penalty = ge_penalty
        self.unaligned_penalty = unaligned_penalty
        
        # Store CpG site information for the current region.
        self.methylation_processor = methylation_processor
        self.cpg_sites = {}
        self.cpg_meth_levels = {}
        
        # Extract regional methylation data only when a processor is provided.
        if self.methylation_processor:
            self._prepare_region_methylation_data()

        # Open the BAM/SAM file.
        try:
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".bam":
                self.file = pysam.AlignmentFile(filename, "rb",
                                                reference_filename=reference_filename if reference_filename else None)
            else:
                self.file = pysam.AlignmentFile(filename, "r",
                                                reference_filename=reference_filename if reference_filename else None)
        except Exception as e:
            self.logger.error(f"Unable to open alignment file {filename}: {str(e)}")
            raise

        # Retrieve chromosome information.
        # Use names without the chr prefix internally; map back to original BAM names when fetching reads.
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

        # Open the reference genome.
        if reference_filename and os.path.isfile(reference_filename):
            try:
                self.fasta = pysam.FastaFile(reference_filename)
            except Exception as e:
                self.logger.error(f"Unable to open reference genome: {str(e)}")
                self.fasta = None
        else:
            self.fasta = None

        # Initialize the set of CpG positions.
        self.cpg_ids = set()
        if self.fasta and self.cpg_island_file:
            self.load_reference_and_cpg()

    def _prepare_region_methylation_data(self):
        """Prepare methylation data structures for the current region."""
        if not self.methylation_processor or not self.region:
            return
            
        self.logger.info("Preparing methylation data for the current region from MethylationProcessor...")
        start_time = time.time()
        
        try:
            # Get the current region boundaries.
            region_start, region_end = self.region
            chrom = None  # Determined later in process_region.
            
            # Initialize chromosome data structures.
            self.cpg_sites = {}
            self.cpg_meth_levels = {}
            
            # The chromosome will be specified when process_region is called.
            self.logger.info(f"Regional methylation data preparation completed in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            self.logger.error(f"Error preparing regional methylation data: {str(e)}")

    def extract_region_cpg_data(self, chr_name):
        """Extract CpG site data for the specified chromosome and current region."""
        if not self.methylation_processor:
            return
            
        start_time = time.time()
        region_start, region_end = self.region
        
        # Filter methylation data for the current region.
        filtered_data = self.methylation_processor.meth_data[
            (self.methylation_processor.meth_data['Chr'] == chr_name) & 
            (self.methylation_processor.meth_data['Start'] >= region_start) & 
            (self.methylation_processor.meth_data['End'] <= region_end)
        ]
        
        # Initialize data structures.
        if chr_name not in self.cpg_sites:
            self.cpg_sites[chr_name] = []
            self.cpg_meth_levels[chr_name] = {}
        
        # Extract positions and methylation levels.
        for _, row in filtered_data.iterrows():
            pos = row['Start']
            meth_percent = row['MethPercent'] / 100.0  # Convert to the 0-1 range.
            
            # Store CpG positions and methylation levels.
            self.cpg_sites[chr_name].append(pos)
            self.cpg_meth_levels[chr_name][pos] = meth_percent
        
        # Sort positions for subsequent lookups.
        if chr_name in self.cpg_sites:
            self.cpg_sites[chr_name].sort()
            
        self.logger.info(f"Extracted methylation data for {chr_name}:{region_start}-{region_end}: {len(self.cpg_sites.get(chr_name, []))} CpG sites in {time.time() - start_time:.2f} seconds")

    def is_cpg_site(self, chrom, position):
        """Check whether the given genomic position is a CpG site."""
        if self.methylation_processor:
            if chrom not in self.cpg_meth_levels:
                return False
            return position in self.cpg_meth_levels[chrom]
        else:
            return (chrom, position) in self.cpg_ids

    def get_methylation_level(self, chrom, position):
        """Return the methylation level at the specified CpG site."""
        if self.methylation_processor and chrom in self.cpg_meth_levels and position in self.cpg_meth_levels[chrom]:
            return self.cpg_meth_levels[chrom][position]
        
        # Return the default value of 0.5 when methylation information is unavailable.
        return 0.5

    def load_reference_and_cpg(self):
        """Read CpG island intervals and scan the reference genome for CpG positions.
        Store positions as (chr_name, pos) tuples in self.cpg_ids."""
        self.logger.info("Loading CpG island data and scanning the reference sequence for CG dinucleotides...")
        cpg_island_regions = []
        try:
            with open(self.cpg_island_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    fields = line.strip().split()
                    if len(fields) < 3:
                        continue
                    c_chr = fields[0].replace("chr", "")
                    c_start = int(fields[1])
                    c_end = int(fields[2])
                    cpg_island_regions.append((c_chr, c_start, c_end))
        except Exception as e:
            self.logger.warning(f"Failed to read cpg_island_file: {str(e)}")

        for chrom_name in self.references:
            try:
                seq_length = self.chrom_lengths[chrom_name]
                refseq = self.fasta.fetch(chrom_name, 0, seq_length).upper()
            except Exception as e:
                self.logger.warning(f"Unable to fetch reference sequence {chrom_name}: {str(e)}")
                continue

            for i in range(len(refseq) - 1):
                if refseq[i] == 'C' and refseq[i + 1] == 'G':
                    self.cpg_ids.add((chrom_name, i))
                    self.cpg_ids.add((chrom_name, i + 1))

    def get_chromosome_length(self, chr_name):
        """Return the length of the specified chromosome."""
        idx = self.references.index(chr_name)
        return self.lengths[idx]

    def read_chromosome(self, chr_name):
        """Count read coverage for the chromosome or region in genomic bins.
        Return rd_p, rd_u, and his_read_frg:
            rd_p: Total read coverage per bin.
            rd_u: Read coverage per bin for mapping_quality > 0.
            his_read_frg: Reserved histogram output; currently returned as None."""
        if chr_name not in self.chrom_lengths:
            self.logger.warning(f"Chromosome '{chr_name}' was not found in the alignment file.")
            return None, None, None

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

            # Check whether reference_end is None.
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

        return rd_p, rd_u, None

    def calculate_gc_content(self, chr_name, start, end):
        """Calculate GC content in bins of bin_size across the specified interval.
        Return an array containing the GC percentage for each bin."""
        if not self.fasta:
            return None
        try:
            sequence = self.fasta.fetch(chr_name, start, end).upper()
            n_bins = ((end - start) + self.bin_size - 1) // self.bin_size
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
                    gc[i] = (gc_count * 100.0) / total_count
            return gc
        except Exception as e:
            self.logger.error(f"Error calculating GC content for {chr_name}: {str(e)}")
            return None

    def calc_base_nuc_penalty(self, chr_name, ref_pos, read_base):
        """Calculate a base penalty following the approach in align_bs.cpp.
        1. An exact base match receives a penalty of zero.
        2. At a CpG site:
           a) A methylated base without C-to-T or G-to-A conversion receives
              a penalty of (1 - methyl_ratio) * mismatch_penalty.
           b) An unmethylated base with C-to-T or G-to-A conversion receives
              a penalty of methyl_ratio * mismatch_penalty.
        3. Other mismatches receive mismatch_penalty."""
        if not self.fasta:
            return self.unaligned_penalty
            
        try:
            # Fetch the base from the reference genome.
            ref_base_str = self.fasta.fetch(chr_name, ref_pos, ref_pos + 1).upper()
        except:
            return self.unaligned_penalty

        if len(ref_base_str) == 0:
            return self.unaligned_penalty

        ref_b = ref_base_str[0]
        read_b = read_base.upper()
        
        # Step 1: Check for an exact base match.
        if ref_b == read_b:
            return 0
            
        # Step 2: Check for a special conversion at a CpG site.
        is_cpg_site = self.is_cpg_site(chr_name, ref_pos)
        
        if is_cpg_site:
            # Determine methylation status.
            is_ct_conversion = (ref_b == 'C' and read_b == 'T')
            is_ga_conversion = (ref_b == 'G' and read_b == 'A')
            
            # Distinguish methylated from unmethylated bases.
            is_methylated = not (is_ct_conversion or is_ga_conversion)
            
            if (ref_b == 'C' or ref_b == 'G'):  # Consider only C and G positions.
                methyl_ratio = self.get_methylation_level(chr_name, ref_pos)
                
                if is_methylated:
                    # For methylated bases, weight the penalty by (1 - methyl_ratio).
                    return (1 - methyl_ratio) * self.mismatch_penalty
                else:
                    # For unmethylated bases, weight the penalty by methyl_ratio.
                    return methyl_ratio * self.mismatch_penalty
        
        # Step 3: Apply the standard mismatch penalty otherwise.
        return self.mismatch_penalty

    def calculate_methylation_penalty(self, chr_name):
        """Calculate per-read penalties from alignments and the reference, then aggregate by bin."""
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

        self.logger.info("Calculating methylation penalties...")
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
                if op == 0:  # M (match/mismatch)
                    for _ in range(length_op):
                        if read_idx >= len(seq):
                            penalty_sum += self.unaligned_penalty
                            return penalty_sum
                        penalty_sum += self.calc_base_nuc_penalty(chr_name, ref_idx, seq[read_idx])
                        read_idx += 1
                        ref_idx += 1
                elif op == 1:  # insertion
                    penalty_sum += self.go_penalty + self.ge_penalty * (length_op - 1)
                    read_idx += length_op
                elif op == 2:  # deletion
                    penalty_sum += self.go_penalty + self.ge_penalty * (length_op - 1)
                    ref_idx += length_op
                else:
                    if op in [4, 5]:
                        read_idx += length_op
                    elif op == 3:
                        ref_idx += length_op
            return penalty_sum

        count = 0
        for aln in read_iter:
            if count % 10000 == 0:
                self.logger.debug(f"Processed {count} reads")
            count += 1
            
            if aln.is_unmapped or aln.is_secondary or aln.is_duplicate:
                continue
                
            penalty_val = calc_read_penalty(aln)
            overlap_start = aln.reference_start
            overlap_end = aln.reference_end
            
            # Check whether reference_end is None.
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
        
        self.logger.info(f"Methylation penalty calculation completed: {count} reads processed in {time.time() - start_time:.2f} seconds")
        return penalty_by_bin

    def process_region(self, chr_name):
        """Process the specified chromosome within the current region.
        1. Count read coverage.
        2. Calculate GC content.
        3. Calculate align_bs-like methylation penalties per bin.
        Return these results and the derived features in a dictionary."""
        # Extract regional methylation data first if a processor is provided.
        if self.methylation_processor:
            self.extract_region_cpg_data(chr_name)
            
        self.logger.info(f"Processing region {chr_name}:{self.region[0]}-{self.region[1]}")
        start_time = time.time()
            
        rd_p, rd_u, _ = self.read_chromosome(chr_name)
        if rd_p is None:
            return None

        results = {
            'rd_p': rd_p,
            'rd_u': rd_u,
        }

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

            # === Example penalty normalization ===
            # Divide by (rd_p + 1).
            norm_penalty = penalty_by_bin / (rd_p + 1)
            results['methyl_penalty_norm'] = norm_penalty
            
            # === Add depth-to-methylation ratio features ===
            if 'meth_density' in results and 'meth_variance' in results:
                # Add the ratio of depth to methylation variance.
                epsilon = 1e-6  # Avoid division by zero.
                results['depth_var_ratio'] = rd_p / (results['meth_variance'] + epsilon)
                
                # Add the ratio of depth to methylation density.
                results['depth_density_ratio'] = rd_p / (results['meth_density'] + epsilon)
                
                # Add the ratio of methylation penalty to methylation variance.
                results['penalty_var_ratio'] = penalty_by_bin / (results['meth_variance'] + epsilon)
            
            # === Add local context features ===
            # Calculate depth changes using adjacent differences.
            depth_gradient = np.zeros_like(rd_p)
            depth_gradient[1:] = np.diff(rd_p)  # Depth difference between adjacent bins.

            # Calculate the moving-window mean and standard deviation of depth.
            window_size = 5  # Use a window of five bins.
            depth_smooth = np.zeros_like(rd_p)
            depth_local_std = np.zeros_like(rd_p)

            for i in range(len(rd_p)):
                window_start = max(0, i - window_size // 2)
                window_end = min(len(rd_p), i + window_size // 2 + 1)
                window_data = rd_p[window_start:window_end]
                depth_smooth[i] = np.mean(window_data)
                depth_local_std[i] = np.std(window_data)

            # Store features in the results.
            results['depth_gradient'] = depth_gradient
            results['depth_smooth'] = depth_smooth
            results['depth_local_std'] = depth_local_std

            # Detect abrupt depth changes as candidate CNV boundaries.
            depth_change_score = np.abs(depth_gradient) / (depth_smooth + 1e-6)
            results['depth_change_score'] = depth_change_score
            
            # === Retrieve and add methylation processor features ===
            if self.methylation_processor:
                # Reuse the loaded methylation table and process only this region to avoid chromosome-wide misalignment.
                import copy
                region_meth_processor = copy.copy(self.methylation_processor)
                region_meth_processor.region = self.region
                meth_features = region_meth_processor.process_chromosome(chr_name)
                if meth_features:
                    for key, value in meth_features.items():
                        # Merge methylation features directly into results.
                        if key not in ['bin_coords', 'bin_start', 'chrom']:
                            results[key] = value
            
            # === Add methylation-related local context features ===
            if 'meth_variance' in results:
                # Calculate the gradient of methylation variance.
                meth_var = results['meth_variance']
                meth_var_gradient = np.zeros_like(meth_var)
                meth_var_gradient[1:] = np.diff(meth_var)
                
                # Apply moving-window smoothing.
                meth_var_smooth = np.zeros_like(meth_var)
                for i in range(len(meth_var)):
                    window_start = max(0, i - window_size // 2)
                    window_end = min(len(meth_var), i + window_size // 2 + 1)
                    window_data = meth_var[window_start:window_end]
                    meth_var_smooth[i] = np.mean(window_data)
                
                # Construct a feature describing corresponding methylation and depth changes.
                if len(meth_var_gradient) != len(depth_gradient):
                    self.logger.warning(f"Length mismatch: depth_gradient({len(depth_gradient)}) versus meth_var_gradient({len(meth_var_gradient)})")
                    # Keep the shorter length and truncate the longer array.
                    if len(meth_var_gradient) > len(depth_gradient):
                        meth_var_gradient = meth_var_gradient[:len(depth_gradient)]
                    else:
                        depth_gradient = depth_gradient[:len(meth_var_gradient)]
                
                coherence_score = depth_gradient * meth_var_gradient
                
                # Store features in the results.
                results['meth_var_gradient'] = meth_var_gradient
                results['meth_var_smooth'] = meth_var_smooth
                results['depth_meth_coherence'] = coherence_score
            
        # Add bin metadata.
        if self.region:
            bin_count = len(rd_p)
            results['bin_start'] = [self.region[0] + i * self.bin_size for i in range(bin_count)]
            results['chrom'] = [chr_name] * bin_count
            
        self.logger.info(f"Region processing completed in {time.time() - start_time:.2f} seconds")
        return results

    def close(self):
        """Close the associated file handles."""
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
    """Read external methylation data, such as MethylDackel output, and generate bin-level features."""

    def __init__(self,
                 meth_file,
                 reference_filename=None,
                 bin_size=100,
                 region=None):
        self.meth_file = meth_file
        self.reference_file = reference_filename
        self.bin_size = bin_size
        self.region = region
        self.logger = logging.getLogger("methylcnv.methylation")

        if reference_filename and os.path.isfile(reference_filename):
            self.fasta = pysam.FastaFile(reference_filename)
        else:
            self.fasta = None

        self.meth_data = pd.read_csv(
            meth_file, sep='\t',
            dtype={
                'Chr': str,
                'Start': int,
                'End': int,
                'MethPercent': float,
                'MethCount': int,
                'UnmethCount': int
            }
        )
        self.meth_data['Chr'] = self.meth_data['Chr'].str.replace('chr', '')

    def process_chromosome(self, chrom):
        if self.fasta:
            chrom_length = self.fasta.get_reference_length(chrom)
        else:
            chrom_length = None

        chr_data = self.meth_data[self.meth_data['Chr'] == chrom].copy()
        if len(chr_data) == 0:
            self.logger.warning(f"No methylation data found for {chrom}.")
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
        features = {
            'meth_level': np.zeros(num_bins),
            'meth_density': np.zeros(num_bins),
            'meth_variance': np.zeros(num_bins),
            'coverage': np.zeros(num_bins),
            'meth_entropy': np.zeros(num_bins),
            'bin_coords': []
        }

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
                    features['meth_level'][bin_idx] = np.average(
                        bin_data['MethPercent'],
                        weights=coverage_arr
                    )
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
                    
        # Add bin_start and chrom metadata.
        bin_starts = [start + i * self.bin_size for i in range(num_bins)]
        features['bin_start'] = bin_starts
        features['chrom'] = [chrom] * num_bins

        # Remove the bin_coords field, which is no longer needed.
        del features['bin_coords']

        return features 
