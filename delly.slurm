#!/bin/bash
#SBATCH --job-name=delly_author
#SBATCH --partition=kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS
#SBATCH --output=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/delly_author_%j.out
#SBATCH --error=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/delly_author_%j.err

set -euo pipefail
export PATH=/public/home/wangruiqi/micromamba/envs/cnv-all/bin:$PATH
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
BASE=/public/home/wangruiqi/bam/cnv_benchmark
BAM="$BASE/prepared_autosomes/sample.autosomes.sorted.bam"
FASTA="$BASE/prepared_autosomes/GRCh38.autosomes.fa"
HELPER="$BASE/TOOLS/build_delly_exclusions.py"
OUT="$BASE/TOOLS/DELLY_author/${SLURM_JOB_ID}"
ID=sample
for required in "$BAM" "$FASTA" "$HELPER"; do
    test -s "$required" || { echo "Missing input: $required" >&2; exit 1; }
done
mkdir -p "$OUT/resources"
cd "$OUT"
date
delly --help > delly_help.txt 2>&1 || true
cat delly_help.txt
# Generate missing exclusions from the user's existing FASTA only.
python "$HELPER" --fasta "$FASTA" \
    --output resources/GRCh38.autosomes.generated.excl.tsv \
    --min_n_run 1000 --terminal_bp 10000
set -x
# DELLY 2.7.0 short-read entry point, equivalent role to older 'delly call'.
delly sr -x resources/GRCh38.autosomes.generated.excl.tsv \
    -o "$ID.DELLY.bcf" -g "$FASTA" "$BAM"
bcftools view "$ID.DELLY.bcf" -Ov -o "$ID.DELLY.vcf"
python "$BASE/normalize_cnv_calls.py" --format vcf \
    --input "$ID.DELLY.vcf" --output DELLY.author.bed
set +x
printf 'BAM=%s\nReference=%s\nScope=chr1-chr22\nCommand=delly sr -x FASTA_derived_exclusions\n' \
    "$BAM" "$FASTA" > protocol.txt
echo SUCCESS > SUCCESS
echo "Results: $OUT"
date
