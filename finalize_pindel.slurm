#!/bin/bash
#SBATCH --job-name=pindel_finalize
#SBATCH --partition=kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --chdir=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS
#SBATCH --output=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/pindel_finalize_%j.out
#SBATCH --error=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/pindel_finalize_%j.err
set -euo pipefail
export PATH=/public/home/wangruiqi/micromamba/envs/cnv-all/bin:$PATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
BASE=/public/home/wangruiqi/bam/cnv_benchmark
SOURCE="$BASE/TOOLS/Pindel_chromosomes/124532446"
FASTA="$BASE/prepared_autosomes/GRCh38.autosomes.fa"
OUT="$SOURCE/finalized_${SLURM_JOB_ID}"
mkdir -p "$OUT"
date
for i in {1..22}; do
    test -f "$SOURCE/chr$i/SUCCESS"
    test -s "$SOURCE/chr$i/calls.vcf"
done
python -u "$BASE/TOOLS/merge_pindel_vcfs.py" --directory "$SOURCE" --fasta "$FASTA" \
    --output "$OUT/combined.header_fixed.vcf"
bcftools sort "$OUT/combined.header_fixed.vcf" -T "$OUT/sort_tmp.XXXXXX" \
    -Ov -o "$OUT/sample.Pindel.vcf"
python "$BASE/normalize_cnv_calls.py" --format vcf \
    --input "$OUT/sample.Pindel.vcf" --output "$OUT/Pindel.author.bed"
# Publish to the original run directory only after successful validation/conversion.
cp "$OUT/sample.Pindel.vcf" "$SOURCE/sample.Pindel.vcf"
cp "$OUT/Pindel.author.bed" "$SOURCE/Pindel.author.bed"
printf 'Header repair and finalization job: %s\n' "$SLURM_JOB_ID" > "$SOURCE/finalization.txt"
echo SUCCESS > "$SOURCE/SUCCESS"
echo "Completed: $SOURCE"
date
