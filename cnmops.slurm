#!/bin/bash
#SBATCH --job-name=cnmops_author
#SBATCH --partition=kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS
#SBATCH --output=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/cnmops_author_%j.out
#SBATCH --error=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/cnmops_author_%j.err

set -euo pipefail
export PATH=/public/home/wangruiqi/micromamba/envs/cnv-all/bin:$PATH
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
BASE=/public/home/wangruiqi/bam/cnv_benchmark
BAM="$BASE/prepared_autosomes/sample.autosomes.sorted.bam"
CODE="$BASE/TOOLS/cnmops.R"
OUT="$BASE/TOOLS/cn_mops_author/${SLURM_JOB_ID}"
WINDOW="${CNMOPS_WINDOW:-1000}"
test -s "$BAM"
test -s "$BAM.bai"
test -s "$CODE"
mkdir -p "$OUT"
date
echo "BAM: $BAM"
echo "Output: $OUT"
echo "Window: $WINDOW bp"
# 16 allocated CPUs provide the default memory allowance; R processes chromosomes serially.
Rscript "$CODE" "$BAM" "$OUT" "$WINDOW"
test -f "$OUT/cn.mops.author.bed"
echo SUCCESS > "$OUT/SUCCESS"
date
