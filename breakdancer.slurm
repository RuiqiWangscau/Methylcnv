#!/bin/bash
#SBATCH --job-name=breakdancer_q10
#SBATCH --partition=kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS
#SBATCH --output=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/breakdancer_q10_%j.out
#SBATCH --error=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/breakdancer_q10_%j.err

set -euo pipefail
export PATH=/public/home/wangruiqi/micromamba/envs/cnv-all/bin:$PATH
BASE=/public/home/wangruiqi/bam/cnv_benchmark
CFG="$BASE/raw_autosomes/BreakDancer_retry_124441625/bam.cfg"
NORMALIZER="$BASE/normalize_cnv_calls.py"
OUT="$BASE/TOOLS/BreakDancer_q10/${SLURM_JOB_ID}"

test -s "$CFG" || { echo "Configuration missing: $CFG" >&2; exit 1; }
test -f "$NORMALIZER"
mkdir -p "$OUT"
cd "$OUT"
cp "$CFG" sample.cfg
date
echo "Output directory: $OUT"
set -x
breakdancer-max -q 10 sample.cfg > sample.ctx
python "$NORMALIZER" --format breakdancer \
    --input sample.ctx --output BreakDancer.q10.bed
set +x
printf 'SUCCESS\n' > SUCCESS
date
