#!/bin/bash
#SBATCH --job-name=pindel_author
#SBATCH --partition=kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS
#SBATCH --output=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/pindel_author_%j.out
#SBATCH --error=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/pindel_author_%j.err

set -euo pipefail
export PATH=/public/home/wangruiqi/micromamba/envs/cnv-all/bin:$PATH
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS=1
BASE=/public/home/wangruiqi/bam/cnv_benchmark
BAM="$BASE/prepared_autosomes/sample.autosomes.sorted.bam"
FASTA="$BASE/prepared_autosomes/GRCh38.autosomes.fa"
OUT="$BASE/TOOLS/Pindel_author/${SLURM_JOB_ID}"
SAMPLE=sample
# Match the supplied author script unless explicitly overridden at submission.
INSERT_SIZE="${INSERT_SIZE:-250}"
if [[ ! "$INSERT_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo 'INSERT_SIZE must be a positive integer' >&2
    exit 1
fi
for required in "$BAM" "$FASTA" "$BAM.bai" "$FASTA.fai"; do
    test -s "$required" || { echo "Missing input: $required" >&2; exit 1; }
done
command -v pindel
command -v pindel2vcf
command -v bcftools
test -f "$BASE/normalize_cnv_calls.py"
mkdir -p "$OUT"
cd "$OUT"
date
printf '%s\t%s\t%s\n' "$BAM" "$INSERT_SIZE" "$SAMPLE" > config.txt
cat config.txt
set -x
pindel -T "$SLURM_CPUS_PER_TASK" -f "$FASTA" -i "$OUT/config.txt" \
    -c ALL -o "$OUT/pindel.out"
# Prefix mode combines event files in one converter invocation and one VCF header.
# It replaces the original broken 'outnames=(find ...)' and raw VCF concatenation.
shopt -s nullglob
EVENT_FILES=("$OUT"/pindel.out_*)
if (( ${#EVENT_FILES[@]} == 0 )); then
    echo 'Pindel produced no event files; stopping before conversion' >&2
    exit 1
fi
DATE=$(date +%Y%m%d)
pindel2vcf -r "$FASTA" -R GRCh38 -P "$OUT/pindel.out" \
    -d "$DATE" -v "$SAMPLE.Pindel.raw.vcf" > conversion.log 2>&1
cat conversion.log
# Validate syntax using bcftools. Keep the converter output for provenance.
bcftools view "$SAMPLE.Pindel.raw.vcf" -Ov -o "$SAMPLE.Pindel.vcf"
python "$BASE/normalize_cnv_calls.py" --format vcf \
    --input "$SAMPLE.Pindel.vcf" --output Pindel.author.bed
set +x
printf 'BAM=%s\nReference=%s\nScope=chr1-chr22\nInsert_size=%s\nThreads=%s\nConverter_date_argument=%s\n' \
    "$BAM" "$FASTA" "$INSERT_SIZE" "$SLURM_CPUS_PER_TASK" "$DATE" > protocol.txt
echo SUCCESS > SUCCESS
echo "Results: $OUT"
date
