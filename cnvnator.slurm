#!/bin/bash
#SBATCH --job-name=cnvnator_author
#SBATCH --partition=kshcnormal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2-00:00:00
#SBATCH --chdir=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS
#SBATCH --output=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/cnvnator_author_%j.out
#SBATCH --error=/public/home/wangruiqi/bam/cnv_benchmark/TOOLS/cnvnator_author_%j.err
set -euo pipefail
TARGET=/public/home/wangruiqi/micromamba/envs/cnv-all
test -x "$TARGET/bin/cnvnator" || { echo 'Existing cnvnator binary missing'; exit 1; }
BASE=/public/home/wangruiqi/bam/cnv_benchmark
BAM="$BASE/prepared_autosomes/sample.autosomes.sorted.bam"
REF="$BASE/prepared_autosomes/GRCh38.autosomes.fa"
WORK="$BASE/TOOLS/CNVnator_author/${SLURM_JOB_ID}"
mkdir -p "$WORK/vcf"
cd "$WORK"
exec > >(tee -a run.log) 2>&1
unset CPATH CPLUS_INCLUDE_PATH C_INCLUDE_PATH ROOT_INCLUDE_PATH EXTRA_CLING_ARGS
unset ROOTSYS LD_PRELOAD LD_LIBRARY_PATH GCC_EXEC_PREFIX COMPILER_PATH LIBRARY_PATH
unset CC CXX CPP CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
export CONDA_PREFIX="$TARGET"
export PATH="$TARGET/bin:/usr/bin:/bin"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# Compiler activation hooks configure relocatable Conda toolchains/sysroots.
# Scope is this job only.
if [[ -d "$TARGET/etc/conda/activate.d" ]]; then
    set +u
    for hook in "$TARGET"/etc/conda/activate.d/*.sh; do
        [[ -f "$hook" ]] && source "$hook"
    done
    set -u
fi
python "$BASE/TOOLS/configure_root_headers.py" --env "$TARGET" --output compiler_paths.txt
INCLUDES=$(cat compiler_paths.txt)
export ROOT_INCLUDE_PATH="$INCLUDES"
export CPLUS_INCLUDE_PATH="$INCLUDES"
# Avoid Cling's obsolete built-in C++ library search paths.
CLING_ARGUMENTS="-nostdinc++"
IFS=':' read -r -a INCLUDE_DIRECTORIES <<< "$INCLUDES"
for include_dir in "${INCLUDE_DIRECTORIES[@]}"; do
    [[ "$include_dir" != *" "* ]] || { echo 'Include path contains spaces'; exit 1; }
    CLING_ARGUMENTS="$CLING_ARGUMENTS -isystem $include_dir"
done
export EXTRA_CLING_ARGS="$CLING_ARGUMENTS"
echo "Clean Cling startup arguments: $EXTRA_CLING_ARGS"
root-config --version
root-config --cxx
cat > root_probe.C <<'ROOT'
#include <vector>
#include <TFile.h>
#include <TTree.h>
#include <iostream>
void root_probe() {
    std::vector<int> values(1, 7);
    TFile file("probe.root", "RECREATE");
    TTree tree("probe", "probe");
    int value = values[0];
    tree.Branch("value", &value);
    tree.Fill(); tree.Write(); file.Close();
    std::cout << "ROOT_PROBE_OK" << std::endl;
}
ROOT
if ! root -l -b -q root_probe.C > root_probe.log 2>&1; then
    cat root_probe.log
    exit 1
fi
cat root_probe.log
grep -q ROOT_PROBE_OK root_probe.log
checked_cnvnator() {
    local step=$1
    shift
    cnvnator "$@" > "$step.log" 2>&1
    cat "$step.log"
    if grep -Eiq "fatal error:|unresolved while linking|Can't find directory|header .* not found|C system headers .*must be installed" "$step.log"; then
        echo "CNVnator failed at $step" >&2
        return 1
    fi
}
if grep -Eiq 'fatal error:|unresolved while linking|error: header' root_probe.log; then
    exit 1
fi
CHROMS=()
for i in {1..22}; do CHROMS+=("chr$i"); done
export LC_ALL=C
ID=sample.autosomes.sorted
VCF_SCRIPT=$(command -v cnvnator2VCF.pl || true)
if [[ -z "$VCF_SCRIPT" ]]; then
    VCF_SCRIPT=$(find "$TARGET" -type f -iname 'cnvnator2vcf.pl' -print -quit)
fi
[[ -n "$VCF_SCRIPT" ]] || { echo 'cnvnator2VCF.pl missing from environment' >&2; exit 1; }
cp "$VCF_SCRIPT" cnvnator2VCF.used.pl
python -u "$BASE/TOOLS/split_cnvnator_fasta.py" --fasta "$REF" --out "$WORK/chrom_fasta"
set -x
checked_cnvnator tree -root sample.root -tree "$BAM" -chrom "${CHROMS[@]}"
checked_cnvnator histogram -root sample.root -chrom "${CHROMS[@]}" -his 100 -d "$WORK/chrom_fasta"
checked_cnvnator statistics -root sample.root -stat 100
checked_cnvnator partition -root sample.root -partition 100
checked_cnvnator call -root sample.root -chrom "${CHROMS[@]}" -call 100
# Keep the raw log and separate only valid calls for the Perl converter.
awk '$1=="deletion" || $1=="duplication"' call.log > "$ID.cnv"
perl cnvnator2VCF.used.pl -reference GRCh38 "$ID.cnv" "$WORK/chrom_fasta" \
    > "vcf/$ID.cnvnator.vcf" 2> vcf_conversion.log
cat vcf_conversion.log
if grep -Eq "Can't parse sequence|Can't open|not enough arg" vcf_conversion.log; then
    echo 'VCF conversion failed' >&2
    exit 1
fi
bcftools view -Ov "vcf/$ID.cnvnator.vcf" -o "$ID.cnvnator.validated.vcf"
python "$BASE/normalize_cnv_calls.py" --format cnvnator \
    --input "$ID.cnv" --output CNVnator.author.bed
set +x
printf 'Scope=chr1-chr22\nBin_size=100\nBAM=%s\nReference=%s\n' "$BAM" "$REF" > protocol.txt
printf 'SUCCESS\n' > SUCCESS
echo "Results saved to $WORK"
