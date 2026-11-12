# MethylCNV

MethylCNV applies a bidirectional LSTM (BiLSTM) model with attention to genomic features extracted from whole-genome bisulfite sequencing (WGBS) data. The repository provides pretrained prediction from a sample BAM, a CpG methylation table, and the corresponding reference FASTA, together with model-training and result-analysis scripts.

**Users can predict with the supplied `final_model.pth` without training a model.** The checkpoint contains the trained weights, feature order, input scaler, and model configuration.

## Repository files

| File | Purpose |
| --- | --- |
| `predict.py` | Self-contained prediction entry point, including the model definition and feature-extraction code. |
| `final_model.pth` | Pretrained `CNVDetector` weights and associated preprocessing/configuration metadata. |
| `requirements.txt` | Python dependencies for prediction and model training. |
| `ModelTrain.py` | Region-grouped five-fold training, final refitting, and independent holdout evaluation. |
| `Depth_Methy_Features.py` | Feature-extraction module used by the training-data builder. Prediction embeds its feature calculations. |
| `FeatureCollection_pipeline.py` | Build a labeled training-feature NPZ from WGBS inputs and reference CNVs. |
| `analyze_results.py` | Analyze saved labeled classification predictions without training. |
| `train_sklearn.py` | Classical-classifier training and overlap-precision evaluation workflow; edit dataset-specific settings before use. |
| `six_cnv_tools_combined.sh` | Collection of separate CNV-caller job-script examples. |
| `reference_cnvs_human.bed` | Reference CNV annotations used for labeled-data construction and comparison. |
| `merged_predictions_all_methods.xlsx` | Aggregated comparison data. |

## Installation

Use a Python 3 environment on Linux with compatible PyTorch and scikit-learn versions. Python 2 is not supported. Python 3.10 or later is recommended.

```bash
git clone https://github.com/RuiqiWangscau/Methylcnv.git
cd Methylcnv

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The saved scaler records scikit-learn version **1.9.1**. Prefer a matching environment when loading it. `requirements.txt` lists dependencies rather than a validated version lockfile. For CUDA inference, install a PyTorch build compatible with your CUDA environment.

The training and analysis workflows may additionally need plotting packages:

```bash
python -m pip install matplotlib
```

Classical classifiers additionally require `xgboost`; reading the supplied XLSX with pandas may require `openpyxl`. External tools such as SAMtools, BEDtools and the six CNV callers must be installed separately for their respective workflows.

Load only trusted checkpoints and feature NPZ files: the saved scaler and some NPZ arrays use Python object serialization.

## Predict without training

### Required sample inputs

1. A coordinate-sorted BAM and its BAI or CSI index.
2. A CpG methylation table from the same sample.
3. The corresponding reference FASTA and its FAI index.

The BAM and reference must use the same genome assembly. The program accepts chromosome names with or without a `chr` prefix and checks matching chromosome lengths; this does not convert genome assemblies.

If indexes are missing, create them once:

```bash
samtools index sample.sorted.bam
samtools faidx hg38.fa
```

The methylation input must be a **tab-separated file with a header** containing these columns:

```text
Chr	Start	End	MethPercent	MethCount	UnmethCount
chr1	100	101	50	5	5
```

Coordinates are 0-based and end-exclusive. `MethPercent` is a percentage from 0 to 100; `MethCount` and `UnmethCount` are nonnegative integer counts. A `.bed` extension alone does not establish compatibility: headerless bedGraph or coverage files must first be converted using the appropriate coordinate and column conventions.

### Run prediction

Run from the repository directory:

```bash
python predict.py \
  --bam sample.sorted.bam \
  --meth sample_CpG.bed \
  --reference hg38.fa \
  --output sample_predictions
```

The program automatically loads `final_model.pth` beside `predict.py`, extracts the required features, applies the saved training scaler, and performs inference. It does not train, fit a new scaler, or require reference CNV annotations or true labels.

Use a new output directory for each run. By default, the program processes shared canonical chromosomes 1–22, X, Y and MT in blocks. Chromosomes without methylation observations or fewer than one full model window are skipped and recorded. Terminal incomplete bins are omitted.

To process selected chromosomes:

```bash
python predict.py \
  --bam sample.sorted.bam \
  --meth sample_CpG.bed \
  --reference hg38.fa \
  --chromosomes chr1 chr2 \
  --threads 16 \
  --output sample_chr1_chr2_predictions
```

### Prediction options

| Option | Default | Meaning |
| --- | --- | --- |
| `--bam` | Required | Coordinate-sorted, indexed BAM. |
| `--meth` | Required | Headered CpG methylation table. |
| `--reference` | Required | Indexed reference FASTA. |
| `--output` | Required | New output directory. |
| `--checkpoint` | `final_model.pth` beside the script | Compatible checkpoint path. |
| `--chromosomes` | Shared 1–22, X, Y, MT | Chromosomes to process. |
| `--bin_size` | `100` | Genomic bin size in bp. |
| `--stride` | `10` | Spacing between window starts in bins. |
| `--chunk_bins` | `10000` | Block size for grouping window starts, in bins. |
| `--batch_size` | `192` | Inference batch size. |
| `--threads` | `16` | PyTorch CPU thread count. |
| `--device` | `cpu` | `cpu` or `cuda`. |

Model window length is read from the checkpoint, not set by a prediction option. Bin size and stride are not stored in this checkpoint; retain the original training settings when using it. The current prediction entry point accepts raw WGBS inputs, not a `--data_path` NPZ argument.

### Prediction output

- `window_predictions.tsv`: chromosome, window coordinates, center-bin coordinates, predicted class and three class probabilities.
- `inference_config.json`: input paths, checkpoint checksum, model configuration, run settings, chromosome status and completion status.

The TSV columns are:

```text
chrom  window_start  window_end  bin_start  bin_end  prediction  prob_Normal  prob_DEL  prob_DUP
```

Coordinates are 0-based and end-exclusive. Each window receives one `Normal`, `DEL` or `DUP` classification associated with its center bin. **Window spans are not inferred CNV event boundaries.** The predictor does not merge adjacent calls into CNV events. Classification metrics and CNV-event overlap metrics must therefore be distinguished.

Check that `inference_config.json` reports `status: completed`; failed or interrupted runs can leave partial TSV files. Progress messages are written to standard error. Whole-genome processing can be lengthy; on clusters, submit to a compute node using the local scheduler.

## Pretrained model and training compatibility

Both `ModelTrain.py` and `predict.py` define `Attention`, `FeatureEncoder` and `CNVDetector`. Their model-class definitions were checked for structural equality, and the supplied checkpoint loaded with `strict=True`. They currently retain separate copies of the definitions, so changes must be synchronized.

The supplied checkpoint has:

| Setting | Saved value |
| --- | --- |
| Input features | 10, in the order below |
| Sequence length | 300 bins |
| Hidden size | 128 per direction |
| Feature-encoder LSTM layers | 1, bidirectional |
| Feature fusion | Bidirectional LSTM with feature-level attention |
| Dropout configuration | 0.5 |
| Saved final training epoch | 10 |
| Classes | 0 = Normal; 1 = DEL; 2 = DUP |

Ordered inputs:

```text
rd_u
gc_content
methyl_penalty
depth_gradient
depth_smooth
depth_local_std
depth_change_score
meth_var_gradient
meth_var_smooth
depth_meth_coherence
```

This is a 10-feature model, not the historical five-feature `BiLSTM_CNV` release. In particular, the saved input is `methyl_penalty`, not `methyl_penalty_norm`. Do not substitute historical `final_model.pt`, `feature_scaler.pkl` or feature-name files.

Checkpoint SHA-256:

```text
03fc081addddc12b5ab5afb6b4b5552c667c6e390ec62d64949aea829a2f42be
```

Architecture and checkpoint-loading checks establish compatibility; they do not alone reproduce manuscript performance or establish the complete original training history.

## Optional model training

Training is not required for pretrained prediction.

### Build labeled training features

`FeatureCollection_pipeline.py` currently uses paths and settings inside `main()` rather than command-line arguments. Edit its BAM, methylation, reference CNV, FASTA and output paths before running:

```bash
python FeatureCollection_pipeline.py
```

Its current example uses 100-bp bins, 10,000-bp context, `normal_keep_fraction=0.0` and four processes. It writes `features`, `labels` and `feature_names` to an NPZ. The reference-CNV-guided training construction is separate from unlabeled genome-wide prediction.

### Train and evaluate

```bash
python ModelTrain.py \
  --data_path /path/to/train_features.npz \
  --cnv_bed /path/to/reference_cnvs_human.bed \
  --save_root /path/to/new_training_output \
  --seq_length 300 --bin_size 100 --stride 10 \
  --epochs 100 --batch_size 192 --lr 0.0001 --patience 10 --seed 42
```

The NPZ must contain `features`, `feature_names` and `labels`, with coordinate columns `chrom` and `bin_start`. Labels are 0/1/2. An optional row-aligned `region_id` array supplies groups; otherwise the script reconstructs merged extraction regions from the reference CNV table and context.

The script uses an approximately 10% stratified grouped holdout, five-fold grouped cross-validation on the remaining development data, and a scaler fitted only on each training subset. It selects the final refit epoch count from the median best fold epoch and refits on development data before evaluating the holdout. `--epochs` is the maximum per-fold count, not a requirement that the final checkpoint use that number of epochs.

Training selects all numeric feature columns except coordinate/metadata columns. To train a model usable by the current raw predictor, supply exactly the ten ordered inputs listed above; a checkpoint trained with another feature set is not automatically compatible.

Outputs include fold checkpoints and predictions, `split_manifest.npz`, `protocol.json`, `summary.json`, `oof_predictions.npz`, `test_predictions.npz` and `final_model.pth`. Keep the original feature file and row order together with these records for reproducibility.

## Analyze saved results

Analyze a completed training run:

```bash
python analyze_results.py --cv_dir /path/to/training_output --output analysis
```

Or analyze an individual labeled prediction file without plots:

```bash
python analyze_results.py \
  --results /path/to/test_predictions.npz \
  --output test_analysis --no_plots
```

The analyzer expects matching `labels` and `predictions` arrays in an NPZ. It computes sequence-label classification metrics, not CNV-event overlap metrics, and does not directly consume the raw predictor's TSV. Cross-validation scores are model-selection results; use the independent holdout for final evaluation.

## Comparison workflows

`train_sklearn.py` currently trains five classical classifiers using five input features and a shuffled region-based 80/20 split. It also contains a subsequent BEDtools overlap-precision calculation using external prediction TSVs. Edit the hard-coded paths and provide the required region and comparison tables before running it. Its feature set and split are not identical to the current 10-feature grouped-CV deep model; harmonize them before claiming a matched comparison.

`six_cnv_tools_combined.sh` collects separate caller job scripts. Extract and adapt the relevant tool section for the local paths, environment and scheduler instead of submitting the concatenated file as one unified workflow. External software and referenced helper scripts must be available.

## Interpretation and reproducibility

The pretrained interface applies a fixed model without training. Its successful execution does not by itself validate genome-wide CNV-event accuracy or reproduce published scores. The training workflow uses reference-CNV-guided regions, whereas raw prediction scans predefined genomic bins. Evaluate independent data using a clearly specified prediction unit and event-matching protocol, and retain the source version, checkpoint, feature definitions, training configuration, split manifest and evaluation outputs.


