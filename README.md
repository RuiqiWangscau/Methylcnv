# MethylCNV

MethylCNV is a BiLSTM-based framework with attention for predicting copy number variation from whole-genome bisulfite sequencing (WGBS) data. It extracts genomic and methylation-related features, applies saved preprocessing, and predicts Normal, deletion (DEL), or duplication (DUP) labels.

The repository provides a pretrained prediction workflow, training-data construction, model training, result analysis, and comparison workflows. Users can run prediction without training a model.

## Installation

Use Python 3 on Linux:

```bash
git clone https://github.com/RuiqiWangscau/Methylcnv.git
cd Methylcnv
python -m pip install -r requirements.txt
```

Use compatible PyTorch and scikit-learn versions when loading the saved checkpoint, preferably the original training environment. Install SAMtools separately if input indexes need to be generated.

## Predict with the pretrained model

Provide a coordinate-sorted BAM, the corresponding CpG methylation table, and the matching reference FASTA:

```bash
python predict.py \
  --bam sample.sorted.bam \
  --meth sample_CpG.bed \
  --reference hg38.fa \
  --output sample_predictions
```

Keep `predict.py` and `final_model.pth` together. The program automatically loads the trained weights and saved preprocessing information. Training, reference CNV annotations, and true labels are not required for prediction. Use a new output directory for each run.

The BAM must have a BAI/CSI index and the FASTA an FAI index. The BAM and reference must use the same genome assembly. Generate missing indexes with:

```bash
samtools index sample.sorted.bam
samtools faidx hg38.fa
```

The methylation table must be tab-separated and include this header:

```text
Chr	Start	End	MethPercent	MethCount	UnmethCount
chr1	100	101	50	5	5
```

Coordinates are 0-based and end-exclusive; MethPercent uses 0-100. Other methylation formats must be converted before use. The filename extension alone does not determine compatibility.

Prediction extracts features in genomic blocks, uses the training scaler without refitting it, and outputs classification probabilities. Target chromosomes and resource settings can be selected using the options shown by:

```bash
python predict.py --help
```

On a cluster, run long predictions on compute nodes using the local scheduler.

## Output

- `window_predictions.tsv`: genomic window and center-bin coordinates, predicted class, and class probabilities.
- `inference_config.json`: inputs, model configuration, processing status, and completed or skipped chromosomes.

Each prediction corresponds to a window center. Window boundaries are not detected CNV event boundaries. Check that the configuration reports `status: completed`; interrupted runs may leave partial output.

## Repository contents

| File | Function |
| --- | --- |
| `predict.py` | Integrated feature extraction and pretrained prediction. |
| `final_model.pth` | Trained weights, scaler, ordered features and model configuration. |
| `requirements.txt` | Python dependencies. |
| `ModelTrain.py` | Grouped cross-validation, final training and holdout evaluation. |
| `FeatureCollection_pipeline.py` | Labeled training-data construction. Edit its built-in input/output paths before use. |
| `Depth_Methy_Features.py` | Feature calculations used by the training-data builder. |
| `analyze_results.py` | Classification metrics and plots from saved labeled predictions. |
| `train_sklearn.py` | Classical-classifier and comparison workflow with dataset-specific settings. |
| `six_cnv_tools_combined.sh` | Separate CNV-caller job-script examples that require adaptation. |
| `reference_cnvs_human.bed` | Reference CNV annotations. |
| `merged_predictions_all_methods.xlsx` | Aggregated comparison data. |

Prediction and training use architecturally identical BiLSTM-attention model definitions. The supplied checkpoint was checked using strict loading, and the implementations produced identical outputs for identical inputs. These checks establish implementation compatibility, not reproduction of manuscript performance.

## Training and analysis

Pretrained users do not need these steps. To train a model from a compatible labeled feature NPZ:

```bash
python ModelTrain.py \
  --data_path /path/to/train_features.npz \
  --cnv_bed /path/to/reference_cnvs_human.bed \
  --save_root /path/to/new_training_output
```

Training selects feature columns from the input dataset. A model trained with a different feature set is not automatically compatible with the current raw-input predictor. Retain the original data, split manifest, training protocol and results when reproducing an experiment.

Analyze a completed training run:

```bash
python analyze_results.py --cv_dir /path/to/training_output --output analysis
```

Plotting requires matplotlib in addition to requirements.txt. The analysis reports classification metrics; CNV-event overlap evaluation is a separate procedure. Comparison scripts contain local paths and workflow-specific feature selections and splits, which must be reviewed before use.


