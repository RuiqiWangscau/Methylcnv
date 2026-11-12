# MethylCNV

MethylCNV is a BiLSTM-based framework with attention for predicting copy number variation from whole-genome bisulfite sequencing (WGBS) data. It extracts genomic and methylation-related features, applies saved preprocessing, and predicts Normal, deletion (DEL), or duplication (DUP) labels.

The repository provides a pretrained prediction workflow, training-data construction, model training, result analysis, and comparison workflows. Users can run prediction without training a model.

The study compares MethylCNV with six conventional CNV detection tools—BreakDancer, CNVnator, CNVkit, cn.mops, DELLY, and Pindel—and five machine-learning classifiers—K-nearest neighbors (KNN), Random Forest (RF), XGBoost, Decision Tree (DT), and AdaBoost. The repository includes scripts and comparison data supporting these analyses.

## Installation

Use Python 3 on Linux:

```bash
git clone https://github.com/RuiqiWangscau/Methylcnv.git
cd Methylcnv
python3 -m venv methylcnv_env
source methylcnv_env/bin/activate
python -m pip install -r requirements.txt
```

Use compatible PyTorch and scikit-learn versions when loading the saved checkpoint, preferably the original training environment. Install SAMtools separately if input indexes need to be generated.

Run the commands below from the repository root. Only load trusted checkpoint and feature NPZ files, because these formats may contain serialized Python objects.

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

Pretrained users do not need these steps.

### Construct labeled training data

The training-data builder uses a BAM, a methylation table, a reference FASTA, and a tab-separated reference CNV table with the following header:

```text
Chr	Start	End	Type
```

Use consistent genome assemblies and 0-based, end-exclusive coordinates. `Type` must be `DEL` or `DUP`. Record the reference annotation source and any sample filtering, coordinate conversion, or interval-merging steps used to prepare this table.

`FeatureCollection_pipeline.py` has input paths, output paths, and extraction settings inside `main()`. Edit these for the local dataset before running:

```bash
python FeatureCollection_pipeline.py
```

The resulting NPZ contains `features`, `labels`, and `feature_names`. Feature columns include chromosome and bin-start coordinates. An optional row-aligned `region_id` array can be supplied to the training script for explicit grouping. Training-label encoding is `0 = Normal`, `1 = DEL`, and `2 = DUP`.

This labeled-data workflow uses reference CNVs to define training regions; users predicting a new sample should use `predict.py`, which does not require reference labels.

### Train a model

To train a model from a compatible labeled feature NPZ:

```bash
python ModelTrain.py \
  --data_path /path/to/train_features.npz \
  --cnv_bed /path/to/reference_cnvs_human.bed \
  --save_root /path/to/new_training_output
```

Training selects feature columns from the input dataset. A model trained with a different feature set is not automatically compatible with the current raw-input predictor. Retain the original data, split manifest, training protocol and results when reproducing an experiment.

The training workflow keeps region groups separate between subsets, fits preprocessing only on the relevant training data, uses validation results for model selection, and performs the final evaluation on a held-out subset. Validation cross-validation scores and independent holdout scores should be reported separately. A held-out region is not necessarily an independent individual.

Training outputs include:

```text
training_output/
├── protocol.json
├── split_manifest.npz
├── fold_*/
│   ├── best.pth
│   ├── val_predictions.npz
│   └── metrics.json
├── oof_predictions.npz
├── final_model.pth
├── test_predictions.npz
└── summary.json
```

The final checkpoint and holdout results are generated after cross-validation and final refitting complete. Keep fold checkpoints separate from the final checkpoint.

### Analyze results without retraining

Analyze a completed training run:

```bash
python analyze_results.py --cv_dir /path/to/training_output --output analysis
```

Plotting requires matplotlib in addition to requirements.txt. The analysis reports classification metrics; CNV-event overlap evaluation is a separate procedure. Comparison scripts contain local paths and workflow-specific feature selections and splits, which must be reviewed before use.

A single labeled prediction file can also be analyzed without plots:

```bash
python analyze_results.py \
  --results /path/to/training_output/test_predictions.npz \
  --output test_analysis --no_plots
```

The input NPZ must contain matching one-dimensional `labels` and `predictions` arrays. Outputs include per-class precision, recall and F1, summary metrics, and a confusion matrix. Cross-validation analysis additionally summarizes individual fold results. Unlabeled predictions cannot provide precision or recall without independent reference labels. The analyzer does not directly consume the new-sample prediction TSV.

## Comparison workflows

`train_sklearn.py` contains workflows for five machine-learning classifiers: K-nearest neighbors, Random Forest, XGBoost, Decision Tree, and AdaBoost. Install `xgboost` separately, and edit its dataset, region-table, output and comparison paths before use. It requires a region table containing `region_id`, `chrom`, `start`, and `end`. Verify common held-out data, feature definitions and evaluation units before making matched comparisons; the script's existing selections and split do not automatically match the deep-model training workflow.

`six_cnv_tools_combined.sh` collects workflow examples for six conventional CNV detection tools: BreakDancer, CNVnator, CNVkit, cn.mops, DELLY, and Pindel. It contains separate job-script sections: extract and adapt the relevant section rather than submitting the concatenated file as one job. Install the external callers and their required helpers separately; they are not installed by `requirements.txt`.

## Reproducibility and support

Preserve the code version, checkpoint, input-file identity, feature-generation settings, reference annotation provenance, data splits and evaluation outputs for each experiment. Model-loading consistency does not establish CNV-event accuracy. Event-based evaluation requires an explicit interval construction and matching rule, including CNV type and the denominator used for overlap.

For questions or reproducibility reports, open an [issue](https://github.com/RuiqiWangscau/Methylcnv/issues) and include the command, software environment, and relevant log messages.



