# MethylCNV

MethylCNV is a deep learning tool for detecting copy number variations (CNVs) from whole-genome bisulfite sequencing (WGBS) data. It uses a bidirectional LSTM (BiLSTM) model with attention mechanisms to integrate multiple genomic features including read depth, GC content, and methylation penalties to identify deletions (DEL) and duplications (DUP) in genomes.

The repository supports:

- Prediction with a pretrained feature model, without retraining.
- Construction of labeled feature datasets from WGBS data.
- Region-grouped five-fold cross-validation with an independent holdout set.
- Independent statistical analysis and plotting from saved predictions.
- Classical classifier comparisons and example CNV caller workflows.

## Repository structure

```text
Methylcnv/
    predict.py                         Pretrained prediction entry point
    model.py                           feature BiLSTM model definition
    final_model.pt                     Pretrained feature model weights
    Depth_Methy_Features.py            Feature extraction module
    feature_scaler.pkl                 Saved input scaler
    feature_names.npy                  Ordered pretrained feature names
    FeatureCollection_pipeline.py      Labeled dataset construction
    ModelTrain.py                      Standalone region-grouped five-fold training
    analyze_results.py                 Independent metrics and visualization
    train_sklearn.py                   classical classification models
    six_cnv_tools_combined.sh           Collection of six caller job scripts
    reference_cnvs_human.bed            Reference CNV annotations
```

The repository includes the compatible `final_model.pt` pretrained weights. Keep this file beside `predict.py` together with `model.py`, `Depth_Methy_Features.py`, `feature_scaler.pkl`, and `feature_names.npy`. The default prediction command loads these supplied files automatically; users do not need to retrain the model. Use `--model` only when specifying another compatible checkpoint path.



## Installation

### Clone the repository

```bash
git clone https://github.com/RuiqiWangscau/Methylcnv.git
cd Methylcnv
```

### Create an environment

```bash
python -m venv methylcnv_env
source methylcnv_env/bin/activate
```

On Windows:

```powershell
methylcnv_env\Scripts\activate
```

### Install dependencies

```bash
python -m pip install torch numpy pandas pysam scipy \
    "scikit-learn>=1.1" matplotlib xgboost openpyxl
```

Use a compatible modern Python environment. These commands list dependencies; they are not a validated version lockfile. The pretrained scaler was saved with scikit-learn 1.1.1, so using a compatible environment is particularly important. Record the versions used for inference and investigate serialization/version warnings. The prediction script requires PyTorch APIs including `torch.inference_mode()` and `torch.load(..., weights_only=True)`.

Samtools is useful for preparing BAM and FASTA indexes. BWA-meth and MethylDackel are examples of upstream alignment and methylation-calling tools; these upstream steps are not executed by `predict.py`. The six caller workflows require their own external software installations.

Load model weights and pickle artifacts only from trusted sources.

## Quick start: prediction without training

### 1. Prepare user data

Users provide:

| Input | Requirement |
| --- | --- |
| WGBS BAM | Coordinate-sorted, indexed alignment file. |
| BAM index | Matching `.bai` or another index supported by pysam. |
| CpG methylation table | Same sample as the BAM; tab-separated with the header below. |
| Reference FASTA | Same assembly and sequences used for alignment. |
| FASTA index | Matching `.fai`. |
| Target region | A genomic interval to analyze. |

No training NPZ, reference CNV labels, or retraining is required for prediction.

The methylation table must contain these named columns:

```text
Chr	Start	End	MethPercent	MethCount	UnmethCount
```

Use 0-based, half-open coordinates. `MethPercent` is a percentage from 0 to 100. Convert methylation caller output to this format before use; raw bedGraph files may not include the required header or columns. A region with no methylation records causes the script to stop rather than silently produce predictions.

Create indexes if needed:

```bash
samtools index sample.sorted.bam
samtools faidx hg38.fa
```

Chromosome names with or without `chr` are resolved internally, but assembly differences cannot be corrected by renaming chromosomes. BAM and FASTA chromosome lengths must agree.

### 2. Check pretrained files

Keep these files together:

```text
predict.py
model.py
Depth_Methy_Features.py
final_model.pt
feature_scaler.pkl
feature_names.npy
```

The ordered prediction features are:

```text
rd_u
gc_content
methyl_penalty_norm
meth_density
meth_variance
```

The script calculates these features internally, uses the saved scaler with `transform()`, and loads the checkpoint in evaluation mode. It does not fit a new scaler, augment inputs, or update model weights.

### 3. Run prediction

```bash
python predict.py \
    --bam /path/to/sample.sorted.bam \
    --meth /path/to/sample_CpG.bed \
    --reference /path/to/hg38.fa \
    --output ./sample_predictions
```


### 4. Inspect outputs

```text
sample_predictions/
    bin_predictions.tsv
    extracted_features.tsv
    protocol.json
```

`bin_predictions.tsv` contains:

```text
chrom  start  end  prediction_id  prediction  prob_Normal  prob_DEL  prob_DUP  n_windows
```

The implemented class convention is 0=Normal, 1=DEL, and 2=DUP. Check it against the labeled data used for the released checkpoint. Each bin is assigned the class with the highest mean softmax probability across covering windows. A final window covers the region tail. No post-prediction smoothing is applied.

`extracted_features.tsv` records the numerical features in bin order. `protocol.json` records the input paths and inference parameters.

The output consists of **bin-level predictions**, not automatically validated or merged CNV events. No accuracy or precision can be calculated for a new unlabeled sample without independent reference annotations.

## Build a labeled training dataset

This workflow is separate from applying the pretrained prediction model. It requires a BAM, methylation table, FASTA, and a reference CNV table with this header:

```text
Chr	Start	End	Type
```

`Type` must be DEL or DUP. Coordinates and assemblies must be consistent across inputs.

`FeatureCollection_pipeline.py` calls `Depth_Methy_Features.py` and saves an NPZ containing `features`, `labels`, and `feature_names`. Before running it, edit the paths in its `main()` function:

- Reference FASTA.
- Sample BAM.
- Sample methylation table.
- Reference CNV table.
- Output directory and NPZ filename in `np.savez()`.

Then run:

```bash
python FeatureCollection_pipeline.py
```

Dataset options are constructor arguments in `TrainingDataBuilderRegion`, not CLI flags:

```python
builder = TrainingDataBuilderRegion(
    bin_size=100,
    context_size=10000,
    normal_keep_fraction=0.0,
    n_processes=4,
)
```

The default extracts reference-CNV regions with 10 kb flanks and no additional distant-normal intervals. Training bins are initialized as Normal; a bin overlapping a reference CNV by any positive length receives DEL/DUP according to the first overlapping record. This is a training-label rule, not a >50% event-matching rule.

Reference-guided extraction is for labeled dataset construction. It must not be presented as a truth-independent discovery workflow for unknown samples. Match the bin size and context between dataset generation and grouped training. Nonzero distant-region sampling requires compatible explicit region groups because default reconstruction only covers near-CNV intervals.

## Region-grouped five-fold training

`ModelTrain.py` includes the training model and is the standalone five-fold entry point. It does not require a separate `train_cnv_lstm.py`.

Use the exact NPZ generated by the previous step:

```bash
python ModelTrain.py \
    --data_path /path/to/generated_training_data.npz \
    --cnv_bed /path/to/reference_cnvs.bed \
    --save_root ./results/region_cv \
    --bin_size 100 \
    --context 10000 \
    --seq_length 300 \
    --stride 10 \
    --epochs 100 \
    --batch_size 192 \
    --lr 0.0001 \
    --patience 10 \
    --seed 42
```

This command trains a new model; it does not load the five-feature prediction checkpoint.

### Data partitioning and augmentation

1. Merge overlapping reference CNV intervals extended by their context into genomic groups, unless explicit NPZ region IDs are supplied.
2. Construct continuous windows within each group. The target is the label at window index `seq_length // 2`.
3. Reserve approximately 10% of windows as an independent holdout using the first fold of a shuffled 10-fold `StratifiedGroupKFold`.
4. Run five-fold `StratifiedGroupKFold` on the remaining development data. Groups remain intact while class proportions are approximately balanced.
5. Fit the feature scaler on unique training bins only.
6. Apply augmentation on the fly only to DEL/DUP training windows: Gaussian noise with mean 0 and standard deviation 0.05, with probability 0.5 after standardization. Validation and holdout data are not augmented.
7. Select fold checkpoints using validation weighted F1 and early stopping. Refit on all development data for the median selected epoch count, then evaluate the holdout once.

Checks verify that region groups are disjoint across subsets. `protocol.json` records the actual holdout window fraction. Validation-selected CV scores are model-selection results; use the independent holdout for final evaluation. This is region-level partitioning, not necessarily independent-individual or independent-chromosome validation.

### Training outputs

```text
results/region_cv/
    protocol.json
    split_manifest.npz
    fold_1/ ... fold_5/
        best.pth
        val_predictions.npz
        metrics.json
    oof_predictions.npz
    final_model.pth
    test_predictions.npz
    summary.json
```

The current training architecture uses per-feature BiLSTM encoders with bin attention, a feature-fusion BiLSTM with feature attention, and a three-class classifier. Input feature count is determined from the dataset after excluding metadata. It differs from the pretrained model in `model.py`.

## Independent analysis and plotting

No retraining is required to analyze saved labeled predictions:

```bash
python analyze_results.py \
    --results ./results/region_cv/test_predictions.npz \
    --output ./results/region_cv/test_analysis
```

For a completed five-fold experiment:

```bash
python analyze_results.py \
    --cv_dir ./results/region_cv \
    --output ./results/region_cv/analysis
```

The analysis input NPZ must contain one-dimensional `labels` and `predictions` arrays with IDs 0/1/2. The script does not directly read the TSV produced by `predict.py`.

Outputs include per-class TP/FP/FN, precision, recall, F1, macro/weighted metrics, and a confusion matrix. Five-fold analysis additionally reports per-fold metrics and mean/sample standard deviation. Attention plots require saved `feature_attention`; the current five-fold prediction files do not contain it. Training curves require a compatible CSV supplied with `--history`; the current five-fold training script does not export that CSV. Use `--no_plots` for numerical analysis only.

## Other methods and comparison results

`train_sklearn.py` contains KNN, Random Forest, XGBoost, Decision Tree, and AdaBoost. Update its `NPZ`, `REGIONS_TSV`, and `OUT` constants before running. It requires five features named `rd_u`, `gc_content`, `methyl_penalty`, `meth_density`, and `meth_variance`, plus coordinates and labels. The required region table contains `region_id`, `chrom`, `start`, and `end` and is not supplied as a standalone `regions.tsv` in this checkout.

This baseline script uses a seeded 80/20 region split, bin-level predictions, and majority voting per region. It is not the same protocol or evaluation unit as the five-fold sequence-center classifier. Explicitly align evaluation regions and splits before claiming a matched comparison.

`six_cnv_tools_combined.sh` collects the cn.mops, BreakDancer, CNVkit, CNVnator, DELLY, and Pindel workflows. It contains multiple Slurm headers and site-specific paths. Split out the relevant section and adapt it before job submission; do not submit the collection as a single Slurm job. Required helper scripts and external resources must also be supplied; the collection is not a self-contained six-caller installer.

