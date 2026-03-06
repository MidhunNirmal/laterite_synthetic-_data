# Laterite Synthetic Data Generation

A research pipeline for generating high-fidelity synthetic tabular data from real-world laterite (soil/geotechnical) datasets using multiple generative models: **CTGAN**, **TVAE**, **KDE-Copula GAN**, and **Tabular DDPM (Diffusion Model)**.

---

## Project Overview

This project addresses the challenge of limited geotechnical sample sizes by producing statistically faithful synthetic data. The pipeline covers:

1. **Deep statistical analysis** of the raw laterite dataset
2. **Evidence-based preprocessing** (imputation, normalization, encoding)
3. **Multi-model synthetic generation** with comparative evaluation
4. **Rigorous validation** using KS tests, Wasserstein distances, VIF analysis, and correlation preservation

---

## Dataset

| Property | Value |
|---|---|
| Source file | `laterite.csv` |
| Rows | 53 samples |
| Columns | 16 features (13 numeric, 2 categorical) |
| Scenarios | 16 soil classifications (e.g., MI, SM, MH, SC, GM-GC, …) |
| Preprocessed output | `laterite_preprocessed.csv` |

Key soil properties: Specific Gravity, Gravel %, Sand %, Silt %, Clay %, Liquid Limit, Plastic Limit, Plasticity Index, OMC %, MDD kN/m³, CBR %, Soil Classification.

---

## Project Structure

```
laterite/
│
├── laterite.csv                          # Raw dataset
├── laterite_preprocessed.csv             # Preprocessed dataset (final input to models)
│
├── laterite_preprocessing.py             # Preprocessing pipeline
├── laterite_analysis.py                  # Full deep-dive statistical analysis
├── laterite_analysis_simple.py           # Simplified analysis script
├── laterite_analysis.log                 # Structured decision log
├── laterite_analysis_results.json        # Machine-readable analysis output
├── laterite_synthetic_data_readiness_report.md  # Readiness report
│
├── advanced_imputation.py                # Advanced missing-value imputation (KNN, MICE, MissForest)
├── imputed_knn.csv                       # KNN-imputed dataset
├── imputed_mice.csv                      # MICE-imputed dataset
├── imputed_missforest.csv                # MissForest-imputed dataset
│
├── ctgan_imputation.py                   # CTGAN training & synthetic generation
├── generate_ctgan_500.py                 # Generate 500 CTGAN synthetic samples
├── ctgan_synthetic_500.csv               # CTGAN synthetic output (500 rows)
├── cT_gan.csv                            # CTGAN auxiliary dataset
│
├── kdecopula_laterite/                   # KDE-Copula GAN model (custom implementation)
│   ├── train_laterite.py                 # Training entry point
│   ├── trainer.py                        # Training loop & loss management
│   ├── generator.py                      # Generator network
│   ├── discriminator.py                  # Discriminator network
│   ├── gaussian_copula.py                # Gaussian copula transformation layer
│   ├── mixed_kde_encoder.py              # Mixed KDE encoder for numeric/categorical columns
│   ├── categorical_encoder.py            # Categorical feature encoder
│   ├── generate_laterite.py              # Inference / synthetic generation script
│   ├── config.yaml                       # Model hyperparameters
│   ├── laterite_kdecopula_model.pkl      # Saved KDE-Copula GAN model
│   ├── laterite_synthetic.csv            # KDE-Copula synthetic output (small)
│   └── my_synthetic.csv                  # KDE-Copula synthetic output (full)
│
├── comparison/                           # Cross-model evaluation
│   ├── compare_datasets.py               # Evaluation script (KS, Wasserstein, VIF)
│   ├── distance_metrics.csv              # KS / Wasserstein per feature (core models)
│   ├── distance_metrics_full.csv         # KS / Wasserstein per feature (all models)
│   ├── distribution_comparison.png       # Univariate distribution overlay plots
│   ├── boxplot_comparison.png            # Box plots per model
│   ├── correlation_comparison.png        # Correlation matrix heatmaps
│   ├── qq_plots.png                      # Q-Q plots
│   └── …                                 # Additional plots
│
├── report_of_laterite.md                 # Comprehensive research report
├── report_of_laterite.pdf                # PDF version of the report
└── prompt.txt                            # Analysis prompt specification
```

---

## Models

### 1. CTGAN
Conditional Tabular GAN. Uses mode-specific normalization and a conditional generator to handle mixed-type tabular data.

- Script: `ctgan_imputation.py`, `generate_ctgan_500.py`
- Output: `ctgan_synthetic_500.csv`

### 2. TVAE
Tabular Variational Autoencoder. Preferred over CTGAN for small datasets (<100 rows) due to stable latent space learning.

### 3. KDE-Copula GAN *(custom)*
A hybrid model combining:
- **Mixed KDE Encoder** to model marginal distributions of each feature
- **Gaussian Copula** to capture inter-feature dependency structure
- **Adversarial training** (GAN) to refine synthetic fidelity

- Entry point: `kdecopula_laterite/train_laterite.py`
- Config: `kdecopula_laterite/config.yaml`
- Generation: `kdecopula_laterite/generate_laterite.py`

### 4. Tabular DDPM (Diffusion Model)
MLP-based Denoising Diffusion Probabilistic Model adapted for tabular data. Generates samples by iteratively denoising from Gaussian noise.

---

## Preprocessing Pipeline

Run preprocessing before any model training:

```bash
python laterite_preprocessing.py
```

Steps applied (evidence-based, not heuristic):

| Step | Action |
|---|---|
| Special values | Replace `'NP'` → `0` for plasticity columns |
| High-missingness columns | Drop columns with >50% missing (e.g., `wPI`) |
| KNN imputation | `Specific Gravity`, `OMC %`, `MDD kN/m³` (11–15% missing) |
| Median imputation | `Gravel %`, `Liquid Limit %` (<10% missing) |
| Category retention | `Clay %` (35.8% missing), `CBR %` (49.1% missing) |
| Normalization | `StandardScaler` (scale ratio = 230×) |
| Encoding | Label encoding for soil classifications |

Advanced imputation alternatives (KNN, MICE, MissForest):

```bash
python advanced_imputation.py
```

---

## Running the Models

### CTGAN

```bash
python ctgan_imputation.py
python generate_ctgan_500.py
```

### KDE-Copula GAN

```bash
# Train
python kdecopula_laterite/train_laterite.py

# Generate synthetic samples
python kdecopula_laterite/generate_laterite.py
```

---

## Evaluation

Run the comparative evaluation across all models:

```bash
python comparison/compare_datasets.py
```

Metrics computed:
- **KS Statistic** — univariate distributional similarity (per feature)
- **Wasserstein Distance** — distributional shift magnitude
- **VIF (Variance Inflation Factor)** — multicollinearity preservation
- **Correlation matrix** comparison
- **Q-Q plots** and boxplots per model

---

## Key Findings

| Model | Avg. KS Stat ↓ | Avg. Wasserstein ↓ | Multicollinearity |
|---|---|---|---|
| KDE-Copula GAN | Best | Best | Well-preserved |
| TVAE | Good | Good | Moderate |
| CTGAN | Moderate | Moderate | Partially lost |
| Tabular DDPM | Moderate | Moderate | Moderate |

> Full results in `comparison/distance_metrics_full.csv` and `report_of_laterite.md`.

---

## Reports

| File | Description |
|---|---|
| [`laterite_synthetic_data_readiness_report.md`](laterite_synthetic_data_readiness_report.md) | Pre-training readiness assessment |
| [`report_of_laterite.md`](report_of_laterite.md) | Full research report with all model results |
| [`report_of_laterite.pdf`](report_of_laterite.pdf) | PDF version of the full report |

---

## Requirements

```bash
pip install pandas numpy scipy scikit-learn sdv ctgan torch pyyaml imbalanced-learn
```

Key dependencies:

| Library | Purpose |
|---|---|
| `sdv` / `ctgan` | CTGAN & TVAE models |
| `torch` | KDE-Copula GAN & Diffusion Model |
| `scipy` | KS tests, Wasserstein distance |
| `scikit-learn` | Preprocessing, KNN imputation |
| `pyyaml` | Model configuration |

---

## Citation / Context

This project is part of a research study evaluating synthetic tabular data generation methods for geotechnical laterite datasets with limited sample sizes. The KDE-Copula GAN model is a custom architecture designed to better preserve the marginal distributions and inter-feature correlations characteristic of geotechnical data.
#   l a t e r i t e _ s y n t h e t i c - _ d a t a  
 #   l a t e r i t e _ s y n t h e t i c - _ d a t a  
 