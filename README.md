# Laterite Synthetic Data Generation

> Generating high-fidelity synthetic tabular data from a small real-world geotechnical (laterite soil) dataset using multiple generative models.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Models](#models)
- [Setup](#setup)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Results](#results)

---

## Overview

Geotechnical datasets are often small and hard to collect. This project builds a reproducible pipeline to:

1. Deeply analyze and preprocess a real laterite soil dataset
2. Generate synthetic samples using four generative models
3. Evaluate synthetic data fidelity using statistical metrics

**Models compared:** CTGAN · TVAE · KDE-Copula GAN (custom) · Tabular DDPM

---

## Dataset

| Property | Detail |
|---|---|
| File | `laterite.csv` |
| Samples | 53 rows |
| Features | 16 (13 numeric, 2 categorical) |
| Soil classifications | 16 scenarios (MI, SM, MH, SC, GM-GC, …) |

**Soil properties:** Specific Gravity, Gravel %, Sand %, Silt %, Clay %, Liquid Limit %, Plastic Limit %, Plasticity Index %, OMC %, MDD kN/m³, CBR %, Soil Classification.

> ⚠️ CSV files are excluded from version control via `.gitignore`. Place `laterite.csv` in the root directory before running.

---

## Project Structure

```
laterite/
│
├── laterite.csv                          # Raw dataset (not tracked)
├── laterite_preprocessed.csv             # Preprocessed output (not tracked)
│
├── laterite_preprocessing.py             # Preprocessing pipeline
├── laterite_analysis.py                  # Full statistical analysis
├── laterite_analysis_simple.py           # Simplified analysis
├── laterite_analysis_results.json        # Analysis output
├── laterite_synthetic_data_readiness_report.md
│
├── advanced_imputation.py                # KNN / MICE / MissForest imputation
├── verify_imputation.py                  # Imputation quality checks
│
├── ctgan_imputation.py                   # CTGAN training
├── generate_ctgan_500.py                 # CTGAN synthetic generation (500 samples)
│
├── kdecopula_laterite/                   # KDE-Copula GAN (custom model)
│   ├── config.yaml                       # Hyperparameters
│   ├── train_laterite.py                 # Training entry point
│   ├── generate_laterite.py              # Synthetic generation
│   ├── trainer.py                        # WGAN-GP training loop
│   ├── generator.py                      # Generator network
│   ├── discriminator.py                  # Discriminator network
│   ├── gaussian_copula.py                # Gaussian copula layer
│   ├── mixed_kde_encoder.py              # Mixed KDE encoder
│   └── categorical_encoder.py            # Categorical feature encoder
│
├── comparison/
│   ├── compare_datasets.py               # Statistical comparison across models
│   └── plots/                            # Generated plots
│
├── report_of_laterite.md                 # Full research report
├── report_of_laterite.pdf
├── requirements.txt
└── .gitignore
```

---

## Models

### CTGAN
Conditional Tabular GAN with mode-specific normalization. Standard baseline for synthetic tabular data generation.

### TVAE
Tabular Variational Autoencoder. More stable than CTGAN on small datasets.

### KDE-Copula GAN *(custom)*
A three-stage hybrid model:
1. **Mixed KDE Encoder** — models each feature's marginal distribution via kernel density estimation
2. **Gaussian Copula** — captures inter-feature dependency structure
3. **WGAN-GP** — adversarially refines samples in copula space

Architecture: `noise_dim=128`, `generator/discriminator=[256, 256, 256]`, `lr=0.0002`, `epochs=300`

### Tabular DDPM
MLP-based Denoising Diffusion Probabilistic Model adapted for tabular data.

---

## Setup

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| `sdv`, `ctgan` | CTGAN & TVAE generation |
| `torch` | KDE-Copula GAN & Diffusion Model |
| `scikit-learn` | Preprocessing & KNN imputation |
| `scipy` | KS tests, Wasserstein distance |
| `matplotlib`, `seaborn` | Visualisation |
| `pyyaml` | Model config parsing |

---

## Usage

### 1. Preprocess

```bash
python laterite_preprocessing.py
```

Applies evidence-based preprocessing: NP → 0 substitution, KNN/median imputation, StandardScaler normalization, label encoding.

Advanced imputation (KNN · MICE · MissForest):

```bash
python advanced_imputation.py
```

### 2. Train Models

**CTGAN:**
```bash
python ctgan_imputation.py
python generate_ctgan_500.py
```

**KDE-Copula GAN:**
```bash
cd kdecopula_laterite
python train_laterite.py      # Trains and saves model to laterite_kdecopula_model.pkl
python generate_laterite.py   # Loads model and generates synthetic samples
```

### 3. Compare Models

```bash
python comparison/compare_datasets.py
```

---

## Evaluation

Metrics computed per model:

- **KS Statistic** — per-feature distributional similarity (lower = better)
- **Wasserstein Distance** — distributional shift magnitude (lower = better)
- **VIF** — multicollinearity preservation
- **Correlation matrix** — pairwise feature relationship preservation

Results saved to `comparison/distance_metrics_full.csv`.

---

## Results

| Model | Avg. KS Stat ↓ | Wasserstein ↓ | Multicollinearity |
|---|---|---|---|
| KDE-Copula GAN | **Best** | **Best** | Well-preserved |
| TVAE | Good | Good | Moderate |
| Tabular DDPM | Moderate | Moderate | Moderate |
| CTGAN | Moderate | Moderate | Partially lost |

See [`report_of_laterite.md`](report_of_laterite.md) for the full analysis.
