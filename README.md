# Dynamic Hydrophobicity for OMP Rejection

Research code accompanying the manuscript:

> **Dynamic hydrophobicity representation for predicting and interpreting ionizable organic micropollutant rejection by polyamide membranes**

Jincheng Ma, Airan Hu, Yanling Liu, and Shengji Xia  
Tongji University, Shanghai, China

## Overview

This repository contains research code and supporting computational materials for predicting and interpreting organic micropollutant (OMP) rejection by polyamide nanofiltration and reverse-osmosis membranes. It is organized around the dynamic hydrophobicity framework developed in the accompanying study.

The current model-training module covers:

1. Load and filter modeling data.
2. Define input features and the rejection target.
3. Repeat train/test splitting across random seeds.
4. Impute missing values in a preprocessing pipeline.
5. Perform cross-validated hyperparameter selection.
6. Train an XGBoost regression model.
7. Calculate adjusted R², RMSE, and MAE.
8. Aggregate and save evaluation results.

## Project scope

This repository is the public code home for the broader study. Its scope includes model development, performance evaluation, SHAP and partial-dependence interpretation, molecular-descriptor analysis, scientific visualization, and related supporting workflows. Modules and reproducibility materials will be added as they are prepared for public release.

The current release contains the model-training and evaluation module. Reproducing numerical results from the manuscript also requires the corresponding study data and experiment configuration.

## Structure

```text
.
├── src/
│   └── model_training.py           # Model training and evaluation
├── .gitignore
├── CITATION.cff
├── LICENSE
├── README.md
└── requirements.txt
```

## Environment

- Python 3.10 or newer
- NumPy
- pandas
- scikit-learn
- XGBoost

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Preparing input data

The model code expects a CSV file at:

```text
data.csv
```

Before running it, replace the placeholders in `BASE_INPUT_COLS` and `OBJ_COL` with the column names in your authorized dataset. If the dataset includes a train/validation indicator, add the appropriate filtering rule in `load_data()`.

No example dataset is included. Users are responsible for ensuring that any supplied data can be legally and ethically shared and used.

## Running the workflow

```bash
python src/model_training.py
```

The script writes summary outputs to:

```text
File/results/
├── metrics_summary.csv
└── selected_params.json
```

## Methodological note

The accompanying study introduces the ionization-induced hydrophobicity shift, `IHS = LogP - LogD`, and combines it with LogP to represent intrinsic hydrophobicity and ionization-related changes separately. Consult the manuscript for the scientific rationale, study design, applicability domain, and interpretation of results.

## Citation

If this repository supports your work, please cite the accompanying manuscript and this software repository. GitHub can generate citation metadata from [`CITATION.cff`](CITATION.cff). Publication details and a DOI will be added after they become available.

## License

The source code is released under the [MIT License](LICENSE). The license applies only to the files in this repository; it does not grant rights to third-party datasets, publications, molecular structures, or other research assets.

## Contact

For scientific questions, please contact the corresponding author listed in the manuscript.
