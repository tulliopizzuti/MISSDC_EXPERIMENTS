# Imputation Experiments

This repository contains a collection of independent experimental pipelines for evaluating different data imputation approaches under controlled missingness scenarios.

Each top-level folder corresponds to a specific imputation approach. Every approach is fully self-contained and includes its own Conda environment, datasets, and execution scripts.



### Shared components in <APPROACH_FOLDER>

- **environment.yml**: Conda environment specification required to reproduce the experiments.
- **data/**: Datasets used for experiments.
- **missing.py**: Generation of missing values under different missingness mechanisms (MCAR, MAR, MNAR).
- **impute.py**: Implementation of the imputation of a dirty dataset.
- **score.py**: Evaluation metrics and scoring functions.
- **utils.py**: Shared utility functions.
- **run_experiment.py**: Main script used to run experiments.


## Environment Setup

To create the environment for a specific approach:

```bash
cd DATAWIG
conda env create -f environment.yml
```

Then activate it:

```bash
conda activate MISSDC_EXP_DATAWIG
```

### Environment names

- **TRIARD**: MISSDC_EXP_TRIARD
- **HYPERIMPUTE/GAIN**: MISSDC_EXP_HYPERIMPUTE_GAIN
- **DATAWIG**: MISSDC_EXP_DATAWIG


## Run Experiments

All experiments must be executed from within the corresponding approach folder.

Example:
```bash
python run_experiment.py --dataset iris etc...
```

### HyperImpute and GAIN

For experiments using HyperImpute and GAIN, the method must be specified as the first positional argument:


```bash
python run_experiment.py hyperimpute --dataset iris etc...
python run_experiment.py gain --dataset iris etc...
```

A dedicated field (method) has been added to results.csv to distinguish between these approaches.


## Parallel Execution Constraint

Parallel execution within the same approach directory is not supported due to shared intermediate artifacts generated during execution. This is due to shared temporary artifacts generated during execution, which could lead to concurrency issues, such as the injection of missing values into the datasets.
To run multiple experiments in parallel for the same approach, simply duplicate the entire folder. There is no need to duplicate the Conda environment; environments can be shared across all copies.









