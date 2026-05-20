# Imputation Experiments

This repository contains a collection of independent experimental pipelines for evaluating different data imputation approaches under controlled missingness scenarios.

Each top-level folder corresponds to a specific imputation approach. Every approach is fully self-contained and includes its own Conda environment, datasets, and execution scripts.


## Corrupt datasets
This script generates corrupted versions of CSV datasets by injecting missing values according to different missingness mechanisms and ratios.

### Arguments

| Argument | Description |
|---|---|
| `--datasets_folder` | Folder containing input CSV datasets |
| `--missingness` | List of missingness mechanisms (`MCAR`, `MAR`, `MNAR`) |
| `--ratio` | List of missing value ratios |
| `--repetitions` | Number of repetitions for each configuration |
| `--output` | Output root folder (default: `data`) |



### Input

The script expects a folder containing one or more `.csv` datasets:

```text
datasets/
    adult.csv
    ...
```

## Usage

```bash
python generate_datasets.py \
    --datasets_folder datasets \
    --missingness MCAR MAR MNAR \
    --ratio 0.1 0.2 0.3 \
    --repetitions 5
```


## Output Structure

For each dataset, missingness mechanism, ratio, and repetition, the script creates a dedicated folder:

```text
data/
    adult/
         MCAR/
            0.1/
                1/
                    clean.csv
                    dirty.csv
                    ...
                2/
                    ...
```

# Implementing a New Imputation Approach
Each imputation approach should be isolated in its own folder and Conda environment to avoid dependency conflicts between libraries and frameworks.

## Current imputation approaches structure
```text
\DATAWIG
    data\
        adult/
            MCAR/
                ...
    impute.py
    run_experiments.py
    ...


```

## Setup Steps

### 1. Create the Approach Folder
### 2. Create a Conda Environment
At the moment, the current solution provide an `environment.yml` file. However, any setup that allows the creation of a reproducible Conda environment is acceptable.
After defining the environment configuration file, the environment can be installed and activated with:

```bash
conda env create -f environment.yml
conda activate APPROACH_NAME
```

## 3. Implement `impute.py`

The `impute.py` script should:

- take a corrupted dataset as input, together with all the parameters required by the imputation approach
- execute the imputation method
- write the imputed dataset to disk

## 4. Implement `run_experiment.py`
The `run_experiment.py` script is responsible for iterating over all corrupted datasets, imputing them, executing evaluation scripts, and collecting metrics.

### 1. Define the Approach Name

Update the approach identifier:

```python
APPROACH_NAME = 'DATAWIG'
```

### 2. Define the Imputation Script

Update the script that performs the actual imputation:

```python
IMPUTATION_SCRIPT = "impute.py"
```

### 3. Adapt the Imputation Command

The main section that usually needs customization is the command used to launch the imputation process:

```python
imputation_cmd = [
    sys.executable,
    IMPUTATION_SCRIPT,
    str(dirty_file),
    "--output_file_path", str(repaired_file)
]
```

All the remaining components — dataset iteration, metric computation, result collection, logging, and CSV aggregation — can generally be reused without modification.




# Current environment names

- **TRIARD**: MISSDC_EXP_TRIARD
- **HYPERIMPUTE/GAIN**: MISSDC_EXP_HYPERIMPUTE_GAIN
- **DATAWIG**: MISSDC_EXP_DATAWIG


### Run Experiments

All experiments must be executed from within the corresponding approach folder.


### HyperImpute and GAIN

For experiments using HyperImpute and GAIN, the method must be specified as the first positional argument:


```bash
python run_experiment.py hyperimpute --dataset iris etc...
python run_experiment.py gain --dataset iris etc...
```

## Parallel Execution Constraint

Parallel execution within the same approach directory is not supported due to shared intermediate artifacts generated during execution. This is due to shared temporary artifacts generated during execution, which could lead to concurrency issues, such as the injection of missing values into the datasets.
To run multiple experiments in parallel for the same approach, simply duplicate the entire folder. There is no need to duplicate the Conda environment; environments can be shared across all copies.









