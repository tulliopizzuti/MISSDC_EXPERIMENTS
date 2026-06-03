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

### Usage
Use MISSDC_EXP_CORRUPT_DATASETS conda env
```bash
python corrupt_datasets.py \
    --datasets_folder datasets \
    --missingness MCAR MAR MNAR \
    --ratio 0.01 0.05 0.1 0.2 0.3 \
    --repetitions 5
```

#### for quicker tests

```bash
python corrupt_datasets.py \
    --datasets_folder datasets_sample \
    --missingness MCAR \
    --ratio 0.01 0.05 \
    --repetitions 3
```


### Output Structure

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

## Environment Architecture
MISSDC_EXPERIMENTS uses a main Conda environment called:

```bash
MISSDC_EXP
```

All framework scripts (`run_experiments.py`, `score.py`, etc.) are executed from this environment.

### Repository Structure

```text
MISSDC_EXPERIMENTS/
│
├── config.json
├── run_experiments.py
├── score.py
├── init_env.(bat/sh)
├── environment.yml
│
├── APPROACH_1/
│   ├── environment.yml
│   ├── impute.py
│   └── env_post_setup.(bat/sh)
│
├── APPROACH_2/
│   ├── environment.yml
│   ├── impute.py
│   └── env_post_setup.(bat/sh)
│
├── data/
└── results/
```
### Installing the Framework
Initialize all environments:

Linux/macOS:

```bash
bash init_env.sh
```

Windows:

```bat
init_env.bat
```

The initialization script:

1. Creates the main framework environment (`MISSDC_EXP`).
2. Creates the Conda environment required by each approach.
3. Executes any optional post-installation scripts.

After initialization, activate the main environment:

```bash
conda activate MISSDC_EXP
```

## Adding a New Approach

Adding a new imputation method requires four steps:

1. Create the approach folder.
2. Define its Conda environment.
3. Implement the imputation script.
4. Register the approach in `config.json`.

### Step 1: Create the Approach Folder

Create a new directory:

```text
MY_APPROACH/
```

The folder should contain:

```text
MY_APPROACH/
├── environment.yml
├── impute.py
└── env_post_setup.sh    # optional
```

---

### Step 2: Create `environment.yml`

Each approach must define its own Conda environment.

Example:

```yaml
name: MISSDC_EXP_MY_APPROACH

channels:
  - conda-forge

dependencies:
  - python=3.10
  - pandas
  - numpy
```

The environment name should be unique.

---

### Step 3: Implement `impute.py`

The framework executes the approach through `impute.py`.

The script receives:

- The path of the corrupted dataset as a positional argument.
- The haracter used to represent missing values in the dataset.
- The CSV separator.
- Additional parameters through command-line options.

Example:

```python
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument("dirty_file")
parser.add_argument("--output_file_path", required=True)

args = parser.parse_args()

df = pd.read_csv(args.dirty_file)

# perform imputation

df.to_csv(args.output_file_path, index=False)
```

The script **must create the output file** specified by `--output_file_path`.

If the file is not generated, the execution is considered failed.

---

### Optional step: Post-Installation Setup

Some approaches require additional installation steps after the Conda environment is created.

Examples:

- CUDA-enabled PyTorch
- DGL
- Downloading external models
- Native compilation steps

In these cases, create:

```text
env_post_setup.sh
```

Example:

```bash
conda run -n MISSDC_EXP_MY_APPROACH \
    pip install torch \
    --index-url https://download.pytorch.org/whl/cu121
```

The initialization script automatically executes this file after creating the environment.

---

### Step 4:  Registering the Approach

After creating the implementation, register the approach in `config.json`.

Example:

```json
{
  "approach_configuration": {
    "my_approach": {
      "approach_folder": "MY_APPROACH",
      "imputation_script": "impute.py",
      "sys_exec": [
        "conda",
        "run",
        "-n",
        "MISSDC_EXP_MY_APPROACH",
        "python"
      ]
    }
  }
}
```

---

### Understanding `config.json`

The configuration file contains two main sections:

```json
{
  "common_arguments": {},
  "approach_configuration": {}
}
```

---

#### Common Arguments

Arguments defined inside `common_arguments` are automatically available for every approach.

Example:

```json
{
  "common_arguments": {
    "seed": {
      "type": "int",
      "default": 42,
      "help": "Random seed"
    },
    "timeout": {
      "type": "int",
      "default": 3600,
      "help": "Execution timeout in seconds"
    }
  }
}
```

These arguments:

- are automatically exposed through the command line;
- are automatically forwarded to the imputation script;
- can be overridden when running experiments.

Example:

```bash
python run_experiments.py my_approach \
    --datasets_folder datasets \
    --seed 123
```

The framework automatically forwards:

```bash
--seed 123
```

to the imputation script.

---

#### Approach Configuration

Each approach must define:

```json
{
  "approach_folder": "MY_APPROACH",
  "imputation_script": "impute.py",
  "sys_exec": [...]
}
```

#### `approach_folder`

Working directory used when launching the approach.

Example:

```json
"approach_folder": "MY_APPROACH"
```

#### `imputation_script`

Script implementing the imputation method.

Example:

```json
"imputation_script": "impute.py"
```

#### `sys_exec`

Command prefix used to execute the approach.

Typically:

```json
[
  "conda",
  "run",
  "-n",
  "MISSDC_EXP_MY_APPROACH",
  "python"
]
```

The framework automatically builds commands such as:

```bash
conda run -n MISSDC_EXP_MY_APPROACH \
    python impute.py dirty.csv
```

---

#### Additional Arguments

Approach-specific parameters can be defined through:

```json
{
  "additional_arguments": {
    "epochs": {
      "type": "int",
      "default": 100,
      "help": "Number of training epochs"
    },
    "batch_size": {
      "type": "int",
      "default": 256
    }
  }
}
```

These parameters become automatically available from the command line.

Example:

```bash
python run_experiments.py my_approach \
    --datasets_folder datasets \
    --epochs 200 \
    --batch_size 512
```

The framework automatically forwards them to:

```bash
python impute.py dirty.csv \
    --epochs 200 \
    --batch_size 512
```

No modifications to `run_experiments.py` are required.

---

#### Excluding Common Arguments

Some approaches may not support all common parameters.

Use:

```json
{
  "common_arguments_to_exclude": [
    "seed",
    "timeout"
  ]
}
```

Excluded parameters will not be passed to the approach.

---

## Dataset Structure

The framework expects datasets organized as follows:

```text
datasets/
└── DATASET_NAME/
    └── MISSINGNESS/
        └── RATIO/
            └── REPETITION/
                ├── clean.csv
                └── dirty.csv
```

Example:

```text
datasets/
└── adult/
    └── MCAR/
        └── 0.1/
            └── 1/
                ├── clean.csv
                └── dirty.csv
```

---

## Running Experiments

Basic usage:

```bash
python run_experiments.py APPROACH_NAME \
    --datasets_folder datasets
```

Example:

```bash
python run_experiments.py grimp \
    --datasets_folder datasets
```

Specify a custom result directory:

```bash
python run_experiments.py grimp \
    --datasets_folder datasets \
    --results_folder results
```

Pass common parameters:

```bash
python run_experiments.py grimp \
    --datasets_folder datasets \
    --seed 42
```

Pass approach-specific parameters:

```bash
python run_experiments.py grimp \
    --datasets_folder datasets \
    --epochs 200
```

To see all available parameters for a specific approach:

```bash
python run_experiments.py APPROACH_NAME --help
```

---

## Output

For each dataset instance, the framework:

1. Executes the selected imputation method.
2. Generates `dirty_imputed.csv`.
3. Evaluates the repaired dataset using `score.py`.
4. Appends the results to:

```text
results/<approach>.csv
```

The resulting CSV contains execution metadata, timing information, status flags, and evaluation metrics.

---

## Updating the Environments

Whenever a new approach is added:

1. Create its folder.
2. Add its `environment.yml`.
3. Add its optional `env_post_setup.sh`.
4. Register it in `config.json`.
5. Run the initialization script again:

```bash
bash init_env.sh
```

or

```bat
init_env.bat
```

If an environment definition changes, it is recommended to remove the existing Conda environment and recreate it.


---

# Parallel Execution

Parallel execution of experiments is not recommended.

Since each approach may generate intermediate artifacts and temporary files inside the dataset and/or approach directory (in addition to the final imputed output), running multiple experiments on the same dataset folder can lead to file overwrites, conflicts, and race conditions.

To ensure correctness and reproducibility, experiments should be executed sequentially or on disjoint copies of the repository.