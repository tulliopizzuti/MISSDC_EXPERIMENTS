import argparse
import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import *




TRIARD_DATASET_FOLDER =  "dataset"
SCORE_SCRIPT = Path(__file__).resolve().parent.parent / "score.py"
IMPUTATION_SCRIPT = "impute.py"
APPROACH_NAME = "TRIARD"


def run_experiment(corrupted_dataset_folder, dataset, missingness, ratio, repetition, cwd):
    dataset_dir = corrupted_dataset_folder

    clean_file = dataset_dir / "clean.csv"
    dirty_file = dataset_dir / "dirty.csv"
    repaired_file = dataset_dir / "dirty_imputed.csv"

    timestamp = datetime.now().isoformat(timespec="seconds")



    if not clean_file.exists():
        raise FileNotFoundError(f"clean.csv was not created: {clean_file}")

    if not dirty_file.exists():
        raise FileNotFoundError(f"dirty.csv was not created: {dirty_file}")

    if repaired_file.exists():
        repaired_file.unlink()

    ncols = get_num_columns(dirty_file)


    # Read dataset
    df_with_comma = pd.read_csv(dirty_file, sep=',')
    df_with_semicolon_path = os.path.join(TRIARD_DATASET_FOLDER, f"{dataset}_{str(df_with_comma.isna().sum().sum())}_1.csv")



    # Renuver works with ? & semicolon as CSV separator
    df_with_comma = df_with_comma.fillna("?")

    # Save the dataset to the TRIARD folder using ';'
    df_with_comma.to_csv(df_with_semicolon_path, sep=';', index=None)

    imputation_cmd = [
        sys.executable,
        IMPUTATION_SCRIPT,
        df_with_semicolon_path,
        repaired_file,
        "--dataset_null_char", "?",
        "--iterations", "3",
        "--min_score_to_reach", "0.5",
        # "--remove_log", #Uncomment this line to prevent iteration logs from being saved
    ]

    imputation_returncode, imputation_time, _, _ = run_command(imputation_cmd, cwd=cwd)

    if imputation_returncode != 0:
        row = build_result_row(
            approach=APPROACH_NAME,
            dataset=dataset,
            missingness=missingness,
            ratio=ratio,
            ncols=ncols,
            timestamp=timestamp,
            repetition=repetition,
            status=f"{APPROACH_NAME}_failed",
            clean_file=clean_file,
            dirty_file=dirty_file,
            repaired_file=repaired_file,
            missing_time=None,
            imputation_time=imputation_time,
            score_time=None,
            scores=None,
            error=f"{IMPUTATION_SCRIPT} {'TIME LIMIT' if imputation_returncode == 'TL' else 'failed'}",
        )
        return row

    if not repaired_file.exists():
        row = build_result_row(
            approach=APPROACH_NAME,
            dataset=dataset,
            missingness=missingness,
            ratio=ratio,
            ncols=ncols,
            timestamp=timestamp,
            repetition=repetition,
            status=f"{APPROACH_NAME}_failed",
            clean_file=clean_file,
            dirty_file=dirty_file,
            repaired_file=repaired_file,
            imputation_time=imputation_time,
            score_time=None,
            scores=None,
            missing_time=None,
            error=f"{APPROACH_NAME} failed: Expected repaired file not found",
        )

        return row

    score_cmd = [
        sys.executable,
        SCORE_SCRIPT,
        str(clean_file),
        str(dirty_file),
        str(repaired_file),
    ]

    score_returncode, score_time, score_stdout, _ = run_command(score_cmd, cwd=cwd)

    if score_returncode != 0:
        row = build_result_row(
            approach=APPROACH_NAME,
            dataset=dataset,
            missingness=missingness,
            ratio=ratio,
            ncols=ncols,
            timestamp=timestamp,
            repetition=repetition,
            status="score_failed",
            clean_file=clean_file,
            dirty_file=dirty_file,
            repaired_file=repaired_file,
            missing_time=None,
            imputation_time=imputation_time,
            score_time=score_time,
            scores=None,
            error="score.py failed",
        )
        return row

    scores = parse_score_output(score_stdout)

    row = build_result_row(
        approach=APPROACH_NAME,
        dataset=dataset,
        missingness=missingness,
        ratio=ratio,
        ncols=ncols,
        timestamp=timestamp,
        repetition=repetition,
        status="ok",
        clean_file=clean_file,
        dirty_file=dirty_file,
        repaired_file=repaired_file,
        missing_time=None,
        imputation_time=imputation_time,
        score_time=score_time,
        scores=scores,
        error="",
    )

    return row





def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets_folder", required=True)

    parser.add_argument("--results", default="results/results.csv")


    parser.add_argument("--cwd", default=None)
    parser.add_argument("--java-opts", nargs="*", default=[])

    return parser.parse_args()


def main():
    args = parse_args()
    datasets_folder = Path(args.datasets_folder)
    dataset_files = datasets_folder.rglob("dirty.csv")

    for dirty_file in dataset_files:
        parts = dirty_file.parts
        dataset_name = parts[1]
        missingness = parts[2]
        ratio = float(parts[3])
        repetition = int(parts[4])
        result = run_experiment(dirty_file.parent, dataset_name, missingness, ratio, repetition, args.cwd)
        append_result(args.results, result)


if __name__ == "__main__":
    main()