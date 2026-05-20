from pathlib import Path

import pandas as pd
from typing import List
import numpy as np
import os
import csv
import time
import subprocess
import re


def compute_diff_list(clean_df: pd.DataFrame, dirty_df: pd.DataFrame) -> List[dict]:
    """
    Compare two pandas DataFrames and return a list of dictionaries containing the positions and values where they differ.

    Args:
        clean_df (pd.DataFrame): The clean DataFrame to compare.
        dirty_df (pd.DataFrame): The dirty DataFrame to compare.

    Returns:
        List[dict]: A list of dictionaries containing the positions and values where the DataFrames differ.
    """

    # if not clean_df.columns.equals(dirty_df.columns):
    #     print("The DataFrames have different column names.")
    #     return None

    # reindex the DataFrames to align their columns and index labels
    clean_df = clean_df.reindex(
        columns=dirty_df.columns, index=dirty_df.index)

    # compare the DataFrames and get the positions of different values
    diff = (clean_df != dirty_df)
    diff_pos = diff.stack()[diff.stack()].index.tolist()

    # iterate through the positions where the DataFrames differ
    diff_list = []
    for position in diff_pos:
        clean_val = clean_df.loc[position]
        dirty_val = dirty_df.loc[position]

        # create a dictionary to store the values
        diff_dict = {'position': position,
                     'clean': clean_val, 'dirty': dirty_val}

        diff_list.append(diff_dict)

    return diff_list




def get_num_columns(dataset_file):
    df = pd.read_csv(dataset_file)
    return len(df.columns)



#TODO timeout as parameter
def run_command(cmd, cwd=None, timeout=43200):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )

        elapsed = time.time() - start

        if proc.stdout:
            print("\nSTDOUT:")
            print(proc.stdout)

        if proc.stderr:
            print("\nSTDERR:")
            print(proc.stderr)

        return proc.returncode, elapsed, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        elapsed = time.time() - start

        print(f"\nProcess timed out after {timeout} seconds")

        return (
            "TL",
            elapsed,
            e.stdout if e.stdout else "",
            e.stderr if e.stderr else "",
        )


def parse_score_output(output):
    def extract_float(label):
        match = re.search(rf"{label}:\s*([0-9.]+)", output)
        return float(match.group(1)) if match else None

    def extract_int(label):
        match = re.search(rf"{label}:\s*([0-9]+)", output)
        return int(match.group(1)) if match else None

    return {
        "precision": extract_float("Precision"),
        "recall": extract_float("Recall"),
        "f1": extract_float("F1"),
        "dirty_nulls": extract_int("Number of nulls in dirty"),
        "imputed_nulls": extract_int("Number of nulls in imputed"),
    }


def append_result(results_file, row):
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    exists = results_file.exists()

    with open(results_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)

def build_result_row(
    approach,
    dataset,
    missingness,
    ncols,
    ratio,
    timestamp,
    repetition,
    status,
    clean_file,
    dirty_file,
    repaired_file,
    missing_time,
    imputation_time,
    score_time,
    scores,
    error,

):
    scores = scores or {}

    return {
        "approach": approach,
        "timestamp": timestamp,
        "status": status,
        "dataset": dataset,
        "missingness": missingness,
        "ncols": ncols,
        "ratio": ratio,
        "repetition": repetition,
        "precision": scores.get("precision", ""),
        "recall": scores.get("recall", ""),
        "f1": scores.get("f1", ""),
        "dirty_nulls": scores.get("dirty_nulls", ""),
        "imputed_nulls": scores.get("imputed_nulls", ""),
        "missing_time_seconds": round(missing_time, 4) if missing_time is not None else "",
        f"imputation_time_seconds": round(imputation_time, 4) if imputation_time is not None else "",
        "score_time_seconds": round(score_time, 4) if score_time is not None else "",
        "clean_file": str(clean_file),
        "dirty_file": str(dirty_file),
        "repaired_file": str(repaired_file),
        "jar": 'NONE',
        "error": error,
    }