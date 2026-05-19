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

MISSING_SCRIPT = "missing.py"
SCORE_SCRIPT = "score.py"
IMPUTATION_SCRIPT = "impute.py"



def resolve_data_root(cwd=None):
    """Resolve data root, allowing override via MISSDC_DATA_ROOT."""
    override = os.environ.get("MISSDC_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()

    base = Path(cwd) if cwd else Path(__file__).resolve().parent
    return (base / "data").resolve()


def run_command(cmd, cwd=None):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))

    start = time.time()

    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    elapsed = time.time() - start

    if proc.stdout:
        print("\nSTDOUT:")
        print(proc.stdout)

    if proc.stderr:
        print("\nSTDERR:")
        print(proc.stderr)

    return proc.returncode, elapsed, proc.stdout, proc.stderr


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


def run_experiment(args, repetition):




    data_root = resolve_data_root(args.cwd)
    dataset_dir = data_root / args.dataset

    clean_file = dataset_dir / "clean.csv"
    dirty_file = dataset_dir / "dirty.csv"
    repaired_file = dataset_dir / "dirty_imputed.csv"

    timestamp = datetime.now().isoformat(timespec="seconds")

    # Empty string because missing.py does:
    # map2Int = args[4]
    # Passing "False" would still be truthy.
    map2int_false = ""

    missing_cmd = [
        sys.executable,
        MISSING_SCRIPT,
        args.dataset,
        str(args.ncols),
        args.missingness,
        str(args.ratio),
        map2int_false,
    ]

    missing_returncode, missing_time, _, _ = run_command(missing_cmd, cwd=args.cwd)

    if missing_returncode != 0:
        row = build_result_row(
            args=args,
            timestamp=timestamp,
            repetition=repetition,
            status="missing_failed",
            clean_file=clean_file,
            dirty_file=dirty_file,
            repaired_file=repaired_file,
            missing_time=missing_time,
            imputation_time=None,
            score_time=None,
            scores=None,
            error="missing.py failed",
        )
        append_result(args.results, row)
        return row

    source_dataset_file = dataset_dir / f"{args.dataset}.csv"
    if not source_dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset source file not found: {source_dataset_file}. "
            "Set triard_DATA_ROOT or run with --cwd pointing to the project root."
        )

    if not clean_file.exists():
        raise FileNotFoundError(f"clean.csv was not created: {clean_file}")

    if not dirty_file.exists():
        raise FileNotFoundError(f"dirty.csv was not created: {dirty_file}")

    if repaired_file.exists():
        repaired_file.unlink()


    imputation_cmd = [
        sys.executable,
        IMPUTATION_SCRIPT,
        args.imputation_method,
        dirty_file,
        "--output_file_path", repaired_file
    ]

    imputation_returncode, imputation_time, _, _ = run_command(imputation_cmd, cwd=args.cwd)

    if imputation_returncode != 0:
        row = build_result_row(
            args=args,
            timestamp=timestamp,
            repetition=repetition,
            status=f"{args.imputation_method}_failed",
            clean_file=clean_file,
            dirty_file=dirty_file,
            repaired_file=repaired_file,
            missing_time=missing_time,
            imputation_time=imputation_time,
            score_time=None,
            scores=None,
            error=f"{IMPUTATION_SCRIPT} failed",
        )
        append_result(args.results, row)
        return row

    if not repaired_file.exists():
        raise FileNotFoundError(
            f"Expected repaired file not found: {repaired_file}"
        )

    score_cmd = [
        sys.executable,
        SCORE_SCRIPT,
        str(clean_file),
        str(dirty_file),
        str(repaired_file),
    ]

    score_returncode, score_time, score_stdout, _ = run_command(score_cmd, cwd=args.cwd)

    if score_returncode != 0:
        row = build_result_row(
            args=args,
            timestamp=timestamp,
            repetition=repetition,
            status="score_failed",
            clean_file=clean_file,
            dirty_file=dirty_file,
            repaired_file=repaired_file,
            missing_time=missing_time,
            imputation_time=imputation_time,
            score_time=score_time,
            scores=None,
            error="score.py failed",
        )
        append_result(args.results, row)
        return row

    scores = parse_score_output(score_stdout)

    row = build_result_row(
        args=args,
        timestamp=timestamp,
        repetition=repetition,
        status="ok",
        clean_file=clean_file,
        dirty_file=dirty_file,
        repaired_file=repaired_file,
        missing_time=missing_time,
        imputation_time=imputation_time,
        score_time=score_time,
        scores=scores,
        error="",
    )

    append_result(args.results, row)
    return row


def build_result_row(
    args,
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
        "timestamp": timestamp,
        "method": args.imputation_method,
        "status": status,
        "dataset": args.dataset,
        "missingness": args.missingness,
        "ncols": args.ncols,
        "ratio": args.ratio,
        "repetition": repetition,
        "precision": scores.get("precision", ""),
        "recall": scores.get("recall", ""),
        "f1": scores.get("f1", ""),
        "dirty_nulls": scores.get("dirty_nulls", ""),
        "imputed_nulls": scores.get("imputed_nulls", ""),
        "missing_time_seconds": round(missing_time, 4) if missing_time is not None else "",
        "triard_time_seconds": round(imputation_time, 4) if imputation_time is not None else "",
        "score_time_seconds": round(score_time, 4) if score_time is not None else "",
        "clean_file": str(clean_file),
        "dirty_file": str(dirty_file),
        "repaired_file": str(repaired_file),
        "jar": 'NONE',
        "error": error,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('imputation_method')
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--missingness", required=True, choices=["MCAR", "MAR", "MNAR"])
    parser.add_argument("--ncols", required=True, type=int)
    parser.add_argument("--ratio", required=True, type=float)

    parser.add_argument("--results", default="results/results.csv")

    parser.add_argument("--repetitions", type=int, default=1)

    parser.add_argument("--cwd", default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    for repetition in range(args.repetitions):
        row = run_experiment(args, repetition)
        print("\nSaved result:")
        print(row)


if __name__ == "__main__":
    main()