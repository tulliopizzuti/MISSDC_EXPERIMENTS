import argparse
import builtins
import subprocess
import sys
from datetime import datetime
import json
from pathlib import Path
from utils import *
import os

SCORE_SCRIPT = "score.py"
CONFIG_FILE = "config.json"


def load_config(config_file):
    path = Path(config_file)

    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)


def run_experiment(approach_name,
                   corrupted_dataset_folder,
                   dataset_name,
                   missingness,
                   ratio,
                   repetition,
                   approach_configuration,
                   common_args,
                   additional_args):

    IMPUTATION_SCRIPT = approach_configuration["imputation_script"]


    dataset_dir = corrupted_dataset_folder
    clean_file = dataset_dir / "clean.csv"
    dirty_file = dataset_dir / "dirty.csv"
    repaired_file = dataset_dir / "dirty_imputed.csv"
    ncols = get_num_columns(dirty_file)


    timestamp = datetime.now().isoformat(timespec="seconds")
    if not clean_file.exists():
        raise FileNotFoundError(f"clean.csv was not created: {clean_file}")

    if not dirty_file.exists():
        raise FileNotFoundError(f"dirty.csv was not created: {dirty_file}")

    if repaired_file.exists():
        repaired_file.unlink()


    imputation_cmd = approach_configuration["sys_exec"].copy()
    imputation_cmd += [
        IMPUTATION_SCRIPT,
        str(dirty_file)
    ]
    arguments_to_append = {}
    for k, v in common_args.items():
        if "common_arguments_to_exclude" not in approach_configuration or k not in approach_configuration["common_arguments_to_exclude"]:
            arguments_to_append[k] = v

    if "common_arguments_to_exclude" not in approach_configuration or "output_file_path" not in approach_configuration["common_arguments_to_exclude"]:
        arguments_to_append["output_file_path"] = str(repaired_file)

    for k, v in additional_args.items():
        if "common_arguments_to_exclude" not in approach_configuration or k not in approach_configuration["common_arguments_to_exclude"]:
            arguments_to_append[k] = v

    for k, v in arguments_to_append.items():
        imputation_cmd.append(f"--{k}")
        imputation_cmd.append(v)

    imputation_returncode, imputation_time, _, _ = run_command(imputation_cmd, cwd=approach_configuration["approach_folder"])

    if imputation_returncode != 0:
        row = build_result_row(
            approach=approach_name,
            dataset=dataset_name,
            missingness=missingness,
            ratio=ratio,
            ncols=ncols,
            timestamp=timestamp,
            repetition=repetition,
            status=f"{approach_name}_failed",
            clean_file=clean_file,
            dirty_file=dirty_file,
            repaired_file=repaired_file,
            imputation_time=imputation_time,
            score_time=None,
            scores=None,
            missing_time=None,
            error=f"{IMPUTATION_SCRIPT} {'TIME LIMIT' if imputation_returncode == 'TL' else 'failed'}", )

        return row

    if not repaired_file.exists():
        row = build_result_row(
            approach=approach_name,
            dataset=dataset_name,
            missingness=missingness,
            ratio=ratio,
            ncols=ncols,
            timestamp=timestamp,
            repetition=repetition,
            status=f"{approach_name}_failed",
            clean_file=clean_file,
            dirty_file=dirty_file,
            repaired_file=repaired_file,
            imputation_time=imputation_time,
            score_time=None,
            scores=None,
            missing_time=None,
            error=f"{approach_name} failed: Expected repaired file not found",
        )

        return row

    score_cmd = [
        sys.executable,
        SCORE_SCRIPT,
        str(clean_file),
        str(dirty_file),
        str(repaired_file),
    ]

    score_returncode, score_time, score_stdout, _ = run_command(score_cmd, cwd=None)

    if score_returncode != 0:
        row = build_result_row(
            approach=approach_name,
            dataset=dataset_name,
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
        approach=approach_name,
        dataset=dataset_name,
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


def add_arguments(parser: argparse.ArgumentParser, args_dict: dict):
    for arg_name, v in args_dict.items():
        cfg = v.copy()
        if "type" in cfg:
            cfg["type"] = getattr(builtins, cfg["type"])
        parser.add_argument(f"--{arg_name}", **cfg)


def parse_args():
    CONFIGURATION = load_config(CONFIG_FILE)
    APPROACHES_CONFIG = CONFIGURATION["approach_configuration"]
    COMMON_ARGUMENTS = CONFIGURATION["common_arguments"]
    parser = argparse.ArgumentParser(conflict_handler="resolve")
    parser.add_argument("approach", choices=APPROACHES_CONFIG.keys())
    parser.add_argument("--datasets_folder", required=True)
    parser.add_argument("--results_folder", default="results")
    add_arguments(parser, COMMON_ARGUMENTS)

    args, _ = parser.parse_known_args()
    approach_cfg = APPROACHES_CONFIG[args.approach]
    if "additional_arguments" in approach_cfg:
        add_arguments(parser, approach_cfg["additional_arguments"])
    args = parser.parse_args()

    parameters = {
        "approach": args.approach,
        "results_folder": args.results_folder,
        "datasets_folder": args.datasets_folder,
        "config": approach_cfg.copy()
    }
    if "additional_arguments" in approach_cfg:
        del parameters["config"]["additional_arguments"]

    parameters["common_arguments"] = {}
    for k in COMMON_ARGUMENTS:
        parameters["common_arguments"][k] = args.__dict__[k]
    parameters["additional_arguments"] = {}
    for k in APPROACHES_CONFIG[args.approach].get("additional_arguments", {}):
        parameters["additional_arguments"][k] = args.__dict__[k]

    return parameters


def main():
    approach_parameters = parse_args()

    common_args = approach_parameters["common_arguments"]
    additional_args = approach_parameters["additional_arguments"]
    approach_configuration = approach_parameters["config"]

    approach_name = approach_parameters["approach"]

    result_path = os.path.join(approach_parameters["results_folder"], f"{approach_name}.csv")

    datasets_folder = Path(approach_parameters["datasets_folder"])

    dataset_files = datasets_folder.rglob("dirty.csv")

    for dirty_file in dataset_files:
        parts = dirty_file.parts
        dataset_name = parts[-5]
        missingness = parts[-4]
        ratio = float(parts[-3])
        repetition = int(parts[-2])
        result = run_experiment(
            approach_name,
            dirty_file.parent.absolute(),
            dataset_name,
            missingness,
            ratio,
            repetition,
            approach_configuration,
            common_args,
            additional_args)
        append_result(result_path, result)


if __name__ == "__main__":
    main()
