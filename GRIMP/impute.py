import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import argparse

import toml

from GRIMP import utils as GRIMP_UTILS
from GRIMP.logging import GrimpLogger
import GRIMP.pipeline as pipeline

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import run_command

CLEAN_FILE_FOLDER=os.path.join("grimpdata", "clean")
DIRTY_FILE_FOLDER=os.path.join("grimpdata", "dirty")
EMB_FILE_FOLDER=os.path.join("grimpdata", "pretrained-emb")
PRETRAIN_FILE_FOLDER=os.path.join("grimpdata", "to_pretrain")
RESULTS_FOLDER="grimpresults"
RESULTS_JSON_FOLDER=os.path.join(RESULTS_FOLDER, "json")
IMPUTED_DATASET_FOLDER=os.path.join(RESULTS_FOLDER, "imputed_datasets")
IMPUTED_DATASET_FILENAME = "dirty_allcolumns_0_imputed_grimp_ft.csv"
IMPUTED_DATASET_PATH = os.path.join(IMPUTED_DATASET_FOLDER, IMPUTED_DATASET_FILENAME)

PRETRAIN_SCRIPT="prepare_pretrained_embeddings.py"



def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def reset_folders():
    reset_dir(CLEAN_FILE_FOLDER)
    reset_dir(DIRTY_FILE_FOLDER)
    reset_dir(EMB_FILE_FOLDER)
    reset_dir(PRETRAIN_FILE_FOLDER)
    reset_dir(RESULTS_FOLDER)
    reset_dir(RESULTS_JSON_FOLDER)
    reset_dir(IMPUTED_DATASET_FOLDER)


def build_string_mapping(df: pd.DataFrame) -> dict:
    mapping = {}
    for col in df.columns:
        for val in df[col].dropna().unique():
            if not isinstance(val, (int, float, np.integer, np.floating)):
                normalized = str(val).replace(" ", "-")
                if val != normalized:
                    mapping[val] = normalized
    return mapping

def apply_string_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    return df.replace(mapping)


def grimp_imputation(
    incomplete_df: pd.DataFrame,
        dataset_null_char: str, cwd: str=None):
    reset_folders()

    dirty_df = incomplete_df.replace(dataset_null_char, np.nan)

    str_cols = dirty_df.select_dtypes(include="object").columns
    num_cols = dirty_df.select_dtypes(exclude="object").columns

    clean_df = dirty_df.copy()
    clean_df[str_cols] = dirty_df[str_cols].fillna("null")
    clean_df[num_cols] = dirty_df[num_cols].fillna(-1)



    dirty_df.columns = range(len(dirty_df.columns))
    clean_df=clean_df.copy()
    clean_df.columns = range(len(clean_df.columns))

    string_map = build_string_mapping(clean_df)
    clean_df = apply_string_mapping(clean_df, string_map)
    dirty_df = apply_string_mapping(dirty_df, string_map)

    clean_df.to_csv(os.path.join(CLEAN_FILE_FOLDER, "dirty.csv"), index=False, sep=',')
    dirty_df.to_csv(os.path.join(DIRTY_FILE_FOLDER, "dirty_allcolumns_0.csv"), index=False, sep=',')
    clean_df.to_csv(os.path.join(PRETRAIN_FILE_FOLDER, "dirty_0.csv"), index=False, sep=',')

    pretrain_cmd = [
        sys.executable,
        PRETRAIN_SCRIPT
    ]
    pretrain_returncode, pretrain_time, _, _ = run_command(pretrain_cmd, cwd=cwd)
    base_config = toml.load("default-config.toml")
    run_configs = GRIMP_UTILS.prepare_config_dict(base_config)
    config = run_configs[0]
    logger = GrimpLogger()
    d = GRIMP_UTILS.complete_config(config)
    args = SimpleNamespace(**d)
    logger.add_dict("parameters", vars(args))
    logger.add_run_name()
    logger.add_value("parameters", "num_estimators", 0)
    logger.add_time("start_training")
    graph_dataset, best_state, init_params = pipeline.run_training(args, logger)
    logger.add_time("end_training")
    logger.add_duration("start_training", "end_training", "duration_training")
    logger.print_summary()
    logger.save_json()
    pipeline.run_testing(
        args, graph_dataset, best_state, init_params, logger=logger
    )
    imputed_df = pd.read_csv(IMPUTED_DATASET_PATH, sep=',')
    for i, col in enumerate(imputed_df.columns):
            prefix = f"c{i}_"
            imputed_df[col] = imputed_df[col].apply(
                lambda val: val[len(prefix):] if isinstance(val, str) and val.startswith(prefix) else val
            )
    reverse_string_map = {v: k for k, v in string_map.items()}
    imputed_df = apply_string_mapping(imputed_df, reverse_string_map)
    imputed_df.columns = incomplete_df.columns
    reset_folders()
    return imputed_df




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='GRIMP_Exp',
        description='Run imputation using GRIMP')
    parser.add_argument('dataset_file_path', help="Path to the dataset file")
    parser.add_argument('--dataset_null_char', type=str, default='',
                        help='Character used to represent missing values in the dataset')
    parser.add_argument('--csv_sep', type=str, default=',',
                        help='CSV separator')
    parser.add_argument('--output_file_path', type=str, default='repaired.csv',
                        help='Repaired file path')

    args = parser.parse_args()
    dataset_file_path = args.dataset_file_path
    dataset_null_char = args.dataset_null_char
    csv_sep = args.csv_sep
    output_file_path = args.output_file_path

    # clean_dataset_file_path = os.path.join(
    #     os.path.dirname(args.dataset_file_path),
    #     "clean.csv"
    # )

    df = pd.read_csv(dataset_file_path, sep=csv_sep)
    # df_clean = pd.read_csv(clean_dataset_file_path, sep=csv_sep)
    imputed_df = grimp_imputation(df, dataset_null_char)



    imputed_df.to_csv(
        args.output_file_path,
        index=False,
        sep=args.csv_sep
    )

    print("Imputation completed.")