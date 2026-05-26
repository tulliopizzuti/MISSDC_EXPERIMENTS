
import sys
import os
import pandas as pd
sys.path.insert(0, '../tools')
from utils import parse_args, computeScores, save_score, save_score_col, computeImputationRatio

import toml
import argparse
from types import SimpleNamespace
import GRIMP.utils as utils
import GRIMP.pipeline as pipeline
from GRIMP.logging import GrimpLogger
from pathlib import Path
import time

method_name = 'grimp'


def complete_config(config, clean_file, dirty_file):
    
    config["ground_truth"] = Path(clean_file)
    config["dirty_dataset"] = Path(dirty_file)

    config["text_embs"] = None
    config["imputed_df_tag"] = None

    if "training_columns" not in config:
        config["training_columns"] = None

    if "ignore_columns" not in config:
        config["ignore_columns"] = None

    if "ignore_num_cols" not in config:
        config["ignore_num_cols"] = None

    if "cat_columns" not in config:
        config["cat_columns"] = None

    if "target_columns" not in config:
        config["target_columns"] = None

    if "fd_strategy" not in config:
        config["fd_strategy"] = "attention"

    return config

def curate_imputed_df(df):
    for i, col in enumerate(df.columns):
        # print(i, col)
        if i < 10:

            df[col] = df[col].apply(lambda x: x[3:] if type(
                x) == str and x.startswith("c") else x)
        else:
            df[col] = df[col].apply(lambda x: x[4:] if type(
                x) == str and x.startswith("c") else x)

    return df

def main():

    print('GRIMP Imputer')
    

    dataset, missing_type, missing_ratio, col, outputpath, resultfile = parse_args()
    print('Configuration: ', dataset, missing_type,
          missing_ratio, col, outputpath, resultfile)

    start_time = time.time()

    config_path = '../tools/GRIMP/config/default-config.toml'
    base_config = toml.load(config_path)
    run_configs = utils.prepare_config_dict(base_config)

    dirty_file = os.path.join('..', outputpath, 'dirty.csv')
    clean_file = os.path.join('..', outputpath, 'clean.csv')
    config = run_configs[0]
    config = complete_config(config, clean_file, dirty_file)

    # print(config)
    args = SimpleNamespace(**config)
    print(args)

    if col == 'all':
        args.target_columns = [col]
        
    logger = GrimpLogger()
    logger.add_dict("parameters", vars(args))
    logger.add_run_name()
    logger.add_value("parameters", "num_estimators", 0)
    logger.add_time("start_training")

    graph_dataset, best_state, init_params = pipeline.run_training(args, logger)
    
    logger.add_time("end_training")
    logger.add_duration("start_training", "end_training", "duration_training")
    logger.print_summary()
    logger.save_json()
    
    imputed_df = pipeline.run_testing(
        args, graph_dataset, best_state, init_params, logger=logger
    )

    imputed_df = curate_imputed_df(imputed_df)

    df_dirty = pd.read_csv(dirty_file)
    df_clean = pd.read_csv(clean_file)

    precision, recall, f1 = computeScores(df_dirty, df_clean, imputed_df)
    print('Precision: ', precision, 'Recall: ', recall, 'F1: ', f1)

    runtime = time.time() - start_time

    ncols = len(df_dirty.columns)
    nrows = len(df_dirty.index)

    if col == 'all':
        save_score(resultfile, dataset, ncols, nrows, missing_type, missing_ratio, method_name, precision, recall, f1, runtime)
    else:
        missing, imputed, correcly_imputed = computeImputationRatio(df_dirty, df_clean, imputed_df)
        save_score_col(resultfile, dataset, ncols, nrows, missing_type, missing_ratio, method_name, missing, imputed, correcly_imputed, runtime, col)


    

   


if __name__ == "__main__":
    main()
