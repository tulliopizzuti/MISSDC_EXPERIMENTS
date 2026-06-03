import argparse
import pandas as pd
import numpy as np
import sys
sys.path.append("holoclean")
import holoclean
from repair.featurize import *
from detect import NullDetector, ViolationDetector


def holoclean_imputation(incomplete_df:str, dataset_null_char: str, user: str, pwd: str, host: str, name: str):

    hc = holoclean.HoloClean(
        db_user=user,
        db_pwd=pwd,
        db_host=host,
        db_name=name,
        domain_thresh_1=0,
        domain_thresh_2=0,
        weak_label_thresh=0.99,
        max_domain=10000,
        cor_strength=0.6,
        nb_cor_strength=0.8,
        epochs=10,
        weight_decay=0.01,
        learning_rate=0.001,
        threads=1,
        batch_size=1,
        verbose=True,
        timeout=3*60000,
        feature_norm=False,
        weight_norm=False,
        print_fw=True
    ).session
    hc.load_data('data', incomplete_df)
    detectors = [NullDetector()]
    hc.detect_errors(detectors)
    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer()    
    ]

    hc.repair_errors(featurizers)
    df_imputed = hc.ds.repaired_data.df
    df_imputed = df_imputed.replace("_nan_", np.nan)
    df_imputed = df_imputed.drop('_tid_', axis=1)
    return df_imputed



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='Holoclean_Exp',
        description='Run imputation using Holoclean')
    parser.add_argument('dataset_file_path', help="Path to the dataset file")
    parser.add_argument('--dataset_null_char', type=str, default='',
                        help='Character used to represent missing values in the dataset')
    parser.add_argument('--csv_sep', type=str, default=',',
                        help='CSV separator')
    parser.add_argument('--output_file_path', type=str, default='repaired.csv',
                        help='Repaired file path')
    parser.add_argument('--db_user', type=str, default='holocleanuser',
                        help='User for DB used to persist state.')
    parser.add_argument('--db_pwd', type=str, default='abcd1234',
                        help='Password for DB used to persist state.')
    parser.add_argument('--db_host', type=str, default='localhost',
                        help='Host for DB used to persist state.')
    parser.add_argument('--db_name', type=str, default='holo',
                        help='Name of DB used to persist state.') 

    args = parser.parse_args()
    dataset_file_path = args.dataset_file_path
    dataset_null_char = args.dataset_null_char
    user = args.db_user
    pwd = args.db_pwd
    host = args.db_host
    name = args.db_name
    csv_sep = args.csv_sep
    output_file_path = args.output_file_path


    imputed_df = holoclean_imputation(dataset_file_path, dataset_null_char, user, pwd, host, name)



    imputed_df.to_csv(
        args.output_file_path,
        index=False,
        sep=args.csv_sep
    )

    print("Imputation completed.")
