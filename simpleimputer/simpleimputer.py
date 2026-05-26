from sklearn.impute import SimpleImputer
import sys
import os
import pandas as pd
import time
sys.path.insert(0, '../tools')
from utils import parse_args, computeScores, save_score, save_score_col, computeImputationRatio

method_name = 'simpleimputer'

def main():

    print('Simple Imputer')
    dataset, missing_type, missing_ratio, col, outputpath, resultfile = parse_args()
    print('Configuration: ', dataset, missing_type,
          missing_ratio, col, outputpath, resultfile)

    dirty_file = os.path.join('..', outputpath, 'dirty.csv')
    df_dirty = pd.read_csv(dirty_file)

    start_time = time.time()

    # actual imputation
    imputer = SimpleImputer(strategy="most_frequent")
    imp_values = imputer.fit_transform(df_dirty)
    df_imputed = pd.DataFrame(imp_values, columns=df_dirty.columns)

    # ground truth
    clean_file = os.path.join('..', outputpath, 'clean.csv')
    df_clean = pd.read_csv(clean_file)

    precision, recall, f1 = computeScores(df_dirty, df_clean, df_imputed)
    print('Precision: ', precision, 'Recall: ', recall, 'F1: ', f1)

    runtime = time.time() - start_time

    ncols = len(df_dirty.columns)
    nrows = len(df_dirty.index)


    if col == 'all':
        save_score(resultfile, dataset, ncols, nrows, missing_type, missing_ratio, method_name, precision, recall, f1, runtime)
    else:
        missing, imputed, correcly_imputed = computeImputationRatio(df_dirty, df_clean, df_imputed)
        save_score_col(resultfile, dataset, ncols, nrows, missing_type, missing_ratio, method_name, missing, imputed, correcly_imputed, runtime, col)


if __name__ == "__main__":
    main()
