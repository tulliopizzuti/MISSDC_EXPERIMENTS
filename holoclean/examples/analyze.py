from sklearn.impute import SimpleImputer
import sys
import os
import pandas as pd
import time
sys.path.insert(0, '../tools')
from utils import parse_args, computeScores, save_score

method_name = 'holoclean'

def main():

    print('HoloClean Imputer')
    dataset, missing_type, missing_ratio, col, outputpath, resultfile = parse_args()
    print('Configuration: ', dataset, missing_type,
          missing_ratio, col, outputpath, resultfile)

    
    df_dirty = pd.read_csv('examples/dirty.csv')

    start_time = time.time()

    df_imputed = pd.read_csv('examples/repaired.csv')
    df_imputed = df_imputed.drop('_tid_', axis=1)
    
    # ground truth
    df_clean = pd.read_csv('examples/clean.csv')

    precision, recall, f1 = computeScores(df_dirty, df_clean, df_imputed)
    print('Precision: ', precision, 'Recall: ', recall, 'F1: ', f1)

    runtime = time.time() - start_time

    ncols = len(df_dirty.columns)
    nrows = len(df_dirty.index)

    save_score(resultfile, dataset, ncols, nrows, missing_type, missing_ratio, method_name, precision, recall, f1, runtime)


if __name__ == "__main__":
    main()
