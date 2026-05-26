import sys
import os
import pandas as pd
import time
sys.path.insert(0, '../tools')
sys.path.append('../tools/holoclean/')

from utils import parse_args, computeScores, save_score

import holoclean
from repair.featurize import *
from detect import NullDetector, ViolationDetector

method_name = 'holoclean'


def main():

    print('Holoclean Imputer')
    dataset, missing_type, missing_ratio, col, outputpath, resultfile = parse_args()
    print('Configuration: ', dataset, missing_type,
          missing_ratio, col, outputpath, resultfile)

    dirty_file = os.path.join('..', outputpath, 'dirty.csv')
    df_dirty = pd.read_csv(dirty_file)

    start_time = time.time()

    # actual imputation

    hc = holoclean.HoloClean(
        db_name='holo',
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

    hc.load_data('data', dirty_file)

    # 3. Detect erroneous cells using these two detectors.
    detectors = [NullDetector()]
    hc.detect_errors(detectors)

    # 4. Repair errors utilizing the defined features.
    hc.setup_domain()
    featurizers = [
        InitAttrFeaturizer(),
        OccurAttrFeaturizer(),
        FreqFeaturizer()    
    ]

    hc.repair_errors(featurizers)

    # # ground truth
    # clean_file = os.path.join('..', outputpath, 'clean.csv')
    # df_clean = pd.read_csv(clean_file)

    # precision, recall, f1 = computeScores(df_dirty, df_clean, df_imputed)
    # print('Precision: ', precision, 'Recall: ', recall, 'F1: ', f1)

    # runtime = time.time() - start_time

    # ncols = len(df_dirty.columns)
    # nrows = len(df_dirty.index)

    # save_score(resultfile, dataset, ncols, nrows, missing_type, missing_ratio, method_name, precision, recall, f1, runtime)


 



    

if __name__ == "__main__":
    main()
