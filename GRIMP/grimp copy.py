
import sys
import os
import pandas as pd
sys.path.insert(0, '../tools')
from utils import parse_args, computeScores

import GRIMP.pipeline_simple as pipeline_simple
from GRIMP.logging import GrimpLogger

def main():

    print('GRIMP Imputer')
    

    dataset, missing_type, missing_ratio, col, outputpath = parse_args()
    print('Configuration: ', dataset, missing_type,
          missing_ratio, col, outputpath)

    dirty_file = os.path.join('..', outputpath, 'dirty.csv')
    clean_file = os.path.join('..', outputpath, 'clean.csv')

    pipeline_simple.create_graph_dataset(clean_file, dirty_file)

    

   


if __name__ == "__main__":
    main()
