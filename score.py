import pandas as pd
import os
import random
import sys
import numpy as np




DATAPATH = '/data/'







def computeScores(df_dirty, df_clean, df_imputed):
    
    null_positions = np.where(df_dirty.isnull()) # position in which we have nulls

    

    # print(df_clean.head(10))
    
    correcly_imputed = 0
    imputed = 0

    rec_numerator = 0
    missing = 0

    # print('\n')

    for i in range(len(null_positions[0])):
        row = null_positions[0][i]
        col = null_positions[1][i]
    
        clean_value = df_clean.iloc[row, col]
        inputed_value = df_imputed.iloc[row, col]


        if not pd.isnull(inputed_value):    
            imputed = imputed + 1
            if (clean_value) == (inputed_value):
                correcly_imputed = correcly_imputed + 1

        missing = missing + 1

        # if not pd.isnull(inputed_value):
        #     imputed = imputed + 1
        #     if (clean_value) == (inputed_value):
        #         positive = positive + 1
        # else:
        #     non_inputed = non_inputed + 1
    
    # from what has been inputed, the proportion that is correct
    if correcly_imputed == 0:
        precision = 0
        recall = 0
        f1 = 0
    else:
        precision = correcly_imputed / imputed
        recall = correcly_imputed / missing
        f1 = 2*(precision*recall)/(precision+recall)
        
    
    print("Precision: ", round(precision , 4) )
    print("Recall: ", round(recall , 4))
    print("F1: ", round(f1 , 4))
 
    
    
          
    null_positions_imp = np.where(df_imputed.isnull()) # position in which we have nulls
    imputed_nulls = len(null_positions_imp[0])
    nulls = len(null_positions[0])
    print("Number of nulls in dirty: ", nulls)
    print("Number of nulls in imputed: ", imputed_nulls)

    return precision, recall, f1


def main():


    script_name = sys.argv[0]
    args = sys.argv[1:]

 

    dataset_clean = args[0]
    dataset_dirty = args[1]
    dataset_repaired = args[2]

    # dataset_clean = DATAPATH + 'cancer/clean.csv'
    # dataset_dirty = DATAPATH + 'cancer/dirty.csv'
    # dataset_repaired = DATAPATH + 'cancer/dirty_imputed.csv'


    

    df_clean = pd.read_csv(dataset_clean)
    # print(df_clean.head(10))
    
    df_dirty = pd.read_csv(dataset_dirty)
    # print(df_dirty.head(10))

    df_repaired = pd.read_csv(dataset_repaired)
    # print(df_repaired.head(10))
    
    


    

    print("Parameters", dataset_clean, dataset_dirty, dataset_repaired) 

    computeScores(df_dirty, df_clean, df_repaired)

    


if __name__ == "__main__":
    main()
