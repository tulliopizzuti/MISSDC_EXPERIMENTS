import pandas as pd
import os
import random
import sys
import numpy as np
# sys.path.append('../utils')
from utils import compute_diff_list

from jenga.corruptions.generic import MissingValues
# from jenga.corruptions.generic import CategoricalShift
# from jenga.corruptions.numerical import Scaling


_default_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATAPATH = os.environ.get('MISSDC_DATA_ROOT', _default_data)
# DATAPATH = '../datasets/datasets_missdc/'


def replace_characters(file):
    with open(file, 'r') as f:
        content = f.read()
        replaced_content = ''
        for char in content:
            if char.isdigit():
                replaced_content += chr(ord('a') + int(char))
            else:
                replaced_content += char

    cat_file_name = file.replace('.csv', '_cat.csv')
    with open(cat_file_name, 'w') as f:
        f.write(replaced_content)

    df = pd.read_csv(cat_file_name, sep=',')
    df.to_csv(cat_file_name, index=False)
    # print(df.head(10))


def save_numerical_df(file, dfa):

    df = dfa.copy(deep=True)

    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        df[col] = df[col].astype('category').cat.codes
    
    df = df.replace(-1, np.nan)

    column_names = df.columns.tolist()
    column_names = [col.replace(' str', ' int') for col in column_names]
    df.columns = column_names
    df.to_csv(file, index=False)


def create_datasets(dataset, ncols, missingness, fraction, map2int=False):

    
    clean_path = os.path.join(DATAPATH, dataset, dataset+'.csv')

    # print(clean_path)
    # print(os.getcwd())
    # print(os.path.isfile(clean_path))



  

    if os.path.isfile(clean_path):

        

        clean_df = pd.read_csv(clean_path, sep=',')
        clean_df = clean_df.dropna()  # remove nulls from dataframe

        # print(clean_df.head(10))

        # binninb
        # for col in clean_df.columns.tolist():
        #     if " str" in col.lower():
        #         continue
        #     if clean_df[col].nunique() > 10:
        #         clean_df[col] = pd.cut(clean_df[col], bins=10)
        #         # clean_df[col] = pd.qcut(clean_df[col], q=10)
        #         clean_df[col] = clean_df[col].cat.codes
                
        # clean_df = clean_df.apply(lambda x: x.astype('category').cat.codes)
        print(clean_df.head(10))
       


        clean_df.to_csv(os.path.join(
            DATAPATH, dataset, 'clean.csv'), index=False)

        replace_characters(os.path.join(DATAPATH, dataset, 'clean.csv'))
        if map2int:
            save_numerical_df(os.path.join(DATAPATH, dataset, 'clean_num.csv'), clean_df)

        column_names = clean_df.columns.values.tolist()
        numerical_cols = clean_df.select_dtypes(
            include=['int64', 'float64']).columns.tolist()
        categorical_cols = clean_df.select_dtypes(
            include=['object']).columns.tolist()

        

        if ncols == -1:
            random_cols = column_names
        else:
            # random_cat_cols = random.sample(categorical_cols, ncols)
            # random_num_cols = random.sample(numerical_cols, ncols)
            random_cols = random.sample(column_names, ncols)

        dirty_df = clean_df.copy(deep=True)
        # for col in random_num_cols:
        for col in random_cols:
            # print(col)
            dirty_df = MissingValues(
                col, fraction, missingness=missingness, na_value=None).transform(dirty_df)
            # dirty_df = CategoricalShift(col, fraction, sampling=missingness).transform(dirty_df)
            # dirty_df = Scaling(col, fraction, sampling=missingness).transform(dirty_df)

        num_missing = dirty_df.isna().sum().sum()
        print(f"Number of missing values in dirty_df: {num_missing}")

        
        dirty_df.to_csv(os.path.join(DATAPATH, dataset,
                        'dirty.csv'), index=False)
        replace_characters(os.path.join(DATAPATH, dataset, 'dirty.csv'))
        if map2int:
            save_numerical_df(os.path.join(DATAPATH, dataset,'dirty_num.csv'), dirty_df)

        # diffs = compute_diff_list(clean_df,dirty_df)
        # print('Diffs legth: ' + str(len(diffs)))

        # print("Clean data shape: ", str(clean_df.shape))
        # print("Dirty data shape: ", str(dirty_df.shape))
        
        ## compares the values that are possible to impute
        unique_values_dict = {}
        columns_list = dirty_df.columns.to_list()
        # Iterate over each column in the DataFrame
        for column in columns_list:
            # Get the unique values of the column
            unique_values = dirty_df[column].unique()
            # Store the unique values in the dictionary
            unique_values_dict[column] = unique_values

        missing_indices = np.where(dirty_df.isna())
        
        #print("Missing indices", missing_indices)

        in_domain = 0
        out_domain = 0

        for i in range(len(missing_indices[0])):
            row = missing_indices[0][i]
            col = columns_list[missing_indices[1][i]]

            value = clean_df[col].iloc[row]
            
            if value in unique_values_dict[col]:
                in_domain += 1
            else:
                out_domain += 1
        
        ratio = in_domain / (in_domain + out_domain)

        
        print("In domain: ", in_domain)
        print("Out domain: ", out_domain)
        print("Ratio of in domain values: ", ratio)

    else:
        raise FileNotFoundError(
            f"Source dataset file not found: {clean_path}. "
            "Set MISSDC_DATA_ROOT to the folder containing dataset subfolders."
        )
            





def main():

    # create_datasets('tax', 5, 'MNAR', 0.2)
    # # Check if at least one argument is provided
    # if len(sys.argv) < 2:
    #     print("Usage: python script.py <arg1> [arg2] [arg3] ...")
    #     sys.exit(1)

    # # Access command-line arguments
    script_name = sys.argv[0]
    args = sys.argv[1:]

    dataset = args[0]
    ncols = int(args[1])
    missingness = args[2]
    fraction = float(args[3])
    map2Int = args[4] if len(args)==5 else '' # Fast fix for debug issue
    #map2Int = args[4]

    # dataset = 'zz_insurance'
    # ncols = -1
    # missingness = 'MNAR'
    # fraction = 0.1
    # map2Int = False

    print("Parameters", dataset, ncols, missingness, fraction, map2Int)

    create_datasets(dataset, ncols, missingness, fraction, map2Int)


if __name__ == "__main__":
    main()
