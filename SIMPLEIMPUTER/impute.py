import argparse
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np



def simpleimputer_imputation(
    incomplete_df: pd.DataFrame,
        dataset_null_char: str):
    imputer = SimpleImputer(strategy="most_frequent")
    df_dirty = incomplete_df.replace(dataset_null_char, np.nan)
    imp_values = imputer.fit_transform(df_dirty)
    df_imputed = pd.DataFrame(imp_values, columns=df_dirty.columns)
    return df_imputed





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='SimpleImputer_Exp',
        description='Run imputation using SimpleImputer')
    parser.add_argument('dataset_file_path', help="Path to the dataset file (CSV with semicolon separator)")
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

    df = pd.read_csv(dataset_file_path, sep=csv_sep)

    imputed_df = simpleimputer_imputation(df, dataset_null_char)



    imputed_df.to_csv(
        args.output_file_path,
        index=False,
        sep=args.csv_sep
    )

    print("Imputation completed.")
