import argparse
import numpy as np
import pandas as pd
from missforest import MissForest
from sklearn.preprocessing import OrdinalEncoder


def missforest_imputation(
    incomplete_df: pd.DataFrame,
        dataset_null_char: str):
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )
    df_dirty = incomplete_df.replace(dataset_null_char, np.nan)
    encoded_df = pd.DataFrame(encoder.fit_transform(df_dirty.astype(str)),columns=df_dirty.columns)
    imputer = MissForest()
    imputed_df = imputer.fit_transform(encoded_df)
    for i,c in enumerate(imputed_df.columns):
        max_valid = len(encoder.categories_[i]) - 1
        imputed_df.loc[:, c] = np.clip(imputed_df.loc[:, c], 0, max_valid)

    decoded_df = pd.DataFrame(encoder.inverse_transform(imputed_df),columns=df_dirty.columns)
    return decoded_df




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='MissForest_Exp',
        description='Run imputation using MissForest')
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

    df = pd.read_csv(dataset_file_path, sep=csv_sep)

    imputed_df = missforest_imputation(df, dataset_null_char)
    imputed_df.to_csv(
        args.output_file_path,
        index=False,
        sep=args.csv_sep
    )

    print("Imputation completed.")
