import argparse
import warnings

import pandas as pd
from hyperimpute.plugins.imputers import Imputers
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
import numpy as np
warnings.simplefilter("ignore")

def hyperimpute(
    data: pd.DataFrame,

):
    plugin = Imputers().get("hyperimpute")
    out = plugin.fit_transform(data.copy())
    return out

def gain(
    data: pd.DataFrame

):
    plugin = Imputers().get("gain")
    out = plugin.fit_transform(data.copy())
    return out

def missforest(
    data: pd.DataFrame

):
    plugin = Imputers().get("missforest")
    out = plugin.fit_transform(data.copy())
    return out

def hyperimpute_improved(
    data: pd.DataFrame,
):
    plugin = Imputers().get(
        "hyperimpute",
        classifier_seed=[
            "logistic_regression",
            "random_forest",
            "catboost",
            "neural_nets",
            "xgboost"
        ],
        # regression_seed=[
        #     "logistic_regression",
        #     "random_forest",
        #     "catboost",
        #     "neural_nets",
        #     "xgboost"
        # ],
        imputation_order=2,
        optimizer='hyperband',
        n_inner_iter= 10,
        baseline_imputer=0,
        select_model_by_iteration=True,
        select_model_by_column=True
    )
    return plugin.fit_transform(data.copy())







if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='HyperImpute_GAIN_Exp',
        description='Run imputation using HyperImpute or GAIN')
    parser.add_argument('imputation_method', help="Imputation function to run")
    parser.add_argument('dataset_file_path', help="Path to the dataset file")
    parser.add_argument('--dataset_null_char', type=str, default='',
                        help='Character used to represent missing values in the dataset')
    parser.add_argument('--csv_sep', type=str, default=',',
                        help='CSV separator')
    parser.add_argument('--output_file_path', type=str, default='repaired.csv',
                        help='Repaired file path')

    args = parser.parse_args()
    imputation_method = args.imputation_method
    dataset_file_path = args.dataset_file_path
    dataset_null_char = args.dataset_null_char
    csv_sep = args.csv_sep
    output_file_path = args.output_file_path

    df = pd.read_csv(dataset_file_path, sep=csv_sep)
    df = df.replace(args.dataset_null_char, np.nan)

    columns = df.columns.tolist()
    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan
    )
    encoded_array = encoder.fit_transform(df)
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=columns
    )

    encoded_df=encoded_df.astype("Int64")
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(encoded_df)
    scaled_df = pd.DataFrame(
        scaled_array,
        columns=columns
    )

    imputer = globals().get(args.imputation_method)
    if not imputer: raise ValueError('Imputation method not recognized')

    imputed_scaled_df = imputer(scaled_df)

    imputed_scaled_df.columns = columns

    unscaled_array = scaler.inverse_transform(imputed_scaled_df)
    unscaled_df = pd.DataFrame(
        unscaled_array,
        columns=columns
    )

    rounded_df = unscaled_df.round(0)
    for i, col in enumerate(columns):
        max_category = len(
            encoder.categories_[i]
        ) - 1

        rounded_df[col] = rounded_df[col].clip(
            lower=0,
            upper=max_category
        )

    decoded_array = encoder.inverse_transform(
        rounded_df.to_numpy().copy().astype(np.int64)
    )

    repaired_df = pd.DataFrame(
        decoded_array,
        columns=columns
    )

    repaired_df.to_csv(
        args.output_file_path,
        index=False,
        sep=args.csv_sep
    )

    print("Imputation completed.")
