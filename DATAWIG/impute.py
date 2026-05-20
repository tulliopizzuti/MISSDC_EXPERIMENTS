import argparse
import os
import warnings

import pandas as pd
import numpy as np
from datawig import SimpleImputer, Imputer, CategoricalEncoder, BowEncoder, BowFeaturizer
from sklearn.preprocessing import OrdinalEncoder
import datawig

warnings.simplefilter("ignore")

import os
import numpy as np
import pandas as pd
import datawig


def datawig_imputation(
    incomplete_df: pd.DataFrame,
        dataset_null_char: str,
    model_root: str = "datawig_models",
):
    os.makedirs(model_root, exist_ok=True)
    to_return = incomplete_df.copy()

    counter = 1
    for target in incomplete_df.columns.tolist():
        print(f"Imputing target column: {target} ({counter}/{len(incomplete_df.columns)})")
        counter+=1

        df = incomplete_df.copy()
        input_columns = [c for c in df.columns if c != target]

        null_mask = df[target] == dataset_null_char

        df.loc[null_mask, target] = np.nan
        train_df = df[df[target].notnull()].copy()

        predict_df = df[df[target].isnull()].copy()

        predict_df[target] = predict_df[target].astype(str)
        train_df[target] = train_df[target].astype(str)
        predict_df.replace(dataset_null_char, np.nan, inplace=True)
        train_df.replace(dataset_null_char, np.nan, inplace=True)

        if predict_df.empty:
            continue


        model_path = os.path.join(model_root, target)
        print(f"Input columns: {input_columns}")
        imputer = datawig.SimpleImputer(
            input_columns=input_columns,
            output_column=target,
            output_path=model_path,

        )


        print("Fitting model")

        if train_df.empty:
            continue
        try:
            imputer.fit(train_df=train_df)

            print(f"Predict {target}")
            predicted = imputer.predict(predict_df)
            imputed_col = f"{target}_imputed"
            for idx, val in zip(predicted.index, predicted[imputed_col]):
                to_return.loc[idx, target] = val
        except Exception as e:
            print(f"Imputation failed for {target}")
            print(e)
            pass


    return to_return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='DataWig_Exp',
        description='Run imputation using DataWig')
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

    imputed_df = datawig_imputation(df, dataset_null_char)



    imputed_df.to_csv(
        args.output_file_path,
        index=False,
        sep=args.csv_sep
    )

    print("Imputation completed.")
