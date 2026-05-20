import argparse
import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
from jenga.corruptions.generic import MissingValues

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


def create_datasets(input_dataset_path, output_path, ncols, missingness, fraction, map2int=False):


    if os.path.isfile(input_dataset_path):

        clean_df = pd.read_csv(input_dataset_path, sep=',')
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

        clean_df.to_csv(os.path.join(output_path, 'clean.csv'), index=False)

        replace_characters(os.path.join(output_path, 'clean.csv'))
        if map2int:
            save_numerical_df(os.path.join(output_path, 'clean_num.csv'), clean_df)

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

        dirty_df.to_csv(os.path.join(output_path,
                                     'dirty.csv'), index=False)
        replace_characters(os.path.join(output_path, 'dirty.csv'))
        if map2int:
            save_numerical_df(os.path.join(output_path, 'dirty_num.csv'), dirty_df)

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

        # print("Missing indices", missing_indices)

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
            f"Source dataset file not found: {input_dataset_path}. "
            "Set MISSDC_DATA_ROOT to the folder containing dataset subfolders."
        )


def get_output_folder(output_root, dataset_file, missingness, ratio, repetition):
    output_folder = (
        Path(output_root)
        / dataset_file.stem
        / missingness
        / str(ratio)
        / str(repetition + 1)
    )

    output_folder.mkdir(parents=True, exist_ok=True)

    return output_folder

def get_num_columns(dataset_file):
    df = pd.read_csv(dataset_file)
    return len(df.columns)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets_folder", required=True)

    parser.add_argument(
        "--missingness",
        required=True,
        nargs="+",
        choices=["MCAR", "MAR", "MNAR"],
        help="List of missingness mechanisms"
    )

    parser.add_argument(
        "--ratio",
        required=True,
        nargs="+",
        type=float,
        help="List of missingness ratios"
    )

    parser.add_argument("--output", default="data")

    parser.add_argument("--repetitions", type=int, default=5)

    parser.add_argument("--cwd", default=None)

    return parser.parse_args()

def main():
    args = parse_args()
    datasets_folder = Path(args.datasets_folder)

    dataset_files = sorted(datasets_folder.glob("*.csv"))
    for dataset_file in dataset_files:
        num_columns = get_num_columns(dataset_file)
        for missingness in args.missingness:
            for ratio in args.ratio:
                for repetition in range(args.repetitions):
                    print(f"\nProcessing dataset: {dataset_file.name}, {missingness}, {ratio}, {repetition+1}")
                    output_folder = get_output_folder(
                        output_root=args.output,
                        dataset_file=dataset_file,
                        missingness=missingness,
                        ratio=ratio,
                        repetition=repetition,
                    )
                    create_datasets(dataset_file, output_folder, num_columns, missingness, ratio, '')







if __name__ == "__main__":
    main()