"""
This script is used for injecting random missing values in a given dataframe according to some simple rules.
It is possible to target either a subset, or all columns in the dataframe.
Columns can be converted to string values so they are treated as categorical values.

Author: Riccardo Cappuzzo
"""
import argparse
import os
import os.path as path
import random

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        "-i",
        action="store",
        required=True,
        help="Input file to be modfied by adding noise.",
    )
    parser.add_argument(
        "--method",
        action="store",
        help="Method to be used to add noise.",
        required=True,
        choices=["simple", "detailed", "bart"],
    )
    parser.add_argument(
        "--hivae",
        action="store_true",
        help="Whether to prepare the files required to run hivae on the dataset.",
    )
    parser.add_argument(
        "--error_fraction",
        default=0.1,
        action="store",
        type=float,
        help="Fraction of errors to be added to the columns.",
    )
    parser.add_argument(
        "--detailed_fraction_file",
        action="store",
        help="Path to a file that stores the fraction of error to be injected in each column. ",
    )
    parser.add_argument(
        "--missing_value_flag",
        action="store",
        default=None,
        help="Optional argument to pass to the dataset parser as a null value flag. ",
    )
    parser.add_argument(
        "--target_columns",
        "-t",
        action="store",
        nargs="*",
        help="Columns to be modified by injecting errors. ",
    )
    parser.add_argument(
        "--target_all_columns",
        action="store_true",
        help='Inject "error_fraction/n_columns" errors in each column. ',
    )
    parser.add_argument(
        "--convert_columns",
        action="store",
        nargs="*",
        default=[],
        help='Columns to be converted to type "str". ',
    )
    parser.add_argument(
        "--convert_all_columns",
        action="store_true",
        help='Convert all columns to type "str". ',
    )
    parser.add_argument(
        "--conversion_prefix",
        action="store",
        default="",
        help="Values in columns to be converted to object will use this value as prefix.",
    )
    parser.add_argument(
        "--save_folder",
        action="store",
        default=None,
        help="Force saving the dirty datasets in the given folder.",
    )
    parser.add_argument(
        "--keep_original",
        action="store_true",
        help="Whether to keep the unconverted columns or not.",
    )
    parser.add_argument(
        "--tag",
        action="store",
        default=None,
        help="Tag to be added to the dirty dataset filename.",
    )
    parser.add_argument(
        "--tag_error_frac",
        action="store_true",
        help="Add the error fraction to the tag. ",
    )
    parser.add_argument(
        "--drop_columns",
        action="store",
        default=[],
        nargs="*",
        help="List of columns that should be dropped.",
    )
    parser.add_argument(
        "--keep_mask",
        action="store_true",
        help="Force all errors to be on the same rows. ",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        default=None,
        help="Set a specific random seed for repeatability. ",
    )

    args = parser.parse_args()
    return args


def convert_columns(args, df_base):
    if args.keep_original:
        for col in args.convert_columns:
            df_base[col + "_O"] = df_base[col].apply(
                lambda x: args.conversion_prefix + str(x)
            )
    else:
        for col in args.convert_columns:
            df_base[col] = df_base[col].apply(lambda x: args.conversion_prefix + str(x))
    return df_base


def column_error_injection(column: pd.Series, mask, na_value=np.nan):
    column_dirty = column.copy(deep=True)
    column_dirty.loc[mask] = na_value
    count_injected_values = sum((column_dirty.isna()))
    return column_dirty, count_injected_values


def get_mask(df, error_fraction):
    return df.sample(frac=error_fraction).index


def simple_error_injection(args, df):
    df2 = df.copy(deep=True)
    error_fraction = args.error_fraction
    if args.target_all_columns:
        target_columns = df.columns
    else:
        target_columns = args.target_columns
    error_count = {k: 0 for k in target_columns}

    if args.keep_mask:
        # If arg keep_mask is true, all errors will be injected in the same row for each column.
        # This is a very hard case to be used as a stress test.
        mask = get_mask(df2, error_fraction)
    else:
        mask = None
    for attr in target_columns:
        if not args.keep_mask:
            mask = get_mask(df2, error_fraction)
        df2[attr], injected = column_error_injection(df2[attr], mask)
        error_count[attr] = injected
        # This encodes nan/none as -1
        # df2.loc[df2.sample(frac=args.error_fraction).index, attr] = -1
        print(f"Column {attr}: {injected} errors in {len(df2[attr])} values.")
    tot_errors = df2.isna().sum().sum()
    print(
        f"Injected a total of {tot_errors} errors. {tot_errors}/{len(df2.values.ravel())}={tot_errors / len(df2.values.ravel()):.2f}"
    )
    return df2


def detailed_error_injection(args, df):
    assert path.exists(args.detailed_fraction_file)

    # Read error fractions from file, format: column_name,error_fraction
    with open(args.detailed_fraction_file, "r") as fp:
        fractions = {}
        for idx, row in enumerate(fp):
            column, fraction = row.strip().split(",")
            fractions[column] = fraction

    # Insert errors according to their fraction.
    df2 = df.copy(deep=True)
    for attr in fractions:
        df2[attr], injected = column_error_injection(df2[attr], fractions[attr])
        print(f"Column {attr}: {injected} errors in {len(df2[attr])} values.")
    return df2


def input_columns_validity_check(df, args):
    # Check that all column names supplied by the user are found in the dataframe.
    if args.target_all_columns:
        target_columns = df.columns.tolist()
    else:
        target_columns = args.target_columns
    conversion_columns = args.convert_columns
    for col in set(target_columns + conversion_columns):
        if col not in df.columns:
            raise (ValueError(f"Column {col} was not found in the dataset header."))


def print_info(args):
    print(f"Inserting errors in input file: {args.input_file}")
    print(f"Method: {args.method}")
    if args.hivae:
        print(f"Preparing additional data for HI-VAE.")
    if args.target_all_columns:
        print("Targeting all columns.")
    else:
        s = "\n".join(args.target_columns)
        print(f"Target columns: \n{s}")
    if args.convert_columns:
        s = "\n".join(args.convert_columns)
        print(
            f"Columns to be converted to Object with prefix `{args.conversion_prefix}`:\n{s}"
        )


def write_dirty_df(df_dirty, args):
    input_file_name = args.input_file
    clean_name, ext = path.splitext(input_file_name)
    basename = path.basename(clean_name)
    if args.target_all_columns:
        output_name = f"{basename}_all_columns"
    else:
        output_name = f'{basename}_{"_".join(args.target_columns)}'
    tag = ""
    if args.tag_error_frac:
        tag += f"_{args.error_fraction * 100:g}"
    if args.tag:
        tag += f"_{args.tag}"
    output_name += f"{tag}{ext}"

    if args.save_folder is not None:
        output_path = f"{args.save_folder}"
    else:
        output_path = f"data/{basename}/"
    if not path.exists(output_path):
        print(f"Creating new folder: {output_path}")
        os.makedirs(output_path)

    output_name = os.path.join(output_path, output_name)

    print(f"Output file: {output_name}")
    df_dirty.to_csv(output_name, index=False)
    return output_name


def convert_to_number(df):
    df_copy = df.copy()
    n_uniques = {}
    for idx, col in enumerate(df.columns):
        if df[col].dtype == "O":
            uniques = df_copy[col].unique().tolist()
            dict_uniques = {uniques[_i]: _i for _i in range(len(uniques))}
            df_copy[col] = df_copy[col].apply(lambda x: dict_uniques[x])
            n_uniques[idx] = len(uniques)
    return n_uniques


def convert_to_hivae_format(df_base, df_tgt, path):
    df_copy = df_tgt.copy()
    dict_uniques = dict()
    for col in df_base.columns:
        uniques = df_base[col].unique().tolist()
        dict_uniques[col] = {uniques[_i]: _i for _i in range(len(uniques))}
        dict_uniques[col][np.nan] = ""
        df_copy[col] = df_copy[col].apply(lambda x: dict_uniques[col][x])
    df_copy.to_csv(path, index=False, header=False)
    return dict_uniques


def prepare_dtype_file(df, n_uniques):
    # This function creates a new dataframe that contains a row for each column in the dataframe.
    # Rows are either "categorical" or "real" depending on whether the dtype is "object", or numerical.
    dd = {_: df.columns[_] for _ in range(len(df.columns))}
    for idx, dt in enumerate(df.dtypes):
        # NOTE: this is not robust to weird datatypes: this check assumes that dtypes are either object, or real values.
        if dt == "object" or dt == "string":
            dd[idx] = ["categorical", n_uniques[idx], n_uniques[idx]]
        else:
            dd[idx] = ["real", 1, ""]
    dtypes = pd.DataFrame(columns=["type", "dim", "nclass"]).from_dict(dd)
    return dtypes


def write_hivae_version(df_dirty, df_base, args):
    # HI-VAE requires a specific error format to work take datasets as input. This function prepares all files required
    # by HI-VAE in the proper format.

    print("Writing additional files for HI-VAE.")
    input_file_name = args.input_file

    clean_name, ext = path.splitext(input_file_name)
    basename = path.basename(clean_name)
    # Input dataset is equivalent to the base dataframe, but all values are converted to (numeric) categorical values.
    input_dataset_path = f"data/{basename}_hivae{ext}"
    output_dataset_path = (
        f'data/dirty_datasets/{basename}_{"_".join(args.target_columns)}_hivae'
    )

    if args.tag:
        output_dataset_path += f"_{args.tag}{ext}"
    else:
        output_dataset_path += f"{ext}"

    for df, fpath in zip(
        [df_base, df_dirty], [input_dataset_path, output_dataset_path]
    ):
        n_uniques = convert_to_number(df)
        # convert_to_hivae_format(df_base, df_orig, fpath)

    # All errors (missing values) are listed in a text file that contains the coordinates of the missing values (row, col).
    basename_out, ext = path.splitext(output_dataset_path)
    error_file = f"{basename_out}_errors" + ext
    with open(error_file, "w") as fp:
        for col_idx, col in enumerate(df_dirty.columns):
            error_idx = df_dirty.loc[df_dirty[col].isna()].index.tolist()
            for e in error_idx:
                s = f"{e + 1},{col_idx + 1}\n"
                fp.write(s)

    prepare_dtype_file(df_base, n_uniques)

    dd = {_: list(df_base.columns)[_] for _ in range(len(df_base.columns))}
    for idx, dt in enumerate(df_base.dtypes):
        # col = df_orig.columns[idx]
        if dt == "object":
            dd[idx] = ["cat", str(n_uniques[idx]), str(n_uniques[idx])]
        else:
            dd[idx] = ["real", "1", ""]

    dtype_fname = f"data/{basename}_hivae_data_types.csv"
    with open(dtype_fname, "w") as fp:
        print(f"Datatypes saved in file {dtype_fname}")
        header = "type,dim,nclass"
        fp.write(header)
        for k, v in dd.items():
            s = "\n" + ",".join(v)
            fp.write(s)

    return output_dataset_path


def write_error_file(experiment_path, df_base: pd.DataFrame, frac_dict):
    for k, v in frac_dict.items():
        if k not in df_base.columns:
            raise ValueError(f"Column {k} not found in columns.")
        if not 0 <= v <= 1:
            raise ValueError(f"Fraction {v} for column {k} not in range [0,1].")

    error_file = f"Missing_variable_" + "_".join([str(_) for _ in frac_dict.keys()])
    fw = f"{experiment_path}/{error_file}.csv"
    with open(fw, "w") as fp:
        for col, frac in frac_dict.items():
            n_errors = int(np.round(len(df_base) * frac))
            errors = random.sample(range(len(df_base)), k=n_errors)
            for e in sorted(errors):
                s = f"{e + 1},{col + 1}\n"
                fp.write(s)
    return error_file


if __name__ == "__main__":
    # This script will run BART on a given dataset and will add errors according to the provided values
    args = parse_args()
    # args include
    # functional dependencies
    # general parameters
    # parameters relative to the current experiment

    random.seed(args.seed)

    print_info(args)

    df_base = pd.read_csv(args.input_file, na_values=args.missing_value_flag)

    input_columns_validity_check(df_base, args)

    if args.convert_columns:
        df_base = convert_columns(args, df_base)
        base, ext = path.splitext(args.input_file)
        df_base.to_csv(f"{base}_O{ext}", index=False)
    if args.drop_columns:
        df_base = df_base.drop(args.drop_columns, axis=1)
    method = args.method

    if method == "bart":
        raise NotImplementedError
    elif method == "simple":
        df_dirty = simple_error_injection(args, df_base)
    elif method == "detailed":
        df_dirty = detailed_error_injection(args, df_base)
    else:
        raise ValueError(f"Unknown method {method}.")

    output_name = write_dirty_df(df_dirty, args)

    if args.hivae:
        write_hivae_version(df_dirty, df_base, args)
