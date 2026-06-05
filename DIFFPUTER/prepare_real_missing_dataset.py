
import os
import json
import argparse
import numpy as np
import pandas as pd

MASK_FOLDER = 'real'


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.replace(" ", "-", regex=False)
    return df


def build_category_maps(data_df: pd.DataFrame, col: str, data_dir: str):
    categories = sorted([str(c) for c in data_df[col].dropna().unique()])
    num_bits = max((len(categories) - 1).bit_length(), 1)

    cat2bin = {c: format(i, '0' + str(num_bits) + 'b') for i, c in enumerate(categories)}
    cat2idx = {c: i for i, c in enumerate(categories)}

    with open(f'{data_dir}/{col}_map_bin.json', 'w') as f:
        json.dump(cat2bin, f)
    with open(f'{data_dir}/{col}_map_idx.json', 'w') as f:
        json.dump(cat2idx, f)


def prepare(input_csv: str,
            dataname: str,
            train_ratio: float = 0.7,
            mask_num: int = 10,
            seed: int = 1234):

    DATA_DIR = 'datasets'
    data_dir = f'{DATA_DIR}/{dataname}'
    info_dir = f'{DATA_DIR}/Info'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(info_dir, exist_ok=True)

    raw_df = pd.read_csv(input_csv)


    raw_df = normalize_strings(raw_df)
    for col in raw_df.columns:
        nan_mask = raw_df[col].isna()
        raw_df[col] = raw_df[col].astype(str)
        raw_df.loc[nan_mask, col] = np.nan

    raw_df.to_csv(f'{data_dir}/data.csv', index=False)

    np.random.seed(seed)
    idx = np.arange(len(raw_df))
    np.random.shuffle(idx)
    n_train = int(len(idx) * train_ratio)

    train_df = raw_df.iloc[idx[:n_train]]
    test_df  = raw_df.iloc[idx[n_train:]]
    train_index=list(train_df.index)
    test_index=list(test_df.index)
    with open(f'{data_dir}/indices.json', 'w') as f:
        json.dump({
            "train_index": train_index,
            "test_index": test_index
        }, f, indent=4)
    train_df.to_csv(f'{data_dir}/train.csv', index=False)
    test_df.to_csv(f'{data_dir}/test.csv',   index=False)

    n_cols = len(raw_df.columns)
    info = {
        "name":           dataname,
        "num_col_idx":    [],
        "cat_col_idx":    list(range(n_cols)),
        "target_col_idx": []
    }
    with open(f'{info_dir}/{dataname}.json', 'w') as f:
        json.dump(info, f, indent=2)

    for col in raw_df.columns:
        build_category_maps(raw_df, col, data_dir)


    train_mask = train_df.isna().to_numpy()
    test_mask  = test_df.isna().to_numpy()

    mask_dir = f'{data_dir}/masks/{MASK_FOLDER}'
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(mask_num):
        np.save(f'{mask_dir}/train_mask_{i}.npy', train_mask)
        np.save(f'{mask_dir}/test_mask_{i}.npy',  test_mask)

    return train_df, test_df, train_mask, test_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Prepara un dirty.csv per DiffPuter (tutto categorico, mask dai NaN reali).'
    )
    parser.add_argument('--input',    type=str,   required=True,
                        help='Dataset with NaN')
    parser.add_argument('--name',     type=str,   required=True,
                        help='Nome dataset')
    parser.add_argument('--split',    type=float, default=0.8,
                        help='% train (default 0.7)')
    parser.add_argument('--seed',     type=int,   default=1234,
                        help='Seed (default 1234)')

    args = parser.parse_args()

    prepare(
        input_csv   = args.input,
        dataname    = args.name,
        train_ratio = args.split,
        mask_num    = 1,
        seed        = args.seed,
    )
