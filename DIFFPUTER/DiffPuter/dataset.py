import os
import json
import numpy as np
import pandas as pd


DATA_DIR = 'datasets'

def load_dataset(dataname, idx=0, mask_type='real'):

    data_dir  = f'{DATA_DIR}/{dataname}'
    info_path = f'{DATA_DIR}/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    num_col_idx    = info['num_col_idx']
    cat_col_idx    = info['cat_col_idx']
    target_col_idx = info['target_col_idx']

    train_path = f'{data_dir}/train.csv'
    test_path  = f'{data_dir}/test.csv'
    data_path  = f'{data_dir}/data.csv'


    train_mask_path = f'{data_dir}/masks/{mask_type}/train_mask_{idx}.npy'
    test_mask_path  = f'{data_dir}/masks/{mask_type}/test_mask_{idx}.npy'

    data_df  = pd.read_csv(data_path)
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    train_mask = np.load(train_mask_path)
    test_mask  = np.load(test_mask_path)

    indices_path = f'{data_dir}/indices.json'
    with open(indices_path, "r") as jsonfile:
        indices = json.load(jsonfile)
    cols = train_df.columns.tolist()


    def encode_split(df, data_df, cat_columns, data_dir):

        cat_bin_list = []
        cat_idx_list = []
        bin_num_list = []
        nan_per_col  = []

        for col in cat_columns:
            map_bin_path = f'{data_dir}/{col}_map_bin.json'
            map_idx_path = f'{data_dir}/{col}_map_idx.json'

            with open(map_bin_path, 'r') as f:
                cat2bin = json.load(f)
            with open(map_idx_path, 'r') as f:
                cat2idx = json.load(f)

            raw = df[col].astype(str)
            is_nan = df[col].isna().to_numpy()   # (N,) bool

            first_cat = next(iter(cat2bin))
            raw_filled = raw.where(~is_nan, other=first_cat)

            bin_enc = raw_filled.map(cat2bin).to_numpy()
            bin_arr = np.array(
                [list(map(int, b)) for b in bin_enc], dtype=np.float32
            )                                          # (N, bits)

            bin_arr[is_nan] = 0.0

            idx_enc = raw_filled.map(cat2idx).to_numpy().astype(np.int64)
            idx_enc[is_nan] = -1

            cat_bin_list.append(bin_arr)
            cat_idx_list.append(idx_enc)
            bin_num_list.append(bin_arr.shape[1])
            nan_per_col.append(is_nan)

        X_bin   = np.concatenate(cat_bin_list, axis=1)
        cat_idx = np.stack(cat_idx_list, axis=1)
        bin_num = np.array(bin_num_list, dtype=np.int32)

        return X_bin, cat_idx, bin_num

    cat_columns = [cols[i] for i in cat_col_idx]

    train_X, train_cat_idx, cat_bin_num = encode_split(train_df, data_df, cat_columns, data_dir)
    test_X,  test_cat_idx,  _           = encode_split(test_df,  data_df, cat_columns, data_dir)

    def extend_mask(mask, bin_num):
        n, d = mask.shape
        cum = np.concatenate([[0], bin_num.cumsum()])
        result = np.zeros((n, bin_num.sum()), dtype=bool)
        for k in range(d):
            tile = np.tile(mask[:, k:k+1], bin_num[k])
            result[:, cum[k]:cum[k+1]] = tile
        return result

    extend_train_mask = extend_mask(train_mask[:, cat_col_idx], cat_bin_num)
    extend_test_mask  = extend_mask(test_mask[:, cat_col_idx],  cat_bin_num)

    train_num = np.zeros((train_X.shape[0], 0), dtype=np.float32)
    test_num  = np.zeros((test_X.shape[0],  0), dtype=np.float32)

    return (
        train_X, test_X,
        train_mask, test_mask,
        train_num,  test_num,
        train_cat_idx, test_cat_idx,
        extend_train_mask, extend_test_mask,
        cat_bin_num,
        indices
    )



def mean_std(data: np.ndarray, mask: np.ndarray):

    observed = (~mask).astype(np.float32)
    count = observed.sum(0)
    count = np.where(count == 0, 1.0, count)

    mean = (data * observed).sum(0) / count
    var  = ((data - mean) ** 2 * observed).sum(0) / count
    std  = np.sqrt(var)
    std  = np.where(std == 0, 1.0, std)

    return mean, std



def get_eval(dataname, X_recon, X_true, truth_cat_idx, num_num, cat_bin_num, mask, oos=False):


    info_path = f'{DATA_DIR}/Info/{dataname}.json'
    with open(info_path, 'r') as f:
        info = json.load(f)

    cat_col_idx = info['cat_col_idx']
    cat_mask    = mask[:, cat_col_idx].astype(bool)

    cum = np.concatenate([[0], cat_bin_num.cumsum()])
    n_cat = len(cat_col_idx)

    correct = 0
    total   = 0

    per_col_acc = []

    for k in range(n_cat):
        bits_pred  = X_recon[:, cum[k]:cum[k+1]]
        bits_true  = X_true[:,  cum[k]:cum[k+1]]

        pred_idx = bits_pred.argmax(axis=1) if bits_pred.shape[1] > 1 else (bits_pred[:, 0] > 0.5).astype(int)
        true_idx = truth_cat_idx[:, k]

        col_missing = cat_mask[:, k]

        if col_missing.sum() == 0:
            per_col_acc.append(np.nan)
            continue

        pred_missing = pred_idx[col_missing]
        true_missing = true_idx[col_missing]

        acc_k = (pred_missing == true_missing).mean()
        per_col_acc.append(acc_k)

        correct += (pred_missing == true_missing).sum()
        total   += col_missing.sum()

    overall_acc = correct / total if total > 0 else 0.0

    return overall_acc, per_col_acc



def save_imputed_csv(dataname, X_recon, original_df, cat_bin_num, mask, save_path):

    data_dir  = f'{DATA_DIR}/{dataname}'
    info_path = f'{DATA_DIR}/Info/{dataname}.json'

    with open(info_path, 'r') as f:
        info = json.load(f)

    cat_col_idx = info['cat_col_idx']
    cols        = original_df.columns.tolist()
    cat_columns = [cols[i] for i in cat_col_idx]

    result_df = original_df.copy()

    cum = np.concatenate([[0], cat_bin_num.cumsum()])

    for k, col in enumerate(cat_columns):
        map_idx_path = f'{data_dir}/{col}_map_idx.json'
        with open(map_idx_path, 'r') as f:
            cat2idx = json.load(f)

        idx2cat = {v: k_cat for k_cat, v in cat2idx.items()}

        bits_pred = X_recon[:, cum[k]:cum[k+1]]   # (N, bits_k)

        if bits_pred.shape[1] > 1:
            pred_idx = bits_pred.argmax(axis=1)
        else:
            pred_idx = (bits_pred[:, 0] > 0.5).astype(int)

        col_missing = mask[:, cat_col_idx[k]].astype(bool)

        decoded = np.array([idx2cat.get(i, 'unknown') for i in pred_idx])
        result_df[col] = result_df[col].astype(object)
        result_df.loc[col_missing, col] = decoded[col_missing]

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result_df.to_csv(save_path, index=False)
