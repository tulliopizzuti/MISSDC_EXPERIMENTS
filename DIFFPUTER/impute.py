

import os
import shutil
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time
from tqdm import tqdm

from DiffPuter.model import MLPDiffusion, Model
from DiffPuter.dataset import load_dataset, get_eval, mean_std, save_imputed_csv
from DiffPuter.diffusion_utils import impute_mask

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import run_command

warnings.filterwarnings('ignore')

PRETRAIN_SCRIPT="prepare_real_missing_dataset.py"
DATASETS_FOLDER="datasets"
CKPT_FOLDER="ckpt"

def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
def reset_folders():
    reset_dir(DATASETS_FOLDER)
    reset_dir(CKPT_FOLDER)

def diffputer_imputation(input, device, hid_dim, mask_type, num_trials, num_steps, orig_num_epochs):
    reset_folders()
    dataname=Path(input).stem
    pretrain_cmd = [sys.executable, PRETRAIN_SCRIPT, "--input", input, "--name", dataname]
    pretrain_returncode, pretrain_time, _, _ = run_command(pretrain_cmd)
    (train_X, test_X,
     ori_train_mask, ori_test_mask,
     train_num, test_num,
     train_cat_idx, test_cat_idx,
     train_mask, test_mask,
     cat_bin_num, split_indices) = load_dataset(dataname, split_idx, mask_type)

    train_X = np.nan_to_num(train_X, nan=0.0)
    test_X  = np.nan_to_num(test_X,  nan=0.0)

    mean_X, std_X = mean_std(train_X, train_mask)

    in_dim = train_X.shape[1]

    X      = torch.tensor((train_X - mean_X) / std_X / 2, dtype=torch.float32)
    X_test = torch.tensor((test_X  - mean_X) / std_X / 2, dtype=torch.float32)

    mask_train = torch.tensor(train_mask)
    mask_test  = torch.tensor(test_mask)

    ACCs_in  = []
    ACCs_out = []

    start_time = time.time()

    for iteration in range(args.max_iter):

        ckpt_dir = f'ckpt/{dataname}/rate/{mask_type}/{split_idx}/{num_trials}_{num_steps}'
        os.makedirs(f'{ckpt_dir}/{iteration}', exist_ok=True)



        if iteration == 0:
            X_miss_np = ((1. - mask_train.float()) * X).numpy()
        else:
            X_miss_np = np.load(f'{ckpt_dir}/iter_{iteration}.npy') / 2

        train_data = X_miss_np.astype(np.float32)

        batch_size   = 4096
        train_loader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, num_workers=4)

        num_epochs = orig_num_epochs + 1
        denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)



        model     = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9,
                                      patience=50, verbose=False)

        model.train()
        best_loss = float('inf')
        patience  = 0

        pbar = tqdm(range(num_epochs), desc='Training')
        for epoch in pbar:
            batch_loss, len_input = 0.0, 0
            for batch in train_loader:
                inputs = batch.float().to(device)
                loss   = model(inputs).mean()

                batch_loss += loss.item() * len(inputs)
                len_input  += len(inputs)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            curr_loss = batch_loss / len_input
            scheduler.step(curr_loss)

            if curr_loss < best_loss:
                best_loss = curr_loss
                patience  = 0
                torch.save(model.state_dict(), f'{ckpt_dir}/{iteration}/model.pt')
            else:
                patience += 1
                if patience == 500:
                    break

            pbar.set_postfix(loss=curr_loss)
            if epoch % 1000 == 0:
                torch.save(model.state_dict(),
                           f'{ckpt_dir}/{iteration}/model_{epoch}.pt')


        result_dir = f'results/{dataname}/{mask_type}/{split_idx}/{num_trials}_{num_steps}'
        os.makedirs(result_dir, exist_ok=True)

        rec_Xs = []
        for trial in tqdm(range(num_trials), desc='In-sample imputation'):
            X_miss  = ((1. - mask_train.float()) * X).to(device)
            denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
            model      = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
            model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt'))
            net = model.denose_fn_D if hasattr(model, 'denose_fn_D') else model.denoise_fn_D

            num_samples, dim = X.shape
            rec_X = impute_mask(net, X_miss, mask_train, num_samples, dim,
                                num_steps, device)

            mask_int = mask_train.float().to(device)
            rec_X    = rec_X * mask_int + X_miss * (1 - mask_int)
            rec_Xs.append(rec_X)

        rec_X  = torch.stack(rec_Xs, dim=0).mean(0).cpu().numpy() * 2
        X_true = X.cpu().numpy() * 2
        np.save(f'{ckpt_dir}/iter_{iteration+1}.npy', rec_X)

        acc_in, acc_per_col = get_eval(
            dataname, rec_X, X_true,
            train_cat_idx, train_num.shape[1], cat_bin_num, ori_train_mask
        )
        ACCs_in.append(acc_in)

        train_df = pd.read_csv(f'datasets/{dataname}/train.csv')
        save_imputed_csv(
            dataname, rec_X, train_df, cat_bin_num, ori_train_mask,
            save_path=f'{result_dir}/iter_{iteration}_train_imputed.csv'
        )

        rec_Xs = []
        for trial in tqdm(range(num_trials), desc='Out-of-sample imputation'):
            X_miss  = ((1. - mask_test.float()) * X_test).to(device)
            denoise_fn = MLPDiffusion(in_dim, hid_dim).to(device)
            model      = Model(denoise_fn=denoise_fn, hid_dim=in_dim).to(device)
            model.load_state_dict(torch.load(f'{ckpt_dir}/{iteration}/model.pt'))
            net = model.denoise_fn_D

            num_samples, dim = X_test.shape
            rec_X = impute_mask(net, X_miss, mask_test, num_samples, dim,
                                num_steps, device)

            mask_int = mask_test.float().to(device)
            rec_X    = rec_X * mask_int + X_miss * (1 - mask_int)
            rec_Xs.append(rec_X)

        rec_X   = torch.stack(rec_Xs, dim=0).mean(0).cpu().numpy() * 2
        X_true_test = X_test.cpu().numpy() * 2

        acc_out, acc_per_col_out = get_eval(
            dataname, rec_X, X_true_test,
            test_cat_idx, test_num.shape[1], cat_bin_num, ori_test_mask, oos=True
        )
        ACCs_out.append(acc_out)


        test_df = pd.read_csv(f'datasets/{dataname}/test.csv')
        save_imputed_csv(
            dataname, rec_X, test_df, cat_bin_num, ori_test_mask,
            save_path=f'{result_dir}/iter_{iteration}_test_imputed.csv'
        )

        with open(f'{result_dir}/result.txt', 'a') as f:
            f.write(f'iter {iteration} | acc in-sample: {acc_in:.4f} | acc out-of-sample: {acc_out:.4f}\n')
            f.write(f'  per-col in:  {[f"{a:.3f}" if not np.isnan(a) else "n/a" for a in acc_per_col]}\n')
            f.write(f'  per-col out: {[f"{a:.3f}" if not np.isnan(a) else "n/a" for a in acc_per_col_out]}\n')

    elapsed = time.time() - start_time

    train_final=pd.read_csv(f'{result_dir}/iter_{iteration}_train_imputed.csv')
    test_final=pd.read_csv(f'{result_dir}/iter_{iteration}_test_imputed.csv')

    final_df=pd.concat([train_final, test_final])
    indices=split_indices["train_index"]+split_indices["test_index"]
    final_df.index=indices
    final_df = final_df.sort_index(ascending=True)
    reset_folders()

    return final_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='DiffPuter_Exp',
        description='Run imputation using DiffPuter')
    parser.add_argument('dataset_file_path',   type=str)
    parser.add_argument("--output_file_path", type=str, default="repaired.csv", help="Repaired file path")
    parser.add_argument('--dataset_null_char', type=str, default='', help='Character used to represent missing values in the dataset')
    parser.add_argument('--csv_sep', type=str, default=',', help='CSV separator')
    parser.add_argument('--gpu',        type=int,   default=0)
    parser.add_argument('--max_iter',   type=int,   default=10)
    parser.add_argument('--hid_dim',    type=int,   default=1024)
    parser.add_argument('--num_trials', type=int,   default=20)
    parser.add_argument('--num_steps',  type=int,   default=50)
    parser.add_argument('--num_epochs',  type=int,   default=1000)
    args = parser.parse_args()
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    split_idx = 0
    device = args.device
    hid_dim = args.hid_dim
    mask_type = "real"
    num_trials = args.num_trials
    num_steps = args.num_steps
    num_epochs = args.num_epochs

    imputed_df = diffputer_imputation(args.dataset_file_path, device, hid_dim, mask_type, num_trials, num_steps, num_epochs)

    imputed_df.to_csv(args.output_file_path, index=False, sep=args.csv_sep)
    print("Imputation completed.")




