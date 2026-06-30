import shutil
import sys, os, argparse, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import time
warnings.filterwarnings("ignore")

sys.path.insert(0, 'CAFE')

from fed_imp.sub_modules.client.client_factory import ClientsFactory
from fed_imp.sub_modules.server.load_server import load_server
from fed_imp.sub_modules.strategy.strategy_imp import StrategyImputation
from modules.data_partition import data_partition
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def reset_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
def reset_folders():
    pass



def preprocess(csv_path: str, target_column: str, normalize: bool = True, dataset_null_char: str ="", csv_sep: str=","          ):
    df_orig = pd.read_csv(csv_path, sep=csv_sep)
    df_orig = df_orig.replace(dataset_null_char, np.nan)

    if target_column not in df_orig.columns:
        raise ValueError(f"Target column '{target_column}' missing.")

    df_enc = df_orig.copy()
    scaler = None
    features = [c for c in df_orig.columns if c != target_column]

    for col in features:
        if df_orig[col].dtype == object:
            le = LabelEncoder()
            le.fit(df_enc[col].astype(str))
            df_enc[col] = le.transform(df_enc[col].astype(str)).astype(float)

    target_encoder = LabelEncoder()
    train_mask = df_enc[target_column].notna()
    target_encoder.fit(df_enc.loc[train_mask, target_column].astype(str))
    df_enc.loc[train_mask, target_column] = target_encoder.transform(
        df_enc.loc[train_mask, target_column].astype(str)).astype(float)

    if normalize:
        for col in df_enc.columns:
            sc = MinMaxScaler()
            mask = df_enc[col].notna()
            if mask.sum() < 2:
                continue
            vals = df_enc.loc[mask, col].values.reshape(-1, 1)
            sc.fit(vals)
            df_enc.loc[mask, col] = sc.transform(vals).ravel()
            if col == target_column:
                scaler = sc

    test_mask = df_enc[target_column].isna()

    train_df = df_enc[train_mask].copy()
    test_df = df_enc[test_mask].copy()

    test_idx = np.where(test_mask.values)[0]

    return train_df, test_df, test_idx, target_encoder, scaler


def cafe_train_predict(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str,
        target_encoder: LabelEncoder,
        target_scaler: MinMaxScaler = None,
        num_clients: int = 5,
        imp_rounds: int = 20,
        seed: int = 42,
        partition: str = "sample-evenly",
) -> np.ndarray:
    feature_cols = [c for c in train_df.columns if c != target_column]
    X_train = train_df[feature_cols].values.astype(float)
    X_test = test_df[feature_cols].values.astype(float)

    y_scaled = train_df[target_column].values.astype(float)

    if target_scaler is not None:
        y_unscaled = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).ravel()
    else:
        y_unscaled = y_scaled.copy()


    y_encoded = np.round(y_unscaled).astype(int)

    unique_encoded = np.unique(y_encoded)
    label_map = {val: i for i, val in enumerate(sorted(unique_encoded))}
    y_train = np.array([label_map[val] for val in y_encoded])
    n_classes = len(unique_encoded)

    if n_classes < 2:
        raise ValueError("Target classes < 2.")

    fill_vals = np.nanmean(X_train, axis=0)
    fill_vals = np.where(np.isnan(fill_vals), 0.0, fill_vals)

    for c in range(X_train.shape[1]):
        nan_mask_train = np.isnan(X_train[:, c])
        if nan_mask_train.any():
            X_train[nan_mask_train, c] = fill_vals[c]
        nan_mask_test = np.isnan(X_test[:, c])
        if nan_mask_test.any():
            X_test[nan_mask_test, c] = fill_vals[c]

    n_feat = X_train.shape[1]

    data_config = {
        'task_type': 'classification',
        'num_cols': n_feat,
        'cat_cols': 0,
        'n_classes': int(n_classes),
    }
    imp_config = {
        'initial_strategy_num': 'mean',
        'initial_strategy_cat': 'mode',
        'imp_estimator_num': 'ridge_cv',
        'imp_estimator_cat': 'logistic_cv',
        'clip': True,
        'num_cols': n_feat,
        'imp_evaluation_params': {'tune_params': "gridsearch"},
        "imp_evaluation_model":"logistic"
    }

    train_arr = np.concatenate([X_train, y_train.reshape(-1, 1)], axis=1)
    test_arr = np.concatenate([X_test, np.zeros((len(X_test), 1))], axis=1)

    try:
        partitions = data_partition(
            strategy=partition, params={},
            data=train_arr, n_clients=num_clients,
            seed=seed, regression=False,
        )
        for p in partitions:
            if len(np.unique(p[:, -1])) < 2:
                raise ValueError("single-class partition")
    except Exception:
        partitions = [train_arr.copy() for _ in range(num_clients)]

    partitions_ms = [p.copy() for p in partitions]

    # 6. Crea client e server CAFE
    factory = ClientsFactory(debug=False)
    clients = factory.generate_clients(
        num_clients=num_clients,
        data_partitions=partitions,
        data_ms_clients=partitions_ms,
        test_data=test_arr,
        data_config=data_config,
        imputation_config=imp_config,
        seed=seed,
        client_type='ice',
    )

    strategy_imp = StrategyImputation(strategy='cafe', params={
        'client_thres': 1.0, 'alpha': 0.95, 'gamma': 0.02, 'scale_factor': 4,
    })

    # Configurazione server
    server_cfg = {
        "impute_mode": "instant",
        "imp_round": imp_rounds,
        "imp_local_epochs": 0,
        "pred_round": 0,
        "pred_local_epochs": 0,
        "model_fit_mode": "one_shot",
        "froze_ms_coefs_round": 100,
        "n_cols": n_feat,
    }
    pred_cfg = {
        "model_params": {
            "model": "2nn", "num_hiddens": 32,
            "model_init_config": None, "model_other_params": None,
            "input_feature_dim": n_feat, "output_classes_dim": n_classes,
        },
        "train_params": {
            "batch_size": 128, "learning_rate": 0.001,
            "weight_decay": 0.0001, "pred_round": 0, "pred_local_epochs": 3,
        }
    }

    server = load_server(
        server_type='fedavg_pytorch',
        clients=clients,
        strategy_imp=strategy_imp,
        server_config=server_cfg,
        pred_config=pred_cfg,
        test_data=test_arr,
        seed=seed,
        track=False,
        run_prediction=False,
        persist_data=True,
    )
    server.run()

    X_agg = np.concatenate([c.X_train_filled for c in clients.values()], axis=0)
    y_agg = np.concatenate([c.y_train for c in clients.values()], axis=0)


    col_means = np.nanmean(X_agg, axis=0)
    col_means = np.where(np.isnan(col_means), 0.0, col_means)
    for c in range(X_agg.shape[1]):
        mask = np.isnan(X_agg[:, c])
        if mask.any():
            X_agg[mask, c] = col_means[c]

    try:
        clf = LogisticRegression(max_iter=500, random_state=seed, C=1.0)
        clf.fit(X_agg, y_agg)
        pred_labels = clf.predict(X_test).astype(int)
    except Exception:
        pred_labels = np.full(len(X_test), np.argmax(np.bincount(y_agg.astype(int))))

    reverse_label_map = {i: val for val, i in label_map.items()}
    decoded_predictions = np.empty(len(pred_labels), dtype=object)

    for i, label in enumerate(pred_labels):
        encoded_val = reverse_label_map[label]
        original_val = target_encoder.inverse_transform([encoded_val])[0]
        decoded_predictions[i] = original_val

    return decoded_predictions


def cafe_server_config(n_features: int, n_classes: int, imp_rounds: int):
    server_cfg = {
        "impute_mode": "instant",
        "imp_round": imp_rounds,
        "imp_local_epochs": 0,
        "pred_round": 0,
        "pred_local_epochs": 0,
        "model_fit_mode": "one_shot",
        "froze_ms_coefs_round": 100,
        "n_cols": n_features,
    }
    pred_cfg = {
        "model_params": {
            "model": "2nn", "num_hiddens": 32,
            "model_init_config": None, "model_other_params": None,
            "input_feature_dim": n_features, "output_classes_dim": n_classes,
        },
        "train_params": {
            "batch_size": 128, "learning_rate": 0.001,
            "weight_decay": 0.0001, "pred_round": 0, "pred_local_epochs": 3,
        }
    }
    return server_cfg, pred_cfg


def run_full_imputation(
        csv_path: str,
        num_clients: int = 5,
        partition: str = "sample-evenly",
        imp_rounds: int = 20,
        seed: int = 42,
        normalize: bool = True,
        dataset_null_char:str="",
        csv_sep          :str=","
):
    print("\n" + "=" * 65)
    print("  CAFE  ")
    print("=" * 65)

    df = pd.read_csv(csv_path, sep=csv_sep)

    for target_col in df.columns:
        print(f"\nLoading & preprocessing {target_col}...")
        train_df, test_df, test_idx, target_encoder, scaler = preprocess(
            csv_path, target_col, normalize,dataset_null_char, csv_sep
        )
        if len(test_idx) <= 0: continue
        try:
            imputed_values = cafe_train_predict(train_df,
                                                test_df,
                                                target_col,
                                                target_encoder,
                                                scaler,
                                                num_clients,
                                                imp_rounds,
                                                seed,
                                                partition)
            df.loc[test_idx, target_col] = imputed_values
        except: pass
    return df




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='CAFE_Exp',
        description='Run imputation using CAFE'
    )
    parser.add_argument('dataset_file_path', help="Path to the dataset file")
    parser.add_argument('--dataset_null_char', type=str, default='',
                        help='Character used to represent missing values in the dataset')
    parser.add_argument('--csv_sep', type=str, default=',',
                        help='CSV separator')
    parser.add_argument('--output_file_path', type=str, default='repaired.csv',
                        help='Repaired file path')


    parser.add_argument("--clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", default="sample-evenly")

    args = parser.parse_args()
    print("Start imputation")
    imputed_df = run_full_imputation(
        csv_path=args.dataset_file_path,
        dataset_null_char=args.dataset_null_char,
        csv_sep          =args.csv_sep,
        num_clients=args.clients,
        partition=args.partition,
        imp_rounds=args.rounds,
        seed=args.seed,
        normalize=True,
    )
    imputed_df.to_csv(
        args.output_file_path,
        index=False,
        sep=args.csv_sep
    )

    print("Imputation completed.")
