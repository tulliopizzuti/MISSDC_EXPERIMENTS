import argparse
import os.path as osp
import os
import pickle
import random
import time

import numpy as np
import torch
from numpy import inf

import GRIMP.gnn_models as gnn_models
import GRIMP.multilabel_training as mlt
import GRIMP.testing_utils as testing_utils
from GRIMP.gnn_utils import (
    read_external_features,
    read_features_tensor,
    read_functional_dependencies,
)
from GRIMP.heterograph_dataset import HeterographDataset
from GRIMP.logging import GrimpLogger
from GRIMP.multitask_predictor import MultiTaskPredictor


def set_random_seeds(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_graph_dataset(load_graph_dataset_path):
    graph_dataset = pickle.load(open(load_graph_dataset_path, "rb"))
    return graph_dataset


def create_graph_dataset( original_data_file_name, missing_value_file_name ):
    start = time.time()

    ## defatult params
    graph_layers = 2
    gnn_feats = 64
    h_feats = 128
    predictor_layers = 2
    head_layers = 2
    dropout_clf = 0.2
    batchnorm = False
    shared_model = "attention"
    head_model = "attention"
    k_strat = "weak"
    no_relu = False
    no_sm = False
    
    print("creating graph from scratch")
    graph_dataset = HeterographDataset(
        original_data_file_name,
        missing_value_file_name
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_dataset.graph = graph_dataset.graph.to(device)
    graph_stats = graph_dataset.get_statistics()

    print(graph_stats)

    imputation_columns = graph_dataset.training_columns

    gnn_params = {
            "gnn_in_feats": graph_dataset.num_features,
            "gnn_h_feats": h_feats,
            "num_layers": graph_layers,
            "build_columns": graph_dataset.all_columns,
            "build_fds": graph_dataset.fd_list,
            "heteroconv_aggr": "sum",
            "module_aggr": "gcn",
            "dropout": 0,
        }
    gnn_model = gnn_models.HeteroGraphSAGE(**gnn_params).to(device)

    distinct_values = []
    for col in graph_dataset.all_columns:
        if col in imputation_columns:
            col_id = graph_dataset.col2idx[col]
            distinct_values.append(graph_dataset.head_dims[col_id])
        else:
            distinct_values.append(None)

    gnn_feats = gnn_feats * graph_layers

    mt_params = {
        "shared_in_feats": gnn_feats,
        "shared_out_feats": gnn_feats,
        "shared_h_feats": h_feats,
        "shared_h_layers": predictor_layers,
        "head_h_layers": head_layers,
        "head_out_feats_list": distinct_values,
        "input_tuple_length": graph_dataset.input_tuple_length,
        "dropout": dropout_clf,
        "batchnorm": batchnorm,
        "shared_model": shared_model,
        "head_model": head_model,
        "graph_dataset": graph_dataset,
        "k_strat": k_strat,
        "no_relu": no_relu,
        "no_sm": no_sm,
        "device": device,
    }

    prediction_model = MultiTaskPredictor(**mt_params).to(device)

    prediction_model = prediction_model.to(device)

    init_parameters = {
        "gnn_model": gnn_params,
        "multitask_model": mt_params,
    }

    states = {
        "init_params": init_parameters,
        "checkpoints_gnn": [],
        "checkpoints_mt": [],
    }

    print('aqui')

    return graph_dataset


def run_training(args: argparse.Namespace, logger: GrimpLogger):
    """
    Train the imputation model following the arguments passed in args.
    :param args: User arguments passed on the command line.
    :return:
    """
    if args.seed is not None:
        set_random_seeds(args.seed)

    if args.load_model_file_path:
        graph_dataset = load_graph_dataset(args.load_model_file_path)
    else:
        graph_dataset = create_graph_dataset(args, logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_dataset.graph = graph_dataset.graph.to(device)
    graph_stats = graph_dataset.get_statistics()
    if graph_stats["node_features"] == "external":
        if args.text_embs is not None:
            graph_stats["node_features"] = args.text_embs
        elif args.np_mat is not None:
            graph_stats["node_features"] = args.np_mat
        else:
            raise ValueError
    if args.fd_path is not None:
        graph_stats["fd_path"] = args.fd_path
    else:
        graph_stats["fd_path"] = None
    logger.add_dict("statistics", graph_stats)
    logger.add_value(
        "parameters",
        "training_columns",
        logger.get_value("statistics", "training_columns"),
    )

    imputation_columns = graph_dataset.training_columns

    if not args.skip_gnn:
        gnn_params = {
            "gnn_in_feats": graph_dataset.num_features,
            "gnn_h_feats": args.gnn_feats,
            "num_layers": args.graph_layers,
            "build_columns": graph_dataset.all_columns,
            "build_fds": graph_dataset.fd_list,
            "heteroconv_aggr": args.heteroconv_aggr,
            "module_aggr": args.module_aggr,
            "dropout": args.dropout_gnn,
        }
        gnn_model = gnn_models.HeteroGraphSAGE(**gnn_params).to(device)
    else:
        gnn_model = torch.nn.Linear(graph_dataset.num_features, args.gnn_feats).to(
            device
        )
        gnn_params = {
            "in_features": graph_dataset.num_features,
            "out_features": args.gnn_feats,
        }

    distinct_values = []
    for col in graph_dataset.all_columns:
        if col in imputation_columns:
            col_id = graph_dataset.col2idx[col]
            distinct_values.append(graph_dataset.head_dims[col_id])
        else:
            distinct_values.append(None)

    if args.jumping_knowledge:
        gnn_feats = args.gnn_feats * args.graph_layers
    else:
        gnn_feats = args.gnn_feats

    mt_params = {
        "shared_in_feats": gnn_feats,
        "shared_out_feats": gnn_feats,
        "shared_h_feats": args.h_feats,
        "shared_h_layers": args.predictor_layers,
        "head_h_layers": args.head_layers,
        "head_out_feats_list": distinct_values,
        "input_tuple_length": graph_dataset.input_tuple_length,
        "dropout": args.dropout_clf,
        "batchnorm": args.batchnorm,
        "shared_model": args.shared_model,
        "head_model": args.head_model,
        "graph_dataset": graph_dataset,
        "k_strat": args.k_strat,
        "no_relu": args.no_relu,
        "no_sm": args.no_sm,
        "device": device,
    }

    prediction_model = MultiTaskPredictor(**mt_params).to(device)
    logger.add_value(
        "parameters", "predictor_structure", prediction_model.get_model_structure()
    )
    prediction_model = prediction_model.to(device)
    if not args.skip_gnn:
        logger.add_value("parameters", "gnn_structure", gnn_model.get_model_name())
    else:
        logger.add_value("parameters", "gnn_structure", "liner")
    init_parameters = {
        "gnn_model": gnn_params,
        "multitask_model": mt_params,
    }

    states = {
        "init_params": init_parameters,
        "checkpoints_gnn": [],
        "checkpoints_mt": [],
    }

    print("Begin training")
    start = time.time()
    logger.add_time("start_training")
    logger.add_dict(
        "curves",
        {
            "loss": [],
            "min": inf,
            "end": 0,
            "min_epoch": -1,
            "loss_valid": [],
            "min_valid": inf,
            "end_valid": 0,
            "min_epoch_valid": -1,
        },
    )

    best_state = mlt.train(
        graph_dataset,
        gnn_model,
        multitask_model=prediction_model,
        args=args,
        device=device,
        logger=logger,
    )

    print("training took ", time.time() - start)
    logger.add_time("end_training")
    logger.add_duration("start_training", "end_training", "training_duration")

    return graph_dataset, best_state, init_parameters


def run_testing(
    args=None,
    graph_dataset=None,
    best_state=None,
    init_params=None,
    logger: GrimpLogger = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Begin testing")
    start = time.time()
    logger.add_time("start_testing")

    df_imputed = testing_utils.generate_imputed_dataset_multitask(
        graph_dataset, best_state, init_params, args.skip_gnn
    )

    imp_acc_dict = testing_utils.measure_imp_accuracy(
        graph_dataset, df_imputed, logger=logger
    )

    logger.add_time("end_testing")
    logger.add_duration("start_testing", "end_testing", "testing_duration")
    logger.pprint()

    logger.save_obj(plot_figures=True)

    print(
        f"Testing took {time.time() - start:2f} seconds",
    )

    if args.save_imputed_df:
        fullname, ext = osp.splitext(args.dirty_dataset)
        basename = osp.basename(fullname)
        os.makedirs("results/imputed_datasets", exist_ok=True)
        df_imp_fname = "results/imputed_datasets/" + basename + "_imputed_grimp"
        if args.imputed_df_tag:
            df_imp_fname += f"_{args.imputed_df_tag}"
        df_imp_fname += ext
        df_imputed.to_csv(df_imp_fname, index=False)
        print(f"Imputed dataset saved in {df_imp_fname}")
