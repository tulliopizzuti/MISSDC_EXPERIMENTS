import itertools

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from numpy import inf
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd

from GRIMP.imputation_tripartite_graph import (
    ImputationTripartiteGraphEdgePrediction,
    ImputationTripartiteGraphTripletPrediction,
    ImputationTripartiteGraphNearestNeighbor,
)
from GRIMP.loss_functions import *
from GRIMP.node_prediction_models import NearestNodePredictor
from copy import deepcopy


def early_stopping(epoch, logger, args):
    lvalids = logger["curves"]["loss_valid"][-15:]
    if epoch > args.grace:
        if np.mean(lvalids) - logger["curves"]["min_valid"] < args.th_stop:
            print("No improvement in the loss.")
            return True
        if (
            sum([lvalids[i + 1] > lvalids[i] for i in range(0, len(lvalids) - 1)])
            / len(lvalids)
            > 0.5
        ):
            print("Validation loss is increasing.")
            # print(f'{[f"{_:.5}" for _ in lvalids]}')
            return True
    return False


def train(
    graph_dataset,
    gnn_model: torch.nn.Module,
    multitask_model,
    args,
    device="cpu",
    logger=None,
):

    epochs = args.epochs

    optimizer = torch.optim.Adam(
        itertools.chain(gnn_model.parameters(), multitask_model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    best_state = {
        "gnn_model": gnn_model.state_dict(),
        "multitask_model": multitask_model.state_dict(),
    }

    if args.skip_gnn:
        feats = graph_dataset.graph.ndata["features"]
        ff = feats["cell"].clone().detach()
        ff[: feats["rid"].shape[0], :] = feats["rid"]
        ff = ff.to(device)
    else:
        ff = None

    # XE or focal
    loss_type = args.loss

    for epoch in range(epochs + 1):
        try:
            loss = torch.tensor(0.0, requires_grad=True)
            loss_valid = torch.tensor(0.0, requires_grad=False)
            graph_dataset.graph.to(device)
            multitask_model.train()
            gnn_model.train()
            if not args.skip_gnn:
                # Train with gnn
                h = gnn_model(
                    graph_dataset.graph,
                    graph_dataset.graph.ndata["features"],
                    graph_dataset.graph.edata["features"],
                )
            else:
                # No GNN: use the pretrained features as they are.
                h = gnn_model(ff)

            h = h.to(device)
            # h = graph_dataset.graph.nodes[:].data['features']

            pos_score = multitask_model(
                h,
                graph_dataset.train_positive_samples,
                graph_dataset.boundaries_train,
                step="train",
            )
            idx2col = {
                idx: col for idx, col in enumerate(graph_dataset.df_missing.columns)
            }
            for idx in pos_score:
                if multitask_model.head_type[idx] == "categorical":
                    if args.loss == "xe":
                        loss = loss + compute_loss_multilabel_attr(
                            pos_score[idx], graph_dataset.labels, idx, device=device
                        )
                    elif args.loss == "focal":
                        loss = loss + compute_focal_loss(
                            pos_score[idx],
                            graph_dataset,
                            "train",
                            target_column=idx,
                            alpha=args.loss_alpha,
                            gamma=args.loss_gamma,
                            # alpha=graph_dataset.loss_alpha[idx2col[idx]], gamma=graph_dataset.loss_gamma[idx2col[idx]],
                            device=device,
                        )
                    else:
                        raise ValueError(f"Unknown loss {args.loss}.")
                else:  # numeric head
                    loss = loss + compute_loss_mse(
                        pos_score[idx], graph_dataset.labels, idx, device=device
                    )
            # loss /= len(graph_dataset.train_positive_samples)

            # Compute validation loss

            multitask_model.eval()
            if gnn_model is not None:
                gnn_model.eval()
            valid_score = multitask_model(
                h,
                graph_dataset.valid_positive_samples,
                graph_dataset.boundaries_valid,
                step="valid",
            )
            for idx in valid_score:
                if multitask_model.head_type[idx] == "categorical":
                    if args.loss == "xe":
                        loss_valid = loss_valid + compute_loss_multilabel_attr(
                            valid_score[idx],
                            graph_dataset.labels_valid,
                            target_column=idx,
                            device=device,
                        )
                    elif args.loss == "focal":
                        loss_valid = loss_valid + compute_focal_loss(
                            valid_score[idx],
                            graph_dataset,
                            "valid",
                            target_column=idx,
                            alpha=args.loss_alpha,
                            gamma=args.loss_gamma,
                            # alpha=graph_dataset.loss_alpha[idx2col[idx]], gamma=graph_dataset.loss_gamma[idx2col[idx]],
                            device=device,
                        )
                    else:
                        raise ValueError(f"Unknown loss {args.loss}.")
                else:  # numeric head
                    loss_valid = loss_valid + compute_loss_mse(
                        valid_score[idx], graph_dataset.labels_valid, idx, device=device
                    )

            # loss_valid /= len(graph_dataset.valid_positive_samples)

            # backward
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            if epoch % 5 == 0:
                print(
                    f"In epoch {epoch}, {loss_type} loss: {loss.item():.5f}, validation {loss_type} loss: {loss_valid.item():.5f}"
                )

            if epoch % 50 == 0:
                print(f"{epoch}")
                # logger['checkpoints']['checkpoints_gnn'].append(deepcopy(gnn_model.state_dict()))
                # logger['checkpoints']['checkpoints_mt'].append(deepcopy(multitask_model.state_dict()))

            if logger:
                with torch.no_grad():
                    l_item = loss.item()
                    l_valid_item = loss_valid.item()
                    logger["curves"]["loss"].append(l_item)
                    logger["curves"]["loss_valid"].append(l_valid_item)

                    if logger["curves"]["min"] > l_item:
                        logger["curves"]["min"] = l_item
                        logger["curves"]["min_epoch"] = epoch
                        best_state["gnn_model"] = gnn_model.state_dict()
                        best_state["multitask_model"] = multitask_model.state_dict()
                        # best_state = h.cpu().detach().numpy()
                    if logger["curves"]["min_valid"] > l_valid_item:
                        logger["curves"]["min_valid"] = l_valid_item
                        logger["curves"]["min_epoch_valid"] = epoch
                        # best_state = h.cpu().detach().numpy()
                    else:
                        if not args.force_training:
                            if early_stopping(epoch, logger, args):
                                logger["curves"]["end"] = l_item
                                logger["curves"]["end_valid"] = l_valid_item
                                print(
                                    f"Early stopping at epoch {epoch}. Training loss: {l_item}\tValidation loss: {l_valid_item}"
                                )
                                break
                logger["parameters"]["epochs"] = epoch
                logger["curves"]["end"] = l_item
                logger["curves"]["end_valid"] = l_valid_item
            else:
                raise NotImplementedError
        except KeyboardInterrupt:
            print("Interrupting by keyboard interrupt. ")
            break

    if logger:
        print(
            f'Minimum loss {logger["curves"]["min"]} @ epoch {logger["curves"]["min_epoch"]}'
        )
        print(
            f'Minimum validation loss {logger["curves"]["min_valid"]} @ epoch {logger["curves"]["min_epoch_valid"]}'
        )
    return best_state
