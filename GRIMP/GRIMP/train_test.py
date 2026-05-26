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

# NUM_EPOCHS = 200


def compute_auc(pos_score, neg_score, args=None):
    # scores = torch.cat([pos_score]).numpy()
    # labels = torch.cat([torch.ones(pos_score.shape[0])]).numpy()
    scores = torch.cat([pos_score, neg_score]).numpy()
    labels = torch.cat(
        [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
    ).numpy()

    roc_auc = roc_auc_score(labels, scores)
    # if args and args.plot_roc:
    #     plt.figure()
    #     ax = plt.gca()
    #     tpr, fpr, thresholds = roc_curve(y_score=scores, y_true=labels)
    #     pp = plt.plot(tpr, fpr)
    #     fig_name, ext = osp.splitext(args.dirty_dataset)
    #     fig_name += '.png'
    #     if fig_name:
    #         plt.savefig(fig_name)
    return roc_auc


def train(
    graph_dataset,
    gnn_model,
    link_predictor,
    node_predictor=None,
    epochs=100,
    device="cpu",
    logger=None,
):
    optimizer = torch.optim.Adam(
        itertools.chain(gnn_model.parameters(), link_predictor.parameters()), lr=0.01
    )
    optimizer2 = torch.optim.Adam(
        itertools.chain(gnn_model.parameters(), link_predictor.parameters()), lr=0.01
    )

    node_predictor = NearestNodePredictor(16, 32, 6, num_layers=3)

    if logger:
        logger.add_dict("curves", {"loss": [], "min": inf, "end": 0, "min_epoch": -1})

    import numpy as np

    s1 = sum([np.prod(p.size()) for p in gnn_model.parameters()])
    s2 = sum([np.prod(p.size()) for p in link_predictor.parameters()])
    print("#Parameters GNN, LP", s1, s2)

    all_logits = []
    for epoch in range(epochs):
        # for epoch in tqdm(range(epochs)):
        # print("Epoch ", epoch)
        # forward
        graph_dataset.graph.to(device)
        graph_dataset.graph.ndata["features"].to(device)
        h = gnn_model(graph_dataset.graph, graph_dataset.graph.ndata["features"])
        h = h.to(device)

        if graph_dataset.graph_name == "edge":
            pos_score = link_predictor(h, graph_dataset.train_positive_samples)
            if link_predictor.name == "truefalse":
                neg_score = link_predictor(h, graph_dataset.train_negative_samples)
                loss = compute_loss_max_score(pos_score, neg_score, device)
                # loss = compute_loss_binary_cross_entropy_logits(pos_score, neg_score, device)
            elif link_predictor.name == "mlp":
                # raise NotImplementedError
                neg_score = link_predictor(h, graph_dataset.train_negative_samples)
                # loss = compute_loss(pos_score, neg_score, device)
                loss = compute_hinge_loss(pos_score, neg_score, device=device)

            else:
                raise NotImplementedError
        elif graph_dataset.graph_name == "triplet":
            pos_score = link_predictor(h, graph_dataset.train_positive_samples)
            if link_predictor.name == "truefalse":
                neg_score = link_predictor(h, graph_dataset.train_negative_samples)
                loss = compute_loss_max_score(pos_score, neg_score, device)
            elif link_predictor.name == "mlp":
                neg_score = link_predictor(h, graph_dataset.train_negative_samples)
                loss = compute_loss(pos_score, neg_score, device)
                # loss = compute_hinge_loss(pos_score, neg_score, device=device)
            else:
                raise NotImplementedError
        elif graph_dataset.graph_name == "nn":
            pos_score = link_predictor(h, graph_dataset.train_positive_samples)

            neg_score = link_predictor(h, graph_dataset.train_negative_samples)

            # loss = compute_loss_cosine(pos_score, graph_dataset, h)
            loss = compute_loss(
                pos_score.squeeze(1), neg_score.squeeze(1), device=device
            )
            # loss = compute_hinge_loss(pos_score.squeeze(1), neg_score.squeeze(1), device=device)
        elif graph_dataset.graph_name == "avg":
            pos_score = link_predictor(
                h,
                graph_dataset.train_positive_samples,
                graph_dataset.train_positive_triplets,
            )

            neg_score = link_predictor(
                h,
                graph_dataset.train_negative_samples,
                graph_dataset.train_negative_triplets,
            )

            # loss = compute_loss_cosine(pos_score, graph_dataset, h)
            loss = compute_loss(
                pos_score.squeeze(1), neg_score.squeeze(1), device=device
            )
            # loss = compute_hinge_loss(pos_score.squeeze(1), neg_score.squeeze(1), device=device)

        elif graph_dataset.graph_name == "multilabel":
            pos_score = link_predictor(h, graph_dataset.train_positive_samples)

            # loss = compute_loss_cosine(pos_score, graph_dataset, h)
            loss = compute_loss_multilabel(
                pos_score.squeeze(1), graph_dataset, device=device
            )
            # loss = compute_hinge_loss(pos_score.squeeze(1), neg_score.squeeze(1), device=device)

        else:
            raise ValueError("Type Error in graph dataset.")
        # backward
        optimizer.zero_grad()
        # print("optimizer zero grad")
        loss.backward()
        # print("loss backward")
        optimizer.step()
        # print("optimizer.step")

        if epoch % 5 == 0:
            print("In epoch {}, loss1: {}".format(epoch, loss))
            # print('In epoch {}, loss2: {}'.format(epoch, loss2))

        if epoch % 50 == 0:
            print("epoch 50")

        if logger:
            with torch.no_grad():
                logger["curves"]["loss"].append(loss.item())
                if logger["curves"]["min"] > loss.item():
                    logger["curves"]["min"] = loss.item()
                    logger["curves"]["min_epoch"] = epoch
                    best_state = h.cpu().detach().numpy()
                logger["curves"]["end"] = loss.item()

    print(
        f'Minimum loss {logger["curves"]["min"]} @ epoch {logger["curves"]["min_epoch"]}'
    )
    return best_state
    return h.cpu().detach().numpy()
