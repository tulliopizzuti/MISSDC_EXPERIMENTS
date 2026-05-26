import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn_models import GraphSAGE
from .multitask_predictor import MultiTaskPredictor
from .multilabel_predictor import MultiLabelPredictor


class GRIMP_model(nn.Module):
    def __init__(
        self,
        architecture,
        input_tuple_length,
        gnn_in_feats,
        gnn_h_feats,
        gnn_layers,
        gnn_aggr="gcn",
        predictor_layers=2,
        predictor_h_feats=32,
        head_h_layers=2,
        head_out_feats_list=[],
        multilabel_out_feats=None,
        multilabel_aggregation="concat",
        dropout=0,
        device="cpu",
    ):
        super(GRIMP_model, self).__init__()
        self.name = "grimp"
        self.model = nn.ModuleList()
        self.model.add_module(
            "gnn_model", GraphSAGE(gnn_in_feats, gnn_h_feats, gnn_layers, gnn_aggr)
        )

        self.architecture = architecture

        if architecture == "multitask":
            self.model.add_module(
                "multitask",
                MultiTaskPredictor(
                    shared_in_feats=gnn_h_feats,
                    shared_out_feats=predictor_h_feats,
                    shared_h_feats=predictor_h_feats,
                    shared_h_layers=predictor_layers,
                    head_h_layers=head_h_layers,
                    head_out_feats_list=head_out_feats_list,
                    shared_out_feats_list=None,
                    input_tuple_length=input_tuple_length,
                    dropout=dropout,
                    device=device,
                ),
            )

        elif architecture == "multilabel":
            self.model.add_module(
                "multilabel",
                MultiLabelPredictor(
                    in_feats=gnn_h_feats,
                    h_feats=predictor_h_feats,
                    out_feats=multilabel_out_feats,
                    input_tuple_length=input_tuple_length,
                    num_layers=predictor_layers,
                    dropout=dropout,
                    aggregation=multilabel_aggregation,
                    device=device,
                ),
            )

        else:
            raise ValueError(f"Unknown architecture {architecture}")

    def forward(self, graph, node_features, train_pos_samples, num_samples=None):
        h = self.model.gnn_model(graph, node_features)
        self.current_state = h.detach()
        if self.architecture == "multitask":
            return self.model.multitask(h, train_pos_samples, num_samples)
        elif self.architecture == "multilabel":
            return self.model.multilabel(h, train_pos_samples)
        else:
            raise NotImplementedError

    def evaluate(self, graph_dataset, sample, target_column=None):
        h = self.model.gnn_model(
            graph_dataset.graph, graph_dataset.graph.ndata["features"]
        )
        if self.architecture == "multitask":
            return self.model.multitask.evaluate(
                h, sample, graph_dataset, target_column
            )
        elif self.architecture == "multilabel":
            return self.model.multilabel(h, sample)
        else:
            raise NotImplementedError

    def get_model_structure(self):
        structure = ""
        for model in self.model:
            structure = structure + "_" + f"{model.get_model_structure}"
        return structure
