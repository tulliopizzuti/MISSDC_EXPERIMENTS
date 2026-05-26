# GiG
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLabelPredictor(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats,
        out_feats,
        input_tuple_length,
        num_layers=2,
        dropout=0,
        aggregation="concat",
        device="cpu",
    ):
        super().__init__()
        self.name = "multilabel"
        self.device = device
        self.h_feats = h_feats
        # The number of out_feats must be the same as the number of feats in a node
        self.out_feats = out_feats
        # The input contains input_tuple_length * in_feats features
        if aggregation == "concat":
            self.in_feats = input_tuple_length * in_feats
        elif aggregation == "average":
            self.in_feats = in_feats
        else:
            raise NotImplementedError
        self.aggr = aggregation
        layers = []

        layers.append(nn.Linear(self.in_feats, self.h_feats))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(1, num_layers - 1):
            layers.append(nn.Linear(self.h_feats, self.h_feats))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(self.h_feats, self.out_feats))
        # layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        for model_layer in range(len(self.model)):
            self.model[model_layer] = self.model[model_layer].to(self.device)

        if self.device == "cuda":
            self.model.cuda(self.device)

        self.tuple_embedding_matrix = None

    def compute_triplet_matrix(self, h, triplets):
        if self.aggr == "average":
            averaged_h = h[[triplets]].mean(dim=1)
            self.triplet_embedding_matrix = averaged_h.to(self.device)
        elif self.aggr == "concat":
            self.triplet_embedding_matrix = h[[triplets]].reshape(
                len(triplets), self.in_feats
            )

    def forward(self, h, triplets):
        self.compute_triplet_matrix(h, triplets)
        h = self.model[0](self.triplet_embedding_matrix).to(self.device)
        for layer in self.model[1:]:
            h = layer(h)
        res = h

        return res

    def get_model_structure(self):
        struct_string = ""
        for module in self.modules():
            mod_name = module._get_name()
            if mod_name == "Linear":
                mod_string = f"{mod_name}_{module.in_features}_{module.out_features}"
            else:
                mod_string = f"{mod_name}"
            struct_string += mod_string + "|"
        return struct_string
