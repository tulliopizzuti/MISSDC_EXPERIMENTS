# GiG
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import SAGEConv
from dgl.nn.pytorch import GATConv
import dgl.nn.pytorch as dglnn
import torch


# Copied from https://docs.dgl.ai/tutorials/blitz/4_link_predict.html
class GraphSAGE(nn.Module):
    def __init__(
        self,
        gnn_in_feats,
        gnn_h_feats,
        num_layers=2,
        aggr="gcn",
        jump_know=False,
        dropout=0,
    ):
        super(GraphSAGE, self).__init__()
        layers = []
        # layer 1
        self.input_feats = gnn_in_feats
        self.hidden_feats = gnn_h_feats
        self.jump_know = jump_know
        layers.append(
            SAGEConv(
                self.input_feats,
                self.hidden_feats,
                aggr,
                bias=False,
                feat_drop=dropout,
                activation=F.relu,
            )
        )

        # # all other layers
        #     layers.append(nn.Dropout(p=0.5))
        for c in range(num_layers - 1):
            layers.append(
                SAGEConv(
                    self.hidden_feats,
                    self.hidden_feats,
                    aggr,
                    bias=False,
                    feat_drop=dropout,
                    activation=F.relu,
                )
            )

        # if self.jump_know:
        #     layers.append(nn.Linear(self.hidden_feats*num_layers, self.hidden_feats))
        # else:
        #     # pass
        #     layers.append(nn.Linear(self.hidden_feats, self.hidden_feats))

        self.model = nn.Sequential(*layers)

    def forward(self, g, in_feat, edge_weights=None):
        # out = torch.zeros(size=(g.num_nodes(), self.hidden_feats*len(self.model)))
        h = in_feat
        if self.jump_know:
            out = []
            for idx, layer in enumerate(self.model, start=1):
                h = layer(g, h, edge_weight=edge_weights)
                out.append(h)
            # return self.model[-1](torch.cat(out, dim=-1))
            return torch.cat(out, dim=-1)

        else:
            for idx, layer in enumerate(self.model, start=1):
                h = layer(g, h, edge_weight=edge_weights)
            return h

    def get_model_structure(self):
        struct_string = ""
        for module in self.modules():
            mod_name = module._get_name()
            mod_string = f"{mod_name}"
            struct_string += mod_string + "|"
        return struct_string


"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""


class GAT(nn.Module):
    def __init__(
        self,
        # g,
        num_layers,
        in_dim,
        hid_dim,
        out_dim,
        heads,  # List of heads to be used in each layer
        activation=None,
        feat_drop=0,
        attn_drop=0,
        negative_slope=0.2,
        residual=False,
    ):
        super(GAT, self).__init__()
        # self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        if activation is None:
            activation = nn.functional.relu
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(
            GATConv(
                in_dim,
                hid_dim,
                heads[0],
                feat_drop,
                attn_drop,
                negative_slope,
                residual=False,
                activation=self.activation,
            )
        )
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = hid_dim * num_heads
            self.gat_layers.append(
                GATConv(
                    hid_dim * heads[l - 1],
                    hid_dim,
                    heads[l],
                    feat_drop,
                    attn_drop,
                    negative_slope,
                    residual,
                    self.activation,
                )
            )
        # output projection
        self.gat_layers.append(
            GATConv(
                hid_dim * heads[-2],
                out_dim,
                heads[-1],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,
                None,
            )
        )

    def forward(self, g, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits

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


class HeteroGraphSAGE(nn.Module):
    def __init__(
        self,
        gnn_in_feats,
        gnn_h_feats,
        num_layers=2,
        build_columns=[],
        build_fds=[],
        module_aggr="gcn",
        heteroconv_aggr="mean",
        dropout=0,
    ):
        super(HeteroGraphSAGE, self).__init__()
        layers = []
        self.hidden_feats = gnn_h_feats
        self.first_layer = {}
        self.inner_modules = {}

        assert len(build_columns) > 0
        self.num_columns = len(build_columns)
        relu = nn.ReLU()

        for col in build_columns:
            self.first_layer[col] = dglnn.SAGEConv(
                gnn_in_feats,
                self.hidden_feats,
                aggregator_type=module_aggr,
                bias=True,
                feat_drop=dropout,
            )
            self.first_layer[f"i_{col}"] = dglnn.SAGEConv(
                gnn_in_feats,
                self.hidden_feats,
                aggregator_type=module_aggr,
                bias=True,
                feat_drop=dropout,
            )
            # self.first_layer[f'n_{col}'] = dglnn.SAGEConv(gnn_in_feats, self.hidden_feats, aggregator_type=module_aggr, bias=True, feat_drop=dropout)
            # self.first_layer[f'n_i_{col}'] = dglnn.SAGEConv(gnn_in_feats, self.hidden_feats, aggregator_type=module_aggr, bias=True, feat_drop=dropout)
        for layer in range(num_layers - 1):
            for col in build_columns:
                self.inner_modules[col] = dglnn.SAGEConv(
                    self.hidden_feats,
                    self.hidden_feats,
                    aggregator_type=module_aggr,
                    bias=True,
                    feat_drop=dropout,
                )
                self.inner_modules[f"i_{col}"] = dglnn.SAGEConv(
                    self.hidden_feats,
                    self.hidden_feats,
                    aggregator_type=module_aggr,
                    bias=True,
                    feat_drop=dropout,
                )
                # self.inner_modules[f'n_{col}'] = dglnn.SAGEConv(self.hidden_feats, self.hidden_feats, aggregator_type=module_aggr, bias=True, feat_drop=dropout)
                # self.inner_modules[f'n_i_{col}'] = dglnn.SAGEConv(self.hidden_feats, self.hidden_feats, aggregator_type=module_aggr, bias=True, feat_drop=dropout)

        # if len(build_fds) > 0:
        #     for fd in build_fds:
        #         self.first_layer[fd] = dglnn.SAGEConv(gnn_in_feats, self.hidden_feats, aggregator_type=module_aggr, bias=True, feat_drop=dropout)
        #     for layer in range(num_layers - 1):
        #         for fd in build_fds:
        #             self.inner_modules[fd] = dglnn.SAGEConv(self.hidden_feats, self.hidden_feats, aggregator_type=module_aggr, bias=True, feat_drop=dropout)
        conv1 = dglnn.HeteroGraphConv(self.first_layer, aggregate=heteroconv_aggr)
        layers.append(conv1)
        for layer in range(num_layers - 1):
            layers.append(
                dglnn.HeteroGraphConv(self.inner_modules, aggregate=heteroconv_aggr)
            )
        # conv2 = dglnn.HeteroGraphConv(self.inner_modules, aggregate=heteroconv_aggr)
        # layers.append(conv2)
        layers.append(relu)
        # for c in range(num_layers-1):
        #     layers.append(SAGEConv(self.hidden_feats, self.hidden_feats, aggregator_type=module_aggr, bias=False, feat_drop=dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, g, in_feat_dict, e_feat_dict=None):
        # x_src = {'rid': in_feat_dict['rid']}
        # x_dst = {'cell': in_feat_dict['cell']}

        h_start = self.model[0](g, in_feat_dict)
        h_start["rid"] = h_start["rid"] - torch.mean(h_start["rid"], dim=0)
        h_start["cell"] = h_start["cell"] - torch.mean(h_start["cell"], dim=0)

        h_start = self.model[1](g, h_start)
        h_start["rid"] = h_start["rid"] - torch.mean(h_start["rid"], dim=0)
        h_start["cell"] = h_start["cell"] - torch.mean(h_start["cell"], dim=0)

        h1 = h_start["cell"]
        h1[: h_start["rid"].shape[0], :] = h_start["rid"]
        # for idx, layer in enumerate(self.model[1:], start=1):
        #     h1 = layer(g, h1)
        return self.model[-1](h1)

    def get_model_structure(self):
        struct_string = ""
        for module in self.modules():
            mod_name = module._get_name()
            if mod_name == "Linear" or mod_name == "Dropout":
                continue
                # mod_string = f'{mod_name}_{module.in_features}_{module.out_features}'
            else:
                mod_string = f"{mod_name}"
            if mod_name == "ModuleDict":
                print(len(mod_name))
            struct_string += mod_string + "|"
        return struct_string

    def get_model_name(self):
        name_string = f"HeteroGraphSAGE|SAGEConv{self.num_columns}"
        return name_string
