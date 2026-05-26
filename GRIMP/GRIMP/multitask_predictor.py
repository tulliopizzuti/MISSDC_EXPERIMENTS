import torch
import torch.nn as nn
import torch.nn.functional as F
from GRIMP.multilabel_graph_dataset import ImputationTripartiteGraphMultilabelClassifier

# debug
import matplotlib.pyplot as plt
import seaborn as sns


def get_q_k(k_strat, graph_dataset: ImputationTripartiteGraphMultilabelClassifier):
    # fill contains the cells that must be filled in each slice t of k, that is k[t,:,:]
    fill = [
        _ if _ not in graph_dataset.flat_fds else [_] + graph_dataset.flat_fds[_]
        for _ in range(graph_dataset.num_columns)
    ]

    if k_strat == "full":
        # Case 1: full diagonal
        k = torch.eye(graph_dataset.num_columns).repeat(
            ((graph_dataset.num_columns, 1, 1))
        )
    elif k_strat == "single":
        # Case 2: Only column t active in slice t, FDs have value 0.5
        k = torch.zeros(
            (
                graph_dataset.num_columns,
                graph_dataset.num_columns,
                graph_dataset.num_columns,
            )
        )
        for row in range(k.shape[0]):
            # If there are no FDs, the line below does nothing.
            k[row, fill[row], fill[row]] = 0.5
            k[row, row, row] = 1
    elif k_strat == "weak":
        # Case 3: Weak diagonal. 0.1 on all values except t, which has value 1
        k = torch.eye(graph_dataset.num_columns).repeat(
            ((graph_dataset.num_columns, 1, 1))
        )
        for row in range(k.shape[0]):
            # Reduce the weight of the full diagonal to 0.1
            k[row, :, :] *= 0.1
            # If there are no FDs, the line below does nothing.
            k[row, fill[row], fill[row]] = 0.5
            # Value t should have weight 1
            k[row, row, row] = 1
    else:
        raise ValueError(f"Unknown strat {k_strat}")
    q = F.normalize(torch.stack(graph_dataset.column_features), dim=0).to(torch.float32)
    return q, k


class MultiTaskPredictor(nn.Module):
    def __init__(
        self,
        shared_in_feats,
        shared_out_feats,
        shared_h_feats=32,
        shared_h_layers=2,
        head_h_layers=2,
        head_out_feats_list=[],
        shared_out_feats_list=None,
        input_tuple_length=None,
        dropout=0,
        batchnorm=False,
        shared_model="linear",
        head_model="attention",
        graph_dataset=None,
        k_strat="single",
        no_relu=False,
        no_sm=False,
        device="cpu",
    ):
        """

        :param shared_in_feats: Input features to the shared layer. This is equal to gnn_feats.
        :param shared_out_feats: Size of the last layer of the shared linear layer.
        :param shared_h_feats: Hidden layer size in the shared step.
        :param shared_h_layers: Number of hidden layers in the shared step.
        :param head_h_layers: Number of hidden layers in each head.
        :param head_out_feats_list: Number of features
        :param shared_out_feats_list:
        :param input_tuple_length:
        :param dropout:
        :param batchnorm:
        :param shared_model:
        :param head_model:
        :param graph_dataset:
        :param device:
        """
        super(MultiTaskPredictor, self).__init__()

        self.name = "multitaskpredictor"
        self.device = device

        self.in_feats = shared_in_feats
        self.out_feats = shared_out_feats
        self.h_feats = shared_h_feats
        self.shared_h_layers = shared_h_layers
        self.head_h_layers = head_h_layers
        self.shared_attention = True if shared_model == "attention" else False
        self.head_attention = True if head_model == "attention" else False
        self.head_type = dict()

        self.h2c = graph_dataset.map_h2c
        self.c2h = graph_dataset.map_c2h

        for idx, feats in enumerate(head_out_feats_list):
            if feats is None:
                continue
            tgt_col = graph_dataset.all_columns[idx]
            if tgt_col in graph_dataset.numerical_columns:
                self.head_type[idx] = "numerical"
            elif tgt_col in graph_dataset.categorical_columns:
                self.head_type[idx] = "categorical"
            else:
                raise ValueError(f"Some issue with column {tgt_col}")

        if self.shared_attention or self.head_attention:
            train_b = len(graph_dataset.train_positive_triplets)
            valid_b = train_b + len(graph_dataset.valid_positive_triplets)
            test_b = valid_b + len(graph_dataset.test_positive_triplets)
            q, k = get_q_k(k_strat, graph_dataset)

        if self.shared_attention:
            self.attn_feats = shared_in_feats
            self.base_layer = AttentionLayer(q, k, test_b, device=device)
            self.base_layer.train_mask = torch.arange(0, train_b, dtype=int)
            self.base_layer.valid_mask = torch.arange(train_b, valid_b, dtype=int)
            self.base_layer.test_mask = torch.arange(valid_b, test_b, dtype=int)

            self.base_layer.bound_start["train"] = 0
            self.base_layer.bound_start["valid"] = train_b
            self.base_layer.bound_start["test"] = valid_b
        else:
            self.attn_feats = shared_in_feats
            if input_tuple_length is not None:
                self.base_layer = SharedLayer(
                    shared_in_feats,
                    shared_h_feats,
                    shared_out_feats,
                    shared_h_layers,
                    dropout,
                    input_tuple_length=input_tuple_length,
                    device=device,
                )
            else:
                self.base_layer = SharedLayer(
                    shared_in_feats,
                    shared_h_feats,
                    shared_out_feats,
                    shared_h_layers,
                    dropout,
                    device,
                )

        if self.head_attention:
            if len(head_out_feats_list) > 0:
                self.heads = nn.ModuleList([])
                for idx, feats in enumerate(head_out_feats_list):
                    if feats is None:
                        new_module = nn.Module()
                        self.heads.append(new_module)
                        continue
                    new_module = AttentionHead(
                        q,
                        k[idx].clone().detach(),
                        None,
                        dropout=dropout,
                        out_feat_size=self.in_feats,
                        head_out=feats,
                        no_relu=no_relu,
                        no_sm=no_sm,
                        device=device,
                    )
                    new_module.train_mask = torch.arange(0, train_b, dtype=int)
                    new_module.valid_mask = torch.arange(train_b, valid_b, dtype=int)
                    new_module.test_mask = torch.arange(valid_b, test_b, dtype=int)

                    new_module.bound_start["train"] = 0
                    new_module.bound_start["valid"] = train_b
                    new_module.bound_start["test"] = valid_b

                    self.heads.append(new_module)
                self.num_heads = len(self.heads)

        else:
            if len(head_out_feats_list) > 0:
                self.heads = nn.ModuleList([])
                for idx, feats in enumerate(head_out_feats_list):
                    if feats is None:
                        continue
                    new_module = AttributeLayer(
                        self.out_feats,
                        h_feats=self.h_feats,
                        num_hidden_layers=head_h_layers,
                        out_feats=feats,
                        dropout=dropout,
                        batchnorm=batchnorm,
                        device=device,
                    )
                    self.heads.append(new_module)
                self.num_heads = len(self.heads)
            else:
                self.heads = None
                self.num_heads = 1
                if shared_out_feats_list is None:
                    raise ValueError(
                        f"Size of output layer required if no heads are present. "
                    )

    def forward(self, h, train_pos_samples, boundaries, step=None):
        if self.shared_attention:
            h_shared = self.base_layer(h, train_pos_samples, step)
        else:
            h_shared = self.base_layer(h, train_pos_samples)
            pass

        pos_score = {self.h2c[idx]: 0 for idx, _ in enumerate(self.h2c)}
        if self.head_attention:
            for head_idx, idx in self.h2c.items():
                head = self.heads[idx]
                start = boundaries[head_idx]
                end = boundaries[head_idx + 1]
                bb = (start, end)
                # v = h[[train_pos_samples]][start:end, :]
                v = h[[train_pos_samples]][start:end, :]
                # v = h_shared[boundaries[idx]: boundaries[idx + 1], :]

                pos_score[idx] = head(v, h_shared[start:end, :], step, bb)
        else:
            for idx, head in enumerate(self.heads):
                subset = h_shared[boundaries[idx] : boundaries[idx + 1], :]
                pos_score[idx] = head(subset)
        return pos_score

    def evaluate(self, h, test_sample, graph_dataset, target_column):
        self.eval()
        h_shared = self.base_layer(h, test_sample)
        head = self.heads[target_column]
        # subset = h_shared[head_idx * graph_dataset.attr_train_positive_samples : (head_idx + 1) * graph_dataset.attr_train_positive_samples,:]
        pos_score = head(h_shared)
        return pos_score

    def get_model_structure(self):
        struct_string = ""

        if self.shared_attention:
            struct_string += f"SharedAttention"
        else:
            struct_string += f"SharedLinear_x{self.shared_h_layers}_{self.h_feats}"
        struct_string += "|"
        if self.head_attention:
            struct_string += f"{len(self.heads)}xHeadAttention_{self.in_feats}"
        else:
            struct_string += (
                f"{len(self.heads)}xHeadLinear_x{self.head_h_layers}_{self.h_feats}"
            )

        # for module in self.modules():
        #     mod_name = module._get_name()
        #     if mod_name == 'Linear':
        #         mod_string = f'{mod_name}_{module.in_features}_{module.out_features}' + '|'
        #     else:
        #         mod_string = ''
        #     struct_string += mod_string
        return struct_string


class AttentionLayer(nn.Module):
    def __init__(
        self,
        q,
        k,
        input_feat_size,
        dropout=0.2,
        out_feat_size=None,
        head_out=None,
        hidden_feat_size=32,
        device="cpu",
    ):
        super(AttentionLayer, self).__init__()
        self.device = device
        self.hidden_size = hidden_feat_size
        self.q = nn.Parameter(q, requires_grad=True)
        # self.Wq = nn.Linear(self.q.shape[1], self.q.shape[0], bias=True, )
        self.k = nn.Parameter(torch.eye(self.q.shape[0]), requires_grad=False)
        self.m = nn.Parameter(
            torch.ones(size=(1, self.hidden_size), dtype=torch.float32, device=device),
            requires_grad=True,
        )
        # self.m = nn.Parameter(torch.ones(size=(1, self.q.shape[0]),dtype=torch.float32, device=device),
        #                       requires_grad=False)

        self.Wq = nn.Linear(
            self.q.shape[1],
            self.hidden_size,
            bias=True,
        )
        self.Wk = nn.Linear(self.q.shape[0], self.hidden_size)
        self.Wm = nn.Linear(self.hidden_size, self.q.shape[0])
        self.reluK = nn.ReLU()
        self.reluQ = nn.ReLU()
        self.reluM = nn.ReLU()

        self.train_mask = None
        self.valid_mask = None
        self.test_mask = None
        self.bound_start = dict()
        # self.do_res = nn.Dropout(dropout)
        # self.do_v = nn.Dropout(dropout)

        if head_out is not None:
            self.out_linear = nn.Linear(out_feat_size, head_out, bias=True)

    def get_q(self):
        return self.q

    def forward(self, h, triplets, step):
        v = h[[triplets]]
        # v = F.normalize(v, dim=2)
        q_res = self.reluQ(self.Wq(self.q))
        k_res = self.reluK(self.Wk(self.k))
        kq = torch.matmul(torch.transpose(k_res, 0, 1), q_res)
        sm = F.softmax(kq, dim=0)
        mkq = self.reluM(self.Wm(torch.matmul(self.m, sm)))
        res = torch.matmul(mkq, v)
        # res = torch.bmm(mkq, v)
        # return F.normalize(res, dim=1).squeeze(1)
        return res.squeeze(1)


class AttentionHead(nn.Module):
    def __init__(
        self,
        q,
        k,
        input_feat_size,
        dropout=0.2,
        out_feat_size=None,
        head_out=None,
        hidden_feat_size=32,
        no_relu=True,
        no_sm=True,
        device="cpu",
    ):
        super(AttentionHead, self).__init__()
        self.device = device
        self.hidden_size = hidden_feat_size
        self.no_relu = no_relu
        self.no_sm = no_sm
        self.q = nn.Parameter(q, requires_grad=True)
        self.Wq = nn.Linear(
            self.q.shape[1],
            self.q.shape[0],
            bias=True,
        )
        self.relu = nn.ReLU()
        self.k = nn.Parameter(k, requires_grad=False)
        # self.m = nn.Parameter(torch.ones((input_feat_size, 1, self.k.shape[1], ), dtype=torch.float32), requires_grad=False)
        self.m = nn.Parameter(
            torch.ones((1, self.hidden_size), dtype=torch.float32), requires_grad=False
        )
        # self.m[:, torch.sum(self.k, dim=1)==1]=0
        # self.m.requires_grad=True

        self.Wq = nn.Linear(
            self.q.shape[1],
            self.hidden_size,
            bias=True,
        )
        self.Wk = nn.Linear(self.q.shape[0], self.hidden_size)
        self.Wm = nn.Linear(self.hidden_size, self.q.shape[0], bias=False)
        self.reluK = nn.ReLU()
        self.reluQ = nn.ReLU()
        self.reluM = nn.ReLU()

        self.bound_start = dict()
        self.do_res = nn.Dropout(dropout)
        self.do_v = nn.Dropout(dropout)
        self.alpha = 0.1
        if head_out is not None:
            self.out_linear = nn.Linear(out_feat_size, head_out, bias=True)
        else:
            raise Exception
        # self.v = torch.tensor(v, requires_grad=True, device=device)

    def forward(self, v, v_shared, step, bb):
        v = F.normalize(v, dim=1)
        # q_res = self.Wq(self.q)

        if self.no_relu:
            q_res = self.Wq(self.q)
            k_res = self.Wk(self.k)
        else:
            q_res = self.reluQ(self.Wq(self.q))
            k_res = self.reluK(self.Wk(self.k))

        kq = torch.matmul(torch.transpose(k_res, 0, 1), q_res)
        if self.no_sm:
            sm = kq
        else:
            sm = F.softmax(kq, dim=0)
        mkq = self.reluM(torch.matmul(self.m, self.Wm(sm)))

        res = torch.matmul(mkq, self.do_v(v)).squeeze(1)
        res = res + v_shared * self.alpha

        # res = self.do_res(self.relu(res))
        res = self.do_res(res)

        res = self.out_linear(res)
        return res


class SharedLayer(nn.Module):
    def __init__(
        self,
        shared_input_feats,
        h_feats,
        out_feats,
        num_hidden_layers=2,
        dropout=0,
        input_tuple_length=None,
        device="cpu",
    ):
        super().__init__()
        self.name = "sharedlayer"
        self.device = device
        self.h_feats = h_feats
        # The number of out_feats must be the same as the number of feats in a node
        self.out_feats = out_feats
        # The input contains input_tuple_length * shared_input_feats features
        if input_tuple_length is not None:
            self.shared_input_feats = input_tuple_length * shared_input_feats
        else:
            self.shared_input_feats = shared_input_feats
        layers = []

        layers.append(nn.Linear(self.shared_input_feats, self.h_feats))
        # layers.append(nn.BatchNorm1d(self.h_feats))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        # Reduce the total number of layers by 2 to account for the number of layers
        for l in range(1, num_hidden_layers - 1):
            layers.append(nn.Linear(self.h_feats, self.h_feats))
            # layers.append(nn.BatchNorm1d(self.h_feats))
            #
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(self.h_feats, self.out_feats))
        layers.append(nn.ReLU())

        self.model = nn.Sequential(*layers)
        # for model_layer in range(len(self.model)):
        #     self.model[model_layer] = self.model[model_layer].to(self.device)

        if self.device == "cuda":
            self.model.cuda(self.device)

        self.tuple_embedding_matrix = None

    def compute_triplet_matrix(self, h, triplets):
        # tuples_next = tuples[:, 1:]
        # averaged_h = h[[triplets]].mean(dim=1)
        # targets = h[[triplets[:, 2]]]
        # self.triplet_embedding_matrix = averaged_h.to(self.device)
        # return averaged_h.to(self.device)

        self.triplet_embedding_matrix = h[[triplets]].reshape(
            len(triplets), self.shared_input_feats
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


class AttributeLayer(nn.Module):
    def __init__(
        self,
        in_feats,
        h_feats,
        out_feats,
        num_hidden_layers=1,
        dropout=0,
        batchnorm=False,
        device="cpu",
    ):
        super().__init__()
        self.name = "attributelayer"
        self.device = device
        self.h_feats = h_feats
        # The number of out_feats must be the same as the number unique values in the attribute domain
        self.out_feats = out_feats
        # The input contains input_tuple_length * in_feats features
        self.attr_in_feats = in_feats
        # self.in_feats = input_tuple_length*in_feats
        layers = []

        layers.append(nn.Linear(self.attr_in_feats, self.h_feats))
        if batchnorm:
            # layers.append(nn.LayerNorm(self.h_feats))
            layers.append(nn.BatchNorm1d(self.h_feats))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        for l in range(1, num_hidden_layers - 1):
            layers.append(nn.Linear(self.h_feats, self.h_feats))
            if batchnorm:
                # layers.append(nn.LayerNorm(self.h_feats))
                layers.append(nn.BatchNorm1d(self.h_feats))
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
        # tuples_next = tuples[:, 1:]
        averaged_h = h[[triplets]].mean(dim=1)
        # targets = h[[triplets[:, 2]]]
        # self.triplet_embedding_matrix = averaged_h.to(self.device)
        # return averaged_h.to(self.device)

        self.triplet_embedding_matrix = h[[triplets]].reshape(
            len(triplets), self.attr_in_feats
        )

    def forward(self, h):
        # self.compute_triplet_matrix(h, triplets)
        # h = self.model[0](self.triplet_embedding_matrix).to(self.device)
        for layer in self.model[0:]:
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
