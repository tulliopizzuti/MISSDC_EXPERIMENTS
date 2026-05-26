# GiG
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPPredictor(nn.Module):
    def __init__(self, h_feats, num_layers=2, predict_edges=False):
        super().__init__()
        self.name = "mlp"
        self.h_feats = h_feats
        self.out_feats = 1
        self.predict_edges = predict_edges
        if self.predict_edges:
            self.in_feats = h_feats * 2
        else:
            self.in_feats = h_feats
            # self.in_feats = h_feats * 3

        layers = []
        # The input is a triplet embedding of dimension 3 * h_feats

        layers.append(nn.Linear(self.in_feats, self.h_feats))
        layers.append(nn.ReLU())
        for l in range(1, num_layers - 1):
            layers.append(nn.Linear(self.h_feats, self.h_feats))
            # layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(self.h_feats, self.out_feats))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)

        # self.W1 = nn.Linear(h_feats * 3, h_feats)
        # self.W2 = nn.Linear(h_feats, 1)
        self.triplet_embedding_matrix = None

    def has_neg(self):
        return True

    def compute_triplet_matrix(self, h, triplets):
        num_triplets = len(triplets)
        # print("no grad")

        if self.predict_edges:
            self.triplet_embedding_matrix = h[[triplets]].reshape(
                len(triplets), self.h_feats * 2
            )
        else:
            # self.triplet_embedding_matrix = h[[triplets]].reshape(len(triplets), self.h_feats * 3)
            self.triplet_embedding_matrix = h[[triplets]].mean(dim=1)
        # for index, triplet in enumerate(triplets):
        #     row_node_id, col_node_id, cell_node_id = triplet
        #
        #     start_col_index, end_col_index = 0, self.h_feats
        #     self.triplet_embedding_matrix[index, start_col_index:end_col_index] = h[row_node_id]
        #
        #     start_col_index, end_col_index = end_col_index, end_col_index+self.h_feats
        #     self.triplet_embedding_matrix[index, start_col_index:end_col_index] = h[col_node_id]
        #
        #     start_col_index, end_col_index = end_col_index, end_col_index+self.h_feats
        #     self.triplet_embedding_matrix[index, start_col_index:end_col_index] = h[cell_node_id]

    def forward(self, h, triplets):
        self.compute_triplet_matrix(h, triplets)
        h = self.model[0](self.triplet_embedding_matrix)
        for layer in self.model[1:]:
            h = layer(h)
        res = h

        return res.squeeze(1)


class TrueFalsePredictor(nn.Module):
    def __init__(self, h_feats, predict_edges=False):
        super().__init__()
        self.name = "truefalse"
        self.h_feats = h_feats
        # The input is a triplet embedding of dimension 3 * h_feats
        # self.W1 = nn.Linear(h_feats * 2, h_feats)
        if predict_edges:
            self.predict_edges = True
            self.W1 = nn.Linear(h_feats * 2, h_feats)
        else:
            self.predict_edges = False
            self.W1 = nn.Linear(h_feats * 3, h_feats)
        self.W2 = nn.Linear(h_feats, 2)
        self.triplet_embedding_matrix = None

    def compute_triplet_matrix(self, h, triplets):
        num_triplets = len(triplets)
        # print("no grad")

        if self.predict_edges:
            self.triplet_embedding_matrix = h[[triplets]].reshape(
                len(triplets), self.h_feats * 2
            )
        else:
            self.triplet_embedding_matrix = h[[triplets]].reshape(
                len(triplets), self.h_feats * 3
            )
        # for index, triplet in enumerate(triplets):
        #     row_node_id, col_node_id, cell_node_id = triplet
        #
        #     start_col_index, end_col_index = 0, self.h_feats
        #     self.triplet_embedding_matrix[index, start_col_index:end_col_index] = h[row_node_id]
        #
        #     start_col_index, end_col_index = end_col_index, end_col_index+self.h_feats
        #     self.triplet_embedding_matrix[index, start_col_index:end_col_index] = h[col_node_id]
        #
        #     start_col_index, end_col_index = end_col_index, end_col_index+self.h_feats
        #     self.triplet_embedding_matrix[index, start_col_index:end_col_index] = h[cell_node_id]

    def forward(self, h, triplets):
        self.compute_triplet_matrix(h, triplets)
        res = self.W2(F.relu(self.W1(self.triplet_embedding_matrix)))
        return res
        return torch.softmax(
            F.relu(self.W2(F.relu(self.W1(self.triplet_embedding_matrix)))), dim=1
        ).squeeze(1)


class TripletPredictor(nn.Module):
    def __init__(self, in_feats, h_feats, out_feat):
        super().__init__()
        self.triplet_embedding_matrix = None
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.W1 = nn.Linear(self.h_feats, self.h_feats // 2)
        self.W2 = nn.Linear(self.h_feats // 2, self.h_feats // 2)
        self.W3 = nn.Linear(self.h_feats // 2, out_feat)

        self.output_layer = nn.Sigmoid()

    def has_neg(self):
        return True

    def compute_triplet_matrix(self, h, triplets):
        items, trip, candidates = triplets.shape
        reshaped = h[triplets.reshape(items, trip * candidates)]
        self.triplet_embedding_matrix = reshaped.reshape(items, self.h_feats).mean(
            dim=1
        )
        # self.triplet_embedding_matrix = h[triplets].reshape(len(triplets), self.h_feats)

    def forward(self, h, triplets):
        # res = self.W2(F.relu(self.W1(h)))
        # h_triplets =
        self.compute_triplet_matrix(h, triplets)
        res = self.W3(F.relu(self.W2(F.relu(self.W1(self.triplet_embedding_matrix)))))
        return F.softmax(res)


class NearestNeighborPredictor(nn.Module):
    def __init__(
        self, in_feats, h_feats, input_tuple_length, num_layers=2, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.name = "nn"
        self.h_feats = h_feats
        # The number of out_feats must be the same as the number of feats in a node
        self.out_feats = 1
        # The input contains input_tuple_length * in_feats features
        self.in_feats = input_tuple_length * in_feats
        layers = []

        layers.append(nn.Linear(self.in_feats, self.h_feats))
        layers.append(nn.ReLU())
        for l in range(1, num_layers - 1):
            layers.append(nn.Linear(self.h_feats, self.h_feats))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(self.h_feats, self.out_feats))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        for model_layer in range(len(self.model)):
            self.model[model_layer] = self.model[model_layer].to(self.device)

        if self.device == "cuda":
            self.model.cuda(self.device)

        self.tuple_embedding_matrix = None

    def compute_tuple_matrix(self, h, tuples):
        return h[[tuples]].reshape(len(tuples), self.in_feats).to(self.device)

    def mask_tuple_matrix(self, step):
        mask_range = [step * self.out_feats, (step + 1) * self.out_feats]
        self.tuple_embedding_matrix[:, range(*mask_range)] = torch.zeros(
            size=(self.tuple_embedding_matrix.shape[0], self.out_feats)
        ).to(self.device)

    def forward(self, h, triplets):
        self.tuple_embedding_matrix = self.compute_tuple_matrix(h, triplets).to(
            self.device
        )
        # self.mask_tuple_matrix(step)
        # self.tuple_embedding_matrix = self.tuple_embedding_matrix.to(self.device)

        h = self.model[0](self.tuple_embedding_matrix).to(self.device)
        for layer in self.model[1:]:
            h = layer(h)
        res = h

        # res = torch.sigmoid(self.W2(F.relu(self.W1(self.tuple_embedding_matrix))))
        return res

    def evaluate(self, h, triplets):
        tuple_matrix = self.compute_tuple_matrix(h, triplets)
        h = self.model[0](tuple_matrix)
        for layer in self.model[1:]:
            h = layer(h)
        res = h
        # res = torch.sigmoid(self.W2(F.relu(self.W1(h))))
        return res


class TupleAveragePredictor(nn.Module):
    def __init__(
        self, in_feats, h_feats, input_tuple_length, num_layers=2, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.name = "nn"
        self.h_feats = h_feats
        # The number of out_feats must be the same as the number of feats in a node
        self.out_feats = 1
        self.in_feats = 2 * in_feats
        # The input contains input_tuple_length * in_feats features
        # self.W1 = nn.Linear(self.in_feats, self.h_feats)
        # self.W2 = nn.Linear(self.h_feats, self.out_feats)
        layers = []

        layers.append(nn.Linear(self.in_feats, self.h_feats))
        layers.append(nn.ReLU())
        for l in range(1, num_layers - 1):
            layers.append(nn.Linear(self.h_feats, self.h_feats))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=0.1))
        layers.append(nn.Linear(self.h_feats, self.out_feats))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        for model_layer in range(len(self.model)):
            self.model[model_layer] = self.model[model_layer].to(self.device)

        if self.device == "cuda":
            self.model.cuda(self.device)

        self.tuple_embedding_matrix = None

    def compute_tuple_matrix(self, h, tuples, triplets):
        tuples_next = tuples[:, 1:]
        averaged_h = h[[tuples_next]].mean(dim=1)
        targets = h[[triplets[:, 2]]]
        return torch.cat([averaged_h, targets], dim=1).to(self.device)
        # return h[[tuples]].reshape(len(tuples), self.in_feats).to(self.device)

    def mask_tuple_matrix(self, step):
        mask_range = [step * self.out_feats, (step + 1) * self.out_feats]
        self.tuple_embedding_matrix[:, range(*mask_range)] = torch.zeros(
            size=(self.tuple_embedding_matrix.shape[0], self.out_feats)
        ).to(self.device)

    def forward(self, gnn_h, tuples, triplets):
        self.tuple_embedding_matrix = self.compute_tuple_matrix(
            gnn_h, tuples, triplets
        ).to(self.device)

        h = self.model[0](self.tuple_embedding_matrix).to(self.device)
        for layer in self.model[1:]:
            h = layer(h)
        res = h
        return res
