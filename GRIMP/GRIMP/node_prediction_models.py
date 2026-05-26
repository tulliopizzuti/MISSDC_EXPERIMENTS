import torch
import torch.nn as nn
import torch.nn.functional as F


class NearestNodePredictor(nn.Module):
    def __init__(
        self, in_feats, h_feats, input_tuple_length, num_layers=2, device="cpu"
    ):
        super().__init__()
        self.device = device
        self.name = "nn"
        self.h_feats = h_feats
        # The number of out_feats must be the same as the number of feats in a node
        self.out_feats = in_feats
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

    def forward(self, h, tuples):
        self.tuple_embedding_matrix = self.compute_tuple_matrix(h, tuples).to(
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
