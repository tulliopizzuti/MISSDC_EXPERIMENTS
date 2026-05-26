import numpy as np
import torch
import torch.nn.functional as F
import pickle
import pandas as pd
import warnings
from sklearn.decomposition import PCA


def convert_embs_to_text(embeddings, graph, emb_path):
    fout = open(emb_path, "w")
    header = f"{embeddings.shape[0]} {embeddings.shape[1]}\n"
    fout.write(header)

    nodes = (
        [f"idx__{rid}" for rid in range(graph.num_rows)]
        + [f"cid__{cid}" for cid in graph.col2idx]
        + [f"tt__{val}" for val in graph.val2idx]
    )

    for idx, node in enumerate(nodes):
        s = f"{node} " + " ".join([str(_) for _ in embeddings[idx, :]]) + "\n"
        fout.write(s)

    fout.close()


def save_model_to_file(model_file, graph_dataset, gnn_model, prediction_model=None):
    with torch.no_grad():
        model = dict()
        model["graph_dataset"] = graph_dataset
        model["gnn_model"] = gnn_model
        model["prediction_model"] = prediction_model
        pickle.dump(model, open(model_file, "wb"))


def load_model_from_file(model_file):
    model = pickle.load(open(model_file, "rb"))
    graph_dataset = model["graph_dataset"]
    gnn_model = model["gnn_model"]
    prediction_model = model["prediction_model"]

    return graph_dataset, gnn_model, prediction_model


class Features:
    def __init__(self, path, categorical_columns, numerical_columns):
        self.path = path

        with open(self.path, "r") as fp:
            self.rows, self.dim = [int(_) for _ in fp.readline().strip().split(" ")]
            self.feats = torch.empty((self.rows, self.dim))
            self.ext_features = dict()
            for idx, row in enumerate(fp):
                token, vector = row.strip().split(" ", maxsplit=1)
                if token.startswith("tt__"):
                    token = token.replace("tt__", "", 1)
                if token.startswith("tn__"):
                    token = token.replace("tn__", "", 1)

                if token in self.ext_features:
                    raise ValueError(f"Found duplicate token {token}")
                if token.startswith("cid__") or token.startswith("idx__"):
                    new_token = token
                else:
                    prefix, val = token.split("_", maxsplit=1)
                    if prefix in numerical_columns:
                        val = str(round(float(val), 8))
                        new_token = f"{prefix}_{val}"
                    else:
                        new_token = token

                tt = torch.Tensor([float(_) for _ in vector.split(" ")]).reshape(
                    1, self.dim
                )
                tt = F.normalize(tt)
                self.ext_features[new_token] = tt
                # self.feats[idx, :] = torch.Tensor([float(_) for _ in vector.split(' ')]).reshape(1,self.dim)

    def get_keys(self):
        return list(self.ext_features.keys())

    def get_shape(self):
        return (self.rows, self.dim)


def read_external_features(df_path, path_list, max_comp=32):
    df = pd.read_csv(df_path)
    categorical_columns = df.select_dtypes(exclude="number").columns.to_list()
    numerical_columns = [_ for _ in df.columns if _ not in categorical_columns]

    cat_idxs = [
        f"c_{idx}" for idx, col in enumerate(df.columns) if col in categorical_columns
    ]
    num_idxs = [
        f"c_{idx}" for idx, col in enumerate(df.columns) if col in numerical_columns
    ]

    full_features = dict()
    for ext_features_path in path_list:
        full_features[ext_features_path] = Features(
            ext_features_path, cat_idxs, num_idxs
        )

    common_tokens = []
    total_num_dims = 0
    for path in path_list:
        common_tokens += full_features[path].get_keys()
        total_num_dims += full_features[path].get_shape()[1]
    common_tokens = list(set(common_tokens))

    ext_features = {token: None for token in common_tokens}
    feat_tensor = torch.zeros((len(common_tokens), total_num_dims))
    for idx, token in enumerate(common_tokens):
        start_features = 0
        full_vector = torch.zeros(1, total_num_dims)
        for _, path in enumerate(full_features, start=1):
            dims = full_features[path].get_shape()[1]
            end_features = start_features + dims
            if token in full_features[path].ext_features:
                vector = full_features[path].ext_features[token]
            else:
                vector = torch.zeros(dims)
            full_vector[:, start_features:end_features] = vector
            start_features = end_features
        ext_features[token] = full_vector
        feat_tensor[idx, :] = full_vector

    if max_comp < total_num_dims:
        pca = PCA(n_components=max_comp)
        feat_final = torch.tensor(pca.fit_transform(feat_tensor))
    else:
        feat_final = feat_tensor

    return list(ext_features.keys()), feat_final


def read_features_tensor(tensor_path):
    return torch.load(tensor_path)


def read_functional_dependencies(fd_filename, df_path):
    df = pd.read_csv(df_path, dtype="object")
    # As implemented in MissForestFD
    with open(fd_filename) as f:
        lines = f.readlines()
    fd_dict = {}
    df_columns = list(df.columns)

    for line in lines:
        # FD is of the form a,b,c->d
        lhs, rhs = line.strip().split("->")
        lhs = lhs.split(",")
        # Find the indices of FDs from the list of attributes
        rhs_attr_index = df_columns.index(rhs)
        lhs_attr_indices = tuple([df_columns.index(lhs_attr) for lhs_attr in lhs])
        if rhs_attr_index not in fd_dict:
            fd_dict[rhs_attr_index] = () + (lhs_attr_indices,)
        else:
            # Insert as tuple of  tuples
            # E.g. fd_dict[1] = ( (2,3), (3,4,5) )
            fd_dict[rhs_attr_index] = fd_dict[rhs_attr_index] + (lhs_attr_indices,)
    return fd_dict
