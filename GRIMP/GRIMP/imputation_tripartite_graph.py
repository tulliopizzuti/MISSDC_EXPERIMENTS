# GiG
import os

import pandas as pd
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset
from abc import ABC, abstractmethod
import random
import copy


class ImputationTripartiteGraph(DGLDataset):
    def __init__(
        self,
        original_data_file_name,
        missing_value_file_name,
        random_init=False,
        node_mapping=None,
        ext_features=None,
        positive_negative_train_scale_factor=20,
    ):
        self.graph_name = ""
        self.original_data_file_name = original_data_file_name
        self.missing_value_file_name = missing_value_file_name
        self.num_rows = self.num_columns = 0
        # num(rows) + num(columns)
        self.num_row_col_nodes = 0
        # num(rows) + num(columns) + num(distinct cell values in original_data_file_name)
        self.num_total_nodes = 0
        self.input_tuple_length = 3
        self.missing_values_by_col = dict()

        self.positive_negative_train_scale_factor = positive_negative_train_scale_factor

        # Additional (optional) data structure that saves labels so they're not computed online
        self.labels = None

        # Additional parameter to specify random initialization of all features
        self.random_init = random_init

        # Additional data structures needed to handle external features.
        self.node_mapping = node_mapping
        self.ext_features = ext_features

        super().__init__(name="EmbDI Graph")

    def prepare_node_dicts(self):
        self.node2idx = {f"idx__{idx}": idx for idx in range(self.num_rows)}
        self.node2idx.update({f"cid__{cid}": v for cid, v in self.col2idx.items()})
        self.node2idx.update({f"tt__{token}": v for token, v in self.val2idx.items()})
        self.idx2node = {v: k for k, v in self.node2idx.items()}

    def compute_stats(self):
        # For efficiency sake, we also piggyback on this data frame traversal
        # to compute some other statistics post cell id creation
        num_edges = 0
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                col_name = self.df_orig.columns[col]
                if pd.isnull(self.df_missing.iloc[row, col]) == False:
                    num_edges = num_edges + 2
                    self.num_non_missing_values += 1
                else:
                    # If there are K values, then 1 of them will be in positive triplet and K-1 in negative
                    # So increment test_positive_samples_size by 1 and test_negative_samples_size by K-1
                    self.num_missing_values += 1

        return num_edges

    def measure_value_freq(self):
        self.counts = {k: None for k in self.df_orig.columns}

        for k in self.counts:
            self.counts[k] = self.df_orig.value_counts(k)
            self.counts[k][0] = self.counts[k][1]

        self.budget = copy.deepcopy(self.counts)
        # Multiplying each number in the budget by the number of negative training samples I want
        for k in self.budget:
            # diff = self.budget[k][0] - sum(self.budget[k][1:])
            # if diff > 0:
            #     self.budget[k][0] *= self.positive_negative_train_scale_factor
            #     offset = self.budget[k][0] // sum(self.budget[k][1:])
            #     self.budget[k][1:] *= (self.positive_negative_train_scale_factor + offset + 2)
            # else:
            #     self.budget[k] *= self.positive_negative_train_scale_factor
            self.budget[k][0] = self.budget[k][1]
            self.budget[k] *= self.positive_negative_train_scale_factor
            self.budget[k] = self.budget[k].reset_index(name="counts")
            # self.budget[k]['idx'] = np.arange(len(self.budget[k]))

    def load_and_compute_stats(self):
        # Read everything as a string
        self.df_orig = pd.read_csv(self.original_data_file_name, dtype=str)
        self.df_missing = pd.read_csv(self.missing_value_file_name, dtype=str)

        if self.df_orig.shape != self.df_missing.shape:
            raise Exception(
                "Error: Input files do not have same number of rows and columns"
            )

        # Convert every attribute to string and treat it as categorical
        # Redundant given the dtype in read_csv but still
        for col in self.df_orig.columns:
            self.df_orig[col] = self.df_orig[col].astype(str)
        for col in self.df_missing.columns:
            self.df_missing[col] = self.df_missing[col].astype(str)

        # Using dtype in read_csv makes missing value as a string 'nan'. So replace it appropriately to nan
        self.df_missing.replace("nan", np.nan, inplace=True)

        # Node id semantics: let n, m, l be number of rows, columns and distinct cell values
        # then node ids 0 to n-1 correspond to rows,
        # node ids n to n+m-1 correspond to columns
        # and n+m to n+m+l-1 correspond to cells
        self.num_rows, self.num_columns = self.df_orig.shape
        self.num_row_col_nodes = self.num_rows + self.num_columns

        self.distinct_value_set = {
            self.df_orig.iloc[row, col]
            for row in range(self.num_rows)
            for col in range(self.num_columns)
        }
        self.val2idx = dict(
            (v, i + self.num_row_col_nodes)
            for (i, v) in enumerate(sorted(self.distinct_value_set))
        )
        self.idx2val = dict(
            (i + self.num_row_col_nodes, v)
            for (i, v) in enumerate(sorted(self.distinct_value_set))
        )

        # For each distinct value associate with it the column in which it is found
        # if a value V is present in two columns A and B, then it will be associated with the latest column
        self.cell_to_dict = {
            self.df_orig.iloc[row, col]: col
            for row in range(self.num_rows)
            for col in range(self.num_columns)
        }

        self.col2idx = dict(
            (col, self.num_rows + i) for (i, col) in enumerate(self.df_orig.columns)
        )
        self.idx2col = dict(
            (self.num_rows + i, col) for (i, col) in enumerate(self.df_orig.columns)
        )

        self.num_total_nodes = self.num_row_col_nodes + len(self.distinct_value_set)

        # Aggregate for each column and then aggregate for entire data frame
        self.num_missing_values = (
            0  # will be computed later. self.df_missing.isna().sum().sum()
        )
        self.num_non_missing_values = (
            0  # will be created later len(self.distinct_value_set)
        )

        # Prepare dict mappings between node name and vector idx
        self.prepare_node_dicts()

        self.measure_value_freq()

        # create a dictionary to avoid repeated cell id lookups for each attribute.
        # The key is col_id and it will return cell_ids of all the values in its domain
        self.attribute_domain_cell_id_dict = {}
        for col in self.df_orig.columns:
            distinct_values = self.df_orig[col].unique()
            cell_id_array = [self.val2idx[val] for val in distinct_values]
            col_id = self.col2idx[col]
            self.attribute_domain_cell_id_dict[col_id] = cell_id_array

    def create_graph(self):
        # Create empty graph
        self.graph = dgl.graph(data=[])

        # Create num_total_nodes isolated nodes
        self.graph.add_nodes(self.num_total_nodes)

        # Create edges
        # Note that we use df_missing for creating the edges
        # So cells corresponding to missing values will be isolated

        num_edges = self.compute_stats()

        # DGL expects the edges to be specified as two tensors containing start and end nodes.
        start_nodes = torch.zeros(num_edges, dtype=torch.int64)
        end_nodes = torch.zeros(num_edges, dtype=torch.int64)

        index = 0

        for row in range(self.num_rows):
            row_node_id = row
            for col in range(self.num_columns):
                col_node_id = col + self.num_rows
                if pd.isnull(self.df_missing.iloc[row, col]) == False:
                    cell_node_id = self.val2idx[self.df_missing.iloc[row, col]]
                    start_nodes[index] = row_node_id
                    end_nodes[index] = cell_node_id
                    start_nodes[index + 1] = col_node_id
                    end_nodes[index + 1] = cell_node_id
                    index = index + 2

        self.graph.add_edges(start_nodes, end_nodes)

        # Make the graph undirected
        self.graph = dgl.add_reverse_edges(self.graph)

        print(
            "Graph has %d nodes and %d edges"
            % (self.graph.number_of_nodes(), self.graph.number_of_edges())
        )

    def compute_node_features(self):
        # ndata stores various features for each node.
        # as a start, lets create a single feature variable dubbed "features"
        # TODO: this is only a start. a node feature of dimension self.num_columns is typically too low
        # typically we need in the order of 100s
        # We need to add additional features  that injects FD related information
        self.num_features = self.num_columns
        self.graph.ndata["features"] = torch.zeros(
            (self.num_total_nodes, self.num_columns)
        )

        # Features for row nodes = it is connected to each column
        self.graph.nodes[range(0, self.num_rows)].data["features"] = torch.ones(
            self.num_rows, self.num_columns
        )

        # Features for column nodes = it is connected to only itself
        self.graph.nodes[range(self.num_rows, self.num_rows + self.num_columns)].data[
            "features"
        ] = torch.eye(self.num_columns)

        ###############TODO: more expensive and probably better set of features
        #        #For cell nodes, create an edge between that node and the column in which it was present
        #        for row in range(self.num_rows):
        #            for col in range(self.num_columns):
        #                col_node_id = col + self.num_rows
        #                if pd.isnull(self.df_missing.iloc[row, col]) == False:
        #                    cell_node_id = self.val2idx[self.df_missing.iloc[row, col]]
        ###############This is the expensive part .... figure out a way to to modify ndata inplace
        ######################perhaps by first using a tensor matrix and then set to  ndata['features'] at the end.
        #                    temp = torch.zeros(1, self.num_columns)
        #                    temp[0][col] = 1.0
        #                    self.graph.nodes[cell_node_id].data['features'] = temp
        # Features for cell nodes = it is connected to each column
        self.graph.nodes[range(self.num_row_col_nodes, self.num_total_nodes)].data[
            "features"
        ] = torch.ones(self.num_total_nodes - self.num_row_col_nodes, self.num_columns)

        if self.random_init:
            self.graph.nodes[range(0, self.num_total_nodes)].data[
                "features"
            ] *= torch.FloatTensor(self.num_total_nodes, self.num_columns).uniform_()

    @abstractmethod
    def create_train_positive_negative_graphs(self):
        pass

    @abstractmethod
    def create_test_positive_negative_graphs(self):
        pass

    def load_external_features(self):
        self.graph.ndata["features"] = torch.zeros(size=self.ext_features.shape)
        self.num_features = self.ext_features.shape[1]
        perm = [self.node2idx[node] for node in self.node_mapping]
        reordered = self.ext_features[perm, :]
        self.graph.nodes[:].data["features"] = reordered

    # @abstractmethod
    def generate_labels(self, samples=None):
        pass

    def process(self):
        print("Loading and computing basic stats")
        self.load_and_compute_stats()
        print("Creating graph structure")
        self.create_graph()
        if self.ext_features is not None:
            print("Loading external features")
            self.load_external_features()
        else:
            print("Computing graph features")
            self.compute_node_features()
        print("Creating positive and negative pos_triplets for training")
        self.create_train_positive_negative_graphs()
        print("Creating positive and negative pos_triplets for testing")
        self.create_test_positive_negative_graphs()


class ImputationTripartiteGraphTripletPrediction(ImputationTripartiteGraph):
    def __init__(
        self,
        original_data_file_name,
        missing_value_file_name,
        random_init=False,
        node_mapping=None,
        ext_features=None,
        compare_edgepred=True,
        pos_neg_scale_factor=20,
    ):
        if compare_edgepred:
            self.compare_limit = 1
        else:
            self.compare_limit = None

        self.has_missing = []

        super().__init__(
            original_data_file_name,
            missing_value_file_name,
            random_init,
            node_mapping,
            ext_features,
            pos_neg_scale_factor,
        )
        self.graph_name = "triplet"

    # self.compare_edgepred = compare_edgepred

    # This function loads the two data frames, does some bookkeeping and computes basic stats
    def load_and_compute_stats(self):
        super().load_and_compute_stats()
        self.test_positive_samples_size = 0
        self.test_negative_samples_size = 0
        # The following two variables store (row_id, col_id, cell_id) pos_triplets
        self.test_positive_samples = None
        self.test_negative_samples = None
        self.test_pos_neg_matches = {}

    def compute_stats(self):
        # For efficiency sake, we also piggyback on this data frame traversal
        # to compute some other statistics post cell id creation
        num_edges = 0
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                col_name = self.df_orig.columns[col]
                if pd.isnull(self.df_missing.iloc[row, col]) == False:
                    num_edges = num_edges + 2
                    self.num_non_missing_values += 1
                else:
                    # If there are K values, then 1 of them will be in positive triplet and K-1 in negative
                    # So increment test_positive_samples_size by 1 and test_negative_samples_size by K-1
                    num_unique_neg_vals = self.df_orig[col_name].nunique() - 1

                    self.num_missing_values += 1
                    self.test_positive_samples_size += 1
                    self.test_negative_samples_size += num_unique_neg_vals
        for col in self.df_orig.columns:
            n_missing = sum(self.df_missing[col].isna())
            if n_missing > 0:
                self.has_missing.append(self.col2idx[col])
                self.missing_values_by_col[self.col2idx[col]] = n_missing
        return num_edges

    def get_replacement(self, step, cell_node_id, col):
        # attribute_domain.remove(cell_node_id)
        self.budget_list = []

        tgt_col = self.df_orig.columns[col]
        tgt_val = self.idx2val[cell_node_id]

        found = looped = False
        start_idx = (
            self.budget[tgt_col].loc[self.budget[tgt_col][tgt_col] == tgt_val].index[0]
        )

        idx = (start_idx + 1 + step) % len(self.budget[tgt_col])
        if idx == start_idx:
            idx = (idx + 1) % len(self.budget[tgt_col])

        # if len(self.budget[tgt_col]) > 2:
        #     idx = start_idx + 1 + step
        # else:
        #     idx = int(not start_idx)
        # if idx > len(self.budget[tgt_col]):
        #     idx = start_idx + 1

        while not found:
            if idx >= len(self.budget[tgt_col]):
                if start_idx == 0:
                    idx = 1
                else:
                    idx = 0
                # idx = (start_idx + 1) % len(self.budget[tgt_col])
                looped = True
            if idx == start_idx and looped:
                raise ValueError(f"Ran out of candidates for value {tgt_val}.")
            budget = self.budget[tgt_col].iloc[idx]["counts"]
            if budget > 0:
                choice = idx
                found = True
            else:
                idx += 1

        self.budget[tgt_col].loc[choice, "counts"] -= 1
        res_val = self.budget[tgt_col].loc[choice, tgt_col]

        return self.val2idx[res_val]

    # train_positive_samples are all (row_id, col_id, cell_id) [excluding missing balues]
    # train_negative_samples are all edges not present in the graph  [excluding the missing values]
    # specifically, we create a negative triplet by retaining (row, col) but choose a random cell
    def get_positive_negative_triplets_for_training(self):
        self.positive_triplet_index, self.negative_triplet_index = 0, 0

        cell_id_start, cell_id_end = (
            self.num_rows + self.num_columns,
            self.num_total_nodes,
        )
        random_cell_ids = torch.randint(
            low=cell_id_start,
            high=cell_id_end,
            size=(self.train_negative_triplets_size,),
        )
        self.pos_edges_to_remove = torch.zeros(self.num_non_missing_values * 2, 2).to(
            int
        )
        for row in range(self.num_rows):
            for col in range(self.num_columns):
                tgt_col = self.df_orig.columns[col]

                if pd.isnull(self.df_missing.iloc[row, col]) == False:
                    col_node_id = self.num_rows + col
                    tgt_val = self.df_missing.iloc[row, col]
                    if self.counts[tgt_col].loc[tgt_val] <= 0:
                        continue

                    cell_node_id = self.val2idx[tgt_val]
                    self.train_positive_samples[self.positive_triplet_index] = (
                        row,
                        col_node_id,
                        cell_node_id,
                    )
                    self.pos_edges_to_remove[
                        2 * self.positive_triplet_index, :
                    ] = torch.IntTensor([row, cell_node_id])
                    self.pos_edges_to_remove[
                        2 * self.positive_triplet_index + 1, :
                    ] = torch.IntTensor([cell_node_id, col_node_id])
                    self.positive_triplet_index += 1

                    # Get all the node_ids for the domain values from that column. The [:] end creates a copy
                    attribute_domain = self.attribute_domain_cell_id_dict[col_node_id][
                        :
                    ]
                    attribute_domain.remove(cell_node_id)

                    for index2 in range(self.positive_negative_train_scale_factor):
                        replacement_cell_id = self.get_replacement(
                            index2, cell_node_id, col
                        )
                        #
                        # replacement_cell_id = random.sample(attribute_domain, k=1)[0]
                        self.train_negative_samples[self.negative_triplet_index] = (
                            row,
                            col_node_id,
                            replacement_cell_id,
                        )
                        # self.test_pos_neg_matches[positive_triplet_index-1].append(self.negative_triplet_index)
                        self.negative_triplet_index += 1
                        self.counts[tgt_col].loc[tgt_val] -= 1

                    # for index2 in range(self.positive_negative_train_scale_factor//2+1, self.positive_negative_train_scale_factor):
                    #     random_cell_id = random_cell_ids[negative_triplet_index]
                    #     self.train_negative_samples[negative_triplet_index] = (row, col_node_id, random_cell_id)
                    #     negative_triplet_index += 1

        self.train_positive_samples = torch.LongTensor(
            self.train_positive_samples[: self.positive_triplet_index]
        )
        self.train_negative_samples = torch.LongTensor(
            self.train_negative_samples[: self.negative_triplet_index]
        )

        # edges_to_remove = self.graph.edge_ids(self.pos_edges_to_remove[:, 0], self.pos_edges_to_remove[:, 1])
        # self.graph.remove_edges(edges_to_remove)
        # edges_to_remove = self.graph.edge_ids(self.pos_edges_to_remove[:, 1], self.pos_edges_to_remove[:, 0])
        # self.graph.remove_edges(edges_to_remove)
        # self.pos_edges_to_remove = None

    # Each missing value becomes a positive triplet
    # every other value frm that attribute domain becomes negative triplet
    def get_positive_negative_triplets_for_testing(self):
        positive_triplet_index = 0
        negative_triplet_index = 0

        start_neg = 0
        end_neg = 0
        self.pos_edges_to_remove = torch.zeros(self.num_missing_values * 2, 2).to(int)

        for row in range(self.num_rows):
            for col in range(self.num_columns):
                if pd.isnull(self.df_missing.iloc[row, col]) == True:
                    col_node_id = self.num_rows + col
                    # NOTE: we change again from df_missing to df_orig to get the correct value
                    cell_node_id = self.val2idx[self.df_orig.iloc[row, col]]
                    self.test_positive_samples[positive_triplet_index] = (
                        row,
                        col_node_id,
                        cell_node_id,
                    )
                    self.pos_edges_to_remove[
                        2 * positive_triplet_index, :
                    ] = torch.IntTensor([row, cell_node_id])
                    self.pos_edges_to_remove[
                        2 * positive_triplet_index + 1, :
                    ] = torch.IntTensor([cell_node_id, col_node_id])

                    self.test_pos_neg_matches[positive_triplet_index] = []
                    positive_triplet_index += 1

                    # Get all the node_ids for the domain values from that column. The [:] end creates a copy
                    attribute_domain = self.attribute_domain_cell_id_dict[col_node_id][
                        :
                    ]
                    attribute_domain.remove(cell_node_id)

                    for cell_node_id in attribute_domain:
                        self.test_negative_samples[negative_triplet_index] = (
                            row,
                            col_node_id,
                            cell_node_id,
                        )
                        # self.test_pos_neg_matches[positive_triplet_index-1].append(negative_triplet_index)
                        negative_triplet_index += 1
                    end_neg = negative_triplet_index - 1
                    self.test_pos_neg_matches[positive_triplet_index - 1] = (
                        start_neg,
                        end_neg,
                    )
                    start_neg = end_neg
        self.test_negative_samples = torch.tensor(self.test_negative_samples)
        self.test_positive_samples = torch.tensor(self.test_positive_samples)
        print("positive test pos_triplets", len(self.test_positive_samples))
        print("negative test pos_triplets", len(self.test_negative_samples))
        # edges_to_remove = self.graph.edge_ids(self.pos_edges_to_remove[:, 0], self.pos_edges_to_remove[:, 1])
        # self.graph.remove_edges(edges_to_remove)
        # self.pos_edges_to_remove = None

    def create_train_positive_negative_graphs(self):
        # number of negative examples per positive example

        self.train_positive_triplets_size = self.num_non_missing_values
        self.train_negative_triplets_size = (
            self.train_positive_triplets_size
            * self.positive_negative_train_scale_factor
        )
        self.train_positive_samples = np.zeros((self.train_positive_triplets_size, 3))
        self.train_negative_samples = np.zeros((self.train_negative_triplets_size, 3))
        print(
            "Size of training positive and negative pos_triplets ",
            self.train_positive_triplets_size,
            self.train_negative_triplets_size,
        )
        self.get_positive_negative_triplets_for_training()

    # test_positive_samples are all the edges for the missing values
    # test_negative_samples consists all the incorrect imputation for each missing value
    def create_test_positive_negative_graphs(self):
        # self.test_positive_samples = torch.zeros(size=(self.test_positive_samples_size,1))
        # self.test_negative_samples = torch.zeros(size=(self.test_positive_samples_size,1))
        self.test_positive_samples = [0 for _ in range(self.test_positive_samples_size)]
        self.test_negative_samples = [0 for _ in range(self.test_negative_samples_size)]
        print(
            "Size of testing positive and negative pos_triplets ",
            self.test_positive_samples_size,
            self.test_negative_samples_size,
        )
        self.get_positive_negative_triplets_for_testing()


class ImputationTripartiteGraphEdgePrediction(ImputationTripartiteGraph):
    def __init__(
        self,
        original_data_file_name,
        missing_value_file_name,
        random_init,
        node_mapping,
        ext_features,
        pos_neg_scale_factor,
    ):
        super(ImputationTripartiteGraphEdgePrediction, self).__init__(
            original_data_file_name,
            missing_value_file_name,
            random_init,
            node_mapping,
            ext_features,
            pos_neg_scale_factor,
        )
        self.graph_name = "edge"

    def load_and_compute_stats(self):
        super(ImputationTripartiteGraphEdgePrediction, self).load_and_compute_stats()
        self.test_pos_neg_matches = dict()

    # All the existing edges will be considered as positive examples
    # We now create a negative example of the same size
    # For now negative sampling works as follows:
    # for a row/column node, we randomly choose a cell node
    # for a cell node, we randomly choose a row node
    # we verify to make sure that edge does not already exists
    def get_negative_edges_for_training(self):
        # DGL expects the edges to be specified as two tensors containing start and end nodes.
        start_nodes = torch.zeros(self.train_negative_triplets_size, dtype=torch.int32)
        end_nodes = torch.zeros(self.train_negative_triplets_size, dtype=torch.int32)

        # IMPORTANT the lower bound is inclusive and upper bound is exclusive
        row_id_start, row_id_end = 0, self.num_rows
        col_id_start, col_id_end = self.num_rows, self.num_rows + self.num_columns
        cell_id_start, cell_id_end = (
            self.num_rows + self.num_columns,
            self.num_total_nodes,
        )

        row_id_range = range(row_id_start, row_id_end)
        col_id_range = range(col_id_start, col_id_end)
        cell_id_range = range(cell_id_start, cell_id_end)

        # Precompute some random cell ids
        random_cell_ids = torch.randint(
            low=cell_id_start,
            high=cell_id_end,
            size=(self.train_negative_triplets_size,),
        )
        random_row_ids = torch.randint(
            low=row_id_start, high=row_id_end, size=(self.train_negative_triplets_size,)
        )
        random_cell_id_index = 0
        random_row_id_index = 0

        for scale_run in range(self.positive_negative_train_scale_factor):
            for node_id in range(self.num_total_nodes):
                nodes_vector_pos = node_id + scale_run * self.num_total_nodes
                # Corresponds to a valid column id: pick a random cell id for negative edge
                if row_id_start <= node_id < row_id_end:
                    start_nodes[nodes_vector_pos] = node_id
                    end_nodes[nodes_vector_pos] = random_cell_ids[random_cell_id_index]
                    random_cell_id_index += 1
                # Corresponds to a valid column id: pick a random cell id for negative edge
                elif col_id_start <= node_id < col_id_end:
                    start_nodes[nodes_vector_pos] = node_id
                    end_nodes[nodes_vector_pos] = random_cell_ids[random_cell_id_index]
                    random_cell_id_index += 1
                # Corresponds to a valid column id: pick a random ROW  id for negative edge
                # TODO: test other things
                elif cell_id_start <= node_id < cell_id_end:
                    start_nodes[nodes_vector_pos] = node_id
                    end_nodes[nodes_vector_pos] = random_row_ids[random_row_id_index]
                    random_row_id_index += 1
                else:
                    print("Error: column id seems erroneous %d " % node_id)

        return start_nodes, end_nodes

    # Each missing value becomes a positive triplet
    # every other value frm that attribute domain becomes negative triplet
    def get_positive_negative_edges_for_testing(self):
        positive_triplet_index = 0
        negative_triplet_index = 0

        self.test_positive_triplets = torch.IntTensor(self.num_missing_values, 3)

        start_pos_edges = torch.zeros(self.num_missing_values, dtype=torch.int32)
        end_pos_edges = torch.zeros(self.num_missing_values, dtype=torch.int32)

        start_neg_edges = []
        end_neg_edges = []
        # start_neg_edges = torch.zeros(self.num_missing_values, dtype=torch.int32)
        # end_neg_edges = torch.zeros(self.num_missing_values, dtype=torch.int32)

        index = 0

        start_neg = 0
        end_neg = 0

        for row in range(self.num_rows):
            for col in range(self.num_columns):
                if pd.isnull(self.df_missing.iloc[row, col]) == True:
                    col_node_id = self.num_rows + col
                    # NOTE: we change again from df_missing to df_orig to get the correct value
                    cell_node_id = self.val2idx[self.df_orig.iloc[row, col]]
                    start_pos_edges[positive_triplet_index] = cell_node_id
                    end_pos_edges[positive_triplet_index] = row
                    # self.test_pos_neg_matches[positive_triplet_index] = []
                    self.test_positive_triplets[
                        positive_triplet_index
                    ] = torch.IntTensor((row, col_node_id, cell_node_id))

                    positive_triplet_index += 1

                    # Get all the node_ids for the domain values from that column. The [:] end creates a copy
                    attribute_domain = self.attribute_domain_cell_id_dict[col_node_id][
                        :
                    ]
                    attribute_domain.remove(cell_node_id)

                    for cell_node_id in attribute_domain:
                        start_neg_edges.append(cell_node_id)
                        end_neg_edges.append(row)
                        # self.test_pos_neg_matches[positive_triplet_index-1].append(negative_triplet_index)
                        negative_triplet_index += 1
                    end_neg = negative_triplet_index - 1
                    self.test_pos_neg_matches[positive_triplet_index - 1] = (
                        start_neg,
                        end_neg,
                    )
                    start_neg = end_neg
        # self.test_negative_samples = torch.tensor(self.test_negative_samples)
        # self.test_positive_samples = torch.tensor(self.test_positive_samples)
        # print("positive test pos_triplets", len(self.test_positive_samples))
        # print("negative test pos_triplets", len(self.test_negative_samples))

        start_neg_edges = torch.IntTensor(start_neg_edges)
        end_neg_edges = torch.IntTensor(end_neg_edges)

        return start_pos_edges, end_pos_edges, start_neg_edges, end_neg_edges

    # This creates four "graphs" on which training and evaluation is done.
    # train_{pos,neg}, test_{pos,neg}
    # train_pos are all edges in the graph (excluding the missing values)
    # train_neg are all edges not present in the graph  (excluding the missing values)
    # test_pos are all the edges for the missing values
    # test_neg is currently treated as optional
    # TODO: find a semantics for that
    def create_train_positive_negative_graphs(self):
        # For now, creating a new graph - but could also use self.graph itself
        print("creating train_positive_samples")
        train_pos_u, train_pos_v = self.graph.edges()

        start_nodes = train_pos_u.reshape(-1, 1)
        end_nodes = train_pos_v.reshape(-1, 1)
        self.train_positive_samples = torch.cat([start_nodes, end_nodes], dim=1).to(int)

        self.train_positive_triplets_size = self.num_non_missing_values
        self.positive_negative_train_scale_factor = 20
        self.train_negative_triplets_size = (
            self.train_positive_triplets_size
            * self.positive_negative_train_scale_factor
        )

        print("creating train_negative_samples")
        start_nodes, end_nodes = self.get_negative_edges_for_training()
        start_nodes = start_nodes.reshape(-1, 1)
        end_nodes = end_nodes.reshape(-1, 1)
        self.train_negative_samples = torch.cat([start_nodes, end_nodes], dim=1).to(int)

        # self.train_negative_samples = dgl.graph((start_nodes, end_nodes), num_nodes=self.graph.number_of_nodes())

    def create_test_positive_negative_graphs(self):
        print("creating test_positive_samples")
        (
            start_pos,
            end_pos,
            start_neg,
            end_neg,
        ) = self.get_positive_negative_edges_for_testing()
        start_pos = start_pos.reshape(-1, 1)
        end_pos = end_pos.reshape(-1, 1)
        self.test_positive_samples = torch.cat([start_pos, end_pos], dim=1).to(int)
        # self.test_positive_samples = dgl.graph((start_pos, end_nodes), num_nodes=self.graph.number_of_nodes())

        print("creating test_negative_samples")
        start_neg = start_neg.reshape(-1, 1)
        end_neg = end_neg.reshape(-1, 1)
        self.test_negative_samples = torch.cat([start_neg, end_neg], dim=1).to(int)
        # self.test_negative_samples = dgl.graph((start_nodes, end_nodes), num_nodes=self.graph.number_of_nodes())


class ImputationTripartiteGraphNearestNeighbor(
    ImputationTripartiteGraphTripletPrediction
):
    def __init__(
        self,
        original_data_file_name,
        missing_value_file_name,
        random_init=False,
        node_mapping=None,
        ext_features=None,
        pos_neg_scale_factor=20,
        with_cid=False,
    ):
        self.with_cid = with_cid
        super().__init__(
            original_data_file_name,
            missing_value_file_name,
            random_init,
            node_mapping,
            ext_features,
            False,
            pos_neg_scale_factor,
        )
        self.graph_name = "nn"
        # self.input_tuple_length = 0
        # self.compare_edgepred = compare_edgepred

    def load_and_compute_stats(self):
        super().load_and_compute_stats()
        self.test_positive_samples_size = 0
        self.test_negative_samples_size = 0
        # The following two variables store (row_id, col_id, cell_id) pos_triplets
        self.test_positive_samples = None
        self.test_negative_samples = None
        self.test_pos_neg_matches = {}

    def create_train_positive_negative_graphs(self):
        # Find number of values in each tuple: num_col * 2 (from col_id and value) + 1 (from row id)
        # Removing one of the columns
        self.input_tuple_length = len(self.df_missing.columns)
        self.df_dropped = self.df_missing.dropna(axis=0).copy().reset_index(drop=True)
        self.train_positive_samples_size = len(self.df_dropped)
        self.train_negative_samples_size = (
            self.train_positive_samples_size * self.positive_negative_train_scale_factor
        )

        self.train_positive_samples = np.zeros(
            (self.train_positive_samples_size, self.input_tuple_length)
        )
        self.train_negative_samples = np.zeros(
            (self.train_negative_samples_size, self.input_tuple_length)
        )
        print(
            "Size of training positive pos_triplets ", self.train_positive_samples_size
        )
        print(
            "Size of training negative pos_triplets ", self.train_negative_samples_size
        )
        self.get_positive_negative_samples_for_training()

    def get_positive_negative_samples_for_training(self):
        positive_samples = negative_samples = 0
        for row_num, row in self.df_dropped.iterrows():
            self.train_positive_samples[positive_samples, 0] = row_num
            for col_num, col in enumerate(self.df_dropped.columns):
                val = self.df_dropped.iloc[row_num, col_num]
                val_id = self.val2idx[val]
                self.train_positive_samples[row_num, col_num + 1] = val_id
            positive_samples += 1

            for scaling in range(self.positive_negative_train_scale_factor):
                self.train_negative_samples[negative_samples, 0] = row_num
                for col_num, col in enumerate(self.df_dropped.columns):
                    col_id = self.col2idx[col]

                    val = self.df_dropped.iloc[row_num, col_num]
                    val_id = self.val2idx[val]
                    attribute_domain = self.attribute_domain_cell_id_dict[col_id][:]
                    attribute_domain.remove(val_id)
                    self.train_negative_samples[negative_samples, col_num + 1] = val_id
                negative_samples += 1

        self.labels = self.train_positive_samples.reshape(-1, 1)

    def get_positive_negative_triplets_for_testing(self):
        super(
            ImputationTripartiteGraphNearestNeighbor, self
        ).get_positive_negative_triplets_for_testing()
        self.test_positive_tuples = torch.IntTensor(
            self.test_positive_samples_size, self.input_tuple_length
        )
        self.test_negative_tuples = torch.IntTensor(
            self.test_negative_samples_size, self.input_tuple_length
        )
        negative_samples = positive_samples = 0

        for sample in self.test_positive_samples:
            row_num, col_id, val_id = map(torch.Tensor.item, sample)
            self.test_positive_tuples[positive_samples, 0] = row_num
            for col_num, col in enumerate(self.df_orig.columns):

                val = self.df_orig.iloc[positive_samples, col_num]
                inner_val_id = self.val2idx[val]
                self.test_positive_tuples[positive_samples, col_num + 1] = inner_val_id
            positive_samples += 1

            attribute_domain = self.attribute_domain_cell_id_dict[col_id][:]
            attribute_domain.remove(val_id)
            for scaling in range(len(attribute_domain)):
                self.test_negative_tuples[negative_samples, 0] = row_num
                for col_num, col in enumerate(self.df_dropped.columns):
                    inner_col_id = self.col2idx[col]
                    if inner_col_id != col_id:
                        val = self.df_orig.iloc[row_num, col_num]
                        val_id = self.val2idx[val]
                    else:
                        val_id = attribute_domain[scaling]
                    self.test_negative_tuples[negative_samples, col_num + 1] = val_id

                negative_samples += 1
        self.test_positive_tuples = self.test_positive_tuples.to(int)
        self.test_negative_tuples = self.test_negative_tuples.to(int)

    # test_positive_samples are all the edges for the missing values
    # test_negative_samples consists all the incorrect imputation for each missing value
    def create_test_positive_negative_graphs(self):
        # self.test_positive_samples = torch.zeros(size=(self.test_positive_samples_size,1))
        # self.test_negative_samples = torch.zeros(size=(self.test_positive_samples_size,1))
        self.test_positive_samples = [0 for _ in range(self.test_positive_samples_size)]
        self.test_negative_samples = [0 for _ in range(self.test_negative_samples_size)]
        print(
            "Size of testing positive and negative pos_triplets ",
            self.test_positive_samples_size,
            self.test_negative_samples_size,
        )
        self.get_positive_negative_triplets_for_testing()


class ImputationTripartiteGraphAverageTuple(ImputationTripartiteGraphTripletPrediction):
    def __init__(
        self,
        original_data_file_name,
        missing_value_file_name,
        random_init=False,
        node_mapping=None,
        ext_features=None,
        pos_neg_scale_factor=20,
        with_cid=False,
    ):
        self.with_cid = with_cid
        super().__init__(
            original_data_file_name,
            missing_value_file_name,
            random_init,
            node_mapping,
            ext_features,
            False,
            pos_neg_scale_factor,
        )
        self.graph_name = "avg"
        # self.input_tuple_length = 0
        # self.compare_edgepred = compare_edgepred

    def load_and_compute_stats(self):
        super().load_and_compute_stats()
        self.test_positive_samples_size = 0
        self.test_negative_samples_size = 0
        # The following two variables store (row_id, col_id, cell_id) pos_triplets
        self.test_positive_samples = None
        self.test_negative_samples = None
        self.test_pos_neg_matches = {}

    def create_train_positive_negative_graphs(self):
        # Find number of values in each tuple: num_col * 2 (from col_id and value) + 1 (from row id)
        if self.with_cid:
            self.input_tuple_length = len(self.df_missing.columns) * 2 + 1
        else:
            # Removing one of the columns
            self.input_tuple_length = len(self.df_missing.columns)
        self.df_dropped = self.df_missing.dropna(axis=0).copy().reset_index(drop=True)
        self.train_positive_samples_size = len(self.df_dropped.values.ravel())
        self.train_negative_samples_size = (
            self.train_positive_samples_size * self.positive_negative_train_scale_factor
        )

        self.train_positive_samples = np.zeros(
            (self.train_positive_samples_size, self.input_tuple_length)
        )
        self.train_negative_samples = np.zeros(
            (self.train_negative_samples_size, self.input_tuple_length)
        )

        self.train_positive_triplets_size = len(self.df_dropped.values.ravel())
        self.train_negative_triplets_size = (
            self.train_positive_triplets_size
            * self.positive_negative_train_scale_factor
        )

        self.train_positive_triplets = np.zeros((self.train_positive_triplets_size, 3))
        self.train_negative_triplets = np.zeros((self.train_negative_triplets_size, 3))

        print("Size of training positive samples ", self.train_positive_samples_size)
        print("Size of training negative samples ", self.train_negative_samples_size)
        self.get_positive_negative_samples_for_training()

    def get_positive_negative_samples_for_training(self):
        positive_samples = negative_samples = 0
        positive_triplets = negative_triplets = 0
        for row_num, row in self.df_dropped.iterrows():
            self.train_positive_samples[positive_samples, 0] = row_num
            for col_num, col in enumerate(self.df_dropped.columns):
                val = self.df_dropped.iloc[row_num, col_num]
                val_id = self.val2idx[val]
                col_id = self.col2idx[col]
                full_tuple = self.df_dropped.iloc[row_num].tolist()
                full_tuple.remove(val)

                self.train_positive_samples[positive_samples, 1:] = [
                    self.val2idx[v] for v in full_tuple
                ]
                self.train_positive_triplets[positive_triplets, :] = (
                    row_num,
                    col_id,
                    val_id,
                )

                self.train_negative_samples[positive_samples, 1:] = [
                    self.val2idx[v] for v in full_tuple
                ]
                attribute_domain = self.attribute_domain_cell_id_dict[col_id][:]
                attribute_domain.remove(val_id)
                random.shuffle(attribute_domain)
                for attr_val in attribute_domain[
                    : self.positive_negative_train_scale_factor
                ]:
                    self.train_negative_triplets[negative_triplets, :] = (
                        row_num,
                        col_id,
                        attr_val,
                    )
                    negative_triplets += 1
                positive_triplets += 1
                positive_samples += 1
                negative_samples += 1

        self.labels = self.train_positive_triplets[:, -1]
        # self.labels = self.train_positive_samples.reshape(-1, 1)

    def get_positive_negative_triplets_for_testing(self):
        super(
            ImputationTripartiteGraphAverageTuple, self
        ).get_positive_negative_triplets_for_testing()
        self.test_positive_tuples = torch.zeros(
            (self.test_positive_samples_size, self.input_tuple_length), dtype=int
        )
        # self.test_negative_tuples = torch.zeros((self.test_negative_samples_size, self.input_tuple_length), dtype=int)
        self.test_positive_triplets = torch.IntTensor(
            self.test_positive_samples_size, 3
        )
        self.test_negative_triplets = torch.IntTensor(
            self.test_negative_samples_size, 3
        )

        negative_samples = positive_samples = 0
        positive_triplets = negative_triplets = 0

        for sample in self.test_positive_samples:
            row_num, col_id, val_id = map(torch.Tensor.item, sample)
            self.test_positive_tuples[positive_samples, 0] = row_num
            full_tuple = self.df_orig.iloc[row_num].tolist()
            full_tuple.remove(self.idx2val[val_id])
            self.test_positive_tuples[positive_samples, 1:] = torch.IntTensor(
                [self.val2idx[v] for v in full_tuple]
            )
            self.test_positive_triplets[positive_samples, :] = torch.IntTensor(
                (row_num, col_id, val_id)
            )
            # for col_num, col in enumerate(self.df_orig.columns):
            #     val = self.df_orig.iloc[positive_samples, col_num]
            #     inner_val_id = self.val2idx[val]
            #     self.test_positive_tuples[positive_samples, col_num+1] = inner_val_id
            positive_samples += 1

            attribute_domain = self.attribute_domain_cell_id_dict[col_id][:]
            attribute_domain.remove(val_id)
            for domain_value in range(len(attribute_domain)):
                inner_val_id = attribute_domain[domain_value]
                if inner_val_id != val_id:
                    # self.test_negative_tuples[negative_samples, 0] = row_num
                    full_tuple = self.df_orig.iloc[row_num].tolist()
                    full_tuple.remove(self.idx2val[val_id])
                    # self.test_negative_tuples[negative_samples, 1:] = torch.IntTensor([self.val2idx[v] for v in full_tuple])
                    self.test_negative_triplets[negative_samples, :] = torch.IntTensor(
                        (row_num, col_id, inner_val_id)
                    )

                # if self.with_cid:
                #     for col_num, col in enumerate(self.df_dropped.columns):
                #         inner_col_id = self.col2idx[col]
                #         self.test_negative_tuples[negative_samples, 2*col_num+1] = col_id
                #         if inner_col_id != col_id:
                #             val = self.df_orig.iloc[row_num, col_num]
                #             val_id = self.val2idx[val]
                #         else:
                #             val_id = attribute_domain[domain_value]
                #         self.test_negative_tuples[negative_samples, 2*col_num+2] = val_id
                # else:
                #     for col_num, col in enumerate(self.df_dropped.columns):
                #         inner_col_id = self.col2idx[col]
                #         if inner_col_id != col_id:
                #             val = self.df_orig.iloc[row_num, col_num]
                #             val_id = self.val2idx[val]
                #         else:
                #             val_id = attribute_domain[domain_value]
                #         self.test_negative_tuples[negative_samples, col_num+1] = val_id

                negative_samples += 1
        self.test_positive_tuples = self.test_positive_tuples.to(int)
        # self.test_negative_tuples = self.test_negative_tuples.to(int)

    # test_positive_samples are all the edges for the missing values
    # test_negative_samples consists all the incorrect imputation for each missing value
    def create_test_positive_negative_graphs(self):
        # self.test_positive_samples = torch.zeros(size=(self.test_positive_samples_size,1))
        # self.test_negative_samples = torch.zeros(size=(self.test_positive_samples_size,1))
        self.test_positive_samples = [0 for _ in range(self.test_positive_samples_size)]
        self.test_negative_samples = [0 for _ in range(self.test_negative_samples_size)]
        print(
            "Size of testing positive and negative pos_triplets ",
            self.test_positive_samples_size,
            self.test_negative_samples_size,
        )
        self.get_positive_negative_triplets_for_testing()


class ImputationTripartiteGraphMultilabelClassifier(
    ImputationTripartiteGraphAverageTuple
):
    def __init__(
        self,
        original_data_file_name,
        missing_value_file_name,
        random_init=False,
        node_mapping=None,
        ext_features=None,
        pos_neg_scale_factor=20,
        with_cid=False,
    ):
        self.with_cid = with_cid
        super().__init__(
            original_data_file_name,
            missing_value_file_name,
            random_init,
            node_mapping,
            ext_features,
            pos_neg_scale_factor,
        )
        self.graph_name = "multilabel"

    def create_train_positive_negative_graphs(self):
        self.target_columns = self.df_missing.columns[
            self.df_missing.isna().any()
        ].to_list()

        # Find number of values in each tuple: num_col * 2 (from col_id and value) + 1 (from row id)
        # Removing one of the columns
        self.input_tuple_length = len(self.df_missing.columns) - 1
        self.df_dropped = self.df_missing.dropna(axis=0).copy().reset_index(drop=True)
        self.train_positive_samples_size = len(self.df_dropped) * len(
            self.target_columns
        )
        self.train_negative_samples_size = (
            self.train_positive_samples_size * self.positive_negative_train_scale_factor
        )

        self.train_positive_samples = np.zeros(
            (self.train_positive_samples_size, self.input_tuple_length)
        )
        self.train_negative_samples = np.zeros(
            (self.train_negative_samples_size, self.input_tuple_length)
        )

        self.train_positive_triplets_size = len(self.df_dropped) * len(
            self.target_columns
        )
        self.train_negative_triplets_size = (
            self.train_positive_triplets_size
            * self.positive_negative_train_scale_factor
        )

        self.train_positive_triplets = np.zeros((self.train_positive_triplets_size, 3))
        self.train_negative_triplets = np.zeros((self.train_negative_triplets_size, 3))

        print("Size of training positive samples ", self.train_positive_samples_size)
        print("Size of training negative samples ", self.train_negative_samples_size)
        self.get_positive_negative_samples_for_training()

    def create_test_positive_negative_graphs(self):
        super(
            ImputationTripartiteGraphMultilabelClassifier, self
        ).create_test_positive_negative_graphs()
        # self.labels = self.generate_labels(self.test_positive_samples)

    def generate_labels(self, samples=None):
        labels = torch.zeros(self.train_positive_samples.shape[0], dtype=int)
        idx_conv = {col_id: None for col_id in self.attribute_domain_cell_id_dict}
        self.target_col = None
        for col_id in self.attribute_domain_cell_id_dict:
            attribute_domain = self.attribute_domain_cell_id_dict[col_id]
            idx_conv[col_id] = {v: idx for idx, v in enumerate(attribute_domain)}

        for idx, sample in enumerate(self.train_positive_triplets):
            row_id, col_id, val = sample
            labels[idx] = int(val) - self.num_row_col_nodes
            # labels[idx] = idx_conv[col_id][val]

            self.size_target_col = len(self.attribute_domain_cell_id_dict[col_id])
        return labels

    def get_positive_negative_samples_for_training(self):
        positive_samples = negative_samples = 0
        positive_triplets = negative_triplets = 0
        for row_num, row in self.df_dropped.iterrows():
            # self.train_positive_samples[positive_samples, 0] = row_num
            for col in self.target_columns:
                # col = self.idx2col[target_column]
                col_num = self.df_missing.columns.to_list().index(col)
                val = self.df_dropped.iloc[row_num, col_num]
                val_id = self.val2idx[val]
                col_id = self.col2idx[col]
                full_tuple = self.df_dropped.iloc[row_num].tolist()
                full_tuple.remove(val)

                self.train_positive_samples[positive_samples, :] = [
                    self.val2idx[v] for v in full_tuple
                ]
                self.train_positive_triplets[positive_triplets, :] = (
                    row_num,
                    col_id,
                    val_id,
                )

                positive_triplets += 1
                positive_samples += 1
                negative_samples += 1

        # self.labels = self.train_positive_samples.reshape(-1, 1)

    def get_positive_negative_triplets_for_testing(self):
        super(
            ImputationTripartiteGraphAverageTuple, self
        ).get_positive_negative_triplets_for_testing()
        self.test_positive_tuples = torch.zeros(
            (self.test_positive_samples_size, self.input_tuple_length), dtype=int
        )
        # self.test_negative_tuples = torch.zeros((self.test_negative_samples_size, self.input_tuple_length), dtype=int)
        self.test_positive_triplets = torch.IntTensor(
            self.test_positive_samples_size, 3
        )
        self.test_negative_triplets = torch.IntTensor(
            self.test_negative_samples_size, 3
        )

        negative_samples = positive_samples = 0
        positive_triplets = negative_triplets = 0

        for sample in self.test_positive_samples:
            row_num, col_id, val_id = map(torch.Tensor.item, sample)
            self.test_positive_tuples[positive_samples, 0] = row_num
            full_tuple = self.df_orig.iloc[row_num].tolist()
            full_tuple.remove(self.idx2val[val_id])
            self.test_positive_tuples[positive_samples, :] = torch.IntTensor(
                [self.val2idx[v] for v in full_tuple]
            )
            self.test_positive_triplets[positive_samples, :] = torch.IntTensor(
                (row_num, col_id, val_id)
            )
            # for col_num, col in enumerate(self.df_orig.columns):
            #     val = self.df_orig.iloc[positive_samples, col_num]
            #     inner_val_id = self.val2idx[val]
            #     self.test_positive_tuples[positive_samples, col_num+1] = inner_val_id
            positive_samples += 1

        self.test_positive_tuples = self.test_positive_tuples.to(int)

    def process(self):
        super(ImputationTripartiteGraphMultilabelClassifier, self).process()
        self.labels = self.generate_labels(self.test_positive_samples)
