# GiG
import os

import pandas as pd
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset


class ImputationTripartiteGraphDatasetTripletPrediction(DGLDataset):
    def __init__(
        self,
        original_data_file_name,
        missing_value_file_name,
        random_init=False,
        node_mapping=None,
        ext_features=None,
        predict_edges=False,
    ):
        self.original_data_file_name = original_data_file_name
        self.missing_value_file_name = missing_value_file_name
        self.num_rows = self.num_columns = 0
        # num(rows) + num(columns)
        self.num_row_col_nodes = 0
        # num(rows) + num(columns) + num(distinct cell values in original_data_file_name)
        self.num_total_nodes = 0

        # Additional parameter to specify random initialization of all features
        self.random_init = random_init

        # Additional data structures needed to handle external features.
        self.node_mapping = node_mapping
        self.ext_features = ext_features

        # Flag for predicting edges rather than hyperedges
        self.predict_edges = predict_edges

        super().__init__(name="EmbDI Graph")

    def prepare_node_dicts(self):
        self.node2idx = {f"idx__{idx}": idx for idx in range(self.num_rows)}
        self.node2idx.update({f"cid__{cid}": v for cid, v in self.column_dict.items()})
        self.node2idx.update(
            {f"tt__{token}": v for token, v in self.distinct_value_dict.items()}
        )
        self.idx2node = {v: k for k, v in self.node2idx.items()}

    # This function loads the two data frames, does some bookkeeping and computes basic stats
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
        self.distinct_value_dict = dict(
            (v, i + self.num_row_col_nodes)
            for (i, v) in enumerate(sorted(self.distinct_value_set))
        )
        self.distinct_value_dict_reverse = dict(
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

        self.column_dict = dict(
            (col, self.num_rows + i) for (i, col) in enumerate(self.df_orig.columns)
        )
        self.column_dict_reverse = dict(
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

        self.test_positive_triplets_size = 0
        self.test_negative_triplets_size = 0
        # The following two variables store (row_id, col_id, cell_id) triplets
        self.test_positive_triplets = None
        self.test_negative_triplets = None
        self.test_pos_neg_matches = {}

        # create a dictionary to avoid repeated cell id lookups for each attribute.
        # The key is col_id and it will return cell_ids of all the values in its domain
        self.attribute_domain_cell_id_dict = {}
        for col in self.df_orig.columns:
            distinct_values = self.df_orig[col].unique()
            cell_id_array = [self.distinct_value_dict[val] for val in distinct_values]
            col_id = self.column_dict[col]
            self.attribute_domain_cell_id_dict[col_id] = cell_id_array

    def create_graph(self):
        # Create empty graph
        self.graph = dgl.graph(data=[])

        # Create num_total_nodes isolated nodes
        self.graph.add_nodes(self.num_total_nodes)

        # Create edges
        # Note that we use df_missing for creating the edges
        # So cells corresponding to missing values will be isolated

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
                    self.test_positive_triplets_size += 1
                    self.test_negative_triplets_size = (
                        self.test_negative_triplets_size + num_unique_neg_vals
                    )
        # DGL expects the edges to be specified as two tensors containing start and end nodes.
        start_nodes = torch.zeros(num_edges, dtype=torch.int64)
        end_nodes = torch.zeros(num_edges, dtype=torch.int64)

        index = 0

        for row in range(self.num_rows):
            row_node_id = row
            for col in range(self.num_columns):
                col_node_id = col + self.num_rows
                if pd.isnull(self.df_missing.iloc[row, col]) == False:
                    cell_node_id = self.distinct_value_dict[
                        self.df_missing.iloc[row, col]
                    ]
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

        # Features for cell nodes = it is connected to each column
        self.graph.nodes[range(self.num_row_col_nodes, self.num_total_nodes)].data[
            "features"
        ] = torch.ones(self.num_total_nodes - self.num_row_col_nodes, self.num_columns)

        if self.random_init:
            self.graph.nodes[range(0, self.num_total_nodes)].data[
                "features"
            ] *= torch.FloatTensor(self.num_total_nodes, self.num_columns).uniform_()

    def load_external_features(self):
        self.graph.ndata["features"] = torch.zeros(size=self.ext_features.shape)
        self.num_features = self.ext_features.shape[1]
        perm = [self.node2idx[node] for node in self.node_mapping]
        reordered = self.ext_features[perm, :]
        self.graph.nodes[:].data["features"] = reordered
        # for idx, node in enumerate(self.node2idx):
        #     # node.data['features'] = self.ext_features[node]
        #     self.graph.nodes[idx].data['features'] = self.ext_features[node]
        #     if idx % 100 == 0:
        #         print(f'{idx}/{len(self.node2idx)}')

    # train_positive_samples are all (row_id, col_id, cell_id) [excluding missing balues]
    # train_negative_samples are all edges not present in the graph  [excluding the missing values]
    # specifically, we create a negative triplet by retaining (row, col) but choose a random cell
    def get_positive_negative_triplets_for_training(self):
        positive_triplet_index, negative_triplet_index = 0, 0

        cell_id_start, cell_id_end = (
            self.num_rows + self.num_columns,
            self.num_total_nodes,
        )
        random_cell_ids = torch.randint(
            low=cell_id_start,
            high=cell_id_end,
            size=(self.train_negative_triplets_size,),
        )

        for row in range(self.num_rows):
            for col in range(self.num_columns):
                if pd.isnull(self.df_missing.iloc[row, col]) == False:
                    col_node_id = self.num_rows + col
                    cell_node_id = self.distinct_value_dict[
                        self.df_missing.iloc[row, col]
                    ]
                    self.train_positive_triplets[positive_triplet_index] = (
                        row,
                        col_node_id,
                        cell_node_id,
                    )
                    positive_triplet_index += 1

                    for index2 in range(self.positive_negative_train_scale_factor):
                        random_cell_id = random_cell_ids[negative_triplet_index]
                        self.train_negative_triplets[negative_triplet_index] = (
                            row,
                            col_node_id,
                            random_cell_id,
                        )
                        negative_triplet_index += 1
        self.train_positive_triplets = torch.LongTensor(self.train_positive_triplets)
        self.train_negative_triplets = torch.LongTensor(self.train_negative_triplets)

    # Each missing value becomes a positive triplet
    # every other value frm that attribute domain becomes negative triplet
    def get_positive_negative_triplets_for_testing(self):
        positive_triplet_index = 0
        negative_triplet_index = 0

        start_neg = 0
        end_neg = 0

        for row in range(self.num_rows):
            for col in range(self.num_columns):
                if pd.isnull(self.df_missing.iloc[row, col]) == True:
                    col_node_id = self.num_rows + col
                    # NOTE: we change again from df_missing to df_orig to get the correct value
                    cell_node_id = self.distinct_value_dict[self.df_orig.iloc[row, col]]
                    self.test_positive_triplets[positive_triplet_index] = (
                        row,
                        col_node_id,
                        cell_node_id,
                    )
                    self.test_pos_neg_matches[positive_triplet_index] = []
                    positive_triplet_index += 1

                    # Get all the node_ids for the domain values from that column. The [:] end creates a copy
                    attribute_domain = self.attribute_domain_cell_id_dict[col_node_id][
                        :
                    ]
                    attribute_domain.remove(cell_node_id)
                    for cell_node_id in attribute_domain:
                        self.test_negative_triplets[negative_triplet_index] = (
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
        self.test_negative_triplets = torch.tensor(self.test_negative_triplets)
        self.test_positive_triplets = torch.tensor(self.test_positive_triplets)
        print("positive test triplets", len(self.test_positive_triplets))
        print("negative test triplets", len(self.test_negative_triplets))

    def create_train_positive_negative_graphs(self):
        # number of negative examples per positive example
        self.positive_negative_train_scale_factor = 20

        self.train_positive_triplets_size = self.num_non_missing_values
        self.train_negative_triplets_size = (
            self.train_positive_triplets_size
            * self.positive_negative_train_scale_factor
        )
        self.train_positive_triplets = np.zeros((self.train_positive_triplets_size, 3))
        self.train_negative_triplets = np.zeros((self.train_negative_triplets_size, 3))
        print(
            "Size of training positive and negative triplets ",
            self.train_positive_triplets_size,
            self.train_negative_triplets_size,
        )
        self.get_positive_negative_triplets_for_training()

    # test_positive_samples are all the edges for the missing values
    # test_negative_samples consists all the incorrect imputation for each missing value
    def create_test_positive_negative_graphs(self):
        # self.test_positive_samples = torch.zeros(size=(self.test_positive_samples_size,1))
        # self.test_negative_samples = torch.zeros(size=(self.test_positive_samples_size,1))
        self.test_positive_triplets = [
            0 for _ in range(self.test_positive_triplets_size)
        ]
        self.test_negative_triplets = [
            0 for _ in range(self.test_negative_triplets_size)
        ]
        print(
            "Size of testing positive and negative triplets ",
            self.test_positive_triplets_size,
            self.test_negative_triplets_size,
        )
        self.get_positive_negative_triplets_for_testing()

    def reduce_to_edges(self):
        self.test_positive_triplets = self.test_positive_triplets[:, :2]
        self.test_negative_triplets = self.test_negative_triplets[:, :2]
        self.train_positive_triplets = self.train_positive_triplets[:, :2]
        self.train_negative_triplets = self.train_negative_triplets[:, :2]

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
        print("Creating positive and negative triplets for training")
        self.create_train_positive_negative_graphs()
        print("Creating positive and negative triplets for testing")
        self.create_test_positive_negative_graphs()

        if self.predict_edges:
            self.reduce_to_edges()

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1
