# GiG
import os

import pandas as pd
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset
import random
import copy
import os.path as osp
import pickle
import warnings
from itertools import combinations
from sklearn.preprocessing import minmax_scale

from tqdm import tqdm


class HeterographDataset(DGLDataset):
    def __init__(
        self,
        original_data_file_name,
        missing_value_file_name,
        random_init=False,
        node_mapping=None,
        ext_features=None,
        pos_neg_scale_factor=20,
        smart_sampling=False,
        architecture="multitask",
        training_subset="target",
        target_columns=None,
        training_columns=None,
        ignore_columns=None,
        ignore_num_flag=False,
        convert_columns=None,
        norm_strategy="norm",
        scaling_factor=2,
        keep_mask=False,
        flag_col=False,
        flag_rid=False,
        fd_dict=None,
        fd_strategy=None,
        only_rhs=True,
        max_comb_size=1,
        max_num_comb=15,
        training_sample=1,
        device="cpu",
    ):
        """

        :param original_data_file_name: Path to original (clean) file, to be used as ground truth.
        :param missing_value_file_name: Path to dirty dataset.
        :param random_init: Whether the features should be initialized at random (no pre-trained features).
        :param node_mapping: Prepared automatically if external features are present.
        :param ext_features: Prepared automatically if external features are present.
        :param pos_neg_scale_factor: Not used.
        :param smart_sampling: Not used.
        :param architecture: Fixed to "multitask"
        :param training_subset:
        :param target_columns: Columns to impute on.
        :param training_columns: Columns to train on. They must include target_columns.
        :param ignore_columns: Columns to ignore during training.
        :param ignore_num_flag: If true, numerical columns will not be considered during training and imputation.
        :param convert_columns: Numerical columns to convert to categorical during training.
        :param norm_strategy: {norm, minmax} Normalization method used on numerical columns
        :param scaling_factor: Not used
        :param keep_mask: Not needed
        :param flag_col: If true, replace the null vector with the vector of the attribute the null value is found in.
        :param flag_rid: If true, prepend the row vector to the training vector.
        :param fd_dict: Dictionary that contains the functional dependencies.
        :param fd_strategy: Strategy to use to implement functional dependencies.
        :param only_rhs:
        :param max_comb_size: Not used
        :param max_num_comb: Not used
        :param training_sample: Not used
        :param device: Needed if running on CUDA.
        """
        self.graph_name = ""
        self.original_data_file_name = original_data_file_name
        self.missing_value_file_name = missing_value_file_name
        self.distinguish = True

        self.device = device

        # The following values are initialized here and filled later after loading and studying the datasets.
        self.num_rows = self.num_columns = 0
        # num(rows) + num(columns)
        self.num_row_col_nodes = 0  # used as offset for the dataset values
        # num(rows) + num(columns) + num(distinct cell values in original_data_file_name)
        self.num_total_nodes = 0
        self.input_tuple_length = 0
        self.missing_values_by_col = dict()
        self.scaling_factor = scaling_factor

        # This parameter defines whether the train/valid sets should be kept or if they should be generated randomly
        # during each execution
        self.keep_mask = keep_mask

        # Tuple-building parameters
        self.flag_index = -1
        self.flag_value = "__flag_index__"
        # if flag_col or flag_rid:
        #     raise NotImplementedError(f'These flags don\'t work with the attention layer for the moment. ')
        self.flag_col = flag_col
        self.flag_rid = flag_rid

        # Parameters for size of training set
        assert 0.0 < training_sample <= 1.0
        self.training_sample = training_sample

        # Parameter for finding which values are frequent. All values with freq > quantile_freq are considered as frequent.
        self.quantile_freq = 0.9

        # To handle rows with multiple missing values, sample sets can have a variable length. This list keeps track
        # of the indices of the first/last value of each set of samples.
        self.boundaries_train = []
        self.boundaries_valid = []

        # Parameters for negative sampling
        # Scale factor: number of negative training values for each positive training value.
        self.positive_negative_train_scale_factor = pos_neg_scale_factor
        # Boolean, whether to do smart sampling or not.
        self.smart_sampling = smart_sampling

        # Additional (optional) data structure that saves labels so they're not computed online
        self.labels = None

        # Additional parameter to specify random initialization of all features
        # Initial features are set to 1.0 if random_init is False.
        self.random_init = True

        # Additional data structures needed to handle external features.
        # These are generated by an external function
        self.node_mapping = node_mapping
        self.ext_features = ext_features

        # Parameters for handling FDs
        self.fd_dict = fd_dict
        self.fd_list = []
        if fd_dict is None:
            self.fd_strategy = None
            self.flat_fds = dict()
        else:
            self.fd_strategy = fd_strategy
            self.flat_fds = {}
            for col in self.fd_dict:
                self.flat_fds[col] = self.get_flat_fds(col)
        self.only_rhs = only_rhs
        self.num_fds = 0

        # Model architecture (either multitask or multilabel)
        self.architecture = architecture
        # Subset of columns to be used when performing training. Either all columns, missing columns or columns defined
        # in variable "training_columns".
        self.training_subset = training_subset
        self.target_columns = target_columns
        self.training_columns = training_columns
        self.ignore_columns = ignore_columns
        self.ignore_num_flag = ignore_num_flag
        self.convert_columns = convert_columns if convert_columns is not None else []
        self.norm_strategy = norm_strategy

        self.dirty_columns = []

        # Maximum size of combinations to be used in the generation of additional missing values
        # WARNING: large numbers of dirty columns and/or large values of max_comb_size will cause the system to run out of memory!!!
        self.max_comb_size = max_comb_size
        self.max_num_comb = max_num_comb

        super().__init__(name="EmbDI Graph")
        self.graph_name = "multilabel"

    def _prepare_train_test_columns(self):
        # If target_columns is None, then all columns that contain missing values are assumed to be imputation targets.
        if self.target_columns is None:
            self.target_columns = self.df_missing.columns[
                self.df_missing.isna().any()
            ].to_list()

        if self.ignore_num_flag:
            self.ignore_columns = self.numerical_columns.to_list()

        if self.ignore_columns is not None:
            self.target_columns = [
                target_column
                for target_column in self.target_columns
                if target_column not in self.ignore_columns
            ]

        # If self.training_columns is not None, the training columns have already been supplied.
        if self.training_columns is None:
            if self.training_subset == "target":
                # Train only on the target columns.
                self.training_columns = self.target_columns
            elif self.training_subset == "missing":
                # Train on all columns that contain missing values (this subset might be different from the target subset).
                self.training_columns = self.df_missing.columns[
                    self.df_missing.isna().any()
                ].to_list()
                # Train on all columns in the dataset.
            elif self.training_subset == "all":
                if self.ignore_columns is not None:
                    self.training_columns = [
                        col
                        for col in self.df_missing.columns.to_list()
                        if col not in self.ignore_columns
                    ]
                else:
                    self.training_columns = self.df_missing.columns.to_list()
            else:
                raise ValueError(f"Unknown traing subset {self.training_subset}")

        # if self.target_columns is not None:
        # Check to ensure that all target columns are present among the training columns.
        for col in self.target_columns:
            if col not in self.training_columns:
                raise ValueError(
                    f'Designated target column "f{col}" was not found among training columns. '
                )
        self.training_columns.sort(key=self.df_orig.columns.to_list().index)

        self.map_h2c = {
            idx: self.all_columns.index(col)
            for idx, col in enumerate(self.training_columns)
        }
        self.map_c2h = {v: k for k, v in self.map_h2c.items()}

    def _prepare_node_dictionaries(self):
        # These dictionaries are used to map values between the original dataframes (with strings)
        # and the data in the model (node indices)

        # Map values to their indices, and viceversa
        self.val2idx = dict(
            (v, i + self.num_rows)
            for (i, v) in enumerate(sorted(self.distinct_value_set))
        )
        self.idx2val = dict(
            (i + self.num_rows, v)
            for (i, v) in enumerate(sorted(self.distinct_value_set))
        )

        # # Map columns to their indices, and viceversa
        self.col2idx = dict((col, i) for (i, col) in enumerate(self.df_missing.columns))
        self.idx2col = dict((i, col) for (i, col) in enumerate(self.df_missing.columns))

        self.node2idx = {f"idx__{idx}": idx for idx in range(self.num_rows)}
        # self.node2idx.update({f'cid__{cid}':v for cid, v in self.col2idx.items()})
        self.node2idx.update({f"{token}": v for token, v in self.val2idx.items()})
        self.idx2node = {v: k for k, v in self.node2idx.items()}

        # For each distinct value associate with it the column in which it is found
        # if a value V is present in two columns A and B, then it will be associated with the latest column
        # self.cell_to_dict = {self.df_missing.iloc[row, col]: col for row in range(self.num_rows) for col in range(self.num_columns)}

    def _measure_value_counts(self):
        # Measure the number of occurrences of each unique value in the dataset
        # This is useful to have an idea of the degree distribution in the graph.

        self.counts = {k: None for k in self.df_missing.columns}
        self.quantile_col = {k: 0 for k in self.df_missing.columns}

        self.frequencies = {k: dict() for k in self.df_missing.columns}
        self.frequencies_scaled = {k: dict() for k in self.df_missing.columns}
        self.frequent_values = {k: None for k in self.df_missing.columns}
        for column in self.counts:
            self.counts[column] = self.df_missing.value_counts(column)
            self.quantile_col[column] = np.quantile(
                self.counts[column], self.quantile_freq
            )
            self.frequent_values[column] = [
                val
                for val, count in self.counts[column].items()
                if count > self.quantile_col[column]
            ]
            tmp_df = (self.counts[column] / len(self.df_missing)).reset_index()
            tmp_df.columns = [column, "frequency"]
            tmp_df[column] = tmp_df[column].apply(lambda x: self.val2idx[x])
            self.frequencies[column].update(
                {int(row[0]): row[1] for row in tmp_df.values}
            )
            lower_bound = max(tmp_df["frequency"])
            upper_bound = self.scaling_factor * lower_bound
            s = (
                minmax_scale(
                    list(self.frequencies[column].values()),
                    feature_range=(lower_bound, upper_bound),
                )
                / upper_bound
            )
            self.frequencies_scaled[column] = dict(
                zip(self.frequencies[column].keys(), s)
            )

            # self.frequencies[column] = tmp_df

            if self.smart_sampling:
                # Clamping most frequent value to perform smart sampling of negative samples
                self.counts[column][0] = self.counts[column][1]

        self.loss_gamma = {k: None for k in self.df_missing.columns}
        self.loss_alpha = {k: None for k in self.df_missing.columns}
        for col in self.df_missing.columns:
            self.loss_gamma[col] = -np.log2(
                len(self.frequent_values[col]) / len(self.counts[col])
            )
            self.loss_alpha[col] = len(self.frequent_values[col]) / len(
                self.counts[col]
            )

        if self.smart_sampling:
            self.budget = copy.deepcopy(self.counts)
            for column in self.budget:
                self.budget[column][0] = self.budget[column][1]
                self.budget[column] *= self.positive_negative_train_scale_factor
                self.budget[column] = self.budget[column].reset_index(name="counts")

    def _load_external_features(self):
        """
        This function uses external features provided by the user to initialize the node features.
        :return:
        """
        # Get the number of dimensions in the external features
        self.num_features = self.ext_features.shape[1]

        # Convenience list for storing the column features, since their nodes are not in the graph anymore.
        self.column_features = []

        # Recombining nodes and their vectors.
        ext_feature_dict = dict(zip(self.node_mapping, self.ext_features))
        # Adding the null value flag.
        ext_feature_dict["__flag_index__"] = torch.zeros(self.num_features)
        ext_feature_dict[np.nan] = torch.zeros(self.num_features)
        for col in self.all_columns:
            self.column_features.append(ext_feature_dict[f"cid__{col}"])

        rid_feats = {
            f"idx__{idx}": ext_feature_dict[f"idx__{idx}"]
            for idx in range(self.num_rows)
        }

        def ufunc(x, ext_feat_dict, flag_index):
            try:
                return ext_feat_dict[x]
            except KeyError:
                print(f"Value {x} was not found among the external features. ")
                return flag_index

        cell_feats = {
            idx: ufunc(idx, ext_feature_dict, self.flag_index) for idx in self.val2idx
        }
        cell_feats_t = torch.zeros(
            size=(self.graph.number_of_nodes("cell"), self.num_features)
        )
        # torch.stack(...) converts a list of lists into a bidimensional tensor.
        rid_feats_t = torch.stack(list(rid_feats.values()))
        # Due to how DGL handles classes, there are self.num_rows null vectors before the actual cell_feats vectors.
        cell_feats_t[self.num_rows :, :] = torch.stack(list(cell_feats.values()))

        self.graph.nodes["rid"].data["features"] = rid_feats_t.to(torch.float32)
        self.graph.nodes["cell"].data["features"] = cell_feats_t.to(torch.float32)

    def _remove_target_edges(self, target_triplets):
        """
        This function removes all the edges involved in target_triplets to make sure that the training algorithm is not
        aware of the validation and testing edges.
        :return:
        """

        tgt_by_col = {col_id: {"u": [], "v": []} for col_id in self.idx2col}
        for triplet in target_triplets:
            row_id, col_id_t, val_id = triplet
            col_id = col_id_t.item()
            tgt_by_col[col_id]["u"].append(row_id)
            tgt_by_col[col_id]["v"].append(val_id)

        for col_id in tgt_by_col:
            col = self.idx2col[col_id]
            u = torch.tensor(tgt_by_col[col_id]["u"]).to(int)
            v = torch.tensor(tgt_by_col[col_id]["v"]).to(int)
            target_edges = self.graph.edge_ids(u, v, etype=col)
            self.graph.remove_edges(target_edges, etype=col)

    def _prepare_valid_mask(self, valid_fraction=0.2):
        path, ext = osp.splitext(self.missing_value_file_name)
        if self.distinguish:
            # Split nodes
            mask_filename = path + "_mask_d.pkl"
        else:
            # Merged nodes
            mask_filename = path + "_mask_m.pkl"
        n_rows = len(self.df_train)

        if self.keep_mask and osp.exists(mask_filename):
            mask_dict = pickle.load(open(mask_filename, "rb"))
            try:
                _ = self.df_train.loc[mask_dict["train_idx"]]
                _ = self.df_train.loc[mask_dict["valid_idx"]]
                if len(_) == 0:
                    raise ValueError(f"Something went wrong while importing the mask. ")
                return mask_dict["train_idx"], mask_dict["valid_idx"]
            except KeyError:
                warnings.warn(
                    f"The saved Train/Valid mask is not appropriate for the current combination "
                    f"of training columns.\nA new mask will be generated."
                )
        else:
            # np.random.choice(self.df_train, int(len(self.df_train) * valid_fraction))
            indexes = np.random.choice(np.arange(n_rows), int(n_rows * valid_fraction))
            mask = np.ones(n_rows, dtype=bool)
            mask[indexes] = False
            train_idx = self.df_train.loc[mask].index
            valid_idx = self.df_train.loc[~mask].index
            mask_dict = {"train_idx": train_idx, "valid_idx": valid_idx}
            if self.keep_mask:
                pickle.dump(mask_dict, open(mask_filename, "wb"))
            return train_idx, valid_idx

    def _build_validation_df(self):
        return self.df_train.reindex(self.train_idx), self.df_train.reindex(
            self.valid_idx
        )

    def _align_floats(self, source_df: pd.DataFrame):
        """
        This function iterates over the columns and tries to convert them to floats and then back to strings.
        This ensures consistency between datasets when null values are injected in integer columns.
        :return:
        """

        for col in source_df.columns:
            # if source_df[col].dtype ==
            if col in self.numerical_columns or col in self.convert_columns:
                # if source_df[col].str.isnumeric().all():
                source_df[col] = (
                    source_df[col].astype(float).round(decimals=8).astype("str")
                )
            else:
                source_df[col] = source_df[col].astype("str")
        return source_df

    def _distinguish_homographs(self, df):
        for i, col in enumerate(df.columns):
            df[col] = df[col].apply(
                lambda x: f"c{i}_{x}" if x == x and x != "nan" else np.nan
            )
        return df

    def _normalize_numeric_columns(self):
        # TODO refactor this code so it's not doing the same thing twice
        self.normalized_mapping = {col: {} for col in self.numerical_columns}
        if self.norm_strategy == "minmax":
            self.norm_max = dict()
            self.norm_min = dict()
            for col in self.numerical_columns:
                mmax = self.df_missing[col].max()
                mmin = self.df_missing[col].min()
                self.norm_max[col] = mmax
                self.norm_min[col] = mmin
                for x in self.df_missing[col].unique():
                    self.normalized_mapping[col][round(x, 8)] = (x - mmin) / (
                        mmax - min
                    )
                # self.df_missing[col] = self.[col].apply(lambda x: )
                # self.df_orig[col] = self.df_orig[col]df_missing.apply(lambda x: (x-mean)/std)
        elif self.norm_strategy == "norm":
            self.norm_mean = dict()
            self.norm_std = dict()
            for col in self.numerical_columns:
                mean = self.df_missing[col].mean()
                std = self.df_missing[col].std()
                self.norm_mean[col] = mean
                self.norm_std[col] = std
                for x in self.df_missing[col].unique():
                    self.normalized_mapping[col][round(x, 8)] = (x - mean) / std

                # self.df_missing[col] = self.df_missing[col].apply(lambda x: (x-mean)/std)
                # self.df_orig[col] = self.df_orig[col].apply(lambda x: (x-mean)/std)
        else:
            raise ValueError(f"Unknown normalization strategy {self.norm_strategy}")

    def denormalize_column(self, column, column_values):
        if self.norm_strategy == "norm":
            return column_values * self.norm_std[column] + self.norm_mean[column]
        if self.norm_strategy == "minmax":
            return (column_values + self.norm_min[column]) * (
                self.norm_max[column] - self.norm_min[column]
            )

    def load_and_compute_stats(self):
        # Ground truth
        self.df_orig = pd.read_csv(self.original_data_file_name)
        # Dirty dataset. All data structures are built starting from this.
        self.df_missing = pd.read_csv(self.missing_value_file_name)
        self.all_columns = list(self.df_missing.columns)

        # Inferring column data type.
        self.numerical_columns = self.df_missing.select_dtypes(
            include="number"
        ).columns.to_list()
        self.categorical_columns = self.df_missing.select_dtypes(
            exclude="number"
        ).columns.to_list()

        # Check if the list self.convert_columns contains something. If it does, all columns in the list will be treated as categorical.
        if self.convert_columns:
            for col in self.convert_columns:
                self.numerical_columns.remove(col)
                self.categorical_columns.append(col)
        self._normalize_numeric_columns()
        # TODO: This needs fixing for when I am converting columns to numeric.
        self.df_orig = self._align_floats(self.df_orig)
        self.df_missing = self._align_floats(self.df_missing)
        if self.distinguish:
            self.df_orig = self._distinguish_homographs(self.df_orig)
            self.df_missing = self._distinguish_homographs(self.df_missing)
        self.dirty_columns = self.df_missing.columns[
            self.df_missing.isna().any()
        ].to_list()
        self._prepare_train_test_columns()

        if self.df_orig.shape != self.df_missing.shape:
            raise Exception(
                "Error: Input files do not have same number of rows and columns"
            )

        self.df_train = self.df_missing.dropna(
            subset=self.training_columns, axis=0, how="all"
        ).copy()
        self.train_idx, self.valid_idx = self._prepare_valid_mask()
        self.df_train, self.df_valid = self._build_validation_df()

        # Convert every attribute to string and treat it as categorical
        # Redundant given the dtype in read_csv but still
        for col in self.df_missing.columns:
            self.df_missing[col] = self.df_missing[col].astype(str)
        for col in self.df_missing.columns:
            self.df_missing[col] = self.df_missing[col].astype(str)

        # Using dtype in read_csv makes missing value as a string 'nan'. So replace it appropriately to nan
        self.df_missing.replace("nan", np.nan, inplace=True)

        self.num_rows, self.num_columns = self.df_missing.shape
        self.num_row_col_nodes = self.num_rows + self.num_columns

        self.distinct_value_set = set(self.df_missing.values.ravel().tolist())
        try:
            self.distinct_value_set.remove(np.nan)
            # self.distinct_value_set.remove('nan')
        except KeyError:
            raise KeyError(
                "No null values found in the dataset. Are you sure the correct dirty dataset has been supplied?"
            )
        self._prepare_node_dictionaries()

        # Adding one dummy node for indicization later on (needed for array operations)
        self.num_total_nodes = self.num_rows + len(self.val2idx) + 1
        self.flag_index = self.num_total_nodes - 1
        self.val2idx[self.flag_value] = self.flag_index
        self.val2idx[np.nan] = self.flag_index
        self.idx2val[self.flag_index] = self.flag_value

        # Aggregate for each column and then aggregate for entire data frame
        self.num_missing_values = (
            0  # will be computed later. self.df_missing.isna().sum().sum()
        )
        self.num_non_missing_values = (
            0  # will be created later len(self.distinct_value_set)
        )

        # Prepare dict mappings between node name and vector idx
        self._measure_value_counts()

        # create a dictionary to avoid repeated cell id lookups for each attribute.
        # The key is col_id and it will return cell_ids of all the values in its domain
        self.attribute_domain_cell_id_dict = {}
        self.has_missing = []

        for idx, col in enumerate(self.df_missing.columns):
            n_missing = sum(self.df_missing[col].isna())
            if n_missing > 0:
                self.has_missing.append(col)
                self.missing_values_by_col[col] = n_missing
            distinct_values = self.df_missing[col].unique()
            cell_id_array = [self.val2idx[val] for val in distinct_values if val == val]
            # col_id = self.col2idx[col]
            self.attribute_domain_cell_id_dict[idx] = cell_id_array

        self.head_dims = {col_id: None for col_id in self.attribute_domain_cell_id_dict}
        for idx, col in enumerate(self.training_columns):
            # col_id = self.col2idx[col]
            if col in self.numerical_columns:
                self.head_dims[self.col2idx[col]] = 1
            else:
                self.head_dims[self.col2idx[col]] = len(
                    self.attribute_domain_cell_id_dict[self.col2idx[col]]
                )

    def create_graph(self):
        # Define edge types
        edge_types = [_ for _ in self.all_columns]
        # DGL expects the edges to be specified as two tensors containing start and end nodes.
        start_nodes_d = {col: [] for col in self.all_columns}
        end_nodes_d = {col: [] for col in self.all_columns}

        start_nodes_null_d = {col: [] for col in self.all_columns}
        end_nodes_null_d = {col: [] for col in self.all_columns}

        # Building src-dst nodes for each column.
        for col in self.all_columns:
            for row in range(self.num_rows):
                row_node_id = row
                # Only consider non-null cells.
                if pd.isnull(self.df_missing.loc[row, col]) == False:
                    cell_node_id = self.val2idx[self.df_missing.loc[row, col]]
                    # Reverse edges will be added later.
                    start_nodes_d[col].append(row_node_id)
                    end_nodes_d[col].append(cell_node_id)
                else:
                    cell_node_id = self.val2idx[self.df_missing.loc[row, col]]
                    # Reverse edges will be added later.
                    start_nodes_null_d[col].append(row_node_id)
                    end_nodes_null_d[col].append(cell_node_id)

        # Needed by DGL to define the number of nodes of each class (?)
        num_nodes_dict = {
            "rid": self.num_rows,
            "cell": len(self.val2idx) + self.num_rows,
        }
        weight_tensor_d = [None for _ in self.all_columns]

        data_dict = {}

        for col in self.all_columns:
            this_triplet = ("rid", col, "cell")
            other_triplet = ("cell", f"i_{col}", "rid")
            src = torch.tensor(start_nodes_d[col], dtype=torch.int64)
            dst = torch.tensor(end_nodes_d[col], dtype=torch.int64)
            data_dict[this_triplet] = (src, dst)
            data_dict[other_triplet] = (dst, src)

        # for col in self.all_columns:
        #     this_triplet = ('rid', f'n_{col}', 'cell')
        #     other_triplet = ('cell', f'n_i_{col}', 'rid')
        #     src = torch.tensor(start_nodes_null_d[col], dtype=torch.int64)
        #     dst = torch.tensor(end_nodes_null_d[col], dtype=torch.int64)
        #     data_dict[this_triplet] = (src, dst)
        #     data_dict[other_triplet] = (dst, src)

        # if self.fd_dict is not None:
        #     # pass
        #     fd_edges, fd_list = self.get_fd_edges()
        #     self.fd_list = fd_list
        #     data_dict.update(fd_edges)

        # Building the actual heterograph.
        self.graph = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
        self.graph = self.graph.to_simple()

        print(
            "Graph has %d nodes and %d edges"
            % (self.graph.number_of_nodes(), self.graph.number_of_edges())
        )

    def compute_node_features(self):
        # ndata stores various features for each node.
        scaling = 1
        self.num_features = self.num_columns * scaling

        # self.graph.ndata['features'] = torch.zeros( (self.num_total_nodes, self.num_features) )

        # #Features for row nodes: all rows are connected to all columns
        # self.graph.nodes[range(0, self.num_rows)].data['features'] = torch.ones(self.num_rows, self.num_features)

        # Features for column nodes: columns are connected only to themselves
        self.column_features = []
        for col_feat in torch.eye(self.num_columns).repeat_interleave(scaling, 1):
            self.column_features.append(col_feat)

        # Features for cell nodes: each node is connected to each column
        rid_feats = {
            f"idx__{idx}": torch.zeros(self.num_features).uniform_()
            for idx in range(self.num_rows)
        }
        cell_feats = {
            idx: torch.zeros(self.num_features).uniform_() for idx in self.val2idx
        }
        cell_feats_t = torch.zeros(
            size=(self.graph.number_of_nodes("cell"), self.num_features)
        )
        # torch.stack(...) converts a list of lists into a bidimensional tensor.
        rid_feats_t = torch.stack(list(rid_feats.values()))
        # Due to how DGL handles classes, there are self.num_rows null vectors before the actual cell_feats vectors.
        cell_feats_t[self.num_rows :, :] = torch.stack(list(cell_feats.values()))
        self.graph.nodes["rid"].data["features"] = rid_feats_t.to(torch.float32)
        self.graph.nodes["cell"].data["features"] = cell_feats_t.to(torch.float32)

    def create_train_positive_graph(self):
        self.input_tuple_length = len(self.df_missing.columns)
        if self.flag_rid:
            # RID vector will be added at the beginning of the tuple.
            self.input_tuple_length += 1
        if self.training_sample < 1.0:
            train_df = self.df_train.sample(frac=self.training_sample)
        else:
            train_df = self.df_train.copy()

        (
            self.train_positive_triplets,
            self.train_positive_samples,
            self.train_positive_samples_size,
            self.boundaries_train,
        ) = self.get_positive_samples(train_df)
        print("Size of training positive samples ", self.train_positive_samples_size)

    def create_valid_positive_graph(self):
        (
            self.valid_positive_triplets,
            self.valid_positive_samples,
            self.valid_positive_samples_size,
            self.boundaries_valid,
        ) = self.get_positive_samples(self.df_valid)
        print("Size of validation positive samples ", self.valid_positive_samples_size)

        self._remove_target_edges(self.valid_positive_triplets)

    def create_test_positive_graphs(self):
        (
            self.test_positive_triplets,
            self.test_positive_samples,
            self.test_positive_samples_size,
            self.boundaries_test,
        ) = self.get_samples_for_testing()

        print("Number of testing samples: ", self.test_positive_samples_size)

    def generate_labels(self, triplets):
        labels = {col: [] for col, val in self.attribute_domain_cell_id_dict.items()}
        weights = {col: [] for col, val in self.attribute_domain_cell_id_dict.items()}

        for idx, sample in enumerate(triplets):
            row_id, col_id, val_id = tuple(map(int, sample))
            col = self.idx2col[col_id]
            if val_id == self.flag_index:
                continue
            else:
                if col in self.categorical_columns:
                    labels[col_id].append(
                        self.attribute_domain_cell_id_dict[col_id].index(val_id)
                    )
                    weights[col_id].append(
                        self.frequencies[self.idx2col[col_id]][val_id]
                    )
                elif col in self.numerical_columns:
                    val = self.idx2val[val_id]
                    _, real_value = val.split("_")
                    norm_value = self.normalized_mapping[col][
                        round(float(real_value), 8)
                    ]
                    labels[col_id].append(float(norm_value))
                    weights[col_id].append(0)
                else:
                    raise ValueError(f"Something is wrong with column {col}")
        labels = {col: v for col, v in labels.items() if v is not None}
        labels_t = []
        for col_id in labels:
            if len(labels[col_id]) == 0:
                labels_t.append([])
                continue
            if self.idx2col[col_id] in self.categorical_columns:
                labels_t.append(
                    torch.tensor(labels[col_id], dtype=torch.long, device=self.device)
                )
            else:
                labels_t.append(
                    torch.tensor(
                        labels[col_id], dtype=torch.float32, device=self.device
                    )
                )
        # labels_t = [torch.LongTensor(labels[col]) if self.head_dims[col] > 1 else torch.FloatTensor(labels[col]) for col in labels]
        weights_t = [torch.FloatTensor(weights[col]) for col in weights]
        return labels_t, weights_t

    def _check_triplet(self, triplet):
        """

        :param triplet:
        :return:
        """
        row_id, col_id, val_id = triplet
        if val_id not in self.graph.out_edges(row_id, etype=self.idx2col[col_id]):
            return False
        return True

    @staticmethod
    def get_combinations(target_columns, max_comb_size, max_num_comb, min_comb_size=1):
        """

        :param target_columns:
        :param max_comb_size:
        :param max_num_comb:
        :param min_comb_size:
        :return:
        """
        from operator import itemgetter

        if max_comb_size == 0:
            max_comb_size = len(target_columns) - 1
        if max_comb_size < 1:
            raise ValueError(
                f"Combinations must have at least size 1. Current size: {max_comb_size}"
            )
        if max_comb_size == 1:
            print(f"Max combination size == 1: no additional training samples. ")
            return [[_] for _ in target_columns]
        if max_comb_size >= len(target_columns):
            raise ValueError(
                f"Max combination size must be smaller than the number of target columns. "
                f"\nmax_comb_size={max_comb_size} - len(target_columns)={len(target_columns)}"
            )
        if max_comb_size == len(target_columns) - 1:
            max_num_comb = len(target_columns)
        comb_dict = {l: [] for l in range(max_comb_size, min_comb_size - 1, -1)}
        total_combs = 0
        for comb_len in range(max_comb_size, min_comb_size - 1, -1):
            for comb in combinations(target_columns, comb_len):
                comb_dict[comb_len].append(comb)
                total_combs += 1
            random.shuffle(comb_dict[comb_len])
        n_chosen_combs = min(max(max_num_comb, len(target_columns)), total_combs)

        chosen_combs = []
        counter = 0

        def pick_comb(col):
            for idx, c in enumerate(cl):
                if col in c:
                    yield cl.pop(idx)

        cl = comb_dict[max_comb_size]
        if n_chosen_combs == len(cl):
            comb_list = cl
        else:
            comb_list = []
            counter = 0
            while len(comb_list) < n_chosen_combs:
                col = target_columns[counter]
                tgtc = pick_comb(col).__next__()
                comb_list.append(tgtc)
                counter = (counter + 1) % len(target_columns)

        print(f"Generation of combinations of training columns.")
        print(f"Keeping {n_chosen_combs}/{total_combs} combinations.")
        return comb_list

    def get_flat_fds(self, col):
        """

        :param col:
        :return:
        """
        fds = self.fd_dict[col]
        flat_fds = [l for fd in fds for l in fd]
        return flat_fds

    def get_fd_edges(self):
        """
        This function generates edges for each FD.
        It iterates over each column's FD, then creates an edge between each node in the LHS and each node in the RHS.
        Example:
        :return:
        """
        total_fd_num = sum([len(self.fd_dict[col]) for col in self.fd_dict])
        fd_list = [f"fd{fd}" for fd in range(total_fd_num)]

        flat_fds = []
        for rhs, vals in self.fd_dict.items():
            for lhs in vals:
                flat_fds.append((lhs, rhs))

        fd_edges = dict(zip(flat_fds, [[[], []] for fd in flat_fds]))
        for idx, row in tqdm(self.df_missing.iterrows(), total=len(self.df_missing)):
            for col_id in self.fd_dict:
                col = self.idx2col[col_id]
                rhs_cell = self.val2idx[row[col]]
                fds = self.fd_dict[col_id]
                for lhs in fds:
                    this_fd = (lhs, col_id)
                    lhs_cols = [self.idx2col[c] for c in lhs]
                    lhs_cells = [self.val2idx[val] for val in row[lhs_cols]]
                    for cell in lhs_cells:
                        fd_edges[this_fd][0].append(cell)
                        fd_edges[this_fd][1].append(rhs_cell)

        fd_edges_t = {
            ("cell", f"fd{_}", "cell"): [None, None] for _ in range(total_fd_num)
        }
        for idx, fd in enumerate(fd_edges):
            fd_edges_t[("cell", f"fd{idx}", "cell")][0] = torch.tensor(
                fd_edges[fd][0], dtype=torch.int64
            )
            fd_edges_t[("cell", f"fd{idx}", "cell")][1] = torch.tensor(
                fd_edges[fd][1], dtype=torch.int64
            )
        fd_edges_t = {k: tuple(t) for k, t in fd_edges_t.items()}
        return fd_edges_t, fd_list

    # TODO Check where this is supposed to go.
    def get_fd_context(self, v, col):
        """

        :param v:
        :param col:
        :return:
        """
        for col in self.all_columns:
            if col in self.fd_dict:
                flat_fds = self.get_flat_fds(col)
                fd_tuple = [
                    val if idx in flat_fds else self.flag_index
                    for idx, val in enumerate(v)
                ]
            else:
                fd_tuple = [self.flag_index for idx, val in enumerate(v)]
        return v + fd_tuple

    def get_positive_samples(self, source_df):
        """
        This function prepares the training/validation samples based on the content of the source_df.
        :param source_df:
        :return:
        """
        positive_samples = 0
        positive_triplets = 0
        train_positive_samples = []
        train_positive_triplets = []
        boundaries = []

        problems_by_col = {col: 0 for col in source_df.columns}

        c_idx = {v: k for k, v in enumerate(self.df_orig.columns)}
        comb_list = self.get_combinations(
            self.training_columns, self.max_comb_size, self.max_num_comb
        )
        self.num_combs = len(comb_list)

        for iteration, col in tqdm(
            enumerate(self.training_columns), total=len(self.training_columns)
        ):
            boundaries.append(positive_triplets)
            col_num = source_df.columns.to_list().index(col)
            for row_num, row in source_df.iterrows():
                val = source_df.loc[row_num, col]
                val_id = self.val2idx[val]
                # Check if the current value has the same index as the null value, if so, skip this triplet.
                if val_id == self.flag_index:
                    continue
                full_tuple = source_df.loc[row_num].tolist()
                triplet = (row_num, self.col2idx[col], val_id)
                if not self._check_triplet(triplet):
                    # pass
                    raise ValueError
                for comb in comb_list:
                    if col in comb:
                        v = [self.val2idx[v] for v in full_tuple]
                        for comb_c in comb:
                            # if self.flag_col:
                            #     # Adding column vector to fill in for missing value.
                            #     v[c_idx[comb_c]] = self.col2idx[comb_c]
                            # else:
                            #     # Adding "dummy" vector position for later reindexing
                            #     v[c_idx[comb_c]] = self.flag_index
                            v[c_idx[comb_c]] = self.flag_index
                        train_positive_samples.append(v)
                        train_positive_triplets.append(triplet)
                        positive_triplets += 1
                        positive_samples += 1
        # boundaries.append(positive_triplets)

        og_triplets_tensor = torch.tensor(train_positive_triplets, dtype=int)
        og_samples_tensor = torch.tensor(train_positive_samples, dtype=int)

        uniq_concat = torch.unique(
            torch.cat([og_samples_tensor, og_triplets_tensor], dim=1), dim=0
        )
        sorted_uc = uniq_concat[np.argsort(uniq_concat[:, -2], axis=0)]
        uniques, boundaries = np.unique(sorted_uc[:, -2], return_index=True)
        boundaries = np.concatenate([boundaries, [len(sorted_uc)]])
        samples_tensor = sorted_uc[:, :-3]
        samples_size = len(samples_tensor)
        triplets_tensor = sorted_uc[:, -3:]
        return triplets_tensor, samples_tensor, samples_size, boundaries

    def get_samples_for_testing(self):
        positive_samples = 0
        positive_triplets = 0
        test_samples = []
        test_triplets = []
        boundaries = []
        self.impossible = []
        problems_by_col = {col: 0 for col in self.df_missing.columns}

        for iteration, col in enumerate(self.training_columns):
            # for iteration, col in enumerate(self.target_columns):
            boundaries.append(positive_triplets)
            if col not in self.target_columns:
                continue
            col_num = self.df_missing.columns.to_list().index(col)
            for row_num, row in self.df_missing.iterrows():
                # self.train_positive_samples[positive_samples, 0] = row_num
                # col = self.idx2col[target_column]
                val = self.df_missing.loc[row_num, col]
                if (
                    self.df_missing.columns[col_num] in self.target_columns
                    and pd.isnull(self.df_missing.iloc[row_num, col_num]) == True
                ):
                    true_value = self.df_orig.iloc[row_num, col_num]
                    if col in self.numerical_columns:
                        try:
                            prefix, val = true_value.split("_")
                            true_value_id = float(val)
                        except AttributeError:
                            raise AttributeError(
                                f"The Ground Truth column {col} contains missing values in row{row_num}."
                            )
                    else:
                        try:
                            true_value_id = self.val2idx[true_value]
                        except KeyError:
                            self.impossible.append(true_value)
                            # continue

                    full_tuple = self.df_missing.loc[row_num].tolist()
                    v = [self.val2idx[_v] for _v in full_tuple]
                    v[col_num] = self.flag_index
                    triplet = (row_num, self.col2idx[col], true_value_id)
                    test_samples.append(v)
                    test_triplets.append(triplet)
                    positive_triplets += 1
                    positive_samples += 1
        boundaries.append(positive_triplets)

        print(f"{len(self.impossible)} values cannot be imputed.")
        triplets_tensor = torch.tensor(test_triplets, dtype=float)
        samples_tensor = torch.tensor(test_samples, dtype=int)
        samples_size = len(samples_tensor)
        return triplets_tensor, samples_tensor, samples_size, boundaries

    def get_positive_negative_samples_for_testing(self):
        positive_triplet_index = 0
        self.pos_edges_to_remove = torch.zeros(self.num_missing_values * 2, 2).to(int)
        # "Impossible" imputations are values that are only present in df_orig and do not appear in df_missing.
        self.impossible = []

        for row in range(self.num_rows):
            for col in range(self.num_columns):
                if (
                    self.df_missing.columns[col] in self.target_columns
                    and pd.isnull(self.df_missing.iloc[row, col]) == True
                ):
                    col_node_id = self.num_rows + col
                    tgt_val = self.df_orig.iloc[row, col]
                    try:
                        cell_node_id = self.val2idx[tgt_val]
                        self.test_positive_samples.append(
                            (row, col_node_id, cell_node_id)
                        )
                        positive_triplet_index += 1
                    except KeyError:
                        self.impossible.append(tgt_val)
                        # print(f'{tgt_val} cannot be imputed. ')
        print(f"{len(self.impossible)} imputation values cannot be imputed.")

        self.test_positive_samples = torch.tensor(self.test_positive_samples)
        print("positive test pos_triplets", len(self.test_positive_samples))
        self.test_positive_samples_size = len(self.test_positive_samples)

        self.test_positive_tuples = torch.zeros(
            (self.test_positive_samples_size, self.input_tuple_length), dtype=int
        )
        self.test_positive_triplets = torch.IntTensor(
            self.test_positive_samples_size, 3
        )
        # self.test_negative_triplets = torch.IntTensor(self.test_negative_samples_size, 3)

        positive_samples = 0

        for sample in self.test_positive_samples:
            row_num, col_id, val_id = map(torch.Tensor.item, sample)
            self.test_positive_tuples[positive_samples, 0] = row_num
            full_tuple = self.df_missing.iloc[row_num].tolist()
            # full_tuple.remove(self.idx2val[val_id])
            col = self.idx2col[col_id]
            col_num = self.df_orig.columns.to_list().index(col)
            full_tuple[col_num] = "__flag_index__"
            try:
                self.test_positive_tuples[positive_samples, :] = torch.IntTensor(
                    [
                        self.val2idx[v] if v in self.val2idx else self.flag_index
                        for v in full_tuple
                    ]
                )
                self.test_positive_triplets[positive_samples, :] = torch.IntTensor(
                    (row_num, col_id, val_id)
                )
            except KeyError:
                print("Found a problem")
            positive_samples += 1

        self.test_positive_tuples = self.test_positive_tuples.to(int)

    def get_statistics(self):
        statistics = {
            "num_rows": self.num_rows,
            "num_columns": self.num_columns,
            "num_imputation_columns": len(self.target_columns),
            "training_columns": "_".join(self.training_columns),
            "training_rows": self.train_positive_samples_size,
            "num_missing_values": self.num_missing_values,
            "num_distinct_values": len(self.val2idx),
            "num_fds": self.num_fds,
            "fd_strategy": str(self.fd_strategy),
            "comb_num": self.num_combs,
            "comb_size": self.max_comb_size,
            "unidirectional": False,
        }
        if self.ext_features is not None:
            statistics["node_features"] = "external"
        else:
            if self.random_init:
                statistics["node_features"] = ["random_init"]
            else:
                statistics["node_features"] = "fixed_init"
        return statistics

    def get_triplets_in_long_form(self):
        dict_long_form = {"train": [], "valid": [], "test": []}
        for triplet in self.train_positive_triplets:
            row, col, val = list(map(torch.Tensor.item, triplet))
            long_form = (self.idx2node[row], self.idx2col[col], self.idx2val[val])
            dict_long_form["train"].append(long_form)
        for triplet in self.valid_positive_triplets:
            row, col, val = list(map(torch.Tensor.item, triplet))
            long_form = (self.idx2node[row], self.idx2col[col], self.idx2val[val])
            dict_long_form["valid"].append(long_form)
        for triplet in self.test_positive_triplets:
            row, col, val = list(map(torch.Tensor.item, triplet))
            try:
                long_form = (self.idx2node[row], self.idx2col[col], self.idx2val[val])
            except KeyError:
                long_form = (self.idx2node[row], self.idx2col[col], str(val))

            dict_long_form["test"].append(long_form)

        return dict_long_form

    def process(self):
        print("Loading and computing basic stats")
        self.load_and_compute_stats()
        print("Creating graph structure")
        self.create_graph()
        if self.ext_features is not None:
            print("Loading external features")
            self._load_external_features()
        else:
            print("Computing graph features")
            self.compute_node_features()
        print("Creating positive and negative samples for training")
        self.create_train_positive_graph()
        print("Creating positive and negative samples for validation")
        self.create_valid_positive_graph()
        print("Creating positive and negative samples for testing")
        self.create_test_positive_graphs()
        # if False:
        if self.architecture == "multitask":
            self.labels, self.weights = self.generate_labels(
                self.train_positive_triplets
            )
            self.labels_valid, self.weights_valid = self.generate_labels(
                self.valid_positive_triplets
            )
        else:
            raise ValueError(f"Architecture {self.architecture} is not supported.")
