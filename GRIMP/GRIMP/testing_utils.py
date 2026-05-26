import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from GRIMP.multitask_predictor import MultiTaskPredictor
from GRIMP.gnn_models import HeteroGraphSAGE


def generate_imputed_dataset(graph_dataset, gnn_model, link_predictor):
    """

    :param graph_dataset:
    :param gnn_model:
    :param link_predictor:
    :return:
    """
    h = gnn_model(graph_dataset.graph, graph_dataset.graph.ndata["features"])

    df_imputed = graph_dataset.df_missing.copy(deep=True)
    df_orig = graph_dataset.df_orig.copy(deep=True)
    for idx_sample, sample in enumerate(graph_dataset.test_positive_samples):
        start, end = graph_dataset.test_pos_neg_matches[idx_sample]
        if graph_dataset.graph_name == "nn":
            tuple_sample = graph_dataset.test_positive_tuples[idx_sample]
            test_neg_tuples = graph_dataset.test_negative_tuples[start:end, :]
            test_neg_triplets = graph_dataset.test_negative_samples[start:end, :]
            test_vector = torch.cat([test_neg_tuples, tuple_sample.reshape(1, -1)])
            triplet_vector = torch.cat([test_neg_triplets, sample.unsqueeze(0)])
        elif graph_dataset.graph_name == "avg":
            tuple_sample = graph_dataset.test_positive_tuples[idx_sample]
            test_neg_triplets = graph_dataset.test_negative_samples[start:end, :]
            test_neg_tuples = tuple_sample.repeat(len(test_neg_triplets), 1)
            test_vector = torch.cat([test_neg_tuples, tuple_sample.reshape(1, -1)])
            triplet_vector = torch.cat([test_neg_triplets, sample.unsqueeze(0)])
        elif graph_dataset.graph_name == "multilabel":
            test_vector = graph_dataset.test_positive_tuples[idx_sample].unsqueeze(0)
            triplet_vector = graph_dataset.test_positive_triplets[idx_sample]
            # tuple_sample = graph_dataset.test_positive_tuples[idx_sample]

        else:
            test_neg_triplets = graph_dataset.test_negative_samples[start:end, :]
            test_vector = torch.cat([test_neg_triplets, sample.reshape(1, -1)])
            if graph_dataset.graph_name == "edge":
                (
                    target_row,
                    target_col,
                    target_val,
                ) = graph_dataset.test_positive_triplets[idx_sample]
        if graph_dataset.graph_name == "nn":
            result = link_predictor.evaluate(h, test_vector)
        elif graph_dataset.graph_name == "avg":
            result = link_predictor(h, test_vector, triplet_vector)
        elif graph_dataset.graph_name == "multilabel":
            result = link_predictor(h, test_vector).reshape(-1, 1)
        else:
            result = link_predictor(h, test_vector)
        # mlp
        if len(result.shape) == 1 or result.shape[1] == 1:
            chosen_value = torch.argmax(result).item()
        # truefalse
        else:
            result = F.softmax(result, dim=1)
            chosen_value = torch.argmax(result[:, 0]).item()

        # if graph_dataset.graph_name == 'nn' or graph_dataset.graph_name == 'avg':
        if graph_dataset.graph_name in ["nn", "avg"]:
            imputed_row, imputed_col, imputed_cell = triplet_vector[chosen_value]
        elif graph_dataset.graph_name == "multilabel":
            imputed_row, imputed_col, imputed_cell = triplet_vector
        else:
            if graph_dataset.graph_name == "triplet":
                imputed_row, imputed_col, imputed_cell = test_vector[chosen_value]
            elif graph_dataset.graph_name == "edge":
                imputed_col = torch.IntTensor([target_col])
                imputed_cell, imputed_row = test_vector[chosen_value]
        imputed_col_value = graph_dataset.idx2col[imputed_col.item()]
        if graph_dataset.graph_name != "multilabel":
            imputed_cell_value = graph_dataset.idx2val[imputed_cell.item()]
        else:
            # imputed_cell_value = graph_dataset.attribute_domain_cell_id_dict[graph_dataset.target_column][chosen_value]
            imputed_cell_value = graph_dataset.idx2val[
                chosen_value + graph_dataset.num_row_col_nodes
            ]
            # imputed_cell_value = graph_dataset.idx2val[imputed_cell_value]

        df_imputed.loc[imputed_row.item(), imputed_col_value] = imputed_cell_value

        true_value = df_orig.loc[imputed_row.item(), imputed_col_value]
        # if true_value == imputed_cell_value:
        #     print(f'{true_value} == {imputed_cell_value}')
        # else:
        #     print(f'{true_value} != {imputed_cell_value}')

    return df_imputed


def generate_imputed_dataset_multilabel(graph_dataset, gnn_model, link_predictor):
    # with torch.no_grad():
    h = gnn_model(graph_dataset.graph, graph_dataset.graph.ndata["features"])
    link_predictor.eval()
    df_imputed = graph_dataset.df_missing.copy(deep=True)
    df_orig = graph_dataset.df_orig.copy(deep=True)
    results = np.zeros(
        shape=(
            graph_dataset.test_positive_samples.shape[0],
            len(graph_dataset.distinct_value_set),
        )
    )
    for idx_sample, sample in enumerate(graph_dataset.test_positive_samples):
        test_vector = graph_dataset.test_positive_tuples[idx_sample].unsqueeze(0)
        triplet_vector = graph_dataset.test_positive_triplets[idx_sample]
        imputed_row, imputed_col, imputed_cell = tuple(
            map(torch.Tensor.item, triplet_vector)
        )
        # tuple_sample = graph_dataset.test_positive_tuples[idx_sample]
        result = link_predictor(h, test_vector).reshape(-1, 1)
        results[idx_sample, :] = result.cpu().detach().numpy().squeeze()
        # mlp
        if len(result.shape) == 1 or result.shape[1] == 1:
            chosen_value = torch.argmax(result).item()
        # truefalse
        else:
            result = F.softmax(result, dim=1)
            chosen_value = torch.argmax(result[:, 0]).item()

        imputed_col_value = graph_dataset.idx2col[imputed_col]
        imputed_cell_value = graph_dataset.idx2val[
            chosen_value + graph_dataset.num_row_col_nodes
        ]
        df_imputed.loc[imputed_row, imputed_col_value] = imputed_cell_value
    np.save("results", results)
    return df_imputed


def generate_imputed_dataset_multitask(
    graph_dataset, best_state, init_params, skip_gnn
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not skip_gnn:
        gnn_model = HeteroGraphSAGE(**init_params["gnn_model"]).to(device)
    else:
        gnn_model = torch.nn.Linear(**init_params["gnn_model"]).to(device)

    multitask_model = MultiTaskPredictor(**init_params["multitask_model"]).to(device)

    gnn_model.load_state_dict(best_state["gnn_model"])

    multitask_model.load_state_dict(best_state["multitask_model"])
    gnn_model.eval()
    multitask_model.eval()
    if not skip_gnn:
        h = gnn_model(graph_dataset.graph, graph_dataset.graph.ndata["features"])
    else:
        feats = graph_dataset.graph.ndata["features"]
        ff = feats["cell"].clone().detach().to(device)
        ff[: feats["rid"].shape[0], :] = feats["rid"]
        h = gnn_model(ff)

    df_imputed = graph_dataset.df_missing.copy(deep=True)
    df_orig = graph_dataset.df_orig.copy(deep=True)
    # result = multitask_model(h, graph_dataset.test_positive_samples, graph_dataset)

    multitask_model.eval()
    # result = multitask_model()
    result = multitask_model(
        h,
        graph_dataset.test_positive_samples,
        graph_dataset.boundaries_test,
        step="test",
    )

    for col_idx, col_imputations in result.items():
        iter_col = graph_dataset.map_c2h[col_idx]
        col = graph_dataset.idx2col[col_idx]
        if col_idx in graph_dataset.numerical_columns:
            choices = col_imputations
        else:
            choices = torch.argmax(col_imputations, dim=1)
        l = graph_dataset.boundaries_test[iter_col]
        r = graph_dataset.boundaries_test[iter_col + 1]
        iter_test_triplets = graph_dataset.test_positive_triplets[l:r]
        for idx, triplet_vector in enumerate(iter_test_triplets):
            true_row_idx, true_col_idx, true_cell_idx = tuple(
                map(torch.Tensor.item, triplet_vector)
            )
            if col_idx == true_col_idx:
                if col in graph_dataset.numerical_columns:
                    imputed_cell_value = choices[idx].item()
                else:
                    imputed_cell_value = graph_dataset.idx2val[
                        graph_dataset.attribute_domain_cell_id_dict[true_col_idx][
                            choices[idx].item()
                        ]
                    ]
                df_imputed.loc[true_row_idx, col] = imputed_cell_value

    return df_imputed


def measure_imp_accuracy(graph_dataset, df_imp, logger=None):
    acc_dict = {col: 0 for col in graph_dataset.all_columns}
    acc_type = {col: None for col in graph_dataset.all_columns}
    true_dict = dict()
    missing_dict = dict()
    df_dirty = graph_dataset.df_missing
    df_orig = graph_dataset.df_orig
    target_columns = graph_dataset.target_columns
    for col in target_columns:
        if df_dirty[col].isna().any():
            missing = df_dirty[col].isna()
            test_col = df_imp.loc[missing, col]
            true_col = df_orig.loc[missing, col]

            if col in graph_dataset.numerical_columns:
                true_col = true_col.apply(lambda x: float(x.split("_")[1]))
                denormalized_column = graph_dataset.denormalize_column(col, test_col)
                mse = mean_squared_error(true_col, denormalized_column, squared=False)
                acc_dict[col] = mse
                acc_type[col] = "RMSE"
                true_dict[col] = 0
                missing_dict[col] = sum(missing)
            else:
                try:
                    test_col = test_col.astype("float")
                    true_col = true_col.astype("float")
                    print(
                        f"Column {col} was converted to float before measuring accuracy."
                    )
                except ValueError:
                    pass

                acc = sum(true_col == test_col) / sum(missing)
                true_dict[col] = sum(true_col == test_col)
                missing_dict[col] = sum(missing)
                acc_dict[col] = acc
                acc_type[col] = "Accuracy"

    tot_true = sum(
        [score for col, score in true_dict.items() if acc_type[col] == "Accuracy"]
    )
    tot_missing = sum(
        [score for col, score in missing_dict.items() if acc_type[col] == "Accuracy"]
    )

    header = f'{"Column":^30}{"Measure":^10}{"Score":^16}{"Correct":^8}{"Tot miss":^8}'
    print(header)
    for col in acc_dict:
        if acc_type[col] is None:
            continue
        s = f"{col:^30}{acc_type[col]:^8}{acc_dict[col]:>16.4}{true_dict[col]:^8}{missing_dict[col]:^8}"
        print(s)

    print(f"Correct categorical imputations: {tot_true}")
    print(f"Total missing values: {tot_missing}")
    #print(f"Average imputation accuracy: {tot_true/tot_missing*100:.4f}")

    if logger:
        logger.add_value("results", "accuracy_dict", acc_dict)
        # logger.add_value("results", "imp_accuracy", tot_true / tot_missing)
        logger.add_value("results", "tot_true", tot_true)
        logger.add_value("results", "tot_missing", tot_missing)
        for col in target_columns:
            logger.add_value("results", "col1_imputed", acc_dict[col])

    return acc_dict
