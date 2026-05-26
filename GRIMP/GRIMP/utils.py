import itertools
from pathlib import Path


def convert_to_list(item):
    if isinstance(item, dict):
        return {k: convert_to_list(v) for k, v in item.items()}
    elif isinstance(item, list):
        return item
    else:
        return [item]


def get_comb(config_dict):
    keys, values = zip(*config_dict.items())
    run_variants = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return run_variants


def flatten(key, this_dict):
    flattened_dict = {}
    for k, v in this_dict.items():
        if key == "":
            this_key = f"{k}"
        else:
            this_key = f"{key}.{k}"
        if isinstance(v, list):
            flattened_dict.update({this_key: v})
        else:
            flattened_dict.update(flatten(this_key, v))
    return flattened_dict


def pack(dict_to_pack):
    packed = {}
    for key, value in dict_to_pack.items():
        splits = key.split(".")
        n_splits = len(splits)
        if n_splits == 2:
            s0, s1 = splits
            if s0 not in packed:
                packed[s0] = {s1: value}
            else:
                packed[s0][s1] = value
        elif n_splits > 2:
            s0 = splits[0]
            pp = pack(
                {
                    k_.replace(s0 + ".", ""): v_
                    for k_, v_ in dict_to_pack.items()
                    if k_.startswith(s0)
                }
            )
            packed[s0] = pp
        elif n_splits == 1:
            packed[key] = value
        else:
            raise ValueError
    return packed


def prepare_config_dict(base_config):
    converted_ = convert_to_list(base_config)

    flattened_ = flatten("", converted_)
    config_combinations = get_comb(flattened_)

    config_list = [pack(comb) for comb in config_combinations]

    return config_list


def complete_config(config):
    dataset_name = config["ground_truth"]
    clean_dataset = Path(f"data/clean/{dataset_name}.csv")
    config["ground_truth"] = clean_dataset
    error_fraction = config["error_fraction"]
    dirty_dataset = Path(
        "data/dirty/", f"{dataset_name}_allcolumns_{error_fraction}.csv"
    )
    config["dirty_dataset"] = dirty_dataset
    config["text_embs"] = [
        f"data/pretrained-emb/{dataset_name}_{error_fraction}_{config['emb']}.emb"
    ]
    config["imputed_df_tag"] = config["emb"]

    if "training_columns" not in config:
        config["training_columns"] = None

    if "ignore_columns" not in config:
        config["ignore_columns"] = None

    if "ignore_num_cols" not in config:
        config["ignore_num_cols"] = None

    if "cat_columns" not in config:
        config["cat_columns"] = None

    if "target_columns" not in config:
        config["target_columns"] = None

    if "fd_strategy" not in config:
        config["fd_strategy"] = "attention"

    return config
