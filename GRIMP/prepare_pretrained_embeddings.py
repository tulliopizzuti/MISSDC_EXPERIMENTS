"""This script prepares pre-trained embeddings for all nodes in the graph
(rows, values, column edges) starting from the fasttext pre-trained embeddings.
The embeddings are produced by using the fasttext get_sentence_vector
function.

Author: Riccardo Cappuzzo
"""
import pandas as pd
import fasttext
import numpy as np
from tqdm import tqdm
import os.path as osp
import os
from fasttext.util import download_model
import shutil
import sys


def prepare_ft_model():
    fname = download_model(lang_id="en", if_exists="ignore")
    os.makedirs("data/", exist_ok=True)
    new_fname = osp.join("data", fname)
    shutil.move(fname, new_fname)
    return new_fname


def generate(df, model, n_dim=300):
    """Given the target dataframe and a pre-trained model, produce embeddings for
    all the entities in the dataframe.

    Args:
        df (pd.DataFrame): Target dataframe to convert.
        model (_type_): Fasttext pre-trained model.
        n_dim (int, optional): Number of dimensions to use for the embeddings. Defaults to 300.
    """

    # Replace '-' with spaces for sentence emb generation
    for col in df.columns:
        df[col] = df[col].str.replace("-", " ")

    # Disambiguate values in the dataframe by prefixing them with the column name.
    for idx, col in enumerate(df.columns):
        try:
            df[col] = df[col].astype(float).round(8)
            df[col] = df[col].apply(lambda x: f"c{idx}_{x}")
        except ValueError:
            df[col] = df[col].apply(lambda x: f"c{idx}_{x}")

    # Extract all unique values
    unique_values = [str(_) for _ in set(df.values.ravel())]

    # Generate sentence vectors for all unique values
    print("Generating token embeddings. ")
    val_vectors = []
    missing_vals = []
    for idx, val in enumerate(unique_values):
        # Remove the prefix and generate the vector.
        prefix, true_val = val.split("_", maxsplit=1)
        # Ignore null values.
        if true_val == "nan" or true_val != true_val:
            vector = np.zeros(n_dim)
        else:
            vector = model.get_sentence_vector(str(true_val))
        val_vectors.append(vector)

    # Prepare dict with unique value + sentence vector for that value
    vector_dict = dict(zip(unique_values, val_vectors))
    if "nan" in vector_dict:
        vector_dict.pop("nan")

    vector_dict["nan"] = np.zeros(n_dim)

    print("Generating row embeddings.")
    tot_rows = len(vector_dict) + df.shape[0] + df.shape[1]
    row_vectors = dict()

    for idx, row in df.iterrows():
        tmp_vec = np.zeros(shape=(df.shape[1], n_dim))
        for row_id, word in enumerate(row):
            word = str(word)
            vector = vector_dict[word]
            tmp_vec[row_id] = vector
        row_vectors[idx] = np.mean(tmp_vec, 0)

    print("Generating column embeddings.")
    col_vectors = dict()

    for idx, col in enumerate(df.columns):
        tmp_vec = np.zeros(shape=(df.shape[0], n_dim))
        for row_id, word in enumerate(df[col]):
            vector = vector_dict[str(word)]
            tmp_vec[row_id] = vector
        col_vectors[col] = np.mean(tmp_vec, 0)

    print("Writing embeddings on file. ")
    with open(generated_emb_file, "w", encoding="str") as fp:
        tot_rows = len(row_vectors) + len(col_vectors) + len(vector_dict)
        fp.write(f"{tot_rows} {n_dim}\n")
        for k, vec in tqdm(row_vectors.items(), total=len(row_vectors)):
            s = f"idx__{k} " + " ".join([str(_).strip() for _ in vec]) + "\n"
            fp.write(s)
        for k, vec in tqdm(col_vectors.items(), total=len(col_vectors)):
            s = (
                f'cid__{k.replace(" ", "-")} '
                + " ".join([str(_).strip() for _ in vec])
                + "\n"
            )
            fp.write(s)
        for k, vec in tqdm(vector_dict.items(), total=len(vector_dict)):
            if k == "nan":
                continue
            s = (
                f'tt__{k.replace(" ", "-")} '
                + " ".join([str(_).strip() for _ in vec])
                + "\n"
            )
            fp.write(s)


if __name__ == "__main__":
    fname = "data/cc.en.300.bin"

    # Fetching the model if it is not currently available.
    if not osp.exists(fname):
        print(f"Pre-trained model not found on path {fname}.")
        reply = input("Do you want to download the model? y/[n] : ")
        if reply.lower() == "y":
            fname = prepare_ft_model()
        elif reply == "" or reply.lower() == "n":
            print("Quitting.")
            sys.exit()
        else:
            raise ValueError(f"Unrecognized option {reply}.")

    # Load fasttext model once for all datasets.
    print("Loading fasttext model...")
    model = fasttext.load_model(fname)
    # model=None
    print("Model loaded.")

    # I convert all files present in data/to_pretrain
    for data in os.listdir("data/to_pretrain"):
        df_path = f"data/to_pretrain/{data}"
        basename, ext = osp.splitext(data)
        generated_emb_file = f"data/pretrained-emb/{basename}_ft.emb"
        # Read dirty dataset
        df = pd.read_csv(df_path, dtype="str")

        print(f"Working on dataset {data}.")
        generate(df, model)
