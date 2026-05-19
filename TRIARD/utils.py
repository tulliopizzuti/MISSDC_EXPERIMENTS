import pandas as pd
from typing import List
import numpy as np
import os
import csv


def compute_diff_list(clean_df: pd.DataFrame, dirty_df: pd.DataFrame) -> List[dict]:
    """
    Compare two pandas DataFrames and return a list of dictionaries containing the positions and values where they differ.

    Args:
        clean_df (pd.DataFrame): The clean DataFrame to compare.
        dirty_df (pd.DataFrame): The dirty DataFrame to compare.

    Returns:
        List[dict]: A list of dictionaries containing the positions and values where the DataFrames differ.
    """

    # if not clean_df.columns.equals(dirty_df.columns):
    #     print("The DataFrames have different column names.")
    #     return None

    # reindex the DataFrames to align their columns and index labels
    clean_df = clean_df.reindex(
        columns=dirty_df.columns, index=dirty_df.index)

    # compare the DataFrames and get the positions of different values
    diff = (clean_df != dirty_df)
    diff_pos = diff.stack()[diff.stack()].index.tolist()

    # iterate through the positions where the DataFrames differ
    diff_list = []
    for position in diff_pos:
        clean_val = clean_df.loc[position]
        dirty_val = dirty_df.loc[position]

        # create a dictionary to store the values
        diff_dict = {'position': position,
                     'clean': clean_val, 'dirty': dirty_val}

        diff_list.append(diff_dict)

    return diff_list