import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def impossible_values(ground_truth, target_df, target_columns):
    impossible = {k: None for k in target_columns}
    affected_rows = {k: 0 for k in target_columns}
    for col in target_columns:
        unq_dirty = target_df[col].unique()
        unq_clean = ground_truth[col].unique()
        impossible[col] = [_ for _ in unq_clean if _ not in unq_dirty]
        if len(impossible[col]) > 0:
            print("\n".join(impossible[col]))
        affected_rows[col] = len(
            ground_truth[col].loc[ground_truth[col].isin(impossible[col])]
        )
        print(f"Column {col} has affected rows = {affected_rows[col]}")

    return impossible, affected_rows


def incorrect_imputations(df_imp, df_orig, target_columns, nulls):
    correct_imputations = {k: None for k in target_columns}
    incorrect_imputations = {k: None for k in target_columns}
    stat_df = {k: None for k in target_columns}
    for col in correct_imputations:
        imputations = df_imp[col].loc[nulls[col]]
        v_correct = df_imp[col].loc[nulls[col]] == df_orig.loc[nulls[col], col]
        v_incorrect = df_imp[col].loc[nulls[col]] != df_orig.loc[nulls[col], col]

        correct_imputations[col] = imputations[v_correct].index
        incorrect_imputations[col] = imputations[v_incorrect].index
        count_clean = df_orig.loc[v_correct.index].value_counts(col)
        count_dirty = df_orig.iloc[incorrect_imputations[col]].value_counts(col)
        frac_wrong = count_dirty / count_clean
        dd = pd.DataFrame(columns=["total", "wrong", "ratio"])
        dd["total"] = count_clean
        dd["wrong"] = count_dirty
        dd["ratio"] = frac_wrong
        print(dd)
        stat_df[col] = dd
    return stat_df
