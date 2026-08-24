import argparse
import random
from pathlib import Path

import pandas as pd
from jenga.corruptions.generic import MissingValues


def load_clean_dataset(dataset_dir):
    clean_file = dataset_dir / "clean.csv"
    if not clean_file.is_file():
        raise FileNotFoundError(f"clean.csv not found in {dataset_dir}")
    return pd.read_csv(clean_file, sep=",").dropna()


def iter_dataset_dirs(datasets_folder, dataset_names=None):
    dirs = sorted(
        d for d in Path(datasets_folder).iterdir()
        if d.is_dir() and (d / "clean.csv").is_file()
    )
    if dataset_names:
        dirs = [d for d in dirs if d.name in dataset_names]
    return dirs


def corrupt_dataframe(clean_df, columns, missingness, fraction):
    dirty_df = clean_df.copy(deep=True)
    for col in columns:
        dirty_df = MissingValues(
            col, fraction, missingness=missingness, na_value=None
        ).transform(dirty_df)
    return dirty_df


def save_pair(output_folder, clean_df, dirty_df):
    output_folder.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_folder / "clean.csv", index=False)
    dirty_df.to_csv(output_folder / "dirty.csv", index=False)
    num_missing = dirty_df.isna().sum().sum()
    print(f"  -> {output_folder} (missing values: {num_missing})")


def generate_row_scalability(datasets_folder, output_root, missingness_list, ratios,
                              repetitions, row_fractions, dataset_names=None):
    for dataset_dir in iter_dataset_dirs(datasets_folder, dataset_names):
        dataset_name = dataset_dir.name
        clean_df = load_clean_dataset(dataset_dir)
        total_rows = len(clean_df)
        columns = clean_df.columns.tolist()

        print(f"\n[row scalability] dataset={dataset_name} total_rows={total_rows}")

        for fraction in row_fractions:
            n_rows = max(1, int(round(total_rows * fraction)))
            # keep original row order, take the first n_rows rows
            subset_df = clean_df.iloc[:n_rows].reset_index(drop=True)
            pct = int(round(fraction * 100))

            for missingness in missingness_list:
                for ratio in ratios:
                    for repetition in range(repetitions):
                        dirty_df = corrupt_dataframe(subset_df, columns, missingness, ratio)
                        output_folder = (
                            Path(output_root) / "row_scalability" / dataset_name /
                            f"{pct}pct" / missingness / str(ratio) / str(repetition + 1)
                        )
                        save_pair(output_folder, subset_df, dirty_df)


def generate_column_scalability(datasets_folder, output_root, missingness_list, ratios,
                                 repetitions, min_columns, col_step, col_repetitions,
                                 dataset_names=None, n_rows=10_000):
    for dataset_dir in iter_dataset_dirs(datasets_folder, dataset_names):
        dataset_name = dataset_dir.name
        clean_df = load_clean_dataset(dataset_dir)
        if n_rows is not None:
            # keep original row order, take the first n_rows rows
            clean_df = clean_df.iloc[:n_rows].reset_index(drop=True)
        all_columns = clean_df.columns.tolist()
        total_columns = len(all_columns)

        if total_columns < min_columns:
            print(f"\n[column scalability] Skipping {dataset_name}: only {total_columns} "
                  f"columns available (min_columns={min_columns})")
            continue

        print(f"\n[column scalability] dataset={dataset_name} total_columns={total_columns}")

        for ncols in range(min_columns, total_columns + 1, col_step):
            # multiple random column subsets per ncols to accommodate randomness
            for col_rep in range(col_repetitions):
                random_cols = random.sample(all_columns, ncols)
                subset_df = clean_df[random_cols].reset_index(drop=True)

                for missingness in missingness_list:
                    for ratio in ratios:
                        for repetition in range(repetitions):
                            dirty_df = corrupt_dataframe(subset_df, random_cols, missingness, ratio)
                            output_folder = (
                                Path(output_root) / "column_scalability" / dataset_name /
                                f"{ncols}cols" / str(col_rep + 1) / missingness / str(ratio) /
                                str(repetition + 1)
                            )
                            save_pair(output_folder, subset_df, dirty_df)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate row/column scalability datasets from datasets_large, "
                    "corrupted with jenga missingness mechanisms."
    )

    parser.add_argument("--datasets_folder", default="datasets_large",
                        help="Folder containing dataset subfolders, each with a clean.csv")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Restrict generation to these dataset names (default: all found)")
    parser.add_argument("--output", default="data_scalability",
                        help="Output root folder")

    parser.add_argument("--mode", nargs="+", choices=["row", "column"], default=["row", "column"],
                        help="Which scalability experiment(s) to generate")

    parser.add_argument("--missingness", nargs="+", choices=["MCAR", "MAR", "MNAR"], default=["MCAR"],
                        help="List of missingness mechanisms")
    parser.add_argument("--ratio", nargs="+", type=float, default=[0.1],
                        help="List of missingness ratios")
    parser.add_argument("--repetitions", type=int, default=5,
                        help="Number of corruption repetitions per configuration")

    parser.add_argument("--row_fractions", nargs="+", type=float,
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="Row fractions to generate for row scalability (rows kept in original order)")

    parser.add_argument("--min_columns", type=int, default=7,
                        help="Minimum number of columns to start column scalability from")
    parser.add_argument("--col_step", type=int, default=1,
                        help="Step between successive column counts")
    parser.add_argument("--col_repetitions", type=int, default=10,
                        help="Number of random column subsets to generate for each column count")
    parser.add_argument("--col_n_rows", type=int, default=10_000,
                        help="Number of rows (first rows, original order) to use for column "
                             "scalability. Use -1 to keep all rows")

    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducible column sampling")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    if "row" in args.mode:
        generate_row_scalability(
            datasets_folder=args.datasets_folder,
            output_root=args.output,
            missingness_list=args.missingness,
            ratios=args.ratio,
            repetitions=args.repetitions,
            row_fractions=args.row_fractions,
            dataset_names=args.datasets,
        )

    if "column" in args.mode:
        generate_column_scalability(
            datasets_folder=args.datasets_folder,
            output_root=args.output,
            missingness_list=args.missingness,
            ratios=args.ratio,
            repetitions=args.repetitions,
            min_columns=args.min_columns,
            col_step=args.col_step,
            col_repetitions=args.col_repetitions,
            dataset_names=args.datasets,
            n_rows=None if args.col_n_rows == -1 else args.col_n_rows,
        )


if __name__ == "__main__":
    main()
