import argparse
import os
from pathlib import Path


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--datasets_folder", required=True)
    parser.add_argument("--imputed_datasets_name", default="dirty_imputed.csv")


    args = parser.parse_args()



    return args.datasets_folder, args.imputed_datasets_name


def main():
    datasets_folder, imputed_datasets_name = parse_args()

    datasets_folder = Path(datasets_folder)

    dataset_files = datasets_folder.rglob(imputed_datasets_name)
    removed=0
    error=0
    for imputed_file in dataset_files:
        try:
            os.remove(imputed_file)
            print(f"File '{imputed_file}' deleted successfully.")
            removed+=1

        except:
            print(f"Error occurred while removing: {imputed_file}")
            error+=1
    print(f"Removed {removed} files, error {error} times.")



if __name__ == "__main__":
    main()
