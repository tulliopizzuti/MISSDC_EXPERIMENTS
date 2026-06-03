import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import *

IMPUTATION_SCRIPT = "missdc_hybrid.jar"

def missdc_imputation(
    dirty_file,
        mode, java_opts):
    imputation_cmd = [
        "java",
        *java_opts,
        "-jar",
        IMPUTATION_SCRIPT,
        dirty_file,
        "--mode",mode
    ]
    imputation_returncode, imputation_time, _, _ = run_command(imputation_cmd)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='MISSDC_Exp',
        description='Run imputation using MISSDC')
    parser.add_argument('dataset_file_path', help="Path to the dataset file")
    parser.add_argument("--mode", default="dc", choices=["dc", "hybrid"], type=str)
    parser.add_argument("--java-opts", nargs="*", default=[])

    args = parser.parse_args()
    dataset_file_path = args.dataset_file_path
    mode = args.mode
    java_opts = args.java_opts

    missdc_imputation(dataset_file_path, mode, java_opts)

    print("Imputation completed.")
