import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import *

IMPUTATION_SCRIPT = "missdcv2.jar"

def missdc_imputation(
    dirty_file,
        mode, voting, min_evidence_multiplicity, dc_max_size,  java_opts):
    imputation_cmd = [
        "java",
        *java_opts,
        "-jar",
        IMPUTATION_SCRIPT,
        dirty_file
    ]
    if mode is not None:
        imputation_cmd.append("--mode")
        imputation_cmd.append(mode)
    if voting is not None:
        imputation_cmd.append("--voting")
        imputation_cmd.append(voting)
    if min_evidence_multiplicity is not None:
        imputation_cmd.append("--min-evidence-multiplicity")
        imputation_cmd.append(min_evidence_multiplicity)
    if dc_max_size is not None:
        imputation_cmd.append("--dc-size-max")
        imputation_cmd.append(dc_max_size)
    imputation_returncode, imputation_time, _, _ = run_command(imputation_cmd)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='MISSDC_Exp',
        description='Run imputation using MISSDC')
    parser.add_argument('dataset_file_path', help="Path to the dataset file")
    parser.add_argument("--mode", choices=["dc","approx","dc-ml","approx-ml","ml"], type=str)
    parser.add_argument("--voting", choices=["current","tuple","dc"], type=str)
    parser.add_argument("--min_evidence_multiplicity", type=str)
    parser.add_argument("--dc_size_max", type=str)
    parser.add_argument("--java-opts", nargs="*", default=[])

    args = parser.parse_args()
    dataset_file_path = args.dataset_file_path
    mode = args.mode
    voting = args.voting
    min_evidence_multiplicity = args.min_evidence_multiplicity
    dc_max_size = args.dc_size_max
    java_opts = args.java_opts

    missdc_imputation(dataset_file_path, mode, voting, min_evidence_multiplicity, dc_max_size, java_opts)

    print("Imputation completed.")
