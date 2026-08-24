import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils import *

IMPUTATION_SCRIPT = "missdc_evimultiplicity.jar"

def missdc_imputation(
    dirty_file,
        mode, evidence_multiplicity, java_opts):
    imputation_cmd = [
        "java",
        *java_opts,
        "-jar",
        IMPUTATION_SCRIPT,
        dirty_file,
    ]
    if evidence_multiplicity is not None:
        imputation_cmd.append("--min-evidence-multiplicity")
        imputation_cmd.append(evidence_multiplicity)
    imputation_returncode, imputation_time, _, _ = run_command(imputation_cmd)

def int_or_none(value):
    if value.lower() == "none":
        return None
    return int(value)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='MISSDCEVIDENCE_Exp',
        description='Run imputation using MISSDCEVIDENCE')
    parser.add_argument('dataset_file_path', help="Path to the dataset file")
    parser.add_argument("--mode", default="dc", choices=["dc", "dc-ml"], type=str)
    parser.add_argument("--min_evidence_multiplicity", default=None, type=int_or_none)
    parser.add_argument("--java-opts", nargs="*", default=[])

    args = parser.parse_args()
    dataset_file_path = args.dataset_file_path
    mode = args.mode
    java_opts = args.java_opts
    min_evidence_multiplicity = args.min_evidence_multiplicity

    missdc_imputation(dataset_file_path, mode, min_evidence_multiplicity, java_opts)

    print("Imputation completed.")
