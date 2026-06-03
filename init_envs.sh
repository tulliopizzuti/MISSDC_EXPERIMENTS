#!/bin/bash
export PIP_TIMEOUT=0


#    "CORRUPT_DATASETS"
#    "DATAWIG"
#    "GRIMP"
#    "HOLOCLEAN"
#    "HYPERIMPUTE_GAIN"
#    "MISSDC"
#    "MISSFOREST"
#    "SIMPLEIMPUTER"
#    "TRIARD"

FOLDERS=(
    "CORRUPT_DATASETS"
    "DATAWIG"
    "GRIMP"
    "HOLOCLEAN"
    "HYPERIMPUTE_GAIN"
    "MISSDC"
    "MISSFOREST"
    "SIMPLEIMPUTER"
    "TRIARD"
)

for F in "${FOLDERS[@]}"; do
    echo ""
    echo "========================="
    echo "Creating env in $F"
    echo "========================="

    conda env create -f "$F/environment.yml" -y
    if [ -f "$F/env_post_setup.sh" ]; then
        bash "$F/env_post_setup.sh"
    fi
done

echo "Done"


