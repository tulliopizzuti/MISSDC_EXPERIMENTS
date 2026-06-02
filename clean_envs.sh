#!/bin/bash

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
    echo "Removing $F"
    conda env remove -n MISSDC_EXP_$F -y
done
echo "Done"