
for %%F in (
    CORRUPT_DATASETS
    DATAWIG
    DIFFPUTER
    GRIMP
    HOLOCLEAN
    HYPERIMPUTE_GAIN
    MISSDC
    MISSFOREST
    SIMPLEIMPUTER
    TRIARD
) do (
    echo Removing MISSDC_EXP_%%F
    conda env remove -n MISSDC_EXP_%%F -y
)

echo Done
