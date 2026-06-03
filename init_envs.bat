set PIP_TIMEOUT=0

call conda env create -f environment.yml -y
for %%F in (
    CORRUPT_DATASETS
    DATAWIG
    GRIMP
    HOLOCLEAN
    HYPERIMPUTE_GAIN
    MISSDC
    MISSFOREST
    SIMPLEIMPUTER
    TRIARD
) do (
    echo.
    echo =========================
    echo Creating env in %%F
    echo =========================

    conda env create -f %%F\environment.yml -y
    if exist "%%F\env_post_setup.bat" (
        call %%F\env_post_setup.bat
    )
)
echo Done