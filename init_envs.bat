set PIP_TIMEOUT=0

@REM     CORRUPT_DATASETS
@REM     DATAWIG
@REM     GRIMP
@REM     HOLOCLEAN
@REM     HYPERIMPUTE_GAIN
@REM     MISSDC
@REM     MISSFOREST
@REM     SIMPLEIMPUTER
@REM     TRIARD


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