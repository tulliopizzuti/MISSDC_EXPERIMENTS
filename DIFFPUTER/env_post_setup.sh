conda run -n MISSDC_EXP_DIFFPUTER pip uninstall torch torchvision torchaudio -y
conda run -n MISSDC_EXP_DIFFPUTER pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

