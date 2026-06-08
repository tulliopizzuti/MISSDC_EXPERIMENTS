from fasttext.util import download_model
import os.path as osp
import os
import shutil


def prepare_ft_model():
    fname = download_model(lang_id="en", if_exists="ignore")
    os.makedirs("grimpdata/", exist_ok=True)
    new_fname = osp.join("grimpdata", fname)
    shutil.move(fname, new_fname)
    return new_fname


if __name__ == "__main__":
    fname = "grimpdata/cc.en.300.bin"
    if not osp.exists(fname):
        fname = prepare_ft_model()