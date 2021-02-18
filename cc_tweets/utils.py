import json
import pickle
import shutil
from os import mkdir
from typing import List

from genericpath import exists


def read_txt_as_str_list(filepath: str) -> List[str]:
    with open(filepath) as f:
        ls = f.readlines()
    ls = [l.strip() for l in ls]
    return ls


def write_str_list_as_txt(lst: List[str], filepath: str):
    with open(filepath, "w") as f:
        f.writelines([f"{s}\n" for s in lst])


def save_pkl(obj, save_path: str):
    with open(save_path, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(save_path: str):
    with open(save_path, "rb") as f:
        return pickle.load(f)


def save_json(obj, save_path: str):
    with open(save_path, "w") as f:
        json.dump(obj, f, indent=4)


def load_json(save_path: str):
    with open(save_path, "r") as f:
        return json.load(f)


def load_yaml(save_path: str):
    from yaml import load

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(save_path, "r") as f:
        return load(f, Loader=Loader)


def mkdir_overwrite(path: str):
    if exists(path):
        shutil.rmtree(path)
    mkdir(path)
