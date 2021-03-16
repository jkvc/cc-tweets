from os import makedirs
from os.path import join

from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.polarization import calc_dem_rep_polarization
from cc_tweets.utils import load_pkl, mkdir_overwrite, read_txt_as_str_list, save_json

NUM_TRIALS = 10
VOCAB_FILE = join(DATASET_SAVE_DIR, "vocab", "stems_2gram_1000.txt")
SAVE_DIR = join(DATASET_SAVE_DIR, "polarization")

if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)
    vocab2idx = {gram: i for i, gram in enumerate(read_txt_as_str_list(VOCAB_FILE))}
    avgpol = calc_dem_rep_polarization(tweets, vocab2idx, NUM_TRIALS)
    print(">> avgpol", avgpol)

    makedirs(SAVE_DIR, exist_ok=True)
    save_json(
        {"polarization": avgpol},
        join(SAVE_DIR, "overall.json"),
    )
