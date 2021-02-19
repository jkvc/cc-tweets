import random
from os.path import join

import numpy as np
from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.polarization import calc_dem_rep_polarization
from cc_tweets.utils import get_ngrams, load_pkl, read_txt_as_str_list

SRC_DATASET_NAME = "tweets_downsized100_filtered"
PKL_PATH = join(DATA_DIR, f"{SRC_DATASET_NAME}.pkl")

VOCAB_FILE = join(DATA_DIR, "vocab_3000_2gram.txt")
NUM_TRIALS = 10


if __name__ == "__main__":
    tweets = load_pkl(PKL_PATH)
    vocab2idx = {gram: i for i, gram in enumerate(read_txt_as_str_list(VOCAB_FILE))}
    avgpol = calc_dem_rep_polarization(tweets, vocab2idx, NUM_TRIALS)
    print(">> avgpol", avgpol)
