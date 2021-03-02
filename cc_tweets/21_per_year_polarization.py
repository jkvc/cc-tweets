import random
from os.path import join
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.data_utils import get_tweet_time
from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.polarization import calc_dem_rep_polarization
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json

NUM_TRIALS = 10
VOCAB_FILE = join(DATASET_SAVE_DIR, "vocab_stems_2gram.txt")

NUM_TRIALS = 10

YEARS = [2017, 2018, 2019]

if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)
    vocab2idx = {gram: i for i, gram in enumerate(read_txt_as_str_list(VOCAB_FILE))}

    year2avgpol = {}
    for year in YEARS:
        tweets_in_year = [t for t in tweets if get_tweet_time(t).year == year]
        avgpol = calc_dem_rep_polarization(tweets_in_year, vocab2idx, NUM_TRIALS)
        year2avgpol[year] = avgpol
        print(">> avgpol", avgpol)
    pprint(year2avgpol)

    plt.plot(year2avgpol.keys(), year2avgpol.values())
    plt.show()

    save_json(year2avgpol, join(DATASET_SAVE_DIR, "21_per_year_polarzation.json"))
