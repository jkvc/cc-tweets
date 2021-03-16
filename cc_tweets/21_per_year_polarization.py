from os import makedirs
from os.path import join
from pprint import pprint

from cc_tweets.data_utils import get_tweet_time
from cc_tweets.experiment_config import DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.polarization import calc_dem_rep_polarization
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json

NUM_TRIALS = 10
VOCAB_FILE = join(DATASET_SAVE_DIR, "vocab", "stems_2gram_1000.txt")
SAVE_DIR = join(DATASET_SAVE_DIR, "polarization")


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

    makedirs(SAVE_DIR, exist_ok=True)
    save_json(
        year2avgpol,
        join(SAVE_DIR, "per_year.json"),
    )
