from os.path import join

from config import WORKING_DIR

DATA_SUBSET_SIZE = 1000000
# DATA_SUBSET_SIZE = 25159799  # all retrived tweets

FILTER_UNK = True

SUBSET_NAME = f"subset.{DATA_SUBSET_SIZE}.{'filtered' if FILTER_UNK else 'unfiltered'}"
SUBSET_WORKING_DIR = join(WORKING_DIR, SUBSET_NAME)
SUBSET_PKL_PATH = join(SUBSET_WORKING_DIR, "tweets.pkl")

EMB_DIM = 50
