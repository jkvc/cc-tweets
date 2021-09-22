from os.path import join

from config import WORKING_DIR

DATA_SUBSET_SIZE = 20000
FILTER_UNK = True

SUBSET_NAME = f"subset.{DATA_SUBSET_SIZE}.{'filtered' if FILTER_UNK else 'unfiltered'}"
SUBSET_WORKING_DIR = join(WORKING_DIR, SUBSET_NAME)
SUBSET_PKL_PATH = join(SUBSET_WORKING_DIR, "tweets.pkl")

EMB_DIM = 50
