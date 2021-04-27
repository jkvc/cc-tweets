from os.path import join

from config import WORKING_DIR

DOWNSIZE_FACTOR = 100
FILTER_UNK = True

SUBSET_NAME = f"downsized{DOWNSIZE_FACTOR}.{'filtered' if FILTER_UNK else 'unfiltered'}"
SUBSET_WORKING_DIR = join(WORKING_DIR, SUBSET_NAME)
SUBSET_PKL_PATH = join(SUBSET_WORKING_DIR, "tweets.pkl")

EMB_DIM = 50
