from os.path import join

from config import DATA_DIR

DOWNSIZE_FACTOR = 10
FILTER_UNK = True

DATASET_NAME = (
    f"tweets_downsized{DOWNSIZE_FACTOR}{'_filtered' if FILTER_UNK else '_unfiltered'}"
)
DATASET_PKL_PATH = join(DATA_DIR, f"{DATASET_NAME}.pkl")
DATASET_SAVE_DIR = join(DATA_DIR, DATASET_NAME)
