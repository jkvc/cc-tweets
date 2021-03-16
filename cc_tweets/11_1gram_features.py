from collections import defaultdict
from os.path import join

from config import DATA_DIR
from tqdm import tqdm

from cc_tweets.experiment_config import DATASET_NAME, DATASET_PKL_PATH, DATASET_SAVE_DIR
from cc_tweets.feature_utils import save_features
from cc_tweets.utils import load_pkl, read_txt_as_str_list, write_str_list_as_txt

SAVE_DIR = join(DATA_DIR, DATASET_NAME, "1gram_features")
VOCAB_FILE = join(DATASET_SAVE_DIR, "vocab", "lemmas_1gram_300.txt")


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)
    vocab = set(read_txt_as_str_list(VOCAB_FILE))

    unigram2id2count = defaultdict(lambda: defaultdict(int))
    for t in tqdm(tweets):
        id = t["id"]
        for lemma in t["lemmas"]:
            if lemma in vocab:
                unigram2id2count[lemma][id] += 1
    unigram2id2count = dict(unigram2id2count)

    save_features(
        tweets, unigram2id2count, source_name="1gram", save_features_dir=SAVE_DIR
    )
    write_str_list_as_txt(vocab, join(SAVE_DIR, "_names.txt"))
