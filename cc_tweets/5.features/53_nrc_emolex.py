from collections import defaultdict
from os.path import join

from config import RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from cc_tweets.experiment_config import DATASET_PKL_PATH
from cc_tweets.feature_utils import save_features
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import load_pkl, read_txt_as_str_list

EMOLEX_PATH = join(
    RESOURCES_DIR, "NRC-Emotion-Intensity-Lexicon-v1", "OneFilePerEmotion"
)
EMOLEX_EMOS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]


def load_emolex_emo2lemma2score():
    lemmatizer = WordNetLemmatizer()
    emo2lemma2score = defaultdict(dict)
    for emo in EMOLEX_EMOS:
        lines = read_txt_as_str_list(join(EMOLEX_PATH, f"{emo}-scores.txt"))
        for line in lines:
            word, score = line.split("\t")
            lemma = lemmatizer.lemmatize(word)
            score = float(score)
            emo2lemma2score[emo][lemma] = score
    return dict(emo2lemma2score)


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)
    emo2lemma2score = load_emolex_emo2lemma2score()

    name2id2score = defaultdict(lambda: defaultdict(float))
    for tweet in tqdm(tweets):
        for emo in EMOLEX_EMOS:
            name2id2score[emo][tweet["id"]] = 0
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                score = emo2lemma2score[emo].get(lemma, 0)
                name2id2score[emo][tweet["id"]] += score

    name2id2score = {f"emolex_{k}": v for k, v in name2id2score.items()}

    save_features(tweets, name2id2score, "nrc_emolex")
