from collections import defaultdict
from os.path import join
from pprint import pprint

from cc_tweets.experiment_configs import SUBSET_PKL_PATH
from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, get_feature, register_feature
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS
from cc_tweets.utils import load_pkl, read_txt_as_str_list
from config import RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

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


def extractor_closure(emo):
    def _extract(tweets):

        emo2lemma2score = load_emolex_emo2lemma2score()
        id2score = defaultdict(float)

        for tweet in tweets:
            already_added = set()
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                score = emo2lemma2score[emo].get(lemma, 0)
                if lemma not in already_added:
                    id2score[tweet["id"]] += score
                    already_added.add(lemma)
        return id2score

    return _extract


for emo in EMOLEX_EMOS:
    register_feature(Feature(f"emo.{emo}", extractor_closure(emo)))


if __name__ == "__main__":
    feature_names = [
        "emo.anger",
        "emo.anticipation",
        "emo.disgust",
        "emo.fear",
        "emo.joy",
        "emo.sadness",
        "emo.surprise",
        "emo.trust",
    ]

    tweets = load_pkl(SUBSET_PKL_PATH)

    for name in feature_names:
        feature_dict = get_feature(name).get_feature_dict(tweets)
        dem_text0val = [
            (t["text"], feature_dict[t["id"]]) for t in tweets if t["stance"] == "dem"
        ]
        rep_text0val = [
            (t["text"], feature_dict[t["id"]]) for t in tweets if t["stance"] == "rep"
        ]
        print("\n\n")
        print(name)
        print("--dem")
        for t in sorted(dem_text0val, key=lambda x: x[1], reverse=True)[:5]:
            print(t)
        print("--rep")
        for t in sorted(rep_text0val, key=lambda x: x[1], reverse=True)[:5]:
            print(t)
