from os.path import join

from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.misc import AFFECT_IGNORE_STEMS
from cc_tweets.utils import load_pkl, read_txt_as_str_list
from config import RESOURCES_DIR
from cc_tweets.experiment_configs import SUBSET_PKL_PATH
from nltk.stem.snowball import SnowballStemmer

SUBJECTIVITY_LEXICON_PATH = join(
    RESOURCES_DIR, "subjectivity_clues_hltemnlp05", "subjclueslen1-HLTEMNLP05.tff"
)
SUBJ_LEVELS = ["weak", "strong", "combined"]


stemmer = SnowballStemmer("english")

lines = read_txt_as_str_list(SUBJECTIVITY_LEXICON_PATH)
STRONG_SUBJ_STEMS = set()
WEAK_SUBJ_STEMS = set()
for line in lines:
    segs = line.split(" ")
    word = segs[2].split("=")[1]
    is_stemmed = segs[3].split("=")[1] == "y"
    if not is_stemmed:
        word = stemmer.stem(word)
    type = segs[0].split("=")[1]
    if type == "weaksubj":
        WEAK_SUBJ_STEMS.add(word)
    elif type == "strongsubj":
        STRONG_SUBJ_STEMS.add(word)
    else:
        raise ValueError()


def get_feature_val(subj_level, tweet):
    v = 0
    for stem in tweet["stems"]:
        if subj_level == "weak":
            if stem in WEAK_SUBJ_STEMS:
                v += 1
        if subj_level == "strong":
            if stem in STRONG_SUBJ_STEMS:
                v += 1
        if subj_level == "combined":
            if stem in WEAK_SUBJ_STEMS:
                v -= 1
            if stem in STRONG_SUBJ_STEMS:
                v += 1
    return v


def closure(level):
    def _extract(tweets):
        id2v = {}
        for t in tweets:
            id2v[t["id"]] = get_feature_val(level, t)
        return id2v

    return _extract


for level in SUBJ_LEVELS:

    register_feature(Feature(f"subj.{level}", closure(level)))
