from collections import defaultdict
from os.path import join

from cc_tweets.experiment_configs import SUBSET_PKL_PATH
from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, register_feature
from cc_tweets.misc import AFFECT_IGNORE_STEMS
from cc_tweets.utils import load_pkl, read_txt_as_str_list, unzip
from config import RESOURCES_DIR
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

SUBJECTIVITY_LEXICON_PATH = join(
    RESOURCES_DIR, "subjectivity_clues_hltemnlp05", "subjclueslen1-HLTEMNLP05.tff"
)
SUBJ_LEVELS = ["weak", "strong", "combined"]

lemmatizer = WordNetLemmatizer()


lines = read_txt_as_str_list(SUBJECTIVITY_LEXICON_PATH)
STRONG_SUBJ_LEMMAS = set()
WEAK_SUBJ_LEMMAS = set()
for line in lines:
    segs = line.split(" ")
    word = segs[2].split("=")[1]
    isadj = segs[3].split("=")[1] == "adj"
    if not isadj:
        continue
    type = segs[0].split("=")[1]
    if type == "weaksubj":
        WEAK_SUBJ_LEMMAS.add(word)
    elif type == "strongsubj":
        STRONG_SUBJ_LEMMAS.add(word)
    else:
        raise ValueError()


def get_feature_val(subj_level, tweet):
    v = 0
    found_words = set()
    for lemma in tweet["lemmas"]:
        if lemma in found_words:
            continue
        if subj_level == "weak":
            if lemma in WEAK_SUBJ_LEMMAS:
                v += 1
        if subj_level == "strong":
            if lemma in STRONG_SUBJ_LEMMAS:
                v += 1
        if subj_level == "combined":
            if lemma in WEAK_SUBJ_LEMMAS:
                v -= 1
            if lemma in STRONG_SUBJ_LEMMAS:
                v += 1
        found_words.add(lemma)
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

if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)
    demstrong, demweak, repstrong, repweak, bothstrong, bothweak = (
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
        defaultdict(int),
    )
    for t in tweets:
        found = set()
        for lemma in t["lemmas"]:
            if lemma in found:
                continue
            found.add(lemma)
            if lemma in WEAK_SUBJ_LEMMAS:
                bothweak[lemma] += 1
                if t["stance"] == "dem":
                    demweak[lemma] += 1
                if t["stance"] == "rep":
                    repweak[lemma] += 1
            if lemma in STRONG_SUBJ_LEMMAS:
                bothstrong[lemma] += 1
                if t["stance"] == "dem":
                    demstrong[lemma] += 1
                if t["stance"] == "rep":
                    repstrong[lemma] += 1

    for name, d in zip(
        [
            "demstrong",
            "demweak",
            "repstrong",
            "repweak",
            "bothstrong",
            "bothweak",
        ],
        [demstrong, demweak, repstrong, repweak, bothstrong, bothweak],
    ):
        print(name)
        print(
            ", ".join(
                unzip(
                    sorted([[count, word] for word, count in d.items()], reverse=True)
                )[1][:10]
            )
        )
