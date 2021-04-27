from os.path import join

from config import RESOURCES_DIR
from nltk.stem.snowball import SnowballStemmer

from cc_tweets.experiment_config import DATASET_PKL_PATH
from cc_tweets.feature_utils import save_features
from cc_tweets.misc import AFFECT_IGNORE_STEMS
from cc_tweets.utils import load_pkl, read_txt_as_str_list

SUBJECTIVITY_LEXICON_PATH = join(
    RESOURCES_DIR, "subjectivity_clues_hltemnlp05", "subjclueslen1-HLTEMNLP05.tff"
)


def load_subj_lex():
    stemmer = SnowballStemmer("english")

    lines = read_txt_as_str_list(SUBJECTIVITY_LEXICON_PATH)
    strong_subj_stems = set()
    weak_subj_stems = set()
    for line in lines:
        segs = line.split(" ")
        word = segs[2].split("=")[1]
        is_stemmed = segs[3].split("=")[1] == "y"
        if not is_stemmed:
            word = stemmer.stem(word)
        type = segs[0].split("=")[1]
        if type == "weaksubj":
            weak_subj_stems.add(word)
        elif type == "strongsubj":
            strong_subj_stems.add(word)
        else:
            raise ValueError()

    return strong_subj_stems, weak_subj_stems


if __name__ == "__main__":
    tweets = load_pkl(DATASET_PKL_PATH)

    strong_subj_stems, weak_subj_stems = load_subj_lex()

    id2numstrongsubj = {}
    id2numweaksubj = {}
    id2combinedsubj = {}

    for tweet in tweets:
        strong_subj_count = weak_subj_count = combined_subj_count = 0
        for stem in tweet["stems"]:
            if stem in AFFECT_IGNORE_STEMS:
                continue
            if stem in strong_subj_stems:
                strong_subj_count += 1
                combined_subj_count += 1
            if stem in weak_subj_stems:
                weak_subj_count += 1
                combined_subj_count -= 1
        id2numstrongsubj[tweet["id"]] = strong_subj_count
        id2numweaksubj[tweet["id"]] = weak_subj_count
        id2combinedsubj[tweet["id"]] = combined_subj_count

    save_features(
        tweets,
        {
            "subj_strong": id2numstrongsubj,
            "subj_weak": id2numweaksubj,
            "subj_combined": id2combinedsubj,
        },
        "subjectivity",
    )
