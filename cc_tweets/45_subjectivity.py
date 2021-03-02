from os.path import join

from config import DATA_DIR, RESOURCES_DIR
from nltk.stem.snowball import SnowballStemmer

from cc_tweets.misc import AFFECT_IGNORE_STEMS
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json

DATASET_NAME = "tweets_downsized100_filtered"
PKL_PATH = join(DATA_DIR, f"{DATASET_NAME}.pkl")

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
    tweets = load_pkl(PKL_PATH)

    strong_subj_stems, weak_subj_stems = load_subj_lex()

    id2numstrongsubj = {}
    id2numweaksubj = {}

    for tweet in tweets:
        strong_subj_count = weak_subj_count = 0
        for stem in tweet["stems"]:
            if stem in AFFECT_IGNORE_STEMS:
                continue
            if stem in strong_subj_stems:
                strong_subj_count += 1
            if stem in weak_subj_stems:
                weak_subj_count += 1
        id2numstrongsubj[tweet["id"]] = strong_subj_count
        id2numweaksubj[tweet["id"]] = weak_subj_count

    save_json(
        id2numstrongsubj,
        join(DATA_DIR, DATASET_NAME, "45_subjectivity_strong_subj.json"),
    )
    save_json(
        id2numweaksubj,
        join(DATA_DIR, DATASET_NAME, "45_subjectivity_weak_subj.json"),
    )

    stats = {}

    dem_tweets = [t for t in tweets if t["stance"] == "dem"]
    rep_tweets = [t for t in tweets if t["stance"] == "rep"]

    for strong_or_weak, id2count in [
        ("strong_subj", id2numstrongsubj),
        ("weak_subj", id2numweaksubj),
    ]:
        substat = {}
        substat["mean_count"] = sum(id2count.values()) / len(id2count)
        substat["mean_count_dem"] = sum(id2count[t["id"]] for t in dem_tweets) / len(
            dem_tweets
        )
        substat["mean_count_rep"] = sum(id2count[t["id"]] for t in rep_tweets) / len(
            rep_tweets
        )
        substat["mean_count_adjusted_for_dem_rep_imbalance"] = (
            substat["mean_count_dem"] + substat["mean_count_rep"]
        ) / 2
        stats[strong_or_weak] = substat
    save_json(stats, join(DATA_DIR, DATASET_NAME, "45_subjectivity_stats.json"))
