import re
from collections import defaultdict
from os.path import join

import pandas as pd
from cc_tweets.experiment_configs import SUBSET_PKL_PATH, SUBSET_WORKING_DIR
from cc_tweets.feature_utils import save_features
from cc_tweets.lexical_features.bank import Feature, get_feature, register_feature
from cc_tweets.misc import AFFECT_IGNORE_LEMMAS, AFFECT_IGNORE_STEMS
from cc_tweets.utils import load_pkl, read_txt_as_str_list, save_json, unzip
from cc_tweets.viz import grouped_bars
from config import DATA_DIR, RESOURCES_DIR
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from tqdm import tqdm

MFD_PATH = join(RESOURCES_DIR, "MFD", "MFD2.0.csv")


lemmatizer = WordNetLemmatizer()
df = pd.read_csv(MFD_PATH)
_valencefoundation2lemmas = {}
for i, row in df.iterrows():
    valence = row["valence"]
    foundation = row["foundation"]
    vf = f"{valence}_{foundation}"
    word = row["word"]
    lemma = lemmatizer.lemmatize(word)
    if vf not in _valencefoundation2lemmas:
        _valencefoundation2lemmas[vf] = set()
    _valencefoundation2lemmas[vf].add(lemma)


def closure(vfname):
    def _extract_features(tweets):
        vflemmas = _valencefoundation2lemmas[vfname]
        id2count = defaultdict(int)
        for tweet in tweets:
            found_words = set()
            for lemma in tweet["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                if lemma in vflemmas:
                    if lemma not in found_words:
                        id2count[tweet["id"]] += 1
                        found_words.add(lemma)
        return id2count

    return _extract_features


for vfname in _valencefoundation2lemmas:
    register_feature(Feature(f"mfd.{vfname}", closure(vfname)))

MFD_VOCAB = {
    "vice_authority": {
        "subvers",
        "riot",
        "treacher",
        "disrespect",
        "dissent",
        "transgress",
        "treason",
        "subvert",
        "tradit",
        "rebel",
        "unlaw",
        "heret",
        "overthrown",
        "chao",
        "unruli",
        "disord",
        "order",
        "disarray",
        "upris",
        "lawless",
        "refus",
        "chaotic",
        "treacheri",
        "illeg",
        "overthrow",
        "overpow",
        "permiss",
        "anarchi",
        "rebellion",
        "disobedi",
    },
    "virtue_authority": {
        "institut",
        "rule",
        "respect",
        "duti",
        "presidenti",
        "ceo",
        "governor",
        "tradit",
        "punish",
        "polit",
        "guid",
        "rank",
        "author",
        "order",
        "govern",
        "control",
        "regul",
        "pope",
        "polic",
        "chief",
        "domin",
        "will",
        "proper",
        "protect",
        "elder",
        "arrest",
        "father",
        "manag",
        "honor",
        "leadership",
    },
    "vice_fairness": {
        "deceiv",
        "fraud",
        "cheat",
        "sexism",
        "crook",
        "hypocrisi",
        "bias",
        "racist",
        "lie",
        "exploit",
        "liar",
        "inequ",
        "rob",
        "dishonest",
        "sexist",
        "decept",
        "injustic",
        "mislead",
        "hypocrit",
        "steal",
        "favorit",
        "trick",
        "racism",
        "betray",
        "scam",
        "oppress",
        "bigot",
        "partial",
        "discrimin",
        "disproportion",
    },
    "virtue_fairness": {
        "equiti",
        "due",
        "proport",
        "vengeanc",
        "retribut",
        "reciproc",
        "repar",
        "honesti",
        "refere",
        "justif",
        "tribun",
        "retali",
        "trustworthi",
        "equit",
        "honest",
        "law",
        "aveng",
        "lawyer",
        "imparti",
        "justifi",
        "equal",
        "fair",
        "justic",
        "reveng",
        "right",
        "integr",
        "compens",
        "unbias",
        "object",
        "pariti",
    },
    "vice_harm": {
        "suffer",
        "tortur",
        "harm",
        "hunger",
        "violenc",
        "exploit",
        "kill",
        "destruct",
        "cri",
        "murder",
        "destroy",
        "damag",
        "endang",
        "assault",
        "threaten",
        "ravag",
        "attack",
        "die",
        "victim",
        "fight",
        "brutal",
        "hurt",
        "killer",
        "threat",
        "vulner",
        "violent",
        "abus",
        "rape",
        "bulli",
        "pain",
    },
    "virtue_harm": {
        "rescu",
        "heal",
        "healthcar",
        "care",
        "safe",
        "cloth",
        "health",
        "benefit",
        "nurs",
        "parent",
        "comfort",
        "human",
        "kind",
        "healthi",
        "relief",
        "hospit",
        "patient",
        "share",
        "protect",
        "safeguard",
        "vulner",
        "love",
        "safeti",
        "healthier",
        "feed",
        "compass",
        "mother",
        "child",
        "help",
        "chariti",
    },
    "vice_loyalty": {
        "unpatriot",
        "enemi",
        "treacher",
        "rebel",
        "traitor",
        "infidel",
        "outsid",
        "heret",
        "disloy",
        "rebellion",
        "treacheri",
        "desert",
        "treason",
        "heresi",
        "betray",
    },
    "virtue_loyalty": {
        "follow",
        "coalit",
        "familiar",
        "pledg",
        "cult",
        "herd",
        "wife",
        "fellow",
        "insid",
        "belong",
        "solidar",
        "communiti",
        "homeland",
        "patriot",
        "togeth",
        "collect",
        "player",
        "alli",
        "troop",
        "uniti",
        "nation",
        "join",
        "corp",
        "sacrif",
        "tribe",
        "tribal",
        "famili",
        "war",
        "bow",
        "sacrific",
    },
    "vice_purity": {
        "corrupt",
        "degrad",
        "horror",
        "addict",
        "rat",
        "viral",
        "infect",
        "damn",
        "mar",
        "horrif",
        "diseas",
        "sin",
        "drug",
        "disgust",
        "epidem",
        "trash",
        "sexual",
        "plagu",
        "spread",
        "horrifi",
        "wast",
        "infecti",
        "swear",
        "contamin",
        "corp",
        "rubbish",
        "gross",
        "soil",
        "dirti",
        "garbag",
    },
    "virtue_purity": {
        "faith",
        "christian",
        "blood",
        "bless",
        "bibl",
        "bloodi",
        "church",
        "pray",
        "jesus",
        "prayer",
        "lord",
        "heaven",
        "marriag",
        "holi",
        "mar",
        "angel",
        "religi",
        "cleaner",
        "bodi",
        "mari",
        "elev",
        "pope",
        "christ",
        "soul",
        "pure",
        "religion",
        "god",
        "food",
        "immun",
        "clean",
    },
}


def vocab_closure(word):
    def _extract_features(tweets):
        id2count = defaultdict(int)
        for tweet in tweets:
            for stem in tweet["stems"]:
                if stem in AFFECT_IGNORE_STEMS:
                    continue
                if stem == word:
                    id2count[tweet["id"]] += 1
        return id2count

    return _extract_features


for vfname, words in MFD_VOCAB.items():
    for word in words:
        register_feature(Feature(f"mfd.{vfname}.{word}", vocab_closure(word)))

if __name__ == "__main__":
    tweets = load_pkl(SUBSET_PKL_PATH)
    cat_names = [
        "vice_authority",
        "virtue_authority",
        "vice_fairness",
        "virtue_fairness",
        "vice_harm",
        "virtue_harm",
        "vice_loyalty",
        "virtue_loyalty",
        "vice_purity",
        "virtue_purity",
    ]

    cat2words = {}

    for name in cat_names:

        dem_wcs = defaultdict(int)
        rep_wcs = defaultdict(int)
        cat_lemmas = _valencefoundation2lemmas[name]
        for t in tweets:
            found_words = set()
            for lemma in t["lemmas"]:
                if lemma in AFFECT_IGNORE_LEMMAS:
                    continue
                if lemma in cat_lemmas:
                    if lemma not in found_words:
                        if t["stance"] == "dem":
                            dem_wcs[lemma] += 1
                        elif t["stance"] == "rep":
                            rep_wcs[lemma] += 1
                        found_words.add(lemma)
        print("-" * 69)
        print(name)
        print("dem")
        print(
            ",".join(
                unzip(
                    sorted(
                        [(count, word) for word, count in dem_wcs.items()], reverse=True
                    )
                )[1][:5]
            )
        )
        print("rep")
        print(
            ",".join(
                unzip(
                    sorted(
                        [(count, word) for word, count in rep_wcs.items()], reverse=True
                    )
                )[1][:5]
            )
        )

        stemmer = SnowballStemmer("english")
        cat_stems = [stemmer.stem(w) for w in _valencefoundation2lemmas[name]]
        all_wcs = defaultdict(int)
        for t in tweets:
            found_words = set()
            for stem in t["stems"]:
                if stem in AFFECT_IGNORE_STEMS:
                    continue
                if stem in cat_stems:
                    if stem not in found_words:
                        found_words.add(stem)
                        all_wcs[stem] += 1

        cat2words[name] = {
            w
            for w in unzip(
                sorted([(count, word) for word, count in all_wcs.items()], reverse=True)
            )[1][:30]
        }
    print(cat2words)
