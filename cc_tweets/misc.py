from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

# words ignored when detected affect/sentiment
# Yiwei Luo, https://github.com/yiweiluo/Green-American-Lexicon/blob/master/3_reddit_analysis/reddit_data/blacklist_words.txt
AFFECT_IGNORE_WORDS = set(
    [
        "organization",
        "autumn",
        "hell",
        "president",
        "drive",
        "company",
        "climate",
        "unite",
        "sun",
        "leader",
        "december",
        "deleted",
        "peace",
        "date",
        "electricity",
        "environment",
        "power",
        "country",
        "delete",
        "weekend",
        "change",
        "support",
        "remove",
        "cloudy",
        "warm",
        "global",
        "warming",
        "experiment",
        "confidence",
        "gore",
        "association",
        "rain",
        "karma",
        "trust",
        "responsible",
        "beach",
        "removed",
        "tropical",
        "japan",
        "fuck",
        "contribution",
        "shit",
        "trump",
        "united",
        "summer",
        "crap",
        "lake",
        "environmental",
        "group",
        "rainy",
        "snowfall",
        "energy",
    ]
)

lemmatizer = WordNetLemmatizer()
AFFECT_IGNORE_LEMMAS = set(lemmatizer.lemmatize(w) for w in AFFECT_IGNORE_WORDS)

stemmer = SnowballStemmer("english")
AFFECT_IGNORE_STEMS = set(stemmer.stem(w) for w in AFFECT_IGNORE_WORDS)
