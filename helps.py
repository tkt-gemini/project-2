import string
import textstat
import nltk
import numpy as np
import pandas as pd
import spacy
import contractions
import unicodedata
import re

from collections import Counter
from empath import Empath
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin

with open("archive/psych-categories.csv", "r", encoding="utf-8") as f:
    PSYCH_CATS = [line.strip() for line in f if line.strip()]

class PreprocessText(BaseEstimator, TransformerMixin):
    def __init__(self, tf_idf=True):
        self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        self.tf_idf = tf_idf
        self._url_pattern = re.compile(r"http\S+|www\S+|https\S+")
        self._tag_pattern = re.compile(r"@\w+|#\w+")
        self._number_pattern = re.compile(r"\d+[.,]?\d*")

    def _clean_text(self, text):
        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFC", text)
        text = self._url_pattern.sub("", text)
        text = self._tag_pattern.sub("", text)
        text = self._number_pattern.sub("<NUM>", text)
        text = contractions.fix(text)

        return text.strip()

    def basic_clean(self, text):
        return self._clean_text(text)

    def tfidf_clean(self, text):
        text = self._clean_text(text).lower()
        doc = self.nlp(text)

        return " ".join(
            [token.text for token in doc if not token.is_space and not token.is_punct]
        )

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        docs = (
            [self.tfidf_clean(str(x)) for x in X]
            if self.tf_idf
            else [self.basic_clean(str(x)) for x in X]
        )

        return docs

class NRCEmo:
    _EMOTIONS = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "negative",
        "positive",
        "sadness",
        "surprise",
        "trust",
    ]

    def __init__(self, file_path):
        df = pd.read_csv(
            file_path, sep="\t", header=None, names=["word", "emotion", "val"]
        )
        self.data = df.pivot_table(
            index="word", columns="emotion", values="val", fill_value=0
        )
        self.data = self.data.reindex(columns=self._EMOTIONS, fill_value=0)
        self._default_vector = np.zeros(len(self._EMOTIONS), dtype=int)

    def vector(self, word):
        if word in self.data.index:
            return self.data.loc[word].values.tolist()
        return self._default_vector.tolist()

class NRCVAD:
    _COLS = ["valence", "arousal", "dominance"]

    def __init__(self, file_path):
        df = pd.read_csv(file_path, sep="\t", header=None, names=["word"] + self._COLS)
        self.data = df.set_index("word")[self._COLS]
        self._default_vector = np.zeros(3, dtype=float)

    def vector(self, word):
        if word in self.data.index:
            return self.data.loc[word].values.tolist()
        return self._default_vector.tolist()

class NRCAIL:
    _EMOTIONS = [
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
    ]

    def __init__(self, file_path):
        df = pd.read_csv(
            file_path, sep="\t", header=None, names=["word", "emotion", "score"]
        )
        self.data = df.pivot_table(
            index="word", columns="emotion", values="score", fill_value=0.0
        )
        self.data = self.data.reindex(columns=self._EMOTIONS, fill_value=0.0)
        self._default_vector = np.zeros(len(self._EMOTIONS), dtype=float)

    def vector(self, word):
        if word in self.data.index:
            return self.data.loc[word].values.tolist()
        return self._default_vector.tolist()

class NRCLex:
    def __init__(self):
        print("Loading Lexicons...\n")
        try:
            self.emo = NRCEmo("archive/NRC-Emotion-Lexicon.csv")
            self.vad = NRCVAD("archive/NRC-VAD-Lexicon.csv")
            self.ail = NRCAIL("archive/NRC-Emotion-Intensity-Lexicon.csv")
            self.output_dim = 74
            print("Done!\n")
        except FileNotFoundError as fe:
            print(f"Error: {fe}\n")

    def _text_normalize(self, text: str):
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))

        return text

    def to_array(self, text):
        words = self._text_normalize(text).split()
        features = np.zeros(self.output_dim)

        if not words:
            return features.tolist()

        # Batch vectorization for better performance
        emo = np.array([self.emo.vector(w) for w in words])
        vad = np.array([self.vad.vector(w) for w in words])
        ail = np.array([self.ail.vector(w) for w in words])

        # Calculate statistics efficiently
        if not (np.allclose(emo, 0) and np.allclose(vad, 0) and np.allclose(ail, 0)):
            features = np.concatenate(
                [
                    emo.mean(axis=0),
                    emo.max(axis=0),
                    emo.min(axis=0),
                    vad.mean(axis=0),
                    vad.max(axis=0),
                    vad.min(axis=0),
                    vad.std(axis=0),
                    ail.mean(axis=0),
                    ail.max(axis=0),
                    ail.min(axis=0),
                    ail.std(axis=0),
                ]
            )

        return features.tolist()

class PsychologicalExtractor(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        print("Initializing Psychological System...\n")
        nltk.download("vader_lexicon", quiet=True)
        self.nlp = spacy.load("en_core_web_sm")
        self.sid = SentimentIntensityAnalyzer()
        self.nrc = NRCLex()
        self.emp = Empath()
        print("Done!\n")

    def fit(self, X, y=None):
        return self

    def linguistic_normalize(self, text: str):
        if len(text.strip()) == 0:
            return [0.0] * 5

        doc = self.nlp(text)
        lenght = len(doc) if len(doc) > 0 else 1
        pos_tags = [token.pos_ for token in doc]
        pos_counts = Counter(pos_tags)
        counts = {
            "pronouns": pos_counts.get("PRON", 0),
            "verbs": pos_counts.get("VERB", 0),
            "aux": pos_counts.get("AUX", 0),
            "conjunctions": pos_counts.get("CCONJ", 0) + pos_counts.get("SCONJ", 0),
            "adverbs": pos_counts.get("ADV", 0),
        }

        return [pos / lenght for pos in counts.values()]

    def punctuation_normalize(self, text: str):
        if len(text) == 0:
            return [0.0] * 5

        lenght = len(text) if len(text) > 0 else 1
        counts = {
            "ellipsis": len(re.findall(r"\.{2,}", text)),
            "period": len(re.findall(r"(?<!\.)\.(?!\.)", text)),
            "exclamation": len(re.findall(r"\!", text)),
            "question": len(re.findall(r"\?", text)),
            "comma": len(re.findall(r"\,", text)),
        }

        return [punct / lenght for punct in counts.values()]

    def transform(self, X):
        features = []

        for text in X:
            readability = textstat.flesch_kincaid_grade(text)
            ttr = textstat.text_standard(text, float_output=True)

            nrc_vec = self.nrc.to_array(text)
            txtstat_vec = [readability, ttr]
            lingui_vec = self.linguistic_normalize(text)
            punct_vec = self.punctuation_normalize(text)

            # Vader sentiment
            vader_scores = self.sid.polarity_scores(text)
            if vader_scores is None:
                vader_vec = [0.0, 0.0, 0.0, 0.0]
            else:
                vader_vec = list(vader_scores.values())

            # Empath analysis
            empath_result = self.emp.analyze(
                text, categories=PSYCH_CATS, normalize=True
            )
            if empath_result is None:
                # Create zero vector with same length as expected categories
                empath_vec = [0.0] * len(PSYCH_CATS) if PSYCH_CATS else []
            else:
                empath_vec = list(empath_result.values())

            # Compos vector
            compos_vec = (
                txtstat_vec + lingui_vec + punct_vec + vader_vec + empath_vec + nrc_vec
            )
            features.append(compos_vec)

        return np.array(features)
