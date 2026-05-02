"""
pipeline/features.py — NRC Lexicons, SymptomLexicon, PsychologicalExtractor, FeatureEngineer.

Feature pipeline (Stage 6):
    PsychologicalExtractor → psych vector (txtstat + lingui + punct + vader + empath + nrc + symptom)
    FeatureEngineer        → TF-IDF (word+char) + chi2 select + psych + MaxAbsScaler
"""
from __future__ import annotations

import gc
import json
import re
import string
from collections import Counter
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import scipy.sparse as sp
import spacy
import textstat
from empath import Empath
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MaxAbsScaler
from tqdm import tqdm

from pipeline.config import ARTIFACT_DIR, PSYCH_CATS, SELECT_K, TFIDF_FEATS


# ═════════════════════════════════════════════════════════════════════════════
# NRC LEXICONS
# ═════════════════════════════════════════════════════════════════════════════

class _NRCEmo:
    _EMOTIONS = ["anger","anticipation","disgust","fear","joy",
                 "negative","positive","sadness","surprise","trust"]
    def __init__(self, path: str):
        df = pd.read_csv(path, sep="\t", header=None, names=["word","emotion","val"])
        self.data = (df.pivot_table(index="word", columns="emotion", values="val", fill_value=0)
                     .reindex(columns=self._EMOTIONS, fill_value=0))

class _NRCVAD:
    _COLS = ["valence","arousal","dominance"]
    def __init__(self, path: str):
        df        = pd.read_csv(path, sep="\t", header=None, names=["word"]+self._COLS)
        self.data = df.set_index("word")[self._COLS]

class _NRCAIL:
    _EMOTIONS = ["anger","anticipation","disgust","fear","joy","sadness","surprise","trust"]
    def __init__(self, path: str):
        df = pd.read_csv(path, sep="\t", header=None, names=["word","emotion","score"])
        self.data = (df.pivot_table(index="word", columns="emotion", values="score", fill_value=0.0)
                     .reindex(columns=self._EMOTIONS, fill_value=0.0))


class NRCLex:
    """
    3 lexicon NRC → vector 74 chiều:
        NRCEmo (10) × 3 stats (mean/max/min)      = 30
        NRCVAD  (3) × 4 stats (mean/max/min/std)  = 12
        NRCAIL  (8) × 4 stats (mean/max/min/std)  = 32
    """
    _PUNCT = str.maketrans("", "", string.punctuation)

    def __init__(self):
        self._loaded = False
        try:
            self.emo = _NRCEmo("archive/nrc/emotion-lexicon.csv")
            self.vad = _NRCVAD("archive/nrc/VAD-lexicon.csv")
            self.ail = _NRCAIL("archive/nrc/emotion-intensity-lexicon.csv")
            self.output_dim = (
                len(self.emo._EMOTIONS) * 3
                + len(self.vad._COLS) * 4
                + len(self.ail._EMOTIONS) * 4
            )
            # [PERF] Precompute dict lookups — tránh DataFrame.reindex allocation mỗi lần gọi
            self._emo_dict = dict(zip(self.emo.data.index, self.emo.data.values))
            self._vad_dict = dict(zip(self.vad.data.index, self.vad.data.values))
            self._ail_dict = dict(zip(self.ail.data.index, self.ail.data.values))
            self._emo_zero = np.zeros(len(self.emo._EMOTIONS))
            self._vad_zero = np.zeros(len(self.vad._COLS))
            self._ail_zero = np.zeros(len(self.ail._EMOTIONS))
            self._loaded = True
        except FileNotFoundError as e:
            self.output_dim = 74
            print(f"  NRCLex warning: {e}. Returns zero vectors.")

    def to_array(self, text: str) -> np.ndarray:
        if not self._loaded:
            return np.zeros(self.output_dim)
        words = text.lower().translate(self._PUNCT).split()
        if not words:
            return np.zeros(self.output_dim)
        # [PERF] Dict lookup thay DataFrame.reindex — tránh tạo DataFrame tạm mỗi text
        emo = np.array([self._emo_dict.get(w, self._emo_zero) for w in words], dtype=float)
        vad = np.array([self._vad_dict.get(w, self._vad_zero) for w in words], dtype=float)
        ail = np.array([self._ail_dict.get(w, self._ail_zero) for w in words], dtype=float)
        if np.allclose(emo, 0) and np.allclose(vad, 0) and np.allclose(ail, 0):
            return np.zeros(self.output_dim)
        return np.concatenate([
            emo.mean(0), emo.max(0), emo.min(0),
            vad.mean(0), vad.max(0), vad.min(0), vad.std(0),
            ail.mean(0), ail.max(0), ail.min(0), ail.std(0),
        ])


# ═════════════════════════════════════════════════════════════════════════════
# SYMPTOM LEXICON
# ═════════════════════════════════════════════════════════════════════════════

class SymptomLexicon:
    """
    Rule-based symptom scorer → vector 4 chiều (normalize theo độ dài text).

    Output features:
        depression_score  : hopelessness, anhedonia, fatigue, worthless, empty
        anxiety_score     : worry, panic, racing thoughts, dread, nervous
        isolation_score   : alone, lonely, no one, disconnected, invisible
        trauma_score      : flashback, nightmare, triggered, hypervigilant, numb
    """
    # NOTE: Một số từ (numb, intrusive, avoidance) xuất hiện trong nhiều category
    # — cố ý, phản ánh comorbidity thực tế giữa các rối loạn tâm thần.
    _DEPRESSION_TERMS = {
        "hopeless", "hopelessness", "worthless", "worthlessness", "empty",
        "emptiness", "numb", "numbness", "anhedonia", "joyless", "meaningless",
        "pointless", "despair", "desperation", "hollow", "lifeless",
        "exhausted", "exhaustion", "drained", "fatigue", "fatigued",
        "unmotivated", "apathetic", "apathy", "flat", "blank",
        "crying", "tearful", "weeping", "sobbing",
        "burden", "burdensome", "guilty", "guilt", "ashamed", "shame",
    }
    _ANXIETY_TERMS = {
        "worried", "worrying", "worry", "anxious", "anxiety", "panic",
        "panicking", "dread", "dreading", "fearful", "terrified", "terror",
        "nervous", "nervousness", "tense", "tension", "restless", "restlessness",
        "racing", "overthinking", "ruminating", "rumination",
        "catastrophizing", "intrusive", "obsessive", "compulsive",
        "hyperventilating", "trembling", "shaking", "sweating",
        "avoidance", "avoiding", "phobia", "triggered",
    }
    _ISOLATION_TERMS = {
        "alone", "lonely", "loneliness", "isolated", "isolation",
        "disconnected", "disconnection", "invisible", "unseen", "unheard",
        "unloved", "unwanted", "rejected", "rejection", "abandoned",
        "abandonment", "nobody", "noone", "friendless",
        "excluded", "ostracized", "withdrawn", "withdrawal", "reclusive",
        "hermit", "socially", "avoidant",
    }
    _TRAUMA_TERMS = {
        "flashback", "flashbacks", "nightmare", "nightmares", "trauma",
        "traumatic", "traumatized", "hypervigilant", "hypervigilance",
        "startle", "startled", "dissociate", "dissociation", "dissociating",
        "detached", "detachment", "frozen", "freeze", "numb", "numbing",
        "intrusion", "intrusive", "avoidance", "reexperiencing",
        "survivor", "abuse", "abused", "assault", "assaulted",
        "ptsd", "cptsd", "groomed", "grooming",
    }
    _PUNCT = str.maketrans("", "", string.punctuation)

    def score(self, text: str) -> np.ndarray:
        if not text or not isinstance(text, str):
            return np.zeros(4)
        words = text.lower().translate(self._PUNCT).split()
        n     = len(words) or 1
        # [FIX] Đếm frequency (có lặp) thay vì unique — nhất quán với mẫu số n=len(words)
        return np.array([
            sum(1 for w in words if w in self._DEPRESSION_TERMS) / n,
            sum(1 for w in words if w in self._ANXIETY_TERMS)    / n,
            sum(1 for w in words if w in self._ISOLATION_TERMS)  / n,
            sum(1 for w in words if w in self._TRAUMA_TERMS)     / n,
        ], dtype=float)

    def transform(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self.score(t) for t in texts])

    @staticmethod
    def feature_names() -> list[str]:
        return ["symptom_depression", "symptom_anxiety",
                "symptom_isolation",  "symptom_trauma"]


# ═════════════════════════════════════════════════════════════════════════════
# PSYCHOLOGICAL EXTRACTOR
# ═════════════════════════════════════════════════════════════════════════════

class PsychologicalExtractor(BaseEstimator, TransformerMixin):
    """
    Trích xuất vector đặc trưng tâm lý từ post_clean (KHÔNG masked).

    Output (feature names qua get_feature_names()):
        txtstat (2) + lingui (5) + punct (5) + vader (4) + empath (n) + nrc (74) + symptom (4)
    """
    _TXTSTAT_NAMES = ["flesch_kincaid_grade", "text_standard"]
    _LINGUI_NAMES  = ["pron_ratio","verb_ratio","aux_ratio","conj_ratio","adv_ratio"]
    _PUNCT_NAMES   = ["ellipsis_rate","period_rate","exclaim_rate","question_rate","comma_rate"]
    _VADER_NAMES   = ["vader_neg","vader_neu","vader_pos","vader_compound"]

    _ELLIPSIS = re.compile(r"\.{2,}")
    _PERIOD   = re.compile(r"(?<!\.)\.(?!\.)")
    _EXCLAIM  = re.compile(r"!")
    _QUESTION = re.compile(r"\?")
    _COMMA    = re.compile(r",")

    def __init__(self, batch_size: int = 512) -> None:
        nltk.download("vader_lexicon", quiet=True)
        try:
            self.nlp    = spacy.load("en_core_web_md", disable=["ner","lemmatizer","parser"])
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_md' not found. "
                "Run: python -m spacy download en_core_web_md"
            )
        self.sid        = SentimentIntensityAnalyzer()
        self.nrc        = NRCLex()
        self.emp        = Empath()
        self.symptom    = SymptomLexicon()
        self.batch_size = batch_size
        self._fnames    : list[str] = []

    def fit(self, X, y=None):
        return self

    def get_feature_names(self) -> list[str]:
        if self._fnames:
            return self._fnames
        empath_names = [f"empath_{c}" for c in PSYCH_CATS]
        nrc_names: list[str] = []
        if self.nrc._loaded:
            for s in ["mean","max","min"]:
                nrc_names += [f"nrcemo_{e}_{s}" for e in self.nrc.emo._EMOTIONS]
            for s in ["mean","max","min","std"]:
                nrc_names += [f"nrcvad_{v}_{s}" for v in self.nrc.vad._COLS]
                nrc_names += [f"nrcail_{e}_{s}" for e in self.nrc.ail._EMOTIONS]
        return (self._TXTSTAT_NAMES + self._LINGUI_NAMES + self._PUNCT_NAMES
                + self._VADER_NAMES + empath_names + nrc_names
                + SymptomLexicon.feature_names())

    def _punct_vec(self, text: str) -> list[float]:
        n = len(text) or 1
        return [
            len(self._ELLIPSIS.findall(text)) / n,
            len(self._PERIOD.findall(text))   / n,
            len(self._EXCLAIM.findall(text))  / n,
            len(self._QUESTION.findall(text)) / n,
            len(self._COMMA.findall(text))    / n,
        ]

    def transform(self, X) -> np.ndarray:
        """X: list hoặc Series của post_clean strings. Không truyền post_masked."""
        sample = str(X[0]) if hasattr(X, "__getitem__") and len(X) > 0 else ""
        if "[MH]" in sample:
            raise ValueError(
                "Truyền post_clean vào PsychologicalExtractor, không phải post_masked."
            )
        texts    = [str(x) for x in X]
        features = []
        for text, doc in tqdm(
            zip(texts, self.nlp.pipe(texts, batch_size=self.batch_size)),
            total=len(texts), desc="PsychExtract",
        ):
            n      = len(doc) or 1
            c      = Counter(t.pos_ for t in doc)
            lingui = [c.get("PRON",0)/n, c.get("VERB",0)/n, c.get("AUX",0)/n,
                      (c.get("CCONJ",0)+c.get("SCONJ",0))/n, c.get("ADV",0)/n]
            vader  = self.sid.polarity_scores(text)
            empath = self.emp.analyze(text, categories=PSYCH_CATS, normalize=True)
            features.append(
                [textstat.flesch_kincaid_grade(text),
                 textstat.text_standard(text, float_output=True)]
                + lingui
                + self._punct_vec(text)
                + list(vader.values())
                + (list(empath.values()) if empath else [0.0]*len(PSYCH_CATS))
                + self.nrc.to_array(text).tolist()
                + self.symptom.score(text).tolist()
            )
        result = np.array(features)
        if not self._fnames:
            self._fnames = self.get_feature_names()
            assert len(self._fnames) == result.shape[1], (
                f"Feature name mismatch: {len(self._fnames)} ≠ {result.shape[1]}"
            )
        return result

    def save(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        artifact_dir = Path(artifact_dir)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        nlp_bak, self.nlp = self.nlp, None
        joblib.dump(self, artifact_dir / "psych_extractor.joblib", compress=3)
        self.nlp = nlp_bak
        fnames = self.get_feature_names()
        with open(artifact_dir / "psych_feature_names.json", "w") as f:
            json.dump(fnames, f, indent=2)
        meta = {
            "output_dim": len(fnames),
            "structure" : {
                "txtstat": 2, "lingui": 5, "punct": 5, "vader": 4,
                "empath" : len(PSYCH_CATS),
                "nrc"    : self.nrc.output_dim if self.nrc._loaded else 0,
                "symptom": 4,
            },
        }
        with open(artifact_dir / "psych_meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [Artifact] psych_extractor → {artifact_dir}/")

    @classmethod
    def load(cls, artifact_dir: Path = ARTIFACT_DIR) -> "PsychologicalExtractor":
        obj = joblib.load(Path(artifact_dir) / "psych_extractor.joblib")
        try:
            obj.nlp = spacy.load("en_core_web_md", disable=["ner","lemmatizer","parser"])
        except OSError:
            raise OSError(
                "spaCy model 'en_core_web_md' not found. "
                "Run: python -m spacy download en_core_web_md"
            )
        with open(Path(artifact_dir) / "psych_feature_names.json") as f:
            obj._fnames = json.load(f)
        print(f"  [Artifact] PsychExtractor loaded — dim={len(obj._fnames)}")
        return obj


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEER
# ═════════════════════════════════════════════════════════════════════════════

class FeatureEngineer:
    """
    TF-IDF (word + char) + SelectKBest chi2 + MaxAbsScaler.
    Fit CHỈ trên training data — không có test leakage qua IDF weights.
    (Đã tối ưu RAM: Dùng float32 và chủ động giải phóng bộ nhớ trung gian).
    """
    _TOKEN_PATTERN = {
        "masked": r"(?u)\b[a-zA-Z\[\]][a-zA-Z\[\]]+\b",
        "clean" : r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    }
    _MIN_DF = {"masked": 3, "clean": 5}

    def __init__(
        self,
        tfidf_feats: int = TFIDF_FEATS,
        select_k   : int = SELECT_K,
        variant    : str = "masked",
    ):
        if variant not in ("masked", "clean"):
            raise ValueError(f"variant phải là 'masked' hoặc 'clean', nhận: {variant!r}")
        self.tfidf_feats = tfidf_feats
        self.select_k    = select_k
        self.variant     = variant
        self.tfidf      : TfidfVectorizer | None = None
        self.char_tfidf : TfidfVectorizer | None = None
        self.selector   : SelectKBest     | None = None
        self.scaler     : MaxAbsScaler    | None = None
        self._dims      : dict                   = {}

    def fit_transform(
        self, texts: list[str], psych: np.ndarray, y: np.ndarray
    ) -> sp.csr_matrix:
        print(f"\n[STAGE 6] Feature Engineering ({self.variant.upper()}) — fit on training data")

        # 1. Word TF-IDF
        self.tfidf = TfidfVectorizer(
            ngram_range  = (1, 2),
            max_features = self.tfidf_feats,
            sublinear_tf = True,
            min_df       = self._MIN_DF[self.variant],
            max_df       = 0.95,
            strip_accents= "unicode",
            token_pattern= self._TOKEN_PATTERN[self.variant],
            dtype        = np.float32,
        )
        X_word = self.tfidf.fit_transform(texts)
        print(f"  TF-IDF word  : {X_word.shape[0]:,} × {X_word.shape[1]:,}")

        # 2. Char TF-IDF
        self.char_tfidf = TfidfVectorizer(
            analyzer     = "char_wb",
            ngram_range  = (3, 5),
            max_features = self.tfidf_feats // 3,
            sublinear_tf = True,
            min_df       = 5,
            max_df       = 0.95,
            strip_accents= "unicode",
            dtype        = np.float32,
        )
        X_char = self.char_tfidf.fit_transform(texts)
        print(f"  TF-IDF char  : {X_char.shape[0]:,} × {X_char.shape[1]:,}")

        # 3. Combine TF-IDF
        X_tfidf_combined = sp.hstack([X_word, X_char], format="csr", dtype=np.float32)
        print(f"  TF-IDF total : {X_tfidf_combined.shape[1]:,}")
        del X_word, X_char

        # 4. Feature Selection
        self.selector = SelectKBest(chi2, k=min(self.select_k, X_tfidf_combined.shape[1]))
        X_sel         = self.selector.fit_transform(X_tfidf_combined, y)
        print(f"  chi2 sel     : {X_sel.shape[1]:,} features")
        del X_tfidf_combined

        # 5. Combine với Psych features
        X_comb = sp.hstack([X_sel, sp.csr_matrix(psych, dtype=np.float32)], format="csr", dtype=np.float32)
        del X_sel
        gc.collect()

        # 6. Scaling
        self.scaler = MaxAbsScaler()
        X_final     = self.scaler.fit_transform(X_comb)
        print(f"  Final        : {X_final.shape[0]:,} × {X_final.shape[1]:,}")

        self._dims = {
            "tfidf_word": len(self.tfidf.vocabulary_),
            "tfidf_char": len(self.char_tfidf.vocabulary_),
            "tfidf_sel" : int(X_final.shape[1] - psych.shape[1]),
            "psych"     : int(psych.shape[1]),
            "combined"  : int(X_final.shape[1]),
            "variant"   : self.variant,
        }
        return X_final

    def transform(self, texts: list[str], psych: np.ndarray) -> sp.csr_matrix:
        if self.tfidf is None:
            raise RuntimeError("FeatureEngineer not fitted.")
        X_word = self.tfidf.transform(texts)
        X_char = self.char_tfidf.transform(texts)
        X_tfidf_combined = sp.hstack([X_word, X_char], format="csr", dtype=np.float32)
        del X_word, X_char
        X_sel = self.selector.transform(X_tfidf_combined)
        del X_tfidf_combined
        X_comb = sp.hstack([X_sel, sp.csr_matrix(psych, dtype=np.float32)], format="csr", dtype=np.float32)
        del X_sel
        return self.scaler.transform(X_comb)

    def save(self, artifact_dir: Path = ARTIFACT_DIR) -> None:
        artifact_dir = Path(artifact_dir)
        joblib.dump(self.tfidf,      artifact_dir / "tfidf.joblib",      compress=3)
        joblib.dump(self.char_tfidf, artifact_dir / "char_tfidf.joblib", compress=3)
        joblib.dump(self.selector,   artifact_dir / "selector.joblib",   compress=3)
        joblib.dump(self.scaler,     artifact_dir / "scaler.joblib",     compress=3)
        with open(artifact_dir / "feature_meta.json", "w") as f:
            json.dump(self._dims, f, indent=2)
        print(f"  [Artifact] tfidf, char_tfidf, selector, scaler → {artifact_dir}/")

    @classmethod
    def load(cls, artifact_dir: Path = ARTIFACT_DIR) -> "FeatureEngineer":
        with open(Path(artifact_dir) / "feature_meta.json") as f:
            meta = json.load(f)
        fe              = cls.__new__(cls)
        fe.variant      = meta.get("variant", "masked")
        fe.tfidf_feats  = TFIDF_FEATS
        fe.select_k     = SELECT_K
        fe.tfidf        = joblib.load(Path(artifact_dir) / "tfidf.joblib")
        fe.char_tfidf   = joblib.load(Path(artifact_dir) / "char_tfidf.joblib")
        fe.selector     = joblib.load(Path(artifact_dir) / "selector.joblib")
        fe.scaler       = joblib.load(Path(artifact_dir) / "scaler.joblib")
        fe._dims        = meta
        print(f"  [Artifact] FeatureEngineer [{fe.variant}] loaded — dim={meta.get('combined')}")
        return fe
