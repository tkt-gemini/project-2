import pickle
import re
import unicodedata
import string
from pathlib import Path

import contractions
import spacy
from collections import Counter
from empath import Empath
from nltk.sentiment import SentimentIntensityAnalyzer

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Memory
from sklearn.base import BaseEstimator, TransformerMixin

from config import *

memory = Memory(location=PSYCH_CACHE_DIR, verbose=0)


class Filter:
    def __init__(self):
        pass

    def __cleaning(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = unicodedata.normalize("NFC", text)
        text = re.compile(r"https?://\S+|www\.\S+").sub("", text)
        text = re.sub(r"@\w+|#\w+", "", text)
        text = re.sub(r"\d+[.,]?\d*", "[NUM]", text)
        try:
            text = contractions.fix(text)
        except Exception:
            pass

        return text.strip()

    def __is_self_disclosure(self, text: str) -> bool:
        if not any(p.search(text) for p in FP_COMPILED):
            return False

        return not any(p.search(text) for p in EXCL_COMPILED)
    
    def __masking(self, text:str):
        return MH_PATTERN.sub("[MH]", text)

    def __first_person_ratio(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip():
            return 0.0

        lower = text.lower()
        sentences = [s for s in re.split(r"[.!?\n]+", lower) if s.strip()]
        n_sents   = len(sentences) or 1
        fp_count  = sum(1 for w in lower.split() if w in FIRST_PERSON_PRONOUNS)

        return fp_count / n_sents

    def noise(
            self,
            X:pd.DataFrame,
            min_words:int = MIN_WORDS,
            max_urls:int = MAX_URLS
        ) -> pd.DataFrame:

        n_orig = len(X)
        post = X["post"].fillna("")
        author = X["author"].fillna("") if "author" in X.columns else pd.Series([""] * n_orig)

        m_del = post.str.strip().str.lower().isin({"[removed]", "[deleted]", "", "nan"}) | post.isna()
        m_shrt = post.str.split().str.len() < min_words
        m_bot = author.str.lower().str.contains(BOT_RE, na=False)
        m_mod = post.str.lower().str.contains(MOD_RE, na=False)
        m_spam = post.str.count(r"https?://\S+|www\.\S+") > max_urls
        m_dup = post.duplicated(keep=False)

        stats = {
            "deleted": int(m_del.sum()),
            "short": int(m_shrt.sum()),
            "bot": int(m_bot.sum()),
            "moderator": int(m_mod.sum()),
            "spam": int(m_spam.sum()),
            "duplicated": int(m_dup.sum()),
        }

        m_noise = m_del | m_shrt | m_bot | m_mod | m_spam | m_dup
        X_out = X.copy()
        X_out.loc[m_noise, :] = np.nan
        total_dropped = int(m_noise.sum())
        n_kept = int((~m_noise).sum())

        print("\n[NOISE FILTERING]")
        print(f"  Input: {n_orig:,}")
        for reason, count in stats.items():
            print(f"  Convert to NaN ({reason}): {count:,} ({count/n_orig*100:.1f}%)")

        print(f"  Total NaN: {total_dropped:,} (reasons may overlap)")
        print(f"  Output: {n_kept:,} ({n_kept/n_orig*100:.1f}% retained)")

        return X_out

    def clinical(
            self,
            X:pd.DataFrame,
            fp_threshold:float = FP_THRESHOLD,
            batch_size:int = BATCH_SIZE,
            checkpoint_dir:Path = CHECKPOINT_DIR,
        ) -> pd.DataFrame:
        # Make directory
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # 
        texts = X["post"].fillna("").tolist()
        n_orig = len(texts)
        n_batches = (n_orig + batch_size - 1) // batch_size
        # check last save batch
        ckpts = sorted(
            checkpoint_dir.glob("batch_*.pkl"), 
            key=lambda p: int(p.stem.split('_')[1])
        )
        last_done = int(ckpts[-1].stem.split("_")[1]) if ckpts else -1
        all_results : list[tuple] = []

        if last_done >= n_batches:
            print(f"Stale checkpoints detected (last={last_done}, expected<{n_batches}) - clearing...")
            for _ckpt in checkpoint_dir.glob("batch_*.pkl"):
                _ckpt.unlink()
            last_done = -1

        kept_fp_high = kept_pattern = drop_no_fp = 0

        if last_done >= 0:
            print(f"Resume from batch {last_done + 1}/{n_batches}")
            ckpt_files = sorted(checkpoint_dir.glob("batch_*.pkl"))
            for p in tqdm(ckpt_files, desc="  Load checkpoints", unit="batch"):
                with open(p, "rb") as f:
                    batch = pickle.load(f)

                for _, drop, masked, _ in batch:
                    if drop: drop_no_fp += 1
                    elif pd.notna(masked): kept_fp_high += 1
                    else: kept_pattern += 1

                all_results.extend(batch)

        print(f"\n[CLINICAL FILTER - MASK] {n_orig:,} posts | device: CPU")

        pbar = tqdm(range(last_done + 1, n_batches), desc="Batch", unit="batch", dynamic_ncols=True)
        for batch_i in pbar:
            start = batch_i * batch_size
            end = min(start + batch_size, n_orig)
            batch_results = []

            for post_i, raw in zip(range(start, end), texts[start:end]):
                cleaned = self.__cleaning(raw)
                fp      = self.__first_person_ratio(cleaned)

                if fp >= fp_threshold:
                    kept_fp_high += 1
                    batch_results.append((post_i, False, self.__masking(cleaned), cleaned))
                elif not self.__is_self_disclosure(cleaned):
                    drop_no_fp += 1
                    batch_results.append((post_i, True, np.nan, np.nan))
                else:
                    kept_pattern += 1
                    batch_results.append((post_i, False, self.__masking(cleaned), cleaned))

            all_results.extend(batch_results)
            with open(checkpoint_dir / f"batch_{batch_i}.pkl", "wb") as f:
                pickle.dump(batch_results, f)

        result_df = pd.DataFrame(
            all_results, columns=["_idx", "_drop", "post_masked", "post_cleaned"]
        ).set_index("_idx").sort_index()

        X_out = X.copy()
        X_out["post_masked"] = result_df["post_masked"].reindex(X_out.index)
        X_out["post_cleaned"] = result_df["post_cleaned"].reindex(X_out.index)
        m_drop = result_df["_drop"].reindex(X_out.index).fillna(False).values
        n_dropped = int(m_drop.sum())

        X_out.loc[m_drop, ["post", "post_masked", "post_cleaned"]] = np.nan

        print(f"Kept (fp high): {kept_fp_high:,}")
        print(f"Kept (pattern): {kept_pattern:,}")
        print(f"NaN  (no pattern): {drop_no_fp:,}")
        print(f"Total NaN: {n_dropped:,} ({n_dropped/n_orig*100:.1f}%)")
        print(f"Total kept: {len(X_out):,} ({len(X_out)/n_orig*100:.1f}%)")

        return X_out


class NRCLexicon:
    __PUNCT = str.maketrans("", "", string.punctuation)
    EMOTION = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]
    VAD = ["valence","arousal","dominance"]

    def __init__(self):
        self.__lex = {
            "emo": self.__get_emo("./archive/nrc/emotion-lexicon.csv"),
            "vad": self.__get_vad("./archive/nrc/vad-lexicon.csv"),
        }
        self.__lex["emo"] = dict(zip(self.__lex["emo"].index, self.__lex["emo"].values))
        self.__lex["vad"] = dict(zip(self.__lex["vad"].index, self.__lex["vad"].values))
        self.__emo_zero = np.zeros(len(self.EMOTION))
        self.__vad_zero = np.zeros(len(self.VAD))

    def __get_emo(self, path:str):
        df = pd.read_csv(path, sep="\t", header=None, names=["word","emotion","val"])

        return (df.pivot_table(index="word", columns="emotion", values="val", fill_value=0).reindex(columns=self.EMOTION, fill_value=0))

    def __get_vad(self, path:str):
        df = pd.read_csv(path, sep="\t", header=None, names=["word"] + self.VAD)

        return df.set_index("word")[self.VAD]

    def getlist(self, text:str) -> list[float]:
        words = text.lower().translate(self.__PUNCT).split()

        if len(words) == 0:
            return np.zeros(len(self.EMOTION) + len(self.VAD) * 3).tolist()

        emo = np.array([self.__lex["emo"].get(w, self.__emo_zero.copy()) for w in words], dtype=float)
        vad = np.array([self.__lex["vad"].get(w, self.__vad_zero.copy()) for w in words], dtype=float)

        return np.concatenate([emo.std(0), vad.mean(0), vad.max(0), vad.std(0)]).tolist()


class Psychological(BaseEstimator, TransformerMixin):
    __LINGUI = ['PRON', 'ADJ', 'ADV', 'VERB', 'AUX', 'INTJ', 'NOUN']
    __VADER = ["vader_compound"]

    def __init__(self, batch_size:int = 512) -> None:
        self.nlp = spacy.load("en_core_web_md", disable=["ner","lemmatizer","parser"])
        self.sid = SentimentIntensityAnalyzer()
        self.nrc = NRCLexicon()
        self.emp = Empath()
        self.batch_size = batch_size

    @property
    def fnames(self) -> list[str]:
        empath_names = [f"empath_{c}" for c in PSYCH_CATS]
        nrc_names = [f"nrc_emo_{e}_std" for e in self.nrc.EMOTION] + [f"nrc_vad_{v}_{s}" for v in self.nrc.VAD for s in ["mean", "max", "std"]]

        return ([f"{p.lower()}_ratio" for p in self.__LINGUI] + self.__VADER + empath_names + nrc_names)

    def __lingui(self, doc) -> list[float]:
        n = len(doc) or 1
        count = Counter(t.pos_ for t in doc)

        return [count.get(p, 0) / n for p in self.__LINGUI]

    def fit(self, X, y=None):
        return self

    @memory.cache(ignore=['self'])
    def __cached_extract(self, texts: list[str]) -> np.ndarray:
        features = []
        for text, doc in tqdm(zip(texts, self.nlp.pipe(texts, batch_size=self.batch_size)), total=len(texts), desc="Psychological Extract"):
            lingui = self.__lingui(doc)
            vader  = self.sid.polarity_scores(text)
            
            empath = self.emp.analyze(text, categories=PSYCH_CATS, normalize=True)
            features.append(lingui + [vader["compound"]] + [empath.get(cat, 0.0) for cat in PSYCH_CATS] + self.nrc.getlist(text))

        return np.array(features)

    def transform(self, X):
        texts = [str(x) for x in X]
        result = self.__cached_extract(texts)

        if len(self.fnames) != result.shape[1]:
            raise ValueError(f"Feature name mismatch: {len(self.fnames)} != {result.shape[1]}")

        return result
