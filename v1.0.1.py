"""
v1.0.2 — Two-layer stacking ensemble  (with full-pipeline caching)
─────────────────────────────────────────────────────────────────────────────
Architecture:
  Layer 1 (base models, OOF):
    TF-IDF (word/char n-grams, 32K feats)
    → LogisticRegression  →  16 proba features  (OOF)
    → LinearSVC           →  16 proba features  (OOF via softmax)

  Layer 2 (meta-learner):
    [LR proba (16) | SVC proba (16) | Psych (~270)] → LightGBM

CACHING STRATEGY
────────────────
Each expensive phase saves its output to ./archive/cache/ so that on the
next run *only* the remaining phases are re-executed.  Delete a cache file
to force re-computation of that phase and everything downstream.

  Cache file                  Phase guarded
  ──────────────────────────  ─────────────────────────────────────────────
  cache/01_df_filtered.pkl    Phase 0  – noise + clinical filter + dropna
  cache/02_split.pkl          Phase 0  – train/test split + undersampling
  cache/03_tfidf_Xtrain.npz   Phase 1  – TF-IDF fit-transform
  cache/03_tfidf_Xtest.npz    Phase 1  – TF-IDF transform (test)
  cache/03_tfidf_fe.pkl       Phase 1  – fitted TF-IDF pipeline object
  cache/04_tuning_params.pkl  Phase 2  – Optuna best params (LR + SVC)
  cache/05_oof.pkl            Phase 3  – OOF arrays + test fold averages
  cache/06_psych.pkl          Phase 4  – Psychological features (train+test)

Dependencies: pip install lightgbm
"""

import gc
import pickle
from pathlib import Path

import lightgbm as lgb
import nltk
import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
from imblearn.under_sampling import RandomUnderSampler
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.svm import LinearSVC

from config import (
    ARTIFACT_CLEAN,
    CACHE_DIR,
    CV_FOLDS,
    LABEL_MAPPING,
    RANDOM_SEED,
    SELECT_K,
    TEST_SIZE,
    TFIDF_FEATS,
)
from helper import Filter, Psychological

nltk.download("vader_lexicon", quiet=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Cache helpers ──────────────────────────────────────────────────────────────


def _pkl_save(obj, path: Path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"  [CACHE] Saved → {path}")


def _pkl_load(path: Path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    print(f"  [CACHE] Loaded ← {path}")
    return obj


def _cache_exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


# ── Cache paths ────────────────────────────────────────────────────────────────

C_FILTERED = CACHE_DIR / "01_df_filtered.pkl"
C_SPLIT = CACHE_DIR / "02_split.pkl"
C_TFIDF_TR = CACHE_DIR / "03_tfidf_Xtrain.npz"
C_TFIDF_TE = CACHE_DIR / "03_tfidf_Xtest.npz"
C_TFIDF_FE = CACHE_DIR / "03_tfidf_fe.pkl"
C_TUNING = CACHE_DIR / "04_tuning_params.pkl"
C_OOF = CACHE_DIR / "05_oof.pkl"
C_PSYCH = CACHE_DIR / "06_psych.pkl"


# ══ PHASE 0 — Data loading, preprocessing & splitting ════════════════════════
print("=" * 60)
print("PHASE 0 — Data loading & preprocessing")
print("=" * 60)

if _cache_exists(C_FILTERED, C_SPLIT):
    print("[CACHE HIT] Skipping noise/clinical filter and split.")
    df_info = _pkl_load(C_FILTERED)  # dict with le, N_CLASSES, shapes
    split = _pkl_load(C_SPLIT)
    le = df_info["le"]
    N_CLASSES = df_info["N_CLASSES"]
    X_train_res = split["X_train_res"]
    X_test = split["X_test"]
    y_resampled = split["y_resampled"]
    y_test = split["y_test"]
    print(
        f"[DATA] Train: {len(X_train_res):,} | Test: {len(X_test):,} | Classes: {N_CLASSES}"
    )
else:
    df = pd.read_csv("./archive/raw.csv")
    df["subreddit"] = df["subreddit"].replace(LABEL_MAPPING)

    _filter = Filter()
    df = _filter.noise(df)
    df = _filter.clinical(df)
    df = df.dropna(subset=["post_cleaned", "post_masked", "subreddit"]).reset_index(
        drop=True
    )

    le = LabelEncoder()
    df["_subreddit"] = le.fit_transform(df["subreddit"])
    N_CLASSES = len(le.classes_)

    print("Label encoding:")
    for idx, label in enumerate(le.classes_):
        print(f"  {idx}: {label}")

    # ── Train/test split ───────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        df["post_cleaned"].apply(str.lower),
        df["_subreddit"],
        test_size=TEST_SIZE,
        stratify=df["_subreddit"],
        random_state=RANDOM_SEED,
    )

    # ── Under-sampling ─────────────────────────────────────────────────────────
    target_majority = 15_000
    train_class_counts = pd.Series(y_train).value_counts()
    under_strategy = {
        label: min(count, target_majority)
        for label, count in train_class_counts.items()
    }
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_SEED)
    assert y_train.index.is_unique, "y_train index must be unique"

    idx_2d = np.array(y_train.index).reshape(-1, 1)
    idx_resampled_2d, y_resampled = rus.fit_resample(idx_2d, y_train)
    idx_resampled = idx_resampled_2d.flatten()

    X_train_res = X_train.loc[idx_resampled].reset_index(drop=True)
    y_resampled = np.array(y_resampled)

    print(
        f"\n[DATA] Train: {len(X_train_res):,} | Test: {len(X_test):,} | Classes: {N_CLASSES}"
    )

    # ── Persist ────────────────────────────────────────────────────────────────
    _pkl_save({"le": le, "N_CLASSES": N_CLASSES}, C_FILTERED)
    _pkl_save(
        {
            "X_train_res": X_train_res,
            "X_test": X_test,
            "y_resampled": y_resampled,
            "y_test": y_test,
        },
        C_SPLIT,
    )

    del df, X_train, y_train
    gc.collect()
    print("[CLEANUP] Raw DataFrame removed from RAM.")


# ── Helpers ────────────────────────────────────────────────────────────────────


def get_tfidf_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf_union",
                FeatureUnion(
                    [
                        (
                            "word",
                            Pipeline(
                                [
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            ngram_range=(1, 2),
                                            max_features=TFIDF_FEATS,
                                            sublinear_tf=True,
                                            token_pattern=r"(?u)\b[a-zA-Z\[\]][a-zA-Z\[\]]+\b",
                                        ),
                                    ),
                                    (
                                        "select",
                                        SelectKBest(chi2, k=int(SELECT_K * 0.75)),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "char",
                            Pipeline(
                                [
                                    (
                                        "tfidf",
                                        TfidfVectorizer(
                                            analyzer="char_wb",
                                            ngram_range=(3, 5),
                                            max_features=TFIDF_FEATS // 3,
                                            sublinear_tf=True,
                                        ),
                                    ),
                                    (
                                        "select",
                                        SelectKBest(chi2, k=int(SELECT_K * 0.25)),
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            ("scaler", MaxAbsScaler()),
        ]
    )


def _build_base_model(model_name: str, params: dict):
    if model_name == "LogisticRegression":
        return LogisticRegression(**params)
    return LinearSVC(**params)


def tune_base_model(model_name: str, X, y, n_trials: int = 30) -> dict:
    """Optuna search for LR or LinearSVC hyperparameters."""
    fixed = dict(
        max_iter=2000, dual=False, random_state=RANDOM_SEED, class_weight="balanced"
    )
    max_C = 10.0 if model_name == "LogisticRegression" else 5.0

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-3, max_C, log=True),
            "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True),
            **fixed,
        }
        return cross_val_score(
            _build_base_model(model_name, params),
            X,
            y,
            cv=CV_FOLDS,
            scoring="f1_macro",
            n_jobs=-1,
        ).mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = {**study.best_params, **fixed}
    print(f"  [{model_name}] best params: {best}")
    return best


def svc_proba(svc: LinearSVC, X) -> np.ndarray:
    """Convert LinearSVC decision_function to calibrated pseudo-probabilities."""
    return softmax(svc.decision_function(X), axis=1)


def tune_lgbm(X, y, n_trials: int = 75) -> lgb.LGBMClassifier:
    n_classes = len(np.unique(y))
    lgb_train = lgb.Dataset(X, label=y)

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            # ── Tree structure ─────────────────────────────────────────────
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            # ── Learning ───────────────────────────────────────────────────
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            # ── Regularisation ─────────────────────────────────────────────
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            # ── Subsampling ────────────────────────────────────────────────
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # ── Stability ──────────────────────────────────────────────────
            "feature_pre_filter": False,
            "verbose": -1,
            "n_jobs": -1,
        }

        cv_results = lgb.cv(
            params,
            lgb_train,
            nfold=CV_FOLDS,
            stratified=True,
            metrics="multi_logloss",
            callbacks=[lgb.early_stopping(20, verbose=False)],
        )
        return -min(cv_results["valid multi_logloss-mean"])

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {
        "objective": "multiclass",
        "num_class": n_classes,
        "feature_pre_filter": False,
        "n_jobs": -1,
        "verbose": -1,
        **study.best_params,
    }
    print(f"  [LightGBM] best params: {best_params}")
    clf = lgb.LGBMClassifier(**best_params)
    clf.fit(X, y)
    return clf


# ══ PHASE 1 — TF-IDF pipeline ════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 1 — TF-IDF pipeline")
print("=" * 60)

if _cache_exists(C_TFIDF_TR, C_TFIDF_TE, C_TFIDF_FE):
    print("[CACHE HIT] Loading TF-IDF matrices and pipeline.")
    X_tfidf_train = sp.load_npz(C_TFIDF_TR)
    X_tfidf_test = sp.load_npz(C_TFIDF_TE)
    fe = _pkl_load(C_TFIDF_FE)
else:
    fe = get_tfidf_pipeline()
    X_tfidf_train = fe.fit_transform(X_train_res, y_resampled)
    X_tfidf_test = fe.transform(X_test)

    sp.save_npz(C_TFIDF_TR, X_tfidf_train)
    sp.save_npz(C_TFIDF_TE, X_tfidf_test)
    _pkl_save(fe, C_TFIDF_FE)

print(f"[TF-IDF] Train: {X_tfidf_train.shape} | Test: {X_tfidf_test.shape}")


# ══ PHASE 2 — Hyperparameter tuning for base models ══════════════════════════
print("\n" + "=" * 60)
print("PHASE 2 — Base model hyperparameter tuning")
print("=" * 60)

if _cache_exists(C_TUNING):
    print("[CACHE HIT] Loading Optuna tuning results.")
    tuning = _pkl_load(C_TUNING)
    lr_params = tuning["lr_params"]
    svc_params = tuning["svc_params"]
else:
    print("[TUNING] LogisticRegression...")
    lr_params = tune_base_model("LogisticRegression", X_tfidf_train, y_resampled)

    print("[TUNING] LinearSVC...")
    svc_params = tune_base_model("LinearSVC", X_tfidf_train, y_resampled)

    _pkl_save({"lr_params": lr_params, "svc_params": svc_params}, C_TUNING)


# ══ PHASE 3 — OOF stacking ═══════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3 — Out-of-fold (OOF) base model predictions")
print("=" * 60)

if _cache_exists(C_OOF):
    print("[CACHE HIT] Loading OOF arrays.")
    oof_data = _pkl_load(C_OOF)
    oof_lr = oof_data["oof_lr"]
    oof_svc = oof_data["oof_svc"]
    test_lr_meta = oof_data["test_lr_meta"]
    test_svc_meta = oof_data["test_svc_meta"]
else:
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    oof_lr = np.zeros((len(y_resampled), N_CLASSES))
    oof_svc = np.zeros((len(y_resampled), N_CLASSES))
    test_lr_folds: list[np.ndarray] = []
    test_svc_folds: list[np.ndarray] = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tfidf_train, y_resampled)):
        print(f"  Fold {fold + 1}/{CV_FOLDS}...")

        lr_fold = LogisticRegression(**lr_params).fit(
            X_tfidf_train[tr_idx], y_resampled[tr_idx]
        )
        svc_fold = LinearSVC(**svc_params).fit(
            X_tfidf_train[tr_idx], y_resampled[tr_idx]
        )

        oof_lr[val_idx] = lr_fold.predict_proba(X_tfidf_train[val_idx])
        oof_svc[val_idx] = svc_proba(svc_fold, X_tfidf_train[val_idx])

        test_lr_folds.append(lr_fold.predict_proba(X_tfidf_test))
        test_svc_folds.append(svc_proba(svc_fold, X_tfidf_test))

    test_lr_meta = np.mean(test_lr_folds, axis=0)
    test_svc_meta = np.mean(test_svc_folds, axis=0)

    _pkl_save(
        {
            "oof_lr": oof_lr,
            "oof_svc": oof_svc,
            "test_lr_meta": test_lr_meta,
            "test_svc_meta": test_svc_meta,
        },
        C_OOF,
    )

print(f"  OOF LR  shape: {oof_lr.shape}")
print(f"  OOF SVC shape: {oof_svc.shape}")


# ══ PHASE 4 — Psychological feature extraction ════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 4 — Psychological feature extraction")
print("=" * 60)

if _cache_exists(C_PSYCH):
    print("[CACHE HIT] Loading psychological features.")
    psych_data = _pkl_load(C_PSYCH)
    X_psych_train = psych_data["X_psych_train"]
    X_psych_test = psych_data["X_psych_test"]
    psych = psych_data["psych"]
else:
    psych = Psychological()
    psych.fit(X_train_res, None)

    print("  Extracting training features...")
    X_psych_train = psych.transform(X_train_res)

    print("  Extracting test features...")
    X_psych_test = psych.transform(X_test)

    _pkl_save(
        {
            "X_psych_train": X_psych_train,
            "X_psych_test": X_psych_test,
            "psych": psych,
        },
        C_PSYCH,
    )

print(f"  Psych feature count: {X_psych_train.shape[1]}")


# ══ PHASE 5 — Meta-feature assembly ══════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 5 — Meta-feature assembly")
print("=" * 60)

X_meta_train = np.hstack([oof_lr, oof_svc, X_psych_train])
X_meta_test = np.hstack([test_lr_meta, test_svc_meta, X_psych_test])

n_psych = X_psych_train.shape[1]
print(f"  Meta-train shape : {X_meta_train.shape}")
print(f"  Meta-test  shape : {X_meta_test.shape}")
print(f"  Breakdown        : LR({N_CLASSES}) + SVC({N_CLASSES}) + Psych({n_psych})")


# ══ PHASE 6 — LightGBM meta-learner ══════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 6 — LightGBM meta-learner (Optuna-tuned)")
print("=" * 60)

lgbm = tune_lgbm(X_meta_train, y_resampled)


# ══ PHASE 7 — Evaluation ═════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 7 — Evaluation on held-out test set")
print("=" * 60)

y_pred = lgbm.predict(X_meta_test)
clf_report = classification_report(y_test, y_pred, target_names=le.classes_)
print(clf_report)

report_text = (
    "EVALUATION REPORT — STACKING v1.0.2\n"
    + "=" * 50
    + "\n\n"
    + "Architecture : TF-IDF (LR + SVC via OOF) + Psych → LightGBM\n"
    + f"Meta-features: LR proba ({N_CLASSES}) + SVC proba ({N_CLASSES}) + Psych ({n_psych})\n\n"
    + "Model: LightGBM (meta-learner)\n"
    + "-" * 30
    + "\n"
    + clf_report
    + "\n"
)


# ══ PHASE 8 — Retrain base models on full training data for inference ══════════
print("\n" + "=" * 60)
print("PHASE 8 — Retrain base models on full data")
print("=" * 60)

lr_final = LogisticRegression(**lr_params).fit(X_tfidf_train, y_resampled)
svc_final = LinearSVC(**svc_params).fit(X_tfidf_train, y_resampled)

del X_tfidf_train, X_tfidf_test, X_meta_train, X_meta_test
gc.collect()


# ══ PHASE 9 — Persist artifacts ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"PHASE 9 — Saving artifacts to {ARTIFACT_CLEAN}")
print("=" * 60)

out = ARTIFACT_CLEAN

artifacts = {
    "label_encoder.pkl": le,
    "tfidf_pipeline.pkl": fe,
    "base_lr.pkl": lr_final,
    "base_svc.pkl": svc_final,
    "psych_extractor.pkl": psych,
    "meta_lgbm.pkl": lgbm,
}
for fname, obj in artifacts.items():
    with open(out / fname, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"  Saved {fname}")

with open(out / "evaluation_v1.0.2.txt", "w", encoding="utf-8") as fh:
    fh.write(report_text)

print("\n[DONE] All artifacts saved.")


# ══ Inference helper ══════════════════════════════════════════════════════════


class StackingInferencePipeline:
    """
    Wraps all saved artifacts for single-text or batch inference.

    Usage:
        pipe = StackingInferencePipeline.load(ARTIFACT_CLEAN)
        labels = pipe.predict(["I've been feeling really hopeless lately..."])
        probas = pipe.predict_proba(["..."])  # [{label: prob, ...}]
    """

    def __init__(self, le, tfidf, lr, svc, psych, lgbm):
        self.le = le
        self.tfidf = tfidf
        self.lr = lr
        self.svc = svc
        self.psych = psych
        self.lgbm = lgbm

    @classmethod
    def load(cls, artifact_dir):
        d = Path(artifact_dir)

        def _load(name):
            with open(d / name, "rb") as fh:
                return pickle.load(fh)

        return cls(
            le=_load("label_encoder.pkl"),
            tfidf=_load("tfidf_pipeline.pkl"),
            lr=_load("base_lr.pkl"),
            svc=_load("base_svc.pkl"),
            psych=_load("psych_extractor.pkl"),
            lgbm=_load("meta_lgbm.pkl"),
        )

    def _meta_features(self, texts: list[str]) -> np.ndarray:
        texts_lower = [t.lower() for t in texts]
        X_tf = self.tfidf.transform(texts_lower)
        lr_p = self.lr.predict_proba(X_tf)
        svc_p = svc_proba(self.svc, X_tf)
        psy_p = self.psych.transform(texts_lower)
        return np.hstack([lr_p, svc_p, psy_p])

    def predict(self, texts: list[str]) -> list[str]:
        idx = self.lgbm.predict(self._meta_features(texts))
        return self.le.inverse_transform(idx).tolist()

    def predict_proba(self, texts: list[str]) -> list[dict]:
        probas = self.lgbm.predict_proba(self._meta_features(texts))
        return [dict(zip(self.le.classes_, row)) for row in probas]


# ── Quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[INFERENCE DEMO]")
    pipe = StackingInferencePipeline(le, fe, lr_final, svc_final, psych, lgbm)

    sample = ["I've been feeling really hopeless and can't get out of bed for weeks."]
    print(f"  Input: {sample[0]}")
    print(f"  Prediction: {pipe.predict(sample)}")

    top3 = sorted(pipe.predict_proba(sample)[0].items(), key=lambda x: -x[1])[:3]
    print("  Top 3 probabilities:")
    for label, prob in top3:
        print(f"    {label}: {prob:.3f}")
