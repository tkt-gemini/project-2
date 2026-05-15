"""
v1.0.0 + LightGBM (flat) — ablation study version
────────────────────────────────────────────────────
Thêm LightGBM trực tiếp trên cùng feature set (TF-IDF + Psych) để tách biệt
đóng góp của "classifier tốt hơn" vs "kiến trúc stacking" trong v1.0.1.

CACHING STRATEGY
────────────────
  Cache file               Phase guarded
  ───────────────────────  ─────────────────────────────────────────
  cache/01_df_filtered.pkl Phase 0 – noise + clinical filter + dropna
  cache/02_split.pkl       Phase 0 – train/test split + undersampling
  cache/03_fe_Xtrain.npz   Phase 1 – FeatureUnion fit-transform (train)
  cache/03_fe_Xtest.npz    Phase 1 – FeatureUnion transform (test)
  cache/03_fe_obj.pkl      Phase 1 – fitted FeatureUnion object
  cache/04_tuning.pkl      Phase 2 – Optuna best params (LR + SVC + LightGBM)

Xoá một cache file để force re-compute phase đó và tất cả phase sau.
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler, StandardScaler
from sklearn.svm import LinearSVC

from config import (
    ARTIFACT_DIR,
    CACHE_DIR,
    CV_FOLDS,
    LABEL_MAPPING,
    MODEL_FILE_MAP,
    RANDOM_SEED,
    SELECT_K,
    TEST_SIZE,
    TFIDF_FEATS,
)
from helper import DenseToSparse, Filter, Psychological

nltk.download("vader_lexicon", quiet=True)
optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── Cache helpers ──────────────────────────────────────────────────────────────


def _pkl_save(obj, path: Path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(obj, fh)
    print(f"  [CACHE] Saved -> {path}")


def _pkl_load(path: Path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    print(f"  [CACHE] Loaded <- {path}")
    return obj


def _cache_exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


# ── Cache paths ────────────────────────────────────────────────────────────────

C_FILTERED = CACHE_DIR / "01_df_filtered.pkl"
C_SPLIT = CACHE_DIR / "02_split.pkl"
C_FE_TR = CACHE_DIR / "03_fe_Xtrain.npz"
C_FE_TE = CACHE_DIR / "03_fe_Xtest.npz"
C_FE_OBJ = CACHE_DIR / "03_fe_obj.pkl"
C_TUNING = CACHE_DIR / "04_tuning.pkl"


# ── Helper functions ───────────────────────────────────────────────────────────


def get_feature_engineer():
    return FeatureUnion(
        [
            (
                "dense_branch",
                Pipeline(
                    [
                        ("psych", Psychological()),
                        ("scaler", StandardScaler()),
                        ("to_sparse", DenseToSparse()),
                    ]
                ),
            ),
            (
                "sparse_branch",
                Pipeline(
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
                                                    SelectKBest(
                                                        chi2, k=int(SELECT_K * 0.75)
                                                    ),
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
                                                    SelectKBest(
                                                        chi2, k=int(SELECT_K * 0.25)
                                                    ),
                                                ),
                                            ]
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        ("scaler", MaxAbsScaler()),
                    ]
                ),
            ),
        ]
    )


def tune_linear(model_name: str, X, y, n_trials: int = 30) -> dict:
    """Optuna search cho LR hoac LinearSVC, tra ve best params dict."""
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
        model = (
            LinearSVC(**params)
            if model_name == "LinearSVC"
            else LogisticRegression(**params)
        )
        return cross_val_score(
            model, X, y, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1
        ).mean()

    print(f"[TUNING] {model_name}: {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = {**study.best_params, **fixed}
    print(f"  [{model_name}] best params: {best}")
    return best


def tune_lgbm(X, y, n_trials: int = 50) -> dict:
    """
    Optuna search cho LightGBM tren full feature matrix (TF-IDF + Psych).
    Tach biet tune vs fit de tan dung cache o phase 2.
    """
    n_classes = len(np.unique(y))
    lgb_train = lgb.Dataset(X, label=y)

    def objective(trial):
        params = {
            "objective": "multiclass",
            "num_class": n_classes,
            "metric": "multi_logloss",
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
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

    print(f"[TUNING] LightGBM (flat): {n_trials} trials...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best = {
        "objective": "multiclass",
        "num_class": n_classes,
        "feature_pre_filter": False,
        "n_jobs": -1,
        "verbose": -1,
        **study.best_params,
    }
    print(f"  [LightGBM] best params: {best}")
    return best


# ══ PHASE 0 — Data loading, preprocessing & splitting ════════════════════════
print("=" * 60)
print("PHASE 0 - Data loading & preprocessing")
print("=" * 60)

if _cache_exists(C_FILTERED, C_SPLIT):
    print("[CACHE HIT] Skipping noise/clinical filter and split.")
    df_info = _pkl_load(C_FILTERED)
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

    X_train, X_test, y_train, y_test = train_test_split(
        df["post_cleaned"],
        df["_subreddit"],
        test_size=TEST_SIZE,
        stratify=df["_subreddit"],
        random_state=RANDOM_SEED,
    )

    target_majority = 15_000
    train_class_counts = pd.Series(y_train).value_counts()
    under_strategy = {
        label: min(count, target_majority)
        for label, count in train_class_counts.items()
    }
    rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_SEED)

    assert y_train.index.is_unique, (
        "y_train index must be unique. Call reset_index(drop=True) first."
    )
    idx_2d = np.array(y_train.index).reshape(-1, 1)
    idx_resampled_2d, y_resampled = rus.fit_resample(idx_2d, y_train)
    idx_resampled = idx_resampled_2d.flatten()

    X_train_res = X_train.loc[idx_resampled].reset_index(drop=True)
    y_resampled = np.array(y_resampled)

    print(
        f"\n[DATA] Train: {len(X_train_res):,} | Test: {len(X_test):,} | Classes: {N_CLASSES}"
    )

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


# ══ PHASE 1 — FeatureUnion fit-transform (TF-IDF + Psychological) ════════════
print("\n" + "=" * 60)
print("PHASE 1 - FeatureUnion (TF-IDF + Psychological features)")
print("=" * 60)

if _cache_exists(C_FE_TR, C_FE_TE, C_FE_OBJ):
    print("[CACHE HIT] Loading feature matrices and FeatureUnion object.")
    X_transformed = sp.load_npz(C_FE_TR)
    X_test_transformed = sp.load_npz(C_FE_TE)
    fe = _pkl_load(C_FE_OBJ)
else:
    fe = get_feature_engineer()
    print("Fitting FeatureUnion (TF-IDF + Psychological)...")
    fe.fit(X_train_res, y_resampled)
    X_transformed = fe.transform(X_train_res)
    X_test_transformed = fe.transform(X_test)

    sp.save_npz(C_FE_TR, X_transformed)
    sp.save_npz(C_FE_TE, X_test_transformed)
    _pkl_save(fe, C_FE_OBJ)

print(f"[FE] Train: {X_transformed.shape} | Test: {X_test_transformed.shape}")


# ══ PHASE 2 — Hyperparameter tuning ══════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 2 - Hyperparameter tuning (LR + SVC + LightGBM)")
print("=" * 60)

if _cache_exists(C_TUNING):
    print("[CACHE HIT] Loading Optuna tuning results.")
    tuning = _pkl_load(C_TUNING)
    lr_params = tuning["lr_params"]
    svc_params = tuning["svc_params"]
    lgb_params = tuning["lgb_params"]
else:
    print("[TUNING] LogisticRegression...")
    lr_params = tune_linear("LogisticRegression", X_transformed, y_resampled)

    print("[TUNING] LinearSVC...")
    svc_params = tune_linear("LinearSVC", X_transformed, y_resampled)

    print("[TUNING] LightGBM (flat)...")
    lgb_params = tune_lgbm(X_transformed, y_resampled)

    _pkl_save(
        {"lr_params": lr_params, "svc_params": svc_params, "lgb_params": lgb_params},
        C_TUNING,
    )


# ══ PHASE 3 — Train final models ══════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 3 - Train final models on full training data")
print("=" * 60)

lr_model = LogisticRegression(**lr_params).fit(X_transformed, y_resampled)
svc_model = LinearSVC(**svc_params).fit(X_transformed, y_resampled)

lgb_model = lgb.LGBMClassifier(**lgb_params)
lgb_model.fit(X_transformed, y_resampled)

trained_models = {
    "LogisticRegression": lr_model,
    "LinearSVC": svc_model,
    "LightGBM_flat": lgb_model,
}

del X_transformed
gc.collect()


# ══ PHASE 4 — Evaluation ══════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PHASE 4 - Evaluation on held-out test set")
print("=" * 60)

report_text = "EVALUATION REPORT - CLEANED DATA\n" + "=" * 50 + "\n"

for model_name, model in trained_models.items():
    y_pred = model.predict(X_test_transformed)
    print(f"\n--- {model_name} ---")
    clf_report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(clf_report)
    report_text += f"\nModel: {model_name}\n{'-' * 30}\n{clf_report}\n"

del X_test_transformed
gc.collect()
print("  [CLEANUP] X_test_transformed freed.")


# ══ PHASE 5 — Persist artifacts ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"PHASE 5 - Saving artifacts to {ARTIFACT_DIR}")
print("=" * 60)

with open(ARTIFACT_DIR / "label_encoder.pkl", "wb") as fh:
    pickle.dump(le, fh)
with open(ARTIFACT_DIR / "feature_engineer.pkl", "wb") as fh:
    pickle.dump(fe, fh)

# LR + SVC: luu ca model rieng lan inference pipeline (giu nguyen cau truc goc)
for model_name, model in [("LogisticRegression", lr_model), ("LinearSVC", svc_model)]:
    model_file, pipeline_file = MODEL_FILE_MAP[model_name]
    with open(ARTIFACT_DIR / model_file, "wb") as fh:
        pickle.dump(model, fh)
    inference_pipeline = Pipeline([("preprocessing", fe), ("classifier", model)])
    with open(ARTIFACT_DIR / pipeline_file, "wb") as fh:
        pickle.dump(inference_pipeline, fh)
    print(f"  Saved {model_file} + {pipeline_file}")

# LightGBM flat: luu rieng (khong co trong MODEL_FILE_MAP goc)
with open(ARTIFACT_DIR / "model_lgbm_flat.pkl", "wb") as fh:
    pickle.dump(lgb_model, fh)
print("  Saved model_lgbm_flat.pkl")

with open(ARTIFACT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as fh:
    fh.write(report_text)
print("  Saved evaluation_report.txt")

del fe, trained_models
gc.collect()

print("\n[DONE] All artifacts saved.")
