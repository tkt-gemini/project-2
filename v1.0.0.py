import gc
import pickle

import nltk
import numpy as np
import optuna
import pandas as pd
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

print("Label encoding mapping:")
for idx, label in enumerate(le.classes_):
    print(f"  {idx}: {label}")

X_train, X_test, y_train, y_test = train_test_split(
    df["post_cleaned"],
    df["_subreddit"],
    test_size=TEST_SIZE,
    stratify=df["_subreddit"],
    random_state=RANDOM_SEED,
)


def get_feature_engineer():
    return FeatureUnion(
        [
            (
                "dense_branch",
                Pipeline(
                    [
                        ("psych", Psychological()),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                        (
                            "to_sparse",
                            DenseToSparse(),
                        ),
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
                                                        chi2,
                                                        k=int(SELECT_K * 0.75),
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
                                                        chi2,
                                                        k=int(SELECT_K * 0.25),
                                                    ),
                                                ),
                                            ]
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "scaler",
                            MaxAbsScaler(),
                        ),
                    ]
                ),
            ),
        ]
    )


def tune_and_train(model_name: str, X, y, n_trials: int = 30):
    fixed_params = {
        "max_iter": 2000,
        "dual": False,
        "random_state": RANDOM_SEED,
        "class_weight": "balanced",
    }

    def objective(trial):
        if model_name == "LinearSVC":
            params = {
                "C": trial.suggest_float("C", 1e-3, 5.0, log=True),
                "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True),
            }
            model = LinearSVC(**params, **fixed_params)

        elif model_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
                "tol": trial.suggest_float("tol", 1e-4, 1e-3, log=True),
            }
            model = LogisticRegression(**params, **fixed_params)

        score = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
        return score.mean()

    print(f"[TUNING] {model_name}: {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = {**study.best_params, **fixed_params}

    if model_name == "LinearSVC":
        best_model = LinearSVC(**best_params)

    elif model_name == "LogisticRegression":
        best_model = LogisticRegression(**best_params)

    print(f"  [+] Best params: {best_params}")

    best_model.fit(X, y)
    return best_model


# ── Under-sampling ──
target_majority = 15_000
train_class_counts = pd.Series(y_train).value_counts()
under_strategy = {
    label: min(count, target_majority) for label, count in train_class_counts.items()
}

print("\n[GLOBAL] Undersampling...")
rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_SEED)
print("Done!")

assert y_train.index.is_unique, (
    "y_train index must be unique for index-based resampling. "
    "Call reset_index(drop=True) on the source DataFrame first."
)

idx_2d = np.array(y_train.index).reshape(-1, 1)
idx_resampled_2d, y_resampled = rus.fit_resample(idx_2d, y_train)
idx_resampled = idx_resampled_2d.flatten()

X_train_res = X_train.loc[idx_resampled]

print(f"\n{'=' * 50}\nPIPELINE\n{'=' * 50}")

print("[FEATURE ENGINEER]")
fe = get_feature_engineer()
print("Fitting...")
fe.fit(X_train_res, y_resampled)
X_transformed = fe.transform(X_train_res)
print("Done!")

print("[MODEL]")
trained_models = {
    "LogisticRegression": None,
    "LinearSVC": None,
}

for k in trained_models.keys():
    trained_models[k] = tune_and_train(k, X_transformed, y_resampled)

del X_transformed
gc.collect()

print("Evaluating on test set...")
X_test_transformed = fe.transform(X_test)

report_text = "EVALUATION REPORT\n" + "=" * 50 + "\n"

for model_name, model in trained_models.items():
    y_pred = model.predict(X_test_transformed)
    print(f"\n--- {model_name} ---")
    clf_report = classification_report(y_test, y_pred, target_names=le.classes_)
    print(clf_report)
    report_text += f"\nModel: {model_name}\n{'-' * 30}\n{clf_report}\n"

# ── Persist artefacts ─────────────────────────────────────────────────────
print(f"Saving artefacts to {ARTIFACT_DIR}...")

with open(ARTIFACT_DIR / "label_encoder.pkl", "wb") as fh:
    pickle.dump(le, fh)
with open(ARTIFACT_DIR / "feature_engineer.pkl", "wb") as fh:
    pickle.dump(fe, fh)

for model_name, model in trained_models.items():
    model_file, pipeline_file = MODEL_FILE_MAP[model_name]

    with open(ARTIFACT_DIR / model_file, "wb") as fh:
        pickle.dump(model, fh)

    inference_pipeline = Pipeline(
        [
            ("preprocessing", fe),
            ("classifier", model),
        ]
    )
    with open(ARTIFACT_DIR / pipeline_file, "wb") as fh:
        pickle.dump(inference_pipeline, fh)
    del inference_pipeline

with open(ARTIFACT_DIR / "evaluation_report.txt", "w", encoding="utf-8") as fh:
    fh.write(report_text)

# ── Free RAM before next iteration ────────────────────────────────────────
del X_test_transformed, fe, trained_models
gc.collect()

print("\n[FINISHED] All pipelines complete.")
