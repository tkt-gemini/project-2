import gc
import pickle

import lightgbm as lgb
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
from sklearn.preprocessing import LabelEncoder, MaxAbsScaler
from sklearn.svm import LinearSVC

from config import *
from helper import Filter, Psychological

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

X_cleaned_train, X_cleaned_test, X_masked_train, X_masked_test, y_train, y_test = (
    train_test_split(
        df["post_cleaned"],
        df["post_masked"],
        df["_subreddit"],
        test_size=TEST_SIZE,
        stratify=df["_subreddit"],
        random_state=RANDOM_SEED,
    )
)


def get_feature_engineer():
    return Pipeline(
        [
            (
                "extraction",
                FeatureUnion(
                    [
                        # Branch 1: Psychological features (full — no SelectKBest)
                        ("psych_branch", Psychological()),
                        # Branch 2: Word TF-IDF → chi2, top 24 000 features
                        (
                            "word_tfidf_branch",
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
                        # Branch 3: Char TF-IDF → chi2, top 8 000 features
                        (
                            "char_tfidf_branch",
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


def tune_and_train(model_name: str, X, y, n_trials: int = 10):
    def objective(trial):
        if model_name == "LinearSVC":
            params = {
                "C": trial.suggest_float("C", 1e-4, 1.0, log=True),
                "max_iter": 1000,
                "dual": False,
                "random_state": RANDOM_SEED,
                "class_weight": "balanced",
            }
            model = LinearSVC(**params)

        elif model_name == "LogisticRegression":
            params = {
                "C": trial.suggest_float("C", 1e-4, 1.0, log=True),
                "max_iter": 2000,
                "class_weight": "balanced",
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
            }
            model = LogisticRegression(**params)

        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 300, 800),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 0.1, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 20, 127),
                "max_depth": -1,
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
                "bagging_freq": 5,
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
                "verbose": -1,
                "class_weight": "balanced",
            }
            model = lgb.LGBMClassifier(**params)

        score = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="f1_macro", n_jobs=1)
        return score.mean()

    print(f"[TUNING] {model_name} (max {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED
    best_params["class_weight"] = "balanced"

    # Restore full parallelism for the final one-shot training run.
    if model_name == "LinearSVC":
        best_params["max_iter"] = 1000
        best_params["dual"] = False
        best_model = LinearSVC(**best_params)

    elif model_name == "LogisticRegression":
        best_params["max_iter"] = 1000
        best_params["n_jobs"] = -1
        best_model = LogisticRegression(**best_params)

    elif model_name == "LightGBM":
        best_params["n_jobs"] = -1
        best_params["verbose"] = -1
        best_model = lgb.LGBMClassifier(**best_params)

    print(f"  [+] Best params: {best_params}")

    best_model.fit(X, y)
    return best_model


# ── Under-sampling (computed once, applied to both CLEANED and MASKED runs) ──
target_majority = 15_000
train_class_counts = pd.Series(y_train).value_counts()
under_strategy = {
    label: min(count, target_majority) for label, count in train_class_counts.items()
}

print("\n[GLOBAL] Undersampling...")
rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_SEED)

# FIX: Assert index uniqueness before resampling. train_test_split on a
# reset_index DataFrame always gives unique indices, but this guard prevents
# silent duplicate-row bugs if the calling code changes in the future.
assert y_train.index.is_unique, (
    "y_train index must be unique for index-based resampling. "
    "Call reset_index(drop=True) on the source DataFrame first."
)

idx_2d = np.array(y_train.index).reshape(-1, 1)
idx_resampled_2d, y_resampled = rus.fit_resample(idx_2d, y_train)
idx_resampled = idx_resampled_2d.flatten()

X_cleaned_train_res = X_cleaned_train.loc[idx_resampled]
X_masked_train_res = X_masked_train.loc[idx_resampled]

# Map short names used for file naming
_MODEL_FILE_MAP = {
    "LinearSVC": ("model_svc.pkl", "inference_pipeline_svc.pkl"),
    "LogisticRegression": ("model_lr.pkl", "inference_pipeline_lr.pkl"),
    "LightGBM": ("model_lgb.pkl", "inference_pipeline_lgb.pkl"),
}

run_configs = [
    ("CLEANED", X_cleaned_train_res, X_cleaned_test, ARTIFACT_CLEAN),
    ("MASKED", X_masked_train_res, X_masked_test, ARTIFACT_MASK),
]

for data_type, X_train_res, X_test, out_dir in run_configs:
    print(f"\n{'=' * 50}\nPIPELINE: {data_type}\n{'=' * 50}")

    fe = get_feature_engineer()
    print(f"[{data_type}] Fitting Feature Engineer...")
    X_transformed = fe.fit_transform(X_train_res, y_resampled)

    best_lgb = tune_and_train("LightGBM", X_transformed, y_resampled, 50)
    best_svc = tune_and_train("LinearSVC", X_transformed, y_resampled)
    best_lr = tune_and_train("LogisticRegression", X_transformed, y_resampled)

    del X_transformed
    gc.collect()

    print(f"\n[{data_type}] Evaluating on test set...")
    X_test_transformed = fe.transform(X_test)

    trained_models = {
        "LinearSVC": best_svc,
        "LogisticRegression": best_lr,
        "LightGBM": best_lgb,
    }

    report_text = f"EVALUATION REPORT — {data_type} DATA\n" + "=" * 50 + "\n"

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test_transformed)
        print(f"\n--- {model_name} ({data_type}) ---")
        clf_report = classification_report(y_test, y_pred, target_names=le.classes_)
        print(clf_report)
        report_text += f"\nModel: {model_name}\n{'-' * 30}\n{clf_report}\n"

    # ── Persist artefacts ─────────────────────────────────────────────────────
    print(f"[{data_type}] Saving artefacts to {out_dir}...")

    with open(out_dir / "label_encoder.pkl", "wb") as fh:
        pickle.dump(le, fh)
    with open(out_dir / "feature_engineer.pkl", "wb") as fh:
        pickle.dump(fe, fh)

    for model_name, model in trained_models.items():
        model_file, pipeline_file = _MODEL_FILE_MAP[model_name]

        with open(out_dir / model_file, "wb") as fh:
            pickle.dump(model, fh)

        inference_pipeline = Pipeline(
            [
                ("preprocessing", fe),
                ("classifier", model),
            ]
        )
        with open(out_dir / pipeline_file, "wb") as fh:
            pickle.dump(inference_pipeline, fh)
        del inference_pipeline

    with open(out_dir / "evaluation_report.txt", "w", encoding="utf-8") as fh:
        fh.write(report_text)

    # ── Free RAM before next iteration ────────────────────────────────────────
    del X_test_transformed, fe, best_svc, best_lr, best_lgb, trained_models
    gc.collect()

print("\n[FINISHED] All pipelines complete.")
