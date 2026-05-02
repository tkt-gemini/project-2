"""
pipeline/models.py — Model search, evaluation, and artifact saving (Stages 7–9).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from pipeline.config import (
    ARTIFACT_DIR,
    CV_FOLDS,
    MACRO_F1,
    RANDOM_SEED,
    SELECT_K,
    TEST_SIZE,
    TFIDF_FEATS,
    TIER1_CAP,
)


def get_model_candidates() -> dict:
    return {
        "LinearSVC": (
            LinearSVC(max_iter=2000, class_weight="balanced", dual=True),
            {"C": [0.01, 0.1, 0.5, 1.0, 5.0]},
        ),
        "LogisticRegression": (
            LogisticRegression(
                max_iter=1000, class_weight="balanced", solver="saga", n_jobs=-1,
            ),
            {"C": [0.01, 0.1, 1.0, 10.0], "penalty": ["l1", "l2"]},
        ),
        "LightGBM": (
            lgb.LGBMClassifier(
                n_estimators    = 800,
                class_weight    = "balanced",
                random_state    = RANDOM_SEED,
                n_jobs          = -1,
                verbose         = -1,
                colsample_bytree= 0.8,
                subsample       = 0.8,
                subsample_freq  = 1,
                reg_alpha       = 0.1,
                reg_lambda      = 0.1,
                min_child_weight= 1e-3,
            ),
            {
                "num_leaves"       : [63, 127, 255],
                "learning_rate"    : [0.03, 0.05, 0.1],
                "min_child_samples": [10, 20, 50],
            },
        ),
    }


def run_grid_search(
    X_train: sp.csr_matrix, y_train: np.ndarray,
    model_name: str = "LinearSVC", cv: int = CV_FOLDS,
) -> tuple:
    print(f"\n[STAGE 7] GridSearchCV — {model_name}")
    estimator, param_grid = get_model_candidates()[model_name]
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)
    print(f"  {cv} folds × {n_combos} combos = {cv * n_combos} fits")

    gs = GridSearchCV(
        estimator, param_grid, cv=skf, scoring=MACRO_F1,
        n_jobs=1, verbose=2, return_train_score=False, refit=True,
    )
    gs.fit(X_train, y_train)
    print(f"\n  Best params: {gs.best_params_}")
    print(f"  Best CV macro-F1: {gs.best_score_:.4f}")
    cv_results = pd.DataFrame(gs.cv_results_).sort_values("mean_test_score", ascending=False)

    best = gs.best_estimator_
    if model_name == "LinearSVC":
        best = CalibratedClassifierCV(best, cv="prefit", method="sigmoid")
        best.fit(X_train, y_train)

    return best, cv_results


def run_optuna_search(
    X_train: sp.csr_matrix, y_train: np.ndarray,
    model_name: str = "LinearSVC", n_trials: int = 50, cv: int = CV_FOLDS,
) -> tuple:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\n[STAGE 7] Optuna — {model_name}, {n_trials} trials")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)

    def _base_clf(trial: "optuna.Trial"):
        """Uncalibrated estimator for Optuna CV scoring."""
        if model_name == "LinearSVC":
            return LinearSVC(
                C=trial.suggest_float("C", 1e-3, 10.0, log=True),
                max_iter=2000, class_weight="balanced", dual=True,
            )
        if model_name == "LogisticRegression":
            return LogisticRegression(
                C=trial.suggest_float("C", 1e-3, 10.0, log=True),
                penalty=trial.suggest_categorical("penalty", ["l1", "l2"]),
                max_iter=1000, solver="saga", class_weight="balanced", n_jobs=-1,
            )
        return lgb.LGBMClassifier(
            n_estimators     = trial.suggest_int("n_estimators", 200, 1200),
            num_leaves       = trial.suggest_int("num_leaves", 31, 255),
            learning_rate    = trial.suggest_float("lr", 0.01, 0.2, log=True),
            min_child_samples= trial.suggest_int("min_child_samples", 5, 100),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0),
            subsample        = trial.suggest_float("subsample", 0.6, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            reg_lambda       = trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )

    def _final_clf(best_trial: "optuna.Trial"):
        """Calibrated estimator for final refit — reconstruct từ params dict."""
        p = best_trial.params
        if model_name == "LinearSVC":
            return CalibratedClassifierCV(
                LinearSVC(C=p["C"],
                          max_iter=2000, class_weight="balanced", dual=True),
                cv=3, method="sigmoid",
            )
        if model_name == "LogisticRegression":
            return LogisticRegression(
                C=p["C"], penalty=p["penalty"],
                max_iter=1000, solver="saga", class_weight="balanced", n_jobs=-1,
            )
        return lgb.LGBMClassifier(
            n_estimators     = p["n_estimators"],
            num_leaves       = p["num_leaves"],
            learning_rate    = p["lr"],
            min_child_samples= p["min_child_samples"],
            colsample_bytree = p["colsample_bytree"],
            subsample        = p["subsample"],
            reg_alpha        = p["reg_alpha"],
            reg_lambda       = p["reg_lambda"],
            class_weight="balanced", random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )

    def objective(trial):
        clf    = _base_clf(trial)
        scores = []
        for ti, vi in skf.split(X_train, y_train):
            clf.fit(X_train[ti], y_train[ti])
            scores.append(f1_score(y_train[vi], clf.predict(X_train[vi]),
                                   average="macro", zero_division=0))
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Best macro-F1: {study.best_value:.4f} | params: {study.best_params}")
    best = _final_clf(study.best_trial)
    best.fit(X_train, y_train)
    return best, study


def evaluate(
    model: object, X_test: sp.csr_matrix, y_test: np.ndarray,
    le: LabelEncoder, artifact_dir: Path = ARTIFACT_DIR,
) -> dict:
    print("\n[STAGE 8] Evaluation on holdout test set")
    print("=" * 65)
    y_pred      = model.predict(X_test)
    class_names = le.classes_
    report_str  = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    report_dict = classification_report(y_test, y_pred, target_names=class_names,
                                        zero_division=0, output_dict=True)
    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    print(report_str)
    print(f"  macro-F1    : {macro_f1:.4f}")
    print(f"  weighted-F1 : {weighted_f1:.4f}")
    print("=" * 65)

    cm_df = pd.DataFrame(confusion_matrix(y_test, y_pred), index=class_names, columns=class_names)
    cm_df.to_csv(artifact_dir / "confusion_matrix.csv")

    result = {
        "macro_f1"   : float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class"  : {c: {k: float(v) for k, v in report_dict[c].items()} for c in class_names},
        "n_test"     : int(len(y_test)),
        "timestamp"  : datetime.now().isoformat(),
    }
    with open(artifact_dir / "eval_report.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def save_model(
    model: object, le: LabelEncoder, model_name: str,
    eval_result: dict, artifact_dir: Path = ARTIFACT_DIR,
) -> None:
    print("\n[STAGE 9] Saving model artifacts...")
    joblib.dump(model, artifact_dir / "model.joblib",         compress=3)
    joblib.dump(le,    artifact_dir / "label_encoder.joblib", compress=3)
    meta = {
        "model_name": model_name,
        "classes"   : le.classes_.tolist(),
        "macro_f1"  : eval_result["macro_f1"],
        "trained_at": datetime.now().isoformat(),
        "config"    : {
            "random_seed": RANDOM_SEED, "test_size": TEST_SIZE, "cv_folds": CV_FOLDS,
            "tfidf_feats": TFIDF_FEATS, "select_k": SELECT_K, "tier1_cap": TIER1_CAP,
        },
    }
    with open(artifact_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Artifacts in {artifact_dir}/:")
    for p in sorted(artifact_dir.iterdir()):
        print(f"    {p.name:<35} {p.stat().st_size // 1024:>7} KB")
