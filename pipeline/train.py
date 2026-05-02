"""
pipeline/train.py — Training orchestrators, label mapping, undersampling, and CLI runners.

Orchestration flow:
    _load_and_split     → Load CSV, label mapping, train/test split, undersample
    _get_psych_features → Extract or load cached psych features
    _train_one_variant  → Fit FE → model search → evaluate → save
    run_training        → Single model × single variant
    run_all_models      → 3 models × 2 variants = 6 runs
    run_preprocessing   → Stages 1–4
    run_eval_only       → Load artifacts and re-evaluate
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pipeline.config import (
    ARTIFACT_CLEAN,
    ARTIFACT_DIR,
    ARTIFACT_MASKED,
    CLEAN_CSV,
    LABEL_MAPPING,
    RANDOM_SEED,
    RAW_DATA_DIR,
    TEST_SIZE,
    TIER1_CAP,
    FP_THRESHOLD,
    BATCH_SIZE,
    TrainingConfig,
)
from pipeline.features import FeatureEngineer, PsychologicalExtractor
from pipeline.models import (
    evaluate,
    get_model_candidates,
    run_grid_search,
    run_optuna_search,
    save_model,
)
from pipeline.preprocess import filter_and_mask, leakage_audit, load_raw_dataset, noise_filter


# ═════════════════════════════════════════════════════════════════════════════
# LABEL MAPPING & TIERED UNDERSAMPLING (Stages 5a/5b)
# ═════════════════════════════════════════════════════════════════════════════

def apply_label_mapping(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STAGE 5a] Label Mapping")
    before = df["subreddit"].unique().tolist()
    df     = df.copy()
    df["subreddit"] = df["subreddit"].map(LABEL_MAPPING)
    n_bad = df["subreddit"].isna().sum()
    if n_bad:
        missing = [s for s in before if s not in LABEL_MAPPING]
        print(f"  ⚠ Drop {n_bad:,} unmapped rows: {missing}")
        df = df.dropna(subset=["subreddit"])
    print(f"  {len(before)} → {df['subreddit'].nunique()} labels")
    return df.reset_index(drop=True)


def tiered_undersample(
    df: pd.DataFrame, cap: int = TIER1_CAP, seed: int = RANDOM_SEED
) -> pd.DataFrame:
    """
    Undersample classes > cap xuống đúng cap.
    Classes <= cap giữ nguyên.
    """
    print(f"\n[STAGE 5b] Tiered Undersampling (cap={cap:,})")
    rng    = np.random.default_rng(seed)
    counts = df["subreddit"].value_counts()
    parts  = []

    print(f"  {'Class':<26} {'Before':>8} {'After':>8}")
    print("  " + "─" * 46)
    for lbl, cnt in counts.items():
        sub = df[df["subreddit"] == lbl]
        if cnt > cap:
            sub = sub.iloc[rng.choice(len(sub), size=cap, replace=False)]
        parts.append(sub)
        print(f"  {lbl:<26} {cnt:>8,} {len(sub):>8,}")

    result = pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)
    print("  " + "─" * 46)
    print(f"  {'TOTAL':<26} {len(df):>8,} {len(result):>8,}")
    r_before = counts.max() / counts.min()
    r_after  = min(counts.max(), cap) / min(counts.min(), cap)
    print(f"\n  Ratio: {r_before:.1f}x → {r_after:.1f}x | retained {len(result)/len(df)*100:.1f}%")
    return result


# ═════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _load_and_split(cfg: TrainingConfig) -> tuple:
    """
    Load CSV → label mapping → train/test split → tiered undersampling.
    Returns: (df_train, df_test, le, y_train, y_test)
    """
    df = pd.read_csv(cfg.data_csv)
    df = apply_label_mapping(df)

    le = LabelEncoder()
    y  = le.fit_transform(df["subreddit"].values)
    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    df_test  = df.iloc[idx_test ].reset_index(drop=True)
    df_train = df.iloc[idx_train].reset_index(drop=True)
    y_test   = y[idx_test]

    print(f"\n  Test  (natural dist.): {len(df_test):,}")
    print(f"  Train pool           : {len(df_train):,}")

    if not cfg.no_undersample:
        df_train = tiered_undersample(df_train, cap=cfg.cap)
    y_train = le.transform(df_train["subreddit"].values)

    return df_train, df_test, le, y_train, y_test


def _get_psych_features(
    df_train    : pd.DataFrame,
    df_test     : pd.DataFrame,
    artifact_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract hoặc load từ cache psych features cho train và test.
    Luôn dùng post_clean — PsychExtractor không nhận post_masked.
    Returns: (X_psych_train, X_psych_test)
    """
    train_cache = artifact_dir / "psych_train.npy"
    test_cache  = artifact_dir / "psych_test.npy"

    if train_cache.exists() and test_cache.exists():
        X_tr_cached = np.load(train_cache)
        X_te_cached = np.load(test_cache)
        # [FIX] Validate cache shape — tránh mismatch khi đổi data/cap
        if X_tr_cached.shape[0] == len(df_train) and X_te_cached.shape[0] == len(df_test):
            print("  Loading cached psych features...")
            return X_tr_cached, X_te_cached
        print("  Psych cache shape mismatch — re-extracting...")

    print("\n  Initialising PsychologicalExtractor (once for train+test)...")
    psych_ext     = PsychologicalExtractor()
    X_psych_train = psych_ext.transform(df_train["post_clean"].tolist())
    psych_ext.save(artifact_dir)
    np.save(train_cache, X_psych_train)

    X_psych_test = psych_ext.transform(df_test["post_clean"].tolist())
    np.save(test_cache, X_psych_test)

    return X_psych_train, X_psych_test


def _train_one_variant(
    model_name   : str,
    variant      : str,
    cfg          : TrainingConfig,
    artifact_dir : Path,
    df_train     : pd.DataFrame,
    df_test      : pd.DataFrame,
    le           : LabelEncoder,
    y_train      : np.ndarray,
    y_test       : np.ndarray,
    X_psych_train: np.ndarray,
    X_psych_test : np.ndarray,
) -> dict:
    """
    Fit FeatureEngineer → model search → evaluate → save.
    Đây là đơn vị training nhỏ nhất, gọi được từ cả run_training và run_all_models.
    """
    artifact_dir.mkdir(parents=True, exist_ok=True)
    text_col = "post_masked" if variant == "masked" else "post_clean"

    print("\n" + "═"*65)
    print(f" TRAINING PIPELINE — variant={variant.upper()}  model={model_name}")
    print(f" → {artifact_dir}/")
    print("═"*65)

    fe      = FeatureEngineer(variant=variant, **cfg.fe_kwargs)
    X_train = fe.fit_transform(df_train[text_col].tolist(), X_psych_train, y_train)
    X_test  = fe.transform(    df_test [text_col].tolist(), X_psych_test)
    fe.save(artifact_dir)

    cv_results = None
    if cfg.search == "grid":
        best_model, cv_results = run_grid_search(X_train, y_train, model_name)
    elif cfg.search == "optuna":
        best_model, _          = run_optuna_search(X_train, y_train, model_name, cfg.optuna_trials)
    else:
        print(f"\n[STAGE 7] Training {model_name} — no search")
        estimator, _ = get_model_candidates()[model_name]
        estimator.fit(X_train, y_train)
        best_model   = estimator

    if cv_results is not None:
        cv_results.to_csv(artifact_dir / "cv_results.csv", index=False)

    eval_result = evaluate(best_model, X_test, y_test, le, artifact_dir)
    save_model(best_model, le, model_name, eval_result, artifact_dir)

    print(f"\n[DONE] [{variant}] macro-F1 = {eval_result['macro_f1']:.4f}")
    print(f"       Artifacts → {artifact_dir}/")
    return eval_result


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC ORCHESTRATORS
# ═════════════════════════════════════════════════════════════════════════════

def run_training(
    model_name   : str        = "LinearSVC",
    variant      : str        = "masked",
    cfg          : TrainingConfig | None = None,
    artifact_dir : Path | None = None,
    **cfg_kwargs,
) -> dict:
    """
    Train một model trên một variant.
    Ví dụ:
        run_training("LinearSVC", "masked", TrainingConfig())
        run_training("LightGBM",  "clean",  cap=20_000, search="optuna")
    """
    if cfg is None:
        cfg = TrainingConfig(**cfg_kwargs)
    if artifact_dir is None:
        base = ARTIFACT_MASKED if variant == "masked" else ARTIFACT_CLEAN
        artifact_dir = base / model_name

    df_train, df_test, le, y_train, y_test = _load_and_split(cfg)

    base_dir   = ARTIFACT_MASKED if variant == "masked" else ARTIFACT_CLEAN
    shared_dir = base_dir / "shared"
    psych_src  = shared_dir if (shared_dir / "psych_train.npy").exists() else artifact_dir
    X_psych_train, X_psych_test = _get_psych_features(df_train, df_test, psych_src)

    return _train_one_variant(
        model_name=model_name, variant=variant, cfg=cfg,
        artifact_dir=artifact_dir,
        df_train=df_train, df_test=df_test, le=le,
        y_train=y_train, y_test=y_test,
        X_psych_train=X_psych_train, X_psych_test=X_psych_test,
    )


def run_all_models(cfg: TrainingConfig | None = None, **cfg_kwargs) -> None:
    """
    Train 3 model × 2 variant = 6 runs.
    PsychExtractor extract 1 lần per variant — cache tại shared_dir.
    """
    if cfg is None:
        cfg = TrainingConfig(**cfg_kwargs)

    MODEL_NAMES = ["LinearSVC", "LogisticRegression", "LightGBM"]
    VARIANTS    = ["masked", "clean"]

    all_results: dict = {}
    total = len(MODEL_NAMES) * len(VARIANTS)
    run   = 0

    # [FIX] Load & split 1 lần — kết quả giống nhau vì RANDOM_SEED cố định
    df_train, df_test, le, y_train, y_test = _load_and_split(cfg)

    for variant in VARIANTS:
        all_results[variant] = {}
        base_dir   = ARTIFACT_MASKED if variant == "masked" else ARTIFACT_CLEAN
        shared_dir = base_dir / "shared"
        shared_dir.mkdir(parents=True, exist_ok=True)

        # ── [OPT] Extract psych features 1 lần cho cả variant ─────────────
        print("\n" + "▓"*65)
        print(f"  PSYCH EXTRACTION — variant={variant.upper()} (shared across all models)")
        print("▓"*65)

        X_psych_train, X_psych_test = _get_psych_features(df_train, df_test, shared_dir)

        print(f"  X_psych_train: {X_psych_train.shape}")
        print(f"  X_psych_test : {X_psych_test.shape}")

        for model_name in MODEL_NAMES:
            run += 1
            artifact_dir = base_dir / model_name
            artifact_dir.mkdir(parents=True, exist_ok=True)

            print("\n" + "█"*65)
            print(f"  RUN {run}/{total} — variant={variant.upper()}  model={model_name}")
            print("█"*65)

            result = _train_one_variant(
                model_name=model_name, variant=variant, cfg=cfg,
                artifact_dir=artifact_dir,
                df_train=df_train, df_test=df_test, le=le,
                y_train=y_train, y_test=y_test,
                X_psych_train=X_psych_train, X_psych_test=X_psych_test,
            )
            all_results[variant][model_name] = result

    _print_comparison_report(all_results, MODEL_NAMES, VARIANTS)
    _save_comparison_report(all_results, MODEL_NAMES, VARIANTS)


def run_preprocessing(
    raw_dir      : str   = RAW_DATA_DIR,
    output_csv   : str   = CLEAN_CSV,
    fp_threshold : float = FP_THRESHOLD,
    batch_size   : int   = BATCH_SIZE,
    use_gpu      : bool  = True,
) -> pd.DataFrame:
    """Stages 1–4: load → filter → mask → audit → lưu CSV."""
    print("\n" + "═"*65)
    print(" PREPROCESSING PIPELINE")
    print("═"*65)

    df = load_raw_dataset(raw_dir)
    print(f"  Loaded: {len(df):,} posts")

    df = noise_filter(df)
    df = filter_and_mask(df, fp_threshold=fp_threshold, batch_size=batch_size, use_gpu=use_gpu)
    leakage_audit(df)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n  Saved → {output_csv} ({len(df):,} posts)")
    return df


def run_eval_only(data_csv: str = CLEAN_CSV, variant: str = "masked") -> None:
    """Load artifacts đã lưu và re-evaluate. Không retrain."""
    artifact_dir = ARTIFACT_MASKED if variant == "masked" else ARTIFACT_CLEAN
    text_col     = "post_masked"   if variant == "masked" else "post_clean"
    print(f"\n[EVAL-ONLY] variant={variant} → {artifact_dir}/")

    df = pd.read_csv(data_csv)
    df = apply_label_mapping(df)
    le = joblib.load(artifact_dir / "label_encoder.joblib")
    y  = le.transform(df["subreddit"].values)

    _, idx_test = train_test_split(
        np.arange(len(df)), test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y,
    )
    df_test = df.iloc[idx_test].reset_index(drop=True)
    y_test  = y[idx_test]

    psych_test_cache = artifact_dir / "psych_test.npy"
    if psych_test_cache.exists():
        X_psych_test = np.load(psych_test_cache)
    else:
        psych_ext    = PsychologicalExtractor.load(artifact_dir)
        X_psych_test = psych_ext.transform(df_test["post_clean"].tolist())

    fe     = FeatureEngineer.load(artifact_dir)
    X_test = fe.transform(df_test[text_col].tolist(), X_psych_test)
    model  = joblib.load(artifact_dir / "model.joblib")
    evaluate(model, X_test, y_test, le, artifact_dir)


# ═════════════════════════════════════════════════════════════════════════════
# COMPARISON REPORTS
# ═════════════════════════════════════════════════════════════════════════════

def _save_comparison_report(
    all_results : dict,
    model_names : list[str],
    variants    : list[str],
) -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACT_DIR / "all_models_report.json", "w") as f:
        json.dump(all_results, f, indent=2)

    rows = []
    for variant in variants:
        for model_name in model_names:
            r   = all_results[variant][model_name]
            row = {
                "variant"    : variant,
                "model"      : model_name,
                "macro_f1"   : round(r["macro_f1"], 4),
                "weighted_f1": round(r["weighted_f1"], 4),
            }
            for cls, metrics in r.get("per_class", {}).items():
                row[f"f1_{cls}"] = round(metrics.get("f1-score", 0), 4)
            rows.append(row)
    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "all_models_report.csv", index=False)
    print(f"\n  Reports saved → {ARTIFACT_DIR}/all_models_report.{{json,csv}}")


def _print_comparison_report(
    all_results : dict,
    model_names : list[str],
    variants    : list[str],
) -> None:
    SEP = "─" * 75

    print("\n\n" + "═"*75)
    print("  ALL MODELS COMPARISON REPORT")
    print("═"*75)

    print(f"\n{'Model':<22} {'Variant':<8} {'macro-F1':>10} {'weighted-F1':>13}")
    print(SEP)
    best_macro = 0.0
    best_label = ""
    for variant in variants:
        for model_name in model_names:
            r  = all_results[variant][model_name]
            mf = r["macro_f1"]
            wf = r["weighted_f1"]
            if mf > best_macro:
                best_macro = mf
                best_label = f"{model_name} [{variant}]"
            print(f"  {model_name:<20} {variant:<8} {mf:>10.4f} {wf:>13.4f}")
        print(SEP)
    print(f"\n  Best macro-F1 overall: {best_macro:.4f} — {best_label}")

    print("\n\n" + "═"*75)
    print("  PER-CLASS F1 BREAKDOWN")
    print("═"*75)

    first_result = next(iter(next(iter(all_results.values())).values()))
    class_names  = [c for c in first_result.get("per_class", {}).keys()
                    if c not in ("accuracy", "macro avg", "weighted avg")]
    col_headers  = [f"{m[:4]}-{v[:4]}" for v in variants for m in model_names]
    header_width = 26
    col_width    = 11

    print(f"\n{'Class':<{header_width}}", end="")
    for h in col_headers:
        print(f"{h:>{col_width}}", end="")
    print()
    print(SEP)

    for cls in sorted(class_names):
        print(f"  {cls:<{header_width - 2}}", end="")
        for variant in variants:
            for model_name in model_names:
                f1 = all_results[variant][model_name].get("per_class", {}).get(cls, {}).get("f1-score", 0.0)
                print(f"{f1:>{col_width}.4f}", end="")
        print()

    print(SEP)
    print(f"  {'MACRO AVG':<{header_width - 2}}", end="")
    for variant in variants:
        for model_name in model_names:
            mf = all_results[variant][model_name]["macro_f1"]
            print(f"{mf:>{col_width}.4f}", end="")
    print()
    print("═"*75)
