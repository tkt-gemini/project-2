"""
python -m pipeline — CLI entry point.

Usage:
    python -m pipeline --help
    python -m pipeline --preprocess-only
    python -m pipeline --train-only --variant masked
    python -m pipeline --train-all-models
    python -m pipeline --predict "I've been feeling empty for weeks"
"""
import argparse

from pipeline.config import (
    CLEAN_CSV,
    RAW_DATA_DIR,
    SELECT_K,
    TFIDF_FEATS,
    TIER1_CAP,
    TrainingConfig,
)
from pipeline.inference import MentalHealthClassifier
from pipeline.train import run_all_models, run_eval_only, run_preprocessing, run_training


def main():
    parser = argparse.ArgumentParser(description="Mental Health Classification Pipeline")
    parser.add_argument("--raw-dir",          default=RAW_DATA_DIR)
    parser.add_argument("--data",             default=CLEAN_CSV)
    parser.add_argument("--model",            default="LinearSVC",
                        choices=["LinearSVC","LogisticRegression","LightGBM"])
    parser.add_argument("--variant",          default="masked", choices=["masked","clean"],
                        help="Variant để train, eval hoặc predict")
    parser.add_argument("--search",           default="grid",   choices=["grid","optuna","none"])
    parser.add_argument("--optuna-trials",    default=50,          type=int)
    parser.add_argument("--tfidf-feats",      default=TFIDF_FEATS, type=int)
    parser.add_argument("--select-k",         default=SELECT_K,    type=int)
    parser.add_argument("--cap",              default=TIER1_CAP,   type=int)
    parser.add_argument("--no-undersample",   action="store_true")
    parser.add_argument("--preprocess-only",  action="store_true")
    parser.add_argument("--train-only",       action="store_true")
    parser.add_argument("--train-both",       action="store_true",
                        help="Train cả masked và clean, 1 model")
    parser.add_argument("--train-all-models", action="store_true",
                        help="Train 3 model × 2 variant = 6 runs")
    parser.add_argument("--eval-only",        action="store_true")
    parser.add_argument("--predict",          default=None)

    args = parser.parse_args()

    cfg = TrainingConfig(
        data_csv      = args.data,
        search        = args.search,
        optuna_trials = args.optuna_trials,
        tfidf_feats   = args.tfidf_feats,
        select_k      = args.select_k,
        cap           = args.cap,
        no_undersample= args.no_undersample,
    )

    if args.predict:
        clf    = MentalHealthClassifier.load(variant=args.variant)
        result = clf.predict(args.predict)
        print(f"\n  Variant    : {result['variant']}")
        print(f"  Label      : {result['label']}")
        print(f"  Confidence : {result['confidence']:.4f}" if result["confidence"] else "  Confidence : N/A")
        top3 = sorted(result["probabilities"].items(), key=lambda x: -x[1])[:3]
        print("  Top-3      :", ", ".join(f"{l}={p:.3f}" for l, p in top3))

    elif args.eval_only:
        run_eval_only(args.data, variant=args.variant)

    elif args.preprocess_only:
        run_preprocessing(raw_dir=args.raw_dir, output_csv=args.data)

    elif args.train_all_models:
        run_all_models(cfg)

    elif args.train_both:
        print("\n▶ Training MASKED variant...")
        run_training(args.model, "masked", cfg)
        print("\n▶ Training CLEAN variant...")
        run_training(args.model, "clean",  cfg)

    elif args.train_only:
        run_training(args.model, args.variant, cfg)

    else:
        run_preprocessing(raw_dir=args.raw_dir, output_csv=args.data)
        run_training(args.model, args.variant, cfg)


if __name__ == "__main__":
    main()
