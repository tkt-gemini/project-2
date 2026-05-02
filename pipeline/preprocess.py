"""
pipeline/preprocess.py — Preprocessing Stages 1–4.

Stages:
    1. NoiseFilter      — Remove deleted/short/bot/mod/spam/dup posts
    2+3. FilterAndMask  — Filter by self-disclosure + mask MH keywords
    4. LeakageAudit     — Check top features per class for MH keyword leakage
"""
from __future__ import annotations

import glob
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from pipeline.config import (
    BATCH_SIZE,
    CHECKPOINT_DIR,
    CONTROL_LABELS,
    FP_THRESHOLD,
    MAX_URLS,
    MH_KEYWORDS,
    MIN_WORDS,
    _BOT_RE,
    _MOD_RE,
)
from pipeline.text import clean_text, first_person_ratio, is_self_disclosure, mask_mh_keywords


def load_raw_dataset(path: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Load và chuẩn hoá raw CSVs về schema (post, author, date, subreddit)."""
    files = glob.glob(os.path.join(path, pattern))
    dfs   = []
    for fname in files:
        if "post" in os.path.basename(fname).split("_"):
            continue
        df = pd.read_csv(fname, index_col=None, header=0)
        for alt in ("body", "selftext", "text"):
            if alt in df.columns and "post" not in df.columns:
                df = df.rename(columns={alt: "post"})
                break
        df["subreddit"] = df["subreddit"].map(
            lambda s: "non-mental" if s in CONTROL_LABELS else s
        )
        df = df[~df["subreddit"].isin({"mentalhealth", "COVID19_support"})]
        dfs.append(df)
    if not dfs:
        raise ValueError(f"No valid CSVs in: {path}")
    result    = pd.concat(dfs, axis=0, ignore_index=True)
    available = [c for c in ("post", "author", "date", "subreddit") if c in result.columns]
    return result[available]


def noise_filter(
    df       : pd.DataFrame,
    min_words: int = MIN_WORDS,
    max_urls : int = MAX_URLS,
) -> pd.DataFrame:
    """
    Stage 1 — Lọc nhiễu phi nội dung bằng vectorized pandas operations.
    """
    n_orig = len(df)
    post   = df["post"].fillna("")
    author = df["author"].fillna("") if "author" in df.columns else pd.Series([""] * n_orig)

    _del_set = {"[removed]", "[deleted]", "", "nan"}
    m_del    = post.str.strip().str.lower().isin(_del_set) | post.isna()
    m_shrt   = post.str.split().str.len().fillna(0) < min_words
    m_bot    = author.str.lower().str.contains(_BOT_RE, na=False)
    m_mod    = post.str.lower().str.contains(_MOD_RE, na=False)
    m_spam   = post.str.count(r"https?://\S+|www\.\S+") > max_urls
    m_dup    = post.duplicated(keep="first")

    stats = {
        "deleted"   : int(m_del.sum()),
        "short"     : int(m_shrt.sum()),
        "bot"       : int(m_bot.sum()),
        "moderator" : int(m_mod.sum()),
        "spam"      : int(m_spam.sum()),
        "duplicated": int(m_dup.sum()),
    }

    result = df[~(m_del | m_shrt | m_bot | m_mod | m_spam | m_dup)].reset_index(drop=True)

    print("\n[STAGE 1] Noise Filtering")
    print(f"  Input : {n_orig:,}")
    for reason, count in stats.items():
        print(f"  Drop ({reason:<10}): {count:>7,}  ({count/n_orig*100:.1f}%)")
    print(f"  Output: {len(result):,}  ({len(result)/n_orig*100:.1f}% retained)")
    return result


def filter_and_mask(
    df            : pd.DataFrame,
    fp_threshold  : float = FP_THRESHOLD,
    batch_size    : int   = BATCH_SIZE,
    checkpoint_dir: Path  = CHECKPOINT_DIR,
    use_gpu       : bool  = True,
) -> pd.DataFrame:
    """
    Stage 2+3 — Lọc thảo luận + mask MH keywords trong một pass.

    Mỗi bài:
      A. clean_text() — hàm module-level, dùng chung với inference
      B. fp_ratio >= threshold → GIỮ ngay (bypass self-disclosure check)
      C. is_self_disclosure() → GIỮ nếu True, DROP nếu False
      D. mask_mh_keywords() trên cleaned text đã tính ở bước A

    Output: thêm cột post_clean và post_masked vào df.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    use_gpu     = use_gpu and torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"

    texts     = df["post"].fillna("").tolist()
    n_orig    = len(texts)
    n_batches = (n_orig + batch_size - 1) // batch_size

    def _ckpt_path(idx: int) -> Path:
        return checkpoint_dir / f"batch_{idx:06d}.pkl"

    def _find_last() -> int:
        ckpts = sorted(checkpoint_dir.glob("batch_*.pkl"))
        return int(ckpts[-1].stem.split("_")[1]) if ckpts else -1

    last_done   = _find_last()
    all_results : list[tuple] = []

    # [FIX] Stale checkpoint guard — xóa checkpoint cũ nếu dataset đổi
    if last_done >= n_batches:
        print(f"  Stale checkpoints detected (last={last_done}, expected<{n_batches}) — clearing...")
        for _ckpt in checkpoint_dir.glob("batch_*.pkl"):
            _ckpt.unlink()
        last_done = -1

    if last_done >= 0:
        print(f"  Resume from batch {last_done + 1}/{n_batches}")
        for b in tqdm(range(last_done + 1), desc="  Load checkpoints", unit="batch"):
            p = _ckpt_path(b)
            if p.exists():
                with open(p, "rb") as f:
                    all_results.extend(pickle.load(f))

    print(f"\n[STAGE 2+3] Filter & Mask — {n_orig:,} posts | device={device_name}")

    kept_fp_high = kept_pattern = drop_no_fp = 0

    pbar = tqdm(range(last_done + 1, n_batches), desc="  Batch", unit="batch", dynamic_ncols=True)
    for batch_idx in pbar:
        start         = batch_idx * batch_size
        end           = min(start + batch_size, n_orig)
        batch_results = []

        if use_gpu:
            vram = f"{torch.cuda.memory_reserved(0) // 1024**2}MB"
            pbar.set_postfix({"VRAM": vram, "pos": f"{start:,}"})

        for orig_idx, raw in zip(range(start, end), texts[start:end]):
            cleaned = clean_text(raw)
            fp      = first_person_ratio(cleaned)

            if fp >= fp_threshold:
                kept_fp_high += 1
                batch_results.append((orig_idx, False, mask_mh_keywords(cleaned), cleaned))
            elif not is_self_disclosure(cleaned):
                drop_no_fp += 1
                batch_results.append((orig_idx, True, "", ""))
            else:
                kept_pattern += 1
                batch_results.append((orig_idx, False, mask_mh_keywords(cleaned), cleaned))

        all_results.extend(batch_results)
        with open(_ckpt_path(batch_idx), "wb") as f:
            pickle.dump(batch_results, f)

    result_df = pd.DataFrame(
        all_results, columns=["_idx", "_drop", "post_masked", "post_clean"]
    ).set_index("_idx").sort_index()

    df_out               = df.copy()
    df_out["post_masked"] = result_df["post_masked"].values
    df_out["post_clean"]  = result_df["post_clean"].values
    df_out["_drop"]       = result_df["_drop"].values

    n_dropped = int(df_out["_drop"].sum())
    df_out    = df_out[~df_out["_drop"]].drop(columns=["_drop"]).reset_index(drop=True)

    print(f"\n  Kept  (fp high)   : {kept_fp_high:,}")
    print(f"  Kept  (pattern)   : {kept_pattern:,}")
    print(f"  Drop  (no pattern): {drop_no_fp:,}")
    print(f"  Total drop        : {n_dropped:,}  ({n_dropped/n_orig*100:.1f}%)")
    print(f"  Total kept        : {len(df_out):,}  ({len(df_out)/n_orig*100:.1f}%)")
    return df_out


def leakage_audit(df: pd.DataFrame, top_n: int = 15) -> None:
    """
    Stage 4 — Kiểm tra data leakage: top features mỗi class không được
    là tên bệnh. Dùng per-class LR (class vs. rest) thay vì global TF-IDF.
    """
    from sklearn.linear_model import LogisticRegression as _LR

    text_col = "post_masked" if "post_masked" in df.columns else "post"
    print("\n[STAGE 4] Leakage Audit")
    print("=" * 60)

    dist  = df["subreddit"].value_counts()
    ratio = dist.max() / dist.min()
    print(dist.to_string())
    print(f"\nMax/min ratio = {ratio:.1f}x")
    if ratio > 10:
        print("  ⚠  Imbalance >10x — tiered undersampling sẽ xử lý trong Stage 5b.")

    texts  = df[text_col].fillna("").tolist()
    labels = df["subreddit"].tolist()
    vec    = TfidfVectorizer(max_features=30_000, ngram_range=(1, 2), min_df=5)
    X      = vec.fit_transform(texts)
    fnames = vec.get_feature_names_out()
    mh_set = {k.lower() for k in MH_KEYWORDS}
    leak   = False

    print(f"\nTop-{top_n} discriminative terms per class:")
    for lbl in sorted(set(labels)):
        y = [1 if l == lbl else 0 for l in labels]
        if sum(y) < 10:
            continue
        lr      = _LR(max_iter=200, C=1.0, solver="lbfgs")
        lr.fit(X, y)
        top     = [fnames[i] for i in lr.coef_[0].argsort()[-top_n:][::-1]]
        flagged = [t for t in top if t.lower() in mh_set]
        print(f"\n  [{lbl}] ({sum(y):,}): {', '.join(top)}")
        if flagged:
            print(f"  ⚠ LEAK: {flagged}")
            leak = True

    print("\n" + ("⚠  Leakage detected — expand MH_KEYWORDS." if leak else "✓  Clean."))
    print("=" * 60)
