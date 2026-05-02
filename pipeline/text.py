"""
pipeline/text.py — Shared text utilities for preprocessing and inference.

Functions:
    clean_text        — Basic text cleaning (URL, mentions, numbers, contractions)
    mask_mh_keywords  — Replace MH keywords with [MH] token
    is_self_disclosure — Check for first-person experience patterns
    first_person_ratio — Average first-person pronouns per sentence
"""
from __future__ import annotations

import re
import unicodedata

import contractions

from pipeline.config import (
    FIRST_PERSON_PRONOUNS,
    _EXCL_COMPILED,
    _FP_COMPILED,
    _MH_PATTERN,
    _URL_RE,
)


def clean_text(text: str) -> str:
    """
    Làm sạch cơ bản. Hàm DUY NHẤT cho cả preprocessing lẫn inference.
    Đảm bảo train/inference consistency tuyệt đối.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = _URL_RE.sub("", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"\d+[.,]?\d*", "[NUM]", text)
    try:
        text = contractions.fix(text)
    except Exception:
        pass
    return text.strip()


def mask_mh_keywords(cleaned: str) -> str:
    """Thay MH keywords bằng [MH] trên text đã clean."""
    return _MH_PATTERN.sub("[MH]", cleaned)


def is_self_disclosure(text: str) -> bool:
    """
    True nếu bài có first-person experience pattern VÀ không có exclusion.
    [H1 FIX] Positive check trước → bỏ qua exclusion cho ~60% bài.
    """
    if not any(p.search(text) for p in _FP_COMPILED):
        return False
    return not any(p.search(text) for p in _EXCL_COMPILED)


def first_person_ratio(text: str) -> float:
    """
    Trung bình số đại từ ngôi 1 trên mỗi câu (có thể > 1.0).
    Dùng để bypass self-disclosure check khi >= FP_THRESHOLD.
    Threshold đã calibrate cho metric này — KHÔNG đổi sang per-word ratio.
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0
    lower = text.lower()
    sentences = [s for s in re.split(r"[.!?\n]+", lower) if s.strip()]
    n_sents   = len(sentences) or 1
    fp_count  = sum(1 for w in lower.split() if w in FIRST_PERSON_PRONOUNS)
    return fp_count / n_sents
