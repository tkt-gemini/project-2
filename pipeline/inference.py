"""
pipeline/inference.py — MentalHealthClassifier (Stage 10).

Usage:
    clf = MentalHealthClassifier.load(variant="masked")
    result = clf.predict("I've been feeling empty for weeks.")
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

from pipeline.config import ARTIFACT_CLEAN, ARTIFACT_MASKED
from pipeline.features import FeatureEngineer, PsychologicalExtractor
from pipeline.text import clean_text, mask_mh_keywords


class MentalHealthClassifier:
    """
    Load artifact và predict trên raw text.

    Hỗ trợ hai variant:
        variant="masked" → dùng artifacts/masked/ (train trên post_masked)
        variant="clean"  → dùng artifacts/clean/  (train trên post_clean)
    """
    VARIANT_DIRS = {"masked": ARTIFACT_MASKED, "clean": ARTIFACT_CLEAN}

    def __init__(self):
        self.fe      : FeatureEngineer        | None = None
        self.model   : object                 | None = None
        self.le      = None
        self.psych   : PsychologicalExtractor | None = None
        self.variant : str                           = "masked"

    @classmethod
    def load(
        cls,
        variant     : str       = "masked",
        artifact_dir: Path | None = None,
    ) -> "MentalHealthClassifier":
        if artifact_dir is None:
            if variant not in cls.VARIANT_DIRS:
                raise ValueError(f"variant phải là 'masked' hoặc 'clean', nhận: {variant!r}")
            artifact_dir = cls.VARIANT_DIRS[variant]

        print(f"\n[Stage 10] Loading [{variant}] artifacts from {artifact_dir}/...")
        obj         = cls()
        obj.variant = variant
        obj.fe      = FeatureEngineer.load(artifact_dir)
        obj.model   = joblib.load(Path(artifact_dir) / "model.joblib")
        obj.le      = joblib.load(Path(artifact_dir) / "label_encoder.joblib")
        obj.psych   = PsychologicalExtractor.load(artifact_dir)
        print(f"  Ready [{variant}].\n")
        return obj

    def _prepare(self, text: str) -> tuple[str, str]:
        """Trả về (text_for_tfidf, post_clean)."""
        cleaned = clean_text(text)
        if self.variant == "masked":
            return mask_mh_keywords(cleaned), cleaned
        return cleaned, cleaned

    def predict(self, text: str) -> dict:
        tfidf_text, clean = self._prepare(text)
        psych             = self.psych.transform([clean])
        X                 = self.fe.transform([tfidf_text], psych)
        label             = self.le.inverse_transform(self.model.predict(X))[0]
        try:
            proba = self.model.predict_proba(X)[0]
            return {
                "label"        : label,
                "confidence"   : float(proba.max()),
                "probabilities": {c: float(p) for c, p in zip(self.le.classes_, proba)},
                "variant"      : self.variant,
            }
        except AttributeError:
            return {"label": label, "confidence": None, "probabilities": {}, "variant": self.variant}

    def predict_batch(self, texts: list[str], batch_size: int = 512) -> list[dict]:
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc=f"Inference [{self.variant}]"):
            batch                  = texts[i : i + batch_size]
            tfidf_list, clean_list = zip(*[self._prepare(t) for t in batch])
            psych                  = self.psych.transform(list(clean_list))
            X                      = self.fe.transform(list(tfidf_list), psych)
            labels                 = self.le.inverse_transform(self.model.predict(X))
            try:
                probas = self.model.predict_proba(X)
                for lbl, proba in zip(labels, probas):
                    results.append({
                        "label"        : lbl,
                        "confidence"   : float(proba.max()),
                        "probabilities": {c: float(p) for c, p in zip(self.le.classes_, proba)},
                        "variant"      : self.variant,
                    })
            except AttributeError:
                results.extend(
                    {"label": l, "confidence": None, "probabilities": {}, "variant": self.variant}
                    for l in labels
                )
        return results
