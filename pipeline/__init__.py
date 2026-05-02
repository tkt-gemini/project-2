"""
pipeline — Reddit Mental Health Classification: Unified Pipeline

Sử dụng:
    from pipeline import MentalHealthClassifier
    clf = MentalHealthClassifier.load(variant="masked")
    result = clf.predict("I've been feeling empty for weeks.")

CLI:
    python -m pipeline --help
    python -m pipeline --train-only --variant masked
    python -m pipeline --predict "I've been feeling empty for weeks"
"""
from pipeline.config import *  # noqa: F401,F403
from pipeline.features import (  # noqa: F401
    FeatureEngineer,
    NRCLex,
    PsychologicalExtractor,
    SymptomLexicon,
    _NRCEmo,
    _NRCAIL,
    _NRCVAD,
)
from pipeline.inference import MentalHealthClassifier  # noqa: F401
from pipeline.models import (  # noqa: F401
    evaluate,
    get_model_candidates,
    run_grid_search,
    run_optuna_search,
    save_model,
)
from pipeline.text import (  # noqa: F401
    clean_text,
    first_person_ratio,
    is_self_disclosure,
    mask_mh_keywords,
)
from pipeline.train import (  # noqa: F401
    apply_label_mapping,
    run_all_models,
    run_eval_only,
    run_preprocessing,
    run_training,
    tiered_undersample,
)
