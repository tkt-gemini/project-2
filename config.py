from __future__ import annotations

import re
import warnings
from pathlib import Path

from sklearn.metrics import f1_score, make_scorer

warnings.filterwarnings("ignore", category=UserWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
CLEAN_CSV = "./archive/train.csv"
ARTIFACT_DIR = Path("./archive/model")
ARTIFACT_MASK = Path(ARTIFACT_DIR) / "masked"
ARTIFACT_CLEAN = Path(ARTIFACT_DIR) / "cleaned"
# Cache
CACHE_DIR = Path("./archive/cache")
CHECKPOINT_DIR = Path(CACHE_DIR) / "checkpoint"
PSYCH_CACHE_DIR = Path(CACHE_DIR) / "psych"

for _d in [
    ARTIFACT_DIR,
    ARTIFACT_MASK,
    ARTIFACT_CLEAN,
    CACHE_DIR,
    CHECKPOINT_DIR,
    PSYCH_CACHE_DIR,
]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Training hyperparams ──────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.15
CV_FOLDS = 5
TFIDF_FEATS = 60_000
SELECT_K = 32_000
TIER1_CAP = 30_000
MACRO_F1 = make_scorer(f1_score, average="macro", zero_division=0)

# ── Preprocessing params ──────────────────────────────────────────────────────
FP_THRESHOLD = 0.35
BATCH_SIZE = 256
MIN_WORDS = 10
MAX_URLS = 2

# ── Empath categories ─────────────────────────────────────────────────────────
# FIX: Wrapped in try/except so a missing file raises a clear error at startup
# rather than a confusing AttributeError later when PSYCH_CATS is accessed.
try:
    with open("./archive/psych-categories.csv", "r", encoding="utf-8") as f:
        PSYCH_CATS = [ln.strip() for ln in f if ln.strip()]
except FileNotFoundError as exc:
    raise FileNotFoundError(
        "Could not load './archive/psych-categories.csv'. "
        "Ensure the file exists before importing config."
    ) from exc

# ── Label constants ───────────────────────────────────────────────────────────
CONTROL_LABELS = [
    "conspiracy",
    "divorce",
    "fitness",
    "guns",
    "jokes",
    "legaladvice",
    "meditation",
    "parenting",
    "personalfinance",
    "relationships",
    "teaching",
]

# FIX: This is the single source of truth for label mapping.
# pipeline.py previously had its own copy that was missing adhd, ptsd,
# schizophrenia, and non-mental entries. All callers must import this dict.
LABEL_MAPPING: dict[str, str] = {
    "EDAnonymous": "eating_disorder",
    "addiction": "substance_use_disorder",
    "adhd": "adhd",
    "alcoholism": "alcohol_use_disorder",
    "anxiety": "generalized_anxiety",
    "autism": "autism_spectrum",
    "bipolarreddit": "bipolar_disorder",
    "bpd": "borderline_personality",
    "depression": "major_depressive",
    "healthanxiety": "illness_anxiety",
    "lonely": "social_isolation",
    "socialanxiety": "social_anxiety",
    "suicidewatch": "suicidal_risk",
    "ptsd": "ptsd",
    "schizophrenia": "schizophrenia",
    "non-mental": "non-mental",
}

# ── MH keyword list (DSM-5 + subreddit aliases + slang) ──────────────────────
MH_KEYWORDS: list[str] = [
    "depression",
    "depressed",
    "depressive",
    "mdd",
    "major depressive",
    "anxiety",
    "anxious",
    "generalized anxiety",
    "gad",
    "panic disorder",
    "panic attack",
    "adhd",
    "add",
    "attention deficit",
    "hyperactivity",
    "autism",
    "autistic",
    "asd",
    "asperger",
    "aspie",
    "bipolar",
    "bipolar disorder",
    "mania",
    "manic",
    "hypomania",
    "hypomanic",
    "bpd",
    "borderline",
    "borderline personality",
    "ptsd",
    "post traumatic",
    "posttraumatic",
    "trauma",
    "schizophrenia",
    "schizophrenic",
    "schizoaffective",
    "psychosis",
    "psychotic",
    "hallucination",
    "ocd",
    "obsessive compulsive",
    "obsessive-compulsive",
    "anorexia",
    "anorexic",
    "bulimia",
    "bulimic",
    "binge eating",
    "ednos",
    "arfid",
    "eating disorder",
    "addiction",
    "addicted",
    "alcoholism",
    "alcoholic",
    "substance abuse",
    "suicid",
    "suicidal",
    "suicide",
    "self harm",
    "self-harm",
    "selfharm",
    "cutting",
    "social anxiety",
    "social phobia",
    "social anxiety disorder",
    "health anxiety",
    "hypochondria",
    "hypochondriac",
    "r/depression",
    "r/anxiety",
    "r/adhd",
    "r/autism",
    "r/bipolarreddit",
    "r/bpd",
    "r/ptsd",
    "r/schizophrenia",
    "r/socialanxiety",
    "r/suicidewatch",
    "r/edanonymous",
    "r/addiction",
    "r/alcoholism",
    "r/healthanxiety",
    "r/lonely",
]

MH_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in MH_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

# ── Regex patterns ────────────────────────────────────────────────────────────
FIRST_PERSON_PRONOUNS = {"i", "me", "my", "myself", "mine"}

BOT_RE = re.compile(r"bot$|_bot$|^bot_|automod|automoderator", re.IGNORECASE)
MOD_RE = re.compile(
    r"weekly\s+(thread|check.?in|discussion)"
    r"|daily\s+(thread|check.?in|discussion)"
    r"|monthly\s+(thread|check.?in|discussion)"
    r"|^(rules|announcement|mod\s+post|welcome\s+to|pinned)"
    r"|megathread|resource\s+(thread|list|post)",
    re.IGNORECASE,
)

FP_COMPILED = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bI\b.{0,60}\b(was|have been|got|am|were)\b.{0,40}\bdiagnosed\b",
        r"\bI\b.{0,60}\bhave\b.{0,40}\b(disorder|condition|illness)\b",
        r"\bI\b.{0,50}\b(feel|felt|feeling|been feeling)\b",
        r"\bI\b.{0,50}\b(struggle|struggled|struggling)\b",
        r"\bI\b.{0,50}\b(suffer|suffered|suffering)\b",
        r"\bI\b.{0,50}\b(experience|experienced|experiencing)\b",
        r"\bI\b.{0,50}\b(can'?t|cannot|couldn'?t)\b.{0,30}\b(sleep|eat|focus|function|get out)\b",
        r"\bI\b.{0,50}\b(don'?t|didn'?t)\b.{0,30}\b(want to|feel like|care|know)\b",
        r"\bI'?ve\b.{0,50}\b(been|had|started|noticed)\b",
        r"\bI\b.{0,30}\b(just|recently|always|never|sometimes|still)\b",
        r"\bmy\b.{0,40}\b(life|mind|brain|thoughts|head|body|therapist|doctor|medication)\b",
        r"\b(help me|need help|anyone else|does anyone|can anyone)\b",
    ]
]
EXCL_COMPILED = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bmy\s+(mom|dad|mother|father|brother|sister|son|daughter|friend|partner"
        r"|spouse|husband|wife|boyfriend|girlfriend|colleague|coworker)\b"
        r".{0,60}\b(has|have|was|diagnosed|struggling|suffering)\b",
        r"\b(not sure|unsure)\s+if\s+I\b",
        r"\bI\b.{0,20}\bnot\b.{0,20}\bdiagnosed\b",
        r"\b(asking|posting)\s+(for|on behalf of)\b",
        r"\b(article|study|research|paper|news|published)\b.{0,60}\b(found|shows|suggests)\b",
        r"\bhow\s+to\s+(help|support|talk to)\b.{0,40}\b(someone|person|friend|family)\b",
    ]
]
