# -*- coding: utf-8 -*-
"""
app.py - Streamlit demo for the Reddit Mental Health Classifier
================================================================
Loads the pre-trained inference pipelines produced by main.py and
exposes a UI for classifying free-form text into 16 mental health
categories.

Usage:
    streamlit run app.py

Requirements:
    - Models must already be trained with main.py
    - All training dependencies must be installed (see requirements.txt)
    - spaCy model: python -m spacy download en_core_web_md
"""

import pickle
import warnings
from pathlib import Path

import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.special import softmax

warnings.filterwarnings("ignore")
nltk.download("vader_lexicon", quiet=True)


# ── Page config (must be the FIRST Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="Mental Health Text Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Constants - mirrors config.py so app.py can run standalone ────────────────
VERSION = "v1.0.0"
ARTIFACT_CLEAN = Path(f"./archive/{VERSION}/cleaned")
ARTIFACT_MASK = Path(f"./archive/{VERSION}/masked")

MODEL_FILE_MAP = {
    "LinearSVC": "inference_pipeline_svc.pkl",
    "LogisticRegression": "inference_pipeline_lr.pkl",
}

# Human-readable names for each encoded label
LABEL_DISPLAY: dict[str, str] = {
    "adhd": "ADHD",
    "alcohol_use_disorder": "Alcohol Use Disorder",
    "autism_spectrum": "Autism Spectrum",
    "bipolar_disorder": "Bipolar Disorder",
    "borderline_personality": "Borderline Personality",
    "eating_disorder": "Eating Disorder",
    "generalized_anxiety": "Generalized Anxiety",
    "illness_anxiety": "Illness Anxiety",
    "major_depressive": "Major Depressive",
    "non-mental": "Non-Mental Health",
    "ptsd": "PTSD",
    "schizophrenia": "Schizophrenia",
    "social_anxiety": "Social Anxiety",
    "social_isolation": "Social Isolation",
    "substance_use_disorder": "Substance Use Disorder",
    "suicidal_risk": "Suicidal Risk",
}

# Performance summary pulled from evaluation_report.txt
MODEL_METRICS = {
    ("Cleaned", "LinearSVC"): {"macro_f1": 0.62, "accuracy": 0.80},
    ("Cleaned", "LogisticRegression"): {"macro_f1": 0.61, "accuracy": 0.79},
    ("Masked", "LinearSVC"): {"macro_f1": 0.54, "accuracy": 0.76},
    ("Masked", "LogisticRegression"): {"macro_f1": 0.53, "accuracy": 0.75},
}

# High-recall classes that warrant extra caution even with low confidence
CRISIS_LABELS = {"suicidal_risk"}

# Pre-written example posts to make the demo immediately usable
EXAMPLE_POSTS = [
    {
        "label": "Major Depressive",
        "text": (
            "I've been feeling really low for months now. I can't get out of bed, "
            "I don't enjoy anything anymore. It feels like there's no point to anything "
            "and I'm just going through the motions every day. My friends have stopped "
            "inviting me out because I always cancel anyway."
        ),
    },
    {
        "label": "Generalized Anxiety",
        "text": (
            "My heart keeps racing and I can't stop worrying about everything - work, "
            "money, my health, my relationships. I've been avoiding going out because "
            "I'm scared something bad will happen. Even small decisions feel completely "
            "overwhelming and I lie awake most nights running through worst-case scenarios."
        ),
    },
    {
        "label": "ADHD",
        "text": (
            "I literally cannot focus on anything for more than five minutes. I forget "
            "everything, lose my keys constantly, start tasks and never finish them. "
            "My desk is a disaster and my boss is getting frustrated. I've been this way "
            "my whole life, not just lately - school was the same way."
        ),
    },
    {
        "label": "Bipolar Disorder",
        "text": (
            "Last week I felt on top of the world - barely sleeping, full of energy, "
            "signed up for three new projects. This week I can barely get off the couch. "
            "This happens in cycles and I never know which version of myself I'll wake up as."
        ),
    },
    {
        "label": "Non-Mental Health",
        "text": (
            "Spent the weekend rebuilding my bicycle. Finally got the gear derailleur "
            "dialled in properly. Planning a 50 km ride next Saturday if the weather holds."
        ),
    },
]


# ── Model loading (cached so the heavy pickle loads only once per session) ─────
@st.cache_resource(show_spinner="Loading model - first load may take 30-60 s...")
def load_pipeline(data_type: str, model_name: str):
    """
    Deserialise the sklearn inference Pipeline and LabelEncoder from disk.

    We cache with @st.cache_resource (not @st.cache_data) because the pipeline
    contains a spaCy model and other non-serialisable objects that should live
    in memory as a single shared instance.
    """
    artifact_dir = ARTIFACT_CLEAN if data_type == "Cleaned" else ARTIFACT_MASK
    pipeline_path = artifact_dir / MODEL_FILE_MAP[model_name]
    le_path = artifact_dir / "label_encoder.pkl"

    missing = [str(p) for p in [pipeline_path, le_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s):\n  " + "\n  ".join(missing) + "\n\n"
            "Train the models first by running:\n    python main.py"
        )

    with open(pipeline_path, "rb") as fh:
        pipeline = pickle.load(fh)
    with open(le_path, "rb") as fh:
        le = pickle.load(fh)

    return pipeline, le


# ── Inference ─────────────────────────────────────────────────────────────────
def predict(pipeline, le, text: str, model_name: str) -> tuple[str, pd.DataFrame]:
    """
    Run the inference pipeline on a single text string.

    Returns
    -------
    pred_label : str
        The raw label key (e.g. "major_depressive").
    results : pd.DataFrame
        All classes sorted by confidence descending, with columns
        [label, display, confidence].

    Notes
    -----
    LinearSVC has no predict_proba. We use decision_function (raw margin
    distances from each class hyperplane) and apply softmax to convert them
    into a [0, 1] range that sums to 1. These are *not* calibrated
    probabilities - treat them as relative confidence scores for ranking.
    """
    # The sklearn pipeline expects an iterable of strings, not a bare string
    text_input = pd.Series([text])
    pred_encoded = pipeline.predict(text_input)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]

    if model_name == "LogisticRegression":
        scores = pipeline.predict_proba(text_input)[0]
    else:
        # LinearSVC -> decision function scores -> softmax normalisation
        raw = pipeline.decision_function(text_input)[0]
        scores = softmax(raw)

    results = (
        pd.DataFrame(
            {
                "label": le.classes_,
                "display": [LABEL_DISPLAY.get(lb, lb) for lb in le.classes_],
                "confidence": scores,
            }
        )
        .sort_values("confidence", ascending=False)
        .reset_index(drop=True)
    )

    return pred_label, results


# ── Chart builder ─────────────────────────────────────────────────────────────
def build_bar_chart(results: pd.DataFrame, pred_label: str, top_n: int = 10):
    """
    Horizontal bar chart of the top-N confidence scores.

    The predicted class is highlighted in a distinct colour so it stands
    out at a glance.
    """
    top = results.head(top_n).copy()

    # Colour: red for crisis, teal for prediction, steel-blue for the rest
    colors = []
    for lb in top["label"]:
        if lb in CRISIS_LABELS and lb == pred_label:
            colors.append("#EF5350")  # red - crisis + predicted
        elif lb in CRISIS_LABELS:
            colors.append("#EF9A9A")  # light red - crisis but not top
        elif lb == pred_label:
            colors.append("#00897B")  # teal - predicted class
        else:
            colors.append("#90A4AE")  # neutral grey-blue

    fig = go.Figure(
        go.Bar(
            x=top["confidence"],
            y=top["display"],
            orientation="h",
            marker_color=colors,
            text=[f"{c:.1%}" for c in top["confidence"]],
            textposition="outside",
            cliponaxis=False,
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Confidence score",
            tickformat=".0%",
            range=[0, min(top["confidence"].max() * 1.25, 1.0)],
        ),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=60, t=10, b=30),
        height=360,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")

    st.markdown("#### Classifier")
    model_name = st.radio(
        "Model",
        options=["LinearSVC", "LogisticRegression"],
        help=(
            "**LinearSVC** - slightly higher macro F1 (0.62 cleaned / 0.54 masked). "
            "Confidence scores are derived from decision function + softmax (not calibrated).\n\n"
            "**LogisticRegression** - native probability output; slightly lower F1 "
            "but scores are better calibrated."
        ),
    )

    st.markdown("#### Input mode")
    data_type = st.radio(
        "Preprocessing",
        options=["Cleaned", "Masked"],
        help=(
            "**Cleaned** - text is cleaned (URLs removed, contractions expanded) "
            "but MH keywords like *depression*, *anxiety* are kept. "
            "Use this for real-world text.\n\n"
            "**Masked** - all MH keywords are replaced with `[MH]` before inference. "
            "Lower F1 (-8 pp) but tests whether the model learned *language patterns* "
            "rather than surface keywords."
        ),
    )

    st.markdown("#### Load example")
    example_labels = ["- type your own -"] + [e["label"] for e in EXAMPLE_POSTS]
    example_choice = st.selectbox("Example posts", options=example_labels)

    st.markdown("---")

    # Show metric summary for current selection
    m = MODEL_METRICS.get((data_type, model_name), {})
    if m:
        col_a, col_b = st.columns(2)
        col_a.metric("Macro F1", f"{m['macro_f1']:.2f}")
        col_b.metric("Accuracy", f"{m['accuracy']:.0%}")
        st.caption("Metrics on held-out test set (77,943 posts, 16 classes)")

    st.markdown("---")
    st.caption(
        "⚠️ **Research use only.** "
        "This tool is not a clinical diagnostic instrument and must not be "
        "used as a substitute for professional mental health assessment."
    )


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("🧠 Mental Health Text Classifier")
st.markdown(
    "Classify Reddit-style posts across **16 mental health categories** using a "
    "traditional ML pipeline combining TF-IDF, character n-grams, and "
    "psychological features (VADER, Empath, NRC lexicons, POS ratios). "
    "Trained on the "
    "[Reddit Mental Health Dataset](https://zenodo.org/records/3941387) (2018-2019)."
)

# Populate textarea from example selector
default_text = ""
if example_choice != "- type your own -":
    matched = next((e for e in EXAMPLE_POSTS if e["label"] == example_choice), None)
    if matched:
        default_text = matched["text"]

input_text = st.text_area(
    "Paste or type a post to classify",
    value=default_text,
    height=160,
    placeholder="e.g. \"I've been feeling really low lately and can't get out of bed...\"",
)

word_count = len(input_text.split()) if input_text.strip() else 0
st.caption(f"{word_count} words - minimum recommended: 10")

classify_btn = st.button("▶ Classify", type="primary", disabled=(word_count < 3))


# ── Results ───────────────────────────────────────────────────────────────────
if classify_btn:
    if word_count < 3:
        st.warning("Please enter at least a few words.")
    else:
        try:
            # Load (or retrieve cached) model
            pipeline, le = load_pipeline(data_type, model_name)

            with st.spinner("Running inference..."):
                pred_label, results = predict(pipeline, le, input_text, model_name)

            display_name = LABEL_DISPLAY.get(pred_label, pred_label)
            top_conf = float(results.iloc[0]["confidence"])
            rank_of_pred = int(results[results["label"] == pred_label].index[0]) + 1

            st.markdown("---")

            # ── Prediction banner
            if pred_label in CRISIS_LABELS:
                st.error(
                    f"⚠️ **Predicted class: {display_name}** - confidence {top_conf:.1%}"
                )
                st.warning(
                    "**Crisis resources (Vietnam):** Đường dây hỗ trợ sức khỏe tâm thần "
                    "**1800 599 920** (miễn phí, 24/7)  \n"
                    "**International:** [findahelpline.com](https://findahelpline.com)"
                )
            elif pred_label == "non-mental":
                st.success(
                    f"✅ **Predicted class: {display_name}** "
                    f"- confidence {top_conf:.1%}"
                )
            else:
                st.info(
                    f"🔍 **Predicted class: {display_name}** "
                    f"- confidence {top_conf:.1%}"
                )

            # ── Body: chart + table side by side
            col_chart, col_table = st.columns([3, 2], gap="large")

            with col_chart:
                st.subheader("Top-10 confidence scores")
                fig = build_bar_chart(results, pred_label, top_n=10)
                st.plotly_chart(fig, use_container_width=True)

                # Soft calibration warning for LinearSVC
                if model_name == "LinearSVC":
                    st.caption(
                        "ℹ️ LinearSVC scores are derived via softmax on decision-function "
                        "margins - not calibrated probabilities. Use for ranking only."
                    )

            with col_table:
                st.subheader("Full class ranking")
                tbl = results[["display", "confidence"]].copy()
                tbl.columns = ["Class", "Score"]
                tbl["Score"] = tbl["Score"].map("{:.2%}".format)
                tbl.index = range(1, len(tbl) + 1)

                # Highlight predicted row
                def highlight_pred(row):
                    label_key = results.iloc[row.name - 1]["label"]
                    if label_key == pred_label:
                        return ["background-color: #e3f2fd; font-weight: bold"] * 2
                    return [""] * 2

                st.dataframe(
                    tbl.style.apply(highlight_pred, axis=1),
                    use_container_width=True,
                    height=380,
                )

            # ── Confidence note
            st.markdown("---")
            if top_conf < 0.40:
                st.warning(
                    f"Low confidence ({top_conf:.1%}) - the model is uncertain. "
                    "Consider that the text may exhibit overlapping signals from "
                    "multiple categories, or may be too short for reliable classification."
                )

            # ── Footer metadata
            metrics = MODEL_METRICS.get((data_type, model_name), {})
            macro_f1 = metrics.get("macro_f1", "-")
            accuracy = metrics.get("accuracy", "-")
            n_classes = len(le.classes_)

            st.caption(
                f"Model: **{model_name}** * "
                f"Mode: **{data_type}** * "
                f"Test-set macro F1: **{macro_f1}** * "
                f"Test-set accuracy: **{accuracy:.0%}** * "
                f"Classes: **{n_classes}**"
            )

        except FileNotFoundError as exc:
            st.error("**Model files not found.**")
            st.code(str(exc), language="text")
            st.info(
                "Run `python main.py` to train the models first, then restart this app."
            )

        except Exception as exc:
            st.error(f"Inference failed: {exc}")
            with st.expander("Full traceback"):
                st.exception(exc)


# ── Info expander - dataset & model overview ──────────────────────────────────
with st.expander("ℹ️ About the dataset and model", expanded=False):
    st.markdown("""
**Dataset:** [Reddit Mental Health Dataset](https://zenodo.org/records/3941387)
(Cohan et al., 2018-2019). Posts scraped from 16 MH subreddits + 11 control subreddits.

**Labels:** The subreddit a post was found in is used as a proxy label
(e.g. r/depression -> `major_depressive`). These are *self-reported* community memberships,
not clinical diagnoses - a known limitation of the dataset.

**Pipeline overview:**

| Stage | Detail |
|---|---|
| Noise filter | Remove deleted/short/bot/moderator posts |
| Clinical filter | Keep posts with high first-person ratio or self-disclosure patterns |
| Features | TF-IDF word (1-2 gram) + char (3-5 gram) + VADER + Empath + NRC lexicons + POS ratios |
| Resampling | Random under-sampling (max 15 k per class) |
| Tuning | Optuna (30 trials, 3-fold CV, macro F1) |
| Models | LinearSVC * LogisticRegression |

**Known limitations:**
- Subreddit membership != clinical diagnosis (noisy labels)
- Data from 2018-2019 Reddit; community language may have shifted
- Random post-level split may include the same user in train and test sets
- Masking experiment shows ~8 pp macro F1 drop, indicating reliance on surface keywords
    """)
