"""
Streamlit Demo — Mental Health Text Classification
Two-layer Stacking Ensemble (TF-IDF → LR + SVC + Psych → LightGBM)
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── Must set page config first ────────────────────────────────────────────────
st.set_page_config(
    page_title="MindScan AI — Mental Health Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
    background: linear-gradient(160deg, #0a0a1a 0%, #0d1529 40%, #111827 100%);
}

/* Hero section */
.hero {
    text-align: center;
    padding: 2rem 1rem 1rem;
}
.hero h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
    letter-spacing: -0.5px;
}
.hero p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
}

/* Glass card */
.glass {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}

/* Metric card */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-card .label {
    color: #64748b;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 0.3rem;
}

/* Result badge */
.result-badge {
    display: inline-block;
    padding: 0.6rem 1.5rem;
    border-radius: 50px;
    font-weight: 700;
    font-size: 1.1rem;
    color: white;
    margin: 0.5rem 0;
}
.badge-risk { background: linear-gradient(135deg, #ef4444, #f97316); }
.badge-safe { background: linear-gradient(135deg, #10b981, #34d399); }
.badge-moderate { background: linear-gradient(135deg, #f59e0b, #eab308); }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15,23,42,0.95) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 0.95rem !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    color: #94a3b8;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
    border-color: rgba(99,102,241,0.4) !important;
}

/* Hide default header/footer */
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* Expander */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
}

/* Model switcher */
.model-switcher {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 0.5rem;
}
.model-switcher .version-tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 600;
    margin-left: 4px;
}
.tag-stacking { background: rgba(99,102,241,0.2); color: #a5b4fc; }
.tag-flat { background: rgba(251,191,36,0.2); color: #fbbf24; }

/* Radio buttons styling */
.stRadio > div { gap: 0.3rem; }
.stRadio label {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    padding: 0.4rem 0.8rem !important;
    transition: all 0.2s ease !important;
}
.stRadio label:hover {
    border-color: rgba(99,102,241,0.4) !important;
    background: rgba(99,102,241,0.1) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Labels info ───────────────────────────────────────────────────────────────
LABEL_INFO = {
    "adhd": ("🔵", "ADHD", "Attention Deficit Hyperactivity Disorder"),
    "alcohol_use_disorder": ("🟠", "Alcohol Use", "Alcohol Use Disorder"),
    "autism_spectrum": ("🟣", "Autism", "Autism Spectrum Disorder"),
    "bipolar_disorder": ("🔴", "Bipolar", "Bipolar Disorder"),
    "borderline_personality": ("🟡", "BPD", "Borderline Personality Disorder"),
    "eating_disorder": ("🟤", "Eating Disorder", "Eating Disorder"),
    "generalized_anxiety": ("🟠", "Anxiety", "Generalized Anxiety Disorder"),
    "illness_anxiety": ("⚪", "Health Anxiety", "Illness Anxiety Disorder"),
    "major_depressive": ("🔵", "Depression", "Major Depressive Disorder"),
    "non-mental": ("🟢", "Non-Mental", "No Mental Health Condition Detected"),
    "ptsd": ("🔴", "PTSD", "Post-Traumatic Stress Disorder"),
    "schizophrenia": ("🟣", "Schizophrenia", "Schizophrenia"),
    "social_anxiety": ("🟡", "Social Anxiety", "Social Anxiety Disorder"),
    "social_isolation": ("⚪", "Loneliness", "Social Isolation"),
    "substance_use_disorder": ("🟠", "Substance Use", "Substance Use Disorder"),
    "suicidal_risk": ("🔴", "Suicidal Risk", "Suicidal Ideation Risk"),
}

RISK_LABELS = {"suicidal_risk", "substance_use_disorder", "alcohol_use_disorder"}
SAFE_LABELS = {"non-mental"}

SAMPLE_TEXTS = {
    "💭 Depression": "I've been feeling really hopeless and can't get out of bed for weeks. Nothing seems to matter anymore and I don't enjoy things I used to love.",
    "😰 Anxiety": "My heart keeps racing and I can't stop worrying about everything. I feel like something terrible is about to happen all the time.",
    "🧠 ADHD": "I can never focus on anything for more than a few minutes. My mind just wanders constantly and I forget things all the time. It's affecting my work.",
    "🟢 Non-mental": "Just finished a great workout at the gym today. Feeling energized and ready to tackle the rest of the week!",
    "⚠️ Suicidal Risk": "I don't see the point in going on anymore. Everything feels so dark and empty. I just want the pain to stop.",
}


# ── Model version configs ─────────────────────────────────────────────────────
MODEL_VERSIONS = {
    "v1.0.1 — Stacking Ensemble": {
        "key": "v1.0.1",
        "tag": "stacking",
        "desc": "TF-IDF (LR + SVC via OOF) + Psych → LightGBM",
        "accuracy": "81%",
        "macro_f1": "67%",
        "features": "~302",
        "arch_html": """
            <b style="color:#a5b4fc;">Layer 1</b> — Base Models (OOF)<br>
            &nbsp;&nbsp;• TF-IDF → LogisticRegression<br>
            &nbsp;&nbsp;• TF-IDF → LinearSVC<br>
            &nbsp;&nbsp;• Psychological Features<br><br>
            <b style="color:#a5b4fc;">Layer 2</b> — Meta-Learner<br>
            &nbsp;&nbsp;• LightGBM (32 proba + ~270 psych)
        """,
        "perf": {
            "adhd": 0.81, "alcohol_use_disorder": 0.69, "autism_spectrum": 0.69,
            "bipolar_disorder": 0.55, "borderline_personality": 0.65,
            "eating_disorder": 0.73, "generalized_anxiety": 0.69,
            "illness_anxiety": 0.63, "major_depressive": 0.61,
            "non-mental": 0.95, "ptsd": 0.67, "schizophrenia": 0.66,
            "social_anxiety": 0.57, "social_isolation": 0.53,
            "substance_use_disorder": 0.63, "suicidal_risk": 0.64,
        },
    },
    "v1.0.0 — LogisticRegression": {
        "key": "v1.0.0",
        "sub": "lr",
        "tag": "flat",
        "desc": "TF-IDF + Psych → LogisticRegression (flat)",
        "accuracy": "79%",
        "macro_f1": "61%",
        "features": "~32K",
        "arch_html": """
            <b style="color:#fbbf24;">Flat Architecture</b><br><br>
            &nbsp;&nbsp;• TF-IDF (word + char) + Psych features<br>
            &nbsp;&nbsp;• FeatureUnion → StandardScaler + MaxAbsScaler<br>
            &nbsp;&nbsp;• LogisticRegression (Optuna-tuned)
        """,
        "perf": {
            "adhd": 0.81, "alcohol_use_disorder": 0.61, "autism_spectrum": 0.54,
            "bipolar_disorder": 0.40, "borderline_personality": 0.62,
            "eating_disorder": 0.66, "generalized_anxiety": 0.66,
            "illness_anxiety": 0.56, "major_depressive": 0.60,
            "non-mental": 0.94, "ptsd": 0.57, "schizophrenia": 0.53,
            "social_anxiety": 0.54, "social_isolation": 0.49,
            "substance_use_disorder": 0.56, "suicidal_risk": 0.62,
        },
    },
    "v1.0.0 — LinearSVC": {
        "key": "v1.0.0",
        "sub": "svc",
        "tag": "flat",
        "desc": "TF-IDF + Psych → LinearSVC (flat)",
        "accuracy": "80%",
        "macro_f1": "62%",
        "features": "~32K",
        "arch_html": """
            <b style="color:#fbbf24;">Flat Architecture</b><br><br>
            &nbsp;&nbsp;• TF-IDF (word + char) + Psych features<br>
            &nbsp;&nbsp;• FeatureUnion → StandardScaler + MaxAbsScaler<br>
            &nbsp;&nbsp;• LinearSVC (Optuna-tuned, softmax calibration)
        """,
        "perf": {
            "adhd": 0.82, "alcohol_use_disorder": 0.61, "autism_spectrum": 0.60,
            "bipolar_disorder": 0.45, "borderline_personality": 0.66,
            "eating_disorder": 0.66, "generalized_anxiety": 0.69,
            "illness_anxiety": 0.57, "major_depressive": 0.58,
            "non-mental": 0.94, "ptsd": 0.58, "schizophrenia": 0.58,
            "social_anxiety": 0.56, "social_isolation": 0.50,
            "substance_use_disorder": 0.55, "suicidal_risk": 0.64,
        },
    },
}


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_v101():
    """Load the v1.0.1 stacking inference pipeline."""
    project_root = str(Path(__file__).parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    artifact_dir = Path(__file__).parent / "archive" / "v1.0.1"

    def _load(name):
        with open(artifact_dir / name, "rb") as fh:
            return pickle.load(fh)

    return {
        "le": _load("label_encoder.pkl"),
        "tfidf": _load("tfidf_pipeline.pkl"),
        "lr": _load("base_lr.pkl"),
        "svc": _load("base_svc.pkl"),
        "psych": _load("psych_extractor.pkl"),
        "lgbm": _load("meta_lgbm.pkl"),
    }


@st.cache_resource(show_spinner=False)
def load_v100_lr():
    """Load the v1.0.0 LR inference pipeline."""
    artifact_dir = Path(__file__).parent / "archive" / "v1.0.0"
    with open(artifact_dir / "label_encoder.pkl", "rb") as fh:
        le = pickle.load(fh)
    with open(artifact_dir / "inference_pipeline_lr.pkl", "rb") as fh:
        pipe = pickle.load(fh)
    return {"le": le, "pipeline": pipe}


@st.cache_resource(show_spinner=False)
def load_v100_svc():
    """Load the v1.0.0 SVC inference pipeline."""
    artifact_dir = Path(__file__).parent / "archive" / "v1.0.0"
    with open(artifact_dir / "label_encoder.pkl", "rb") as fh:
        le = pickle.load(fh)
    with open(artifact_dir / "inference_pipeline_svc.pkl", "rb") as fh:
        pipe = pickle.load(fh)
    return {"le": le, "pipeline": pipe}


def predict(text, version_name):
    """Unified prediction function for all model versions."""
    from scipy.special import softmax as sp_softmax

    cfg = MODEL_VERSIONS[version_name]

    if cfg["key"] == "v1.0.1":
        m = load_v101()
        texts = [text.lower()]
        X_tf = m["tfidf"].transform(texts)
        lr_p = m["lr"].predict_proba(X_tf)
        svc_p = sp_softmax(m["svc"].decision_function(X_tf), axis=1)
        psy_p = m["psych"].transform(texts)
        meta = np.hstack([lr_p, svc_p, psy_p])
        probas = m["lgbm"].predict_proba(meta)[0]
        pred_idx = np.argmax(probas)
        pred_label = m["le"].inverse_transform([pred_idx])[0]
        proba_dict = dict(zip(m["le"].classes_, probas))
        return pred_label, proba_dict

    else:  # v1.0.0 flat models
        if cfg.get("sub") == "svc":
            m = load_v100_svc()
            texts = [text]
            # SVC: decision_function → softmax
            decision = m["pipeline"].decision_function(texts)
            probas = sp_softmax(decision, axis=1)[0]
        else:
            m = load_v100_lr()
            texts = [text]
            probas = m["pipeline"].predict_proba(texts)[0]

        pred_idx = np.argmax(probas)
        pred_label = m["le"].inverse_transform([pred_idx])[0]
        proba_dict = dict(zip(m["le"].classes_, probas))
        return pred_label, proba_dict


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(str(Path(__file__).parent / "img" / "logo.png"), width=80)
    st.markdown("### 🧠 MindScan AI")
    st.markdown(
        '<p style="color:#64748b;font-size:0.85rem;">'
        "Mental Health Text Classification</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── Model version selector ────────────────────────────────────────────────
    st.markdown("#### 🔄 Model Version")
    selected_version = st.radio(
        "Select model",
        list(MODEL_VERSIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    vcfg = MODEL_VERSIONS[selected_version]
    tag_cls = "tag-stacking" if vcfg["tag"] == "stacking" else "tag-flat"
    tag_label = "STACKING" if vcfg["tag"] == "stacking" else "FLAT"
    st.markdown(
        f'<p style="color:#64748b;font-size:0.78rem;margin-top:4px;">'
        f'{vcfg["desc"]} '
        f'<span class="version-tag {tag_cls}">{tag_label}</span></p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 📊 Architecture")
    st.markdown(
        f'<div style="color:#94a3b8;font-size:0.82rem;line-height:1.7;">'
        f'{vcfg["arch_html"]}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 🏷️ 16 Categories")
    for key, (emoji, short, full) in LABEL_INFO.items():
        st.markdown(
            f'<span style="font-size:0.8rem;color:#94a3b8;">{emoji} {short}</span>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown(
        '<p style="color:#475569;font-size:0.75rem;text-align:center;">'
        "Project 2 — UTEHY<br>Reddit Mental Health Dataset</p>",
        unsafe_allow_html=True,
    )


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero">
        <h1>🧠 MindScan AI</h1>
        <p>Analyze text for mental health indicators using a two-layer stacking ensemble</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Preload selected model ────────────────────────────────────────────────────
with st.spinner("🔄 Loading model... (first time may take a moment)"):
    if vcfg["key"] == "v1.0.1":
        load_v101()
    elif vcfg.get("sub") == "svc":
        load_v100_svc()
    else:
        load_v100_lr()

# ── Model metrics row (dynamic) ──────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
metrics = [
    (vcfg["accuracy"], "Accuracy"),
    (vcfg["macro_f1"], "Macro F1"),
    ("16", "Categories"),
    (vcfg["features"], "Features"),
]
for col, (val, lab) in zip([m1, m2, m3, m4], metrics):
    col.markdown(
        f'<div class="metric-card"><div class="value">{val}</div>'
        f'<div class="label">{lab}</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Analyze Text", "📋 Batch Analysis", "ℹ️ About"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: Single text analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### ✍️ Enter Text")

        # Sample selector
        sample_choice = st.selectbox(
            "Try a sample:",
            ["— Custom input —"] + list(SAMPLE_TEXTS.keys()),
            label_visibility="collapsed",
        )

        default_text = SAMPLE_TEXTS.get(sample_choice, "")
        user_text = st.text_area(
            "Text to analyze",
            value=default_text,
            height=180,
            placeholder="Type or paste text here...",
            label_visibility="collapsed",
        )

        analyze_btn = st.button("🔍 Analyze", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_result:
        if analyze_btn and user_text.strip():
            with st.spinner("Analyzing..."):
                pred_label, proba_dict = predict(
                    user_text, selected_version
                )

            info = LABEL_INFO.get(pred_label, ("❓", pred_label, pred_label))
            emoji, short_name, full_name = info

            # Badge color
            if pred_label in RISK_LABELS:
                badge_cls = "badge-risk"
            elif pred_label in SAFE_LABELS:
                badge_cls = "badge-safe"
            else:
                badge_cls = "badge-moderate"

            confidence = proba_dict[pred_label] * 100

            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Prediction Result")
            st.markdown(
                f'<div class="result-badge {badge_cls}">'
                f"{emoji} {full_name}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p style="color:#94a3b8;margin-top:0.5rem;">'
                f"Confidence: <b style='color:#a5b4fc;'>{confidence:.1f}%</b></p>",
                unsafe_allow_html=True,
            )

            # Top 5 probabilities chart
            sorted_proba = sorted(proba_dict.items(), key=lambda x: -x[1])[:8]
            labels_top = [LABEL_INFO.get(k, ("", k, k))[1] for k, _ in sorted_proba]
            values_top = [v * 100 for _, v in sorted_proba]

            colors = [
                "#6366f1" if i == 0 else "rgba(99,102,241,0.25)"
                for i in range(len(labels_top))
            ]

            fig = go.Figure(
                go.Bar(
                    x=values_top[::-1],
                    y=labels_top[::-1],
                    orientation="h",
                    marker=dict(
                        color=colors[::-1],
                        line=dict(width=0),
                        cornerradius=6,
                    ),
                    text=[f"{v:.1f}%" for v in values_top[::-1]],
                    textposition="outside",
                    textfont=dict(color="#94a3b8", size=12),
                )
            )
            fig.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                height=300,
                margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(
                    showgrid=False, showticklabels=False, zeroline=False, range=[0, max(values_top) * 1.3]
                ),
                yaxis=dict(showgrid=False),
                bargap=0.3,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Full probability breakdown
            with st.expander("📊 Full Probability Breakdown"):
                for label, prob in sorted(proba_dict.items(), key=lambda x: -x[1]):
                    info = LABEL_INFO.get(label, ("❓", label, label))
                    bar_width = prob * 100
                    color = "#6366f1" if label == pred_label else "#334155"
                    st.markdown(
                        f'<div style="display:flex;align-items:center;margin:4px 0;">'
                        f'<span style="color:#94a3b8;width:140px;font-size:0.8rem;">{info[0]} {info[1]}</span>'
                        f'<div style="flex:1;background:#1e293b;border-radius:6px;height:20px;margin:0 8px;">'
                        f'<div style="width:{bar_width}%;background:{color};height:100%;border-radius:6px;'
                        f'transition:width 0.5s;"></div></div>'
                        f'<span style="color:#64748b;font-size:0.8rem;width:50px;">{prob:.3f}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        elif analyze_btn:
            st.warning("Please enter some text to analyze.")
        else:
            st.markdown('<div class="glass">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Results")
            st.markdown(
                '<p style="color:#64748b;text-align:center;padding:3rem 0;">'
                "Enter text and click <b>Analyze</b> to see results</p>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: Batch analysis
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 📋 Batch Text Analysis")
    st.markdown(
        '<p style="color:#64748b;font-size:0.9rem;">Enter multiple texts, one per line.</p>',
        unsafe_allow_html=True,
    )

    batch_text = st.text_area(
        "Batch input",
        height=200,
        placeholder="Line 1: I feel so anxious all the time...\nLine 2: Had a great day at work today!\nLine 3: I can't stop thinking about death...",
        label_visibility="collapsed",
    )

    if st.button("🚀 Analyze Batch", use_container_width=True):
        lines = [l.strip() for l in batch_text.strip().split("\n") if l.strip()]
        if lines:
            results = []
            progress = st.progress(0)
            for i, line in enumerate(lines):
                pred_label, proba_dict = predict(line, selected_version)
                conf = proba_dict[pred_label]
                info = LABEL_INFO.get(pred_label, ("❓", pred_label, pred_label))
                results.append(
                    {
                        "Text": line[:80] + ("..." if len(line) > 80 else ""),
                        "Prediction": f"{info[0]} {info[1]}",
                        "Confidence": f"{conf:.1%}",
                        "Label": pred_label,
                    }
                )
                progress.progress((i + 1) / len(lines))

            progress.empty()

            import pandas as pd
            df = pd.DataFrame(results)
            st.dataframe(
                df[["Text", "Prediction", "Confidence"]],
                use_container_width=True,
                hide_index=True,
            )

            # Distribution chart
            label_counts = pd.Series([r["Label"] for r in results]).value_counts()
            display_labels = [LABEL_INFO.get(l, ("", l, l))[1] for l in label_counts.index]

            fig2 = go.Figure(
                go.Pie(
                    labels=display_labels,
                    values=label_counts.values,
                    hole=0.5,
                    marker=dict(
                        colors=[
                            "#6366f1", "#8b5cf6", "#a78bfa", "#c4b5fd",
                            "#60a5fa", "#38bdf8", "#34d399", "#fbbf24",
                            "#f472b6", "#fb923c", "#94a3b8", "#64748b",
                        ]
                    ),
                    textfont=dict(color="white", size=12),
                )
            )
            fig2.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#94a3b8"),
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                legend=dict(font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("Please enter at least one line of text.")

    st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: About
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown(f"#### 🏗️ Architecture — {selected_version}")
        st.markdown(
            f'<div style="color:#94a3b8;font-size:0.9rem;line-height:1.8;">'
            f'{vcfg["arch_html"]}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Comparison table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### 🔀 Version Comparison")
        import pandas as pd
        comp_data = []
        for vname, vc in MODEL_VERSIONS.items():
            comp_data.append({
                "Model": vname,
                "Type": vc["tag"].upper(),
                "Accuracy": vc["accuracy"],
                "Macro F1": vc["macro_f1"],
            })
        st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown("#### 📈 Performance")

        perf_data = vcfg["perf"]
        perf_labels = [LABEL_INFO.get(k, ("", k, k))[1] for k in perf_data]
        perf_values = list(perf_data.values())

        fig3 = go.Figure(
            go.Bar(
                x=perf_values,
                y=perf_labels,
                orientation="h",
                marker=dict(
                    color=perf_values,
                    colorscale=[[0, "#ef4444"], [0.5, "#f59e0b"], [1, "#10b981"]],
                    cornerradius=5,
                ),
                text=[f"{v:.0%}" for v in perf_values],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            )
        )
        fig3.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", color="#94a3b8", size=11),
            height=500,
            margin=dict(l=10, r=50, t=10, b=10),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False, range=[0, 1.15]),
            yaxis=dict(showgrid=False, autorange="reversed"),
            bargap=0.25,
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("#### 📚 Dataset")
    st.markdown(
        """
        <div style="color:#94a3b8;font-size:0.9rem;line-height:1.7;">
        <b style="color:#a5b4fc;">Reddit Mental Health Dataset</b>
        (<a href="https://zenodo.org/records/3941387" style="color:#60a5fa;">Zenodo</a>)<br><br>
        • <b>15</b> mental health subreddits (r/depression, r/anxiety, r/adhd, ...)<br>
        • <b>11</b> non-mental health subreddits as control group<br>
        • Preprocessing: noise filtering, clinical relevance filtering, MH keyword masking<br>
        • Train/test split: 85/15 with stratification + undersampling
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center;padding:2rem;color:#475569;font-size:0.8rem;">
        ⚠️ <b>Disclaimer</b>: This tool is for educational and research purposes only.
        It is NOT a substitute for professional mental health diagnosis or treatment.
        If you or someone you know is in crisis, please contact a mental health professional.
        </div>
        """,
        unsafe_allow_html=True,
    )
