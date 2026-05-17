"""
Mental Health Text Classifier — Streamlit Demo
Stacking Ensemble: TF-IDF (LR + LinearSVC) + Psychological → LightGBM
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page config (Phải đặt ở đầu file) ─────────────────────────────────────────
st.set_page_config(
    page_title="MH Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Hằng số và Bảng màu biểu đồ ───────────────────────────────────────────────
LABEL_DISPLAY = {
    "adhd": "ADHD",
    "alcohol_use_disorder": "Alcohol Use Disorder",
    "autism_spectrum": "Autism Spectrum",
    "bipolar_disorder": "Bipolar Disorder",
    "borderline_personality": "Borderline Personality",
    "eating_disorder": "Eating Disorder",
    "generalized_anxiety": "Generalized Anxiety",
    "illness_anxiety": "Illness Anxiety",
    "major_depressive": "Major Depressive",
    "non-mental": "No MH Indicator",
    "ptsd": "PTSD",
    "schizophrenia": "Schizophrenia",
    "social_anxiety": "Social Anxiety",
    "social_isolation": "Social Isolation",
    "substance_use_disorder": "Substance Use Disorder",
    "suicidal_risk": "Suicidal Risk",
}

LABEL_EMOJI = {k: "🧠" for k in LABEL_DISPLAY}
LABEL_EMOJI["non-mental"] = "🌱"
LABEL_EMOJI["suicidal_risk"] = "⚠️"

RISK_LEVEL = {
    "suicidal_risk": ("CRITICAL", "#E11D48"),  # Rose
    "schizophrenia": ("HIGH", "#EA580C"),  # Orange
    "major_depressive": ("HIGH", "#EA580C"),
    "ptsd": ("HIGH", "#EA580C"),
    "bipolar_disorder": ("HIGH", "#EA580C"),
    "borderline_personality": ("MODERATE", "#D97706"),  # Amber
    "eating_disorder": ("MODERATE", "#D97706"),
    "substance_use_disorder": ("MODERATE", "#D97706"),
    "alcohol_use_disorder": ("MODERATE", "#D97706"),
    "generalized_anxiety": ("MODERATE", "#D97706"),
    "illness_anxiety": ("LOW", "#0284C7"),  # Sky Blue
    "social_anxiety": ("LOW", "#0284C7"),
    "social_isolation": ("LOW", "#0284C7"),
    "adhd": ("LOW", "#0284C7"),
    "autism_spectrum": ("LOW", "#0284C7"),
    "non-mental": ("NONE", "#10B981"),  # Emerald
}

BAR_COLORS = {
    "CRITICAL": "#E11D48",
    "HIGH": "#EA580C",
    "MODERATE": "#D97706",
    "LOW": "#0284C7",
    "NONE": "#10B981",
}

SAMPLES = [
    "I've been feeling really hopeless and empty for weeks. I can't get out of bed, I've lost interest in everything I used to love, and I don't see the point anymore.",
    "My mood changes so fast it scares me. Last week I had so much energy I barely slept and started three new projects, now I can't stop crying and don't want to leave the house.",
    "I just saw a really funny meme today and decided to share it here. Life is good, work is going well, and I'm really enjoying my new hobby of gardening.",
    "I was in a car accident two years ago and I still have nightmares about it almost every night. Loud sounds make me panic and I avoid driving completely.",
]

# ── CSS Tùy biến (Chỉ giữ lại các class đặc thù không có trong config) ─────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Ẩn thanh header mặc định của Streamlit */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 800px; }

/* Tiêu đề ứng dụng */
.app-header {
    text-align: center;
    padding: 1.5rem 0;
}
.app-header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    color: #111827;
    margin: 0;
}
.app-header p {
    color: #6B7280;
    font-size: 0.95rem;
    margin-top: 0.5rem;
}

/* Các thành phần hiển thị kết quả phân tích */
.result-label {
    font-size: 1.5rem;
    font-weight: 600;
    margin: 0 0 0.25rem 0;
}
.result-sub {
    color: #6B7280;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
}
.confidence-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 600;
}
.confidence-label {
    font-size: 0.75rem;
    color: #6B7280;
    text-transform: uppercase;
    font-weight: 500;
    letter-spacing: 0.05em;
}

/* Khối cảnh báo */
.disclaimer {
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-radius: 8px;
    padding: 1rem;
    font-size: 0.85rem;
    color: #991B1B;
    margin-top: 1.5rem;
}

/* Thẻ thông tin mô hình ở Sidebar */
.model-tag {
    background: #F3F4F6;
    border: 1px solid #E5E7EB;
    border-radius: 4px;
    padding: 2px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #374151;
}

hr.thin { border: none; border-top: 1px solid #E5E7EB; margin: 1.5rem 0; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Bộ tải Mô hình (Model Loader) ─────────────────────────────────────────────
HF_REPO_ID = "tkt-gemini/mh-classifier-v1"

ARTIFACT_DIR = Path("./archive/v1.0.1")

REQUIRED_FILES = [
    "label_encoder.pkl",
    "tfidf_pipeline.pkl",
    "base_lr.pkl",
    "base_svc.pkl",
    "psych_extractor.pkl",
    "meta_lgbm.pkl",
]


def download_artifacts():
    """Tải model files từ Hugging Face nếu chưa có trên máy."""
    from huggingface_hub import hf_hub_download

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for filename in REQUIRED_FILES:
        dest = ARTIFACT_DIR / filename
        if not dest.exists():
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=str(ARTIFACT_DIR),
            )


@st.cache_resource(show_spinner="Downloading & loading model artifacts…")
def load_pipeline():
    try:
        download_artifacts()
    except Exception as exc:
        return None, f"Không thể tải model từ Hugging Face: {exc}"

    def _load(name):
        with open(ARTIFACT_DIR / name, "rb") as fh:
            return pickle.load(fh)

    try:
        pipeline = {
            "le": _load("label_encoder.pkl"),
            "tfidf": _load("tfidf_pipeline.pkl"),
            "lr": _load("base_lr.pkl"),
            "svc": _load("base_svc.pkl"),
            "psych": _load("psych_extractor.pkl"),
            "lgbm": _load("meta_lgbm.pkl"),
        }
    except ModuleNotFoundError as exc:
        return (
            None,
            f"Missing Python dependency: `{exc.name}`. Vui lòng chạy ứng dụng qua pixi để nhận đúng môi trường.",
        )
    except Exception as exc:
        return None, f"Could not load model artifacts: {exc}"

    return pipeline, None


def predict(pipeline: dict, text: str):
    from scipy.special import softmax as _softmax

    texts_lower = [text.lower()]
    X_tf = pipeline["tfidf"].transform(texts_lower)
    lr_p = pipeline["lr"].predict_proba(X_tf)
    svc_p = _softmax(pipeline["svc"].decision_function(X_tf), axis=1)
    psy_p = pipeline["psych"].transform(texts_lower)
    X_meta = np.hstack([lr_p, svc_p, psy_p])

    probas = pipeline["lgbm"].predict_proba(X_meta)[0]
    label_idx = int(np.argmax(probas))
    pred_label = pipeline["le"].inverse_transform([label_idx])[0]
    probas_dict = dict(zip(pipeline["le"].classes_, probas))
    return pred_label, probas_dict


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 MH Classifier")
    st.markdown("---")
    st.markdown(
        """
<div style="font-size: 0.85rem; line-height: 1.6; color: #4B5563;">
<strong>Architecture</strong><br>
Two-layer stacking ensemble.<br><br>

<strong>Layer 1 — Base Models</strong><br>
<span class="model-tag">TF-IDF</span> (32K)<br>
→ <span class="model-tag">LogReg</span> & <span class="model-tag">LinearSVC</span><br><br>

<strong>Layer 2 — Meta Learner</strong><br>
<span class="model-tag">LightGBM</span> + 270 psych features.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("**Try a sample**")
    sample_labels = [
        "🟠 Depression",
        "🔴 Bipolar",
        "🟢 Non-mental",
        "🟣 PTSD",
    ]
    selected = st.selectbox(
        "Load example text",
        ["— choose —"] + sample_labels,
        label_visibility="collapsed",
    )
    if selected != "— choose —":
        idx = sample_labels.index(selected)
        st.session_state["sample_text"] = SAMPLES[idx]


# ── Vùng hiển thị chính (Main) ────────────────────────────────────────────────
st.markdown(
    """
<div class="app-header">
  <h1>Mental Health Text Classifier</h1>
  <p>Stacking Ensemble · TF-IDF + Psychological Features + LightGBM</p>
</div>
""",
    unsafe_allow_html=True,
)

pipeline, load_error = load_pipeline()
if not pipeline:
    st.error(f"**Lỗi tải mô hình:**\n\n{load_error}")
    st.stop()

# ── Khối Nhập liệu (Input) ───────────────────────────────────────────────────
default_text = st.session_state.get("sample_text", "")

user_text = st.text_area(
    "Văn bản phân tích",
    value=default_text,
    height=140,
    placeholder="Nhập nội dung văn bản hoặc bài đăng cần phân tích tại đây...",
    label_visibility="collapsed",
)

char_count = len(user_text.strip().split()) if user_text.strip() else 0
st.caption(f"Số từ: **{char_count}**")

col_btn, col_clear = st.columns([4, 1])
with col_btn:
    run = st.button(
        "Phân tích dữ liệu",
        use_container_width=True,
        type="primary",
        disabled=(char_count < 3),
    )
with col_clear:
    if st.button("Xóa", use_container_width=True):
        st.session_state["sample_text"] = ""
        st.rerun()

# ── Luồng xử lý kết quả (Inference + Results) ─────────────────────────────────
if run and char_count >= 3:
    with st.spinner("Đang chạy luồng suy luận..."):
        try:
            pred_label, probas_dict = predict(pipeline, user_text)
        except Exception as e:
            st.error(f"Inference error: {e}")
            st.stop()

    confidence = probas_dict[pred_label]
    display_name = LABEL_DISPLAY.get(pred_label, pred_label)
    emoji = LABEL_EMOJI.get(pred_label, "🧠")
    risk_level, risk_color = RISK_LEVEL.get(pred_label, ("UNKNOWN", "#6B7280"))

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 1])
    with col_l:
        st.markdown(
            f"""
        <div>
          <p class="result-label" style="color:{risk_color}">{emoji} {display_name}</p>
          <p class="result-sub">
            <span class="badge" style="background:{risk_color}">{risk_level} RISK</span>
            &nbsp; Predicted class: {pred_label}
          </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    with col_r:
        st.markdown(
            f"""
        <div style="text-align:right;">
          <div class="confidence-label">Confidence</div>
          <div class="confidence-val" style="color:{risk_color}">{confidence:.1%}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)

    # ── Biểu đồ Top-8 Phân phối Xác suất ──────────────────────────────────────
    top_n = sorted(probas_dict.items(), key=lambda x: -x[1])[:8]
    labels_disp = [LABEL_DISPLAY.get(k, k) for k, _ in top_n]
    values = [v for _, v in top_n]
    keys = [k for k, _ in top_n]
    bar_colors = [BAR_COLORS[RISK_LEVEL.get(k, ("LOW", ""))[0]] for k in keys]
    opacities = [1.0 if k == pred_label else 0.3 for k in keys]

    def hex_to_rgba(hex_color, alpha):
        h = hex_color.lstrip("#")
        if len(h) == 3:
            h = "".join(c + c for c in h)
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    final_colors = [hex_to_rgba(c, a) for c, a in zip(bar_colors, opacities)]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels_disp,
            orientation="h",
            marker=dict(color=final_colors, line=dict(width=0, color="rgba(0,0,0,0)")),
            text=[f"{v:.1%}" for v in values],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=11, color="#4B5563"),
            hovertemplate="<b>%{y}</b><br>Probability: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=0, r=50, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            range=[0, max(values) * 1.15],
            showgrid=False,
            showline=False,
            showticklabels=False,
            zeroline=False,
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(family="Inter", size=13, color="#374151"),
            showgrid=False,
        ),
        showlegend=False,
        dragmode=False,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ── Bảng phân phối chi tiết (Expander) ────────────────────────────────────
    with st.expander("Hiển thị toàn bộ phân phối 16 nhãn"):
        all_sorted = sorted(probas_dict.items(), key=lambda x: -x[1])
        df_all = pd.DataFrame(
            [
                {
                    "Nhãn dự đoán": LABEL_DISPLAY.get(k, k),
                    "Xác suất": v,
                    "Mức rủi ro": RISK_LEVEL.get(k, ("—", ""))[0],
                }
                for k, v in all_sorted
            ]
        )

        def color_risk(val):
            colors = {
                "CRITICAL": "#FFE4E6",
                "HIGH": "#FFEDD5",
                "MODERATE": "#FEF3C7",
                "LOW": "#E0F2FE",
                "NONE": "#D1FAE5",
            }
            return f"background-color: {colors.get(val, 'transparent')}; color: #374151"

        st.dataframe(
            df_all.style.format({"Xác suất": "{:.2%}"}).map(
                color_risk, subset=["Mức rủi ro"]
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ── Khối thông báo giới hạn trách nhiệm ───────────────────────────────────
    if risk_level in ("CRITICAL", "HIGH"):
        st.markdown(
            """
<div class="disclaimer">
🚨 <strong>Lưu ý:</strong> Kết quả cho thấy văn bản mang dấu hiệu rủi ro cao. Đây là <em>công cụ phân loại tự động</em> được phát triển cho mục đích nghiên cứu, không phải chẩn đoán y khoa. Vui lòng tham vấn chuyên gia tâm lý nếu cần thiết.
</div>
""",
            unsafe_allow_html=True,
        )
