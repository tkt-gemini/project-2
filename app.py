"""
app.py — Streamlit UI cho Reddit Mental Health Classifier

Hỗ trợ hai model variant:
    Masked  — train trên post_masked (MH keywords bị ẩn) → artifacts/masked/
    Clean   — train trên post_clean  (keywords giữ nguyên) → artifacts/clean/

Chạy:
    streamlit run app.py
"""

import __main__
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import pipeline

# ── Joblib unpickle fix ───────────────────────────────────────────────────────
# Ép class vào __main__ để joblib giải nén đúng khi chạy qua Streamlit
__main__.PsychologicalExtractor  = pipeline.FeatureExtractor
__main__.FeatureEngineer         = pipeline.FeatureEngineer
__main__.MentalHealthClassifier  = pipeline.MentalHealthClassifier
__main__.NRCLex                  = pipeline.NRCLex
__main__._NRCEmo                 = pipeline._NRCEmo
__main__._NRCVAD                 = pipeline._NRCVAD
__main__._NRCAIL                 = pipeline._NRCAIL

# ── Cấu hình trang ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mental Health Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helpers ───────────────────────────────────────────────────────────────────

VARIANT_DIRS = {
    "masked": pipeline.MODEL_MASK,
    "clean" : pipeline.MODEL_CLEAN,
}

VARIANT_LABELS = {
    "masked": "Masked (leakage-free)",
    "clean" : "Clean (with keywords)",
}

VARIANT_HELP = {
    "masked": (
        "Train trên text đã **ẩn tên bệnh** (`[MH]`). "
        "Model buộc phải học từ triệu chứng và cảm xúc. "
        "Đây là variant đánh giá năng lực ngôn ngữ thực sự của model."
    ),
    "clean": (
        "Train trên text **giữ nguyên tên bệnh**. "
        "Model có thể dùng từ khoá như *depression*, *anxiety* làm đặc trưng. "
        "Accuracy thường cao hơn nhưng không phản ánh linguistic patterns thuần tuý."
    ),
}

LABEL_COLORS = {
    "major_depressive"      : "#E57373",
    "suicidal_risk"         : "#B71C1C",
    "generalized_anxiety"   : "#FFB74D",
    "social_anxiety"        : "#FFA726",
    "illness_anxiety"       : "#FF8F00",
    "bipolar_disorder"      : "#BA68C8",
    "borderline_personality": "#AB47BC",
    "ptsd"                  : "#7986CB",
    "schizophrenia"         : "#5C6BC0",
    "adhd"                  : "#4DB6AC",
    "autism_spectrum"       : "#26A69A",
    "eating_disorder"       : "#F48FB1",
    "substance_use_disorder": "#A1887F",
    "alcohol_use_disorder"  : "#8D6E63",
    "social_isolation"      : "#90A4AE",
    "non-mental"            : "#81C784",
}

LABEL_VI = {
    "major_depressive"      : "Trầm cảm",
    "suicidal_risk"         : "Rủi ro tự tử",
    "generalized_anxiety"   : "Lo âu toàn thể",
    "social_anxiety"        : "Lo âu xã hội",
    "illness_anxiety"       : "Lo âu sức khoẻ",
    "bipolar_disorder"      : "Rối loạn lưỡng cực",
    "borderline_personality": "Nhân cách ranh giới",
    "ptsd"                  : "PTSD",
    "schizophrenia"         : "Tâm thần phân liệt",
    "adhd"                  : "ADHD",
    "autism_spectrum"       : "Phổ tự kỷ",
    "eating_disorder"       : "Rối loạn ăn uống",
    "substance_use_disorder": "Lạm dụng chất",
    "alcohol_use_disorder"  : "Lạm dụng rượu",
    "social_isolation"      : "Cô lập xã hội",
    "non-mental"            : "Không có vấn đề tâm thần",
}


def _artifact_ready(variant: str) -> bool:
    d = VARIANT_DIRS[variant]
    return (d / "model.joblib").exists() and (d / "tfidf.joblib").exists()


def _load_eval_report(variant: str) -> dict | None:
    path = VARIANT_DIRS[variant] / "eval_report.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def _load_model_meta(variant: str) -> dict | None:
    path = VARIANT_DIRS[variant] / "model_meta.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


@st.cache_resource(show_spinner="Đang tải model...")
def load_classifier(variant: str) -> "pipeline.MentalHealthClassifier | None":
    try:
        return pipeline.MentalHealthClassifier.load(variant=variant)
    except Exception as e:
        st.error(f"Lỗi khi tải model [{variant}]: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("Setting")
    st.markdown("---")

    # ── Chọn variant ─────────────────────────────────────────────────────────
    st.subheader("Model variant")

    available = [v for v in ["masked", "clean"] if _artifact_ready(v)]
    missing   = [v for v in ["masked", "clean"] if not _artifact_ready(v)]

    if not available:
        st.error("Chưa có artifact nào. Chạy training trước:")
        st.code("python pipeline.py --train-both", language="bash")
        st.stop()

    if missing:
        st.warning(
            f"Variant **{missing[0]}** chưa được train. "
            f"Chỉ có thể dùng **{available[0]}**."
        )

    variant_label = st.radio(
        "Chọn model:",
        options=available,
        format_func=lambda v: VARIANT_LABELS[v],
        index=0,
    )
    st.caption(VARIANT_HELP[variant_label])

    st.markdown("---")

    # ── Thông tin model ───────────────────────────────────────────────────────
    st.subheader("Thông tin model")
    meta   = _load_model_meta(variant_label)
    report = _load_eval_report(variant_label)

    if meta:
        st.markdown(f"**Algorithm:** `{meta.get('model_name', 'N/A')}`")
        st.markdown(f"**Trained at:** {meta.get('trained_at', 'N/A')[:10]}")

    if report:
        m_f1 = report.get("macro_f1", 0)
        w_f1 = report.get("weighted_f1", 0)
        col1, col2 = st.columns(2)
        col1.metric("macro-F1", f"{m_f1:.3f}")
        col2.metric("weighted-F1", f"{w_f1:.3f}")
        st.caption(f"Evaluated on {report.get('n_test', '?'):,} holdout posts")

    st.markdown("---")

    # ── So sánh hai variant ───────────────────────────────────────────────────
    if len(available) == 2:
        st.subheader("So sánh variants")
        rows = []
        for v in ["masked", "clean"]:
            r = _load_eval_report(v)
            m = _load_model_meta(v)
            if r and m:
                rows.append({
                    "Variant"     : VARIANT_LABELS[v],
                    "Algorithm"   : m.get("model_name", "-"),
                    "macro-F1"    : f"{r['macro_f1']:.3f}",
                    "weighted-F1" : f"{r['weighted_f1']:.3f}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Variant"), use_container_width=True)

    st.markdown("---")
    st.caption("Reddit Mental Health Classifier · Zenodo 3941387")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════════════════

st.title("🧠 Reddit Mental Health Classifier")
st.markdown(
    f"Dự đoán nhóm rủi ro tâm thần từ văn bản tiếng Anh &nbsp;·&nbsp; "
    f"Đang dùng: **{VARIANT_LABELS[variant_label]}**"
)

# ── Load model ────────────────────────────────────────────────────────────────
clf = load_classifier(variant_label)
if clf is None:
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch, tab_details = st.tabs(["Phân tích đơn", "Phân tích hàng loạt", "Chi tiết model"])

# ─── Tab 1: Phân tích đơn ─────────────────────────────────────────────────────
with tab_single:
    text_input = st.text_area(
        "Content:",
        height=180,
        placeholder="...",
    )

    col_btn, col_ex = st.columns([1, 4])
    predict_clicked = col_btn.button("Phân tích", type="primary", use_container_width=True)

    with col_ex.expander("Examples"):
        examples = {
            "Depression"  : "I've been feeling empty for weeks. Can't sleep, can't eat, don't want to do anything. Everything feels pointless.",
            "ADHD"        : "My brain just won't stop jumping around. I start five things and finish none of them. Been like this my whole life.",
            "Non-mental"  : "Just got back from the gym, managed 5x5 squats at 100kg today. Feeling great about my progress this month.",
        }
        for label, ex_text in examples.items():
            if st.button(f"Example: {label}", key=f"ex_{label}"):
                st.session_state["example_text"] = ex_text
                st.rerun()

    # Áp dụng ví dụ nếu được chọn
    if "example_text" in st.session_state:
        text_input = st.session_state.pop("example_text")

    if predict_clicked or text_input.strip():
        if not text_input.strip():
            st.warning("Vui lòng nhập văn bản.")
        else:
            with st.spinner("Đang phân tích..."):
                result = clf.predict(text_input)

            label      = result["label"]
            confidence = result["confidence"]
            proba_dict = result["probabilities"]
            color      = LABEL_COLORS.get(label, "#90A4AE")
            label_vi   = LABEL_VI.get(label, label)

            st.divider()

            # Kết quả chính
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(
                    f"""<div style='background:{color}22;border-left:4px solid {color};
                    padding:16px;border-radius:8px;'>
                    <span style='font-size:13px;color:{color};font-weight:600;
                    text-transform:uppercase;letter-spacing:1px;'>Dự đoán</span><br>
                    <span style='font-size:24px;font-weight:700;color:{color};'>{label_vi}</span><br>
                    <span style='font-size:12px;color:#888;'>{label}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )
            with col2:
                if confidence is not None:
                    st.metric("Độ tin cậy", f"{confidence:.1%}")
            with col3:
                st.metric("Variant", variant_label)

            # Biểu đồ xác suất Top 5
            if proba_dict:
                st.markdown("#### Phân bố xác suất (Top 5)")
                top5 = sorted(proba_dict.items(), key=lambda x: -x[1])[:5]
                df_p = pd.DataFrame(top5, columns=["Nhóm", "Xác suất"])
                df_p["Nhãn tiếng Việt"] = df_p["Nhóm"].map(lambda x: LABEL_VI.get(x, x))
                df_p["Xác suất (%)"]    = (df_p["Xác suất"] * 100).round(2)
                df_p = df_p.set_index("Nhãn tiếng Việt")[["Xác suất (%)"]]
                st.bar_chart(df_p, horizontal=True, color="#5C6BC0")

            # Text đã xử lý (debug)
            with st.expander("🔧 Xem text sau tiền xử lý"):
                tfidf_text, clean = clf._prepare(text_input)
                c1, c2 = st.columns(2)
                c1.markdown("**post_clean** (cho NRC/Empath):")
                c1.code(clean, language=None)
                c2.markdown("**text cho TF-IDF:**")
                c2.code(tfidf_text, language=None)
                if variant_label == "masked":
                    n_masked = tfidf_text.count("[MH]")
                    if n_masked:
                        st.info(f"{n_masked} MH keyword(s) bị mask thành `[MH]`")
                    else:
                        st.success("Không phát hiện MH keywords trong text này.")


# ─── Tab 2: Phân tích hàng loạt ───────────────────────────────────────────────
with tab_batch:
    st.markdown("Dán nhiều đoạn văn bản, mỗi dòng một bài. Tối đa 200 dòng.")
    batch_input = st.text_area(
        "Nhập văn bản (mỗi dòng = 1 bài):",
        height=220,
        placeholder="I've been feeling empty for weeks...\nMy brain won't stop jumping around...\nJust got back from the gym today...",
    )

    if st.button("Phân tích hàng loạt", type="primary"):
        lines = [l.strip() for l in batch_input.splitlines() if l.strip()]
        if not lines:
            st.warning("Vui lòng nhập ít nhất một dòng.")
        elif len(lines) > 200:
            st.error("Tối đa 200 dòng mỗi lần.")
        else:
            with st.spinner(f"Đang phân tích {len(lines)} bài..."):
                results = clf.predict_batch(lines)

            rows = []
            for text, res in zip(lines, results):
                rows.append({
                    "Text (50 ký tự)"  : text[:50] + ("..." if len(text) > 50 else ""),
                    "Dự đoán"          : LABEL_VI.get(res["label"], res["label"]),
                    "Nhãn (EN)"        : res["label"],
                    "Độ tin cậy"       : f"{res['confidence']:.1%}" if res["confidence"] else "N/A",
                })

            df_result = pd.DataFrame(rows)
            st.dataframe(df_result, use_container_width=True, height=400)

            # Download
            csv_bytes = df_result.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name=f"mh_predictions_{variant_label}.csv",
                mime="text/csv",
            )

            # Phân phối nhãn
            st.markdown("#### Phân phối dự đoán")
            dist = df_result["Dự đoán"].value_counts().reset_index()
            dist.columns = ["Nhóm", "Số lượng"]
            st.bar_chart(dist.set_index("Nhóm"), color="#4DB6AC")


# ─── Tab 3: Chi tiết model ────────────────────────────────────────────────────
with tab_details:
    report = _load_eval_report(variant_label)
    meta   = _load_model_meta(variant_label)

    if not report or not meta:
        st.info("Chưa có eval_report.json. Chạy training để tạo report.")
    else:
        st.markdown(f"### Model: `{meta.get('model_name')}` · Variant: `{variant_label}`")

        col1, col2, col3 = st.columns(3)
        col1.metric("macro-F1",    f"{report['macro_f1']:.4f}")
        col2.metric("weighted-F1", f"{report['weighted_f1']:.4f}")
        col3.metric("Test posts",  f"{report.get('n_test', '?'):,}")

        st.markdown("#### Per-class Performance")
        per_class = report.get("per_class", {})
        rows = []
        for cls, metrics in per_class.items():
            if cls in ("accuracy", "macro avg", "weighted avg"):
                continue
            rows.append({
                "Class"         : cls,
                "Tiếng Việt"    : LABEL_VI.get(cls, cls),
                "Precision"     : round(metrics.get("precision", 0), 3),
                "Recall"        : round(metrics.get("recall", 0), 3),
                "F1-score"      : round(metrics.get("f1-score", 0), 3),
                "Support"       : int(metrics.get("support", 0)),
            })

        if rows:
            df_cls = pd.DataFrame(rows).set_index("Class")
            st.dataframe(
                df_cls.style.background_gradient(subset=["F1-score"], cmap="RdYlGn"),
                use_container_width=True,
                height=500,
            )

        # Config
        with st.expander("Xem config training"):
            st.json(meta.get("config", {}))

        # Confusion matrix (nếu có)
        cm_path = VARIANT_DIRS[variant_label] / "confusion_matrix.csv"
        if cm_path.exists():
            with st.expander("Xem confusion matrix"):
                cm_df = pd.read_csv(cm_path, index_col=0)
                st.dataframe(cm_df, use_container_width=True)

        # Top features (nếu có)
        fm_path = VARIANT_DIRS[variant_label] / "feature_meta.json"
        if fm_path.exists():
            with st.expander("Xem top TF-IDF features (200)"):
                with open(fm_path) as f:
                    fm = json.load(f)
                top_feats = fm.get("top_200_tfidf_features", [])
                st.write(", ".join(f"`{t}`" for t in top_feats[:50]))
                if len(top_feats) > 50:
                    st.caption(f"... và {len(top_feats)-50} features khác")
    