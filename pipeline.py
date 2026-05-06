import pandas as pd
import numpy as np
import gc

import optuna
import pickle
import nltk
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MaxAbsScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    # f1_score,
    # make_scorer,
    # confusion_matrix,
    # ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_score

from helper import Psychological, Filter
from config import *

nltk.download("vader_lexicon", quiet=True)

df = pd.read_csv('./archive/raw.csv')
df['subreddit'] = df['subreddit'].replace(LABEL_MAPPING)

_filter = Filter()
df = _filter.noise(df)
df = _filter.clinical(df)
df = df.dropna(subset=["post_cleaned", "post_masked", "subreddit"]).reset_index(drop=True)
print(f"Shape: {df.shape}")

le = LabelEncoder()
df["_subreddit"] = le.fit_transform(df["subreddit"])

print("Label encoding mapping:")
for idx, label in enumerate(le.classes_):
    print(f"{idx}: {label}")

X_cleaned_train, X_cleaned_test, X_masked_train, X_masked_test, y_train, y_test = train_test_split(
    df["post_cleaned"], 
    df["post_masked"],
    df["_subreddit"],
    test_size=TEST_SIZE,
    stratify=df["_subreddit"],
    random_state=RANDOM_SEED,
)


def get_feature_engineer():
    return Pipeline([
        (
            "extraction",
            FeatureUnion([
                # Luồng 1: Đặc trưng tâm lý (Giữ nguyên toàn bộ, không qua SelectKBest)
                ("psych_branch", Psychological()),

                # Luồng 2: Word TF-IDF -> Lọc bằng chi2 lấy 24,000 từ tốt nhất
                ("word_tfidf_branch", Pipeline([
                    ("tfidf", TfidfVectorizer(
                        ngram_range=(1, 2),
                        max_features=TFIDF_FEATS,
                        sublinear_tf=True,
                        token_pattern=r"(?u)\b[a-zA-Z\[\]][a-zA-Z\[\]]+\b",
                    )),
                    ("select", SelectKBest(chi2, k=int(SELECT_K * 0.75)))
                ])),

                # Luồng 3: Char TF-IDF -> Lọc bằng chi2 lấy 8,000 char tốt nhất
                ("char_tfidf_branch", Pipeline([
                    ("tfidf", TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        max_features=TFIDF_FEATS//3,
                        sublinear_tf=True,
                    )),
                    ("select", SelectKBest(chi2, k=int(SELECT_K * 0.25)))
                ]))
            ])
        ),
        # Scale toàn bộ ma trận (đã được nối lại từ 3 luồng) ở bước cuối
        ("scaler", MaxAbsScaler()),
    ])

# Xử lý đa luồng an toàn hơn để tránh OOM và tối ưu thời gian
def tune_and_train(model_name, X, y, n_trials=30, timeout_seconds=3600):
    def objective(trial):
        if model_name == "LinearSVC":
            params = {
                "C": trial.suggest_float("C", 1e-4, 1.0, log=True), # Giảm max C xuống 1.0
                "max_iter": 1000, # 1000 là đủ cho dual=False
                "dual": False,    # RẤT QUAN TRỌNG: Tăng tốc cực nhanh khi samples > features
                "random_state": RANDOM_SEED
            }
            model = LinearSVC(**params)

        elif model_name == "LogisticRegression":
            solver = trial.suggest_categorical("solver", ["lbfgs", "liblinear"])
            params = {
                "C": trial.suggest_float("C", 1e-4, 1.0, log=True), # Giảm max C
                "solver": solver,
                "max_iter": 1000,
                "random_state": RANDOM_SEED
            }
            # Kích hoạt đa luồng nếu dùng lbfgs
            if solver == "lbfgs":
                params["n_jobs"] = -1
            model = LogisticRegression(**params)

        elif model_name == "LightGBM":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 300), # Giảm tối đa xuống 300 để tránh quá lâu
                "learning_rate": trial.suggest_float("learning_rate", 1e-2, 0.1, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 20, 60),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 50),
                "random_state": RANDOM_SEED,
                "n_jobs": -1,
                "verbose": -1
            }
            model = lgb.LGBMClassifier(**params)

        score = cross_val_score(model, X, y, cv=CV_FOLDS, scoring='f1_macro', n_jobs=2)
        return score.mean()

    print(f"[TUNING] {model_name} (Max {n_trials} trials or {timeout_seconds//60} mins)...")
    study = optuna.create_study(direction="maximize")

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = RANDOM_SEED

    print(f"    [+] Best params found: {best_params}")

    # Cấu hình lại mô hình tốt nhất để huấn luyện lần cuối
    if model_name == "LinearSVC": 
        best_params["max_iter"] = 1000
        best_params["dual"] = False
        best_model = LinearSVC(**best_params)
    elif model_name == "LogisticRegression": 
        best_params["max_iter"] = 1000
        if best_params.get("solver") == "lbfgs":
            best_params["n_jobs"] = -1
        best_model = LogisticRegression(**best_params)
    elif model_name == "LightGBM": 
        best_params["n_jobs"] = -1
        best_params["verbose"] = -1
        best_model = lgb.LGBMClassifier(**best_params)
        
    best_model.fit(X, y)
    return best_model

# Tính toán Under-sampling strategy một lần
target_majority = 15000
train_class_counts = pd.Series(y_train).value_counts()
under_strategy = {
    label: min(count, target_majority) 
    for label, count in train_class_counts.items()
}

print("\n[GLOBAL] Undersampling...")
rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=RANDOM_SEED)

# 1. Resample trực tiếp trên mảng index (phải reshape thành 2D cho imblearn)
idx_2d = np.array(y_train.index).reshape(-1, 1)
idx_resampled_2d, y_resampled = rus.fit_resample(idx_2d, y_train)
idx_resampled = idx_resampled_2d.flatten()

# 2. Áp dụng index đã lọc cho cả 2 loại dữ liệu
X_cleaned_train_res = X_cleaned_train.loc[idx_resampled]
X_masked_train_res = X_masked_train.loc[idx_resampled]

run_configs = [
    ("CLEANED", X_cleaned_train_res, X_cleaned_test, MODEL_CLEAN),
    ("MASKED", X_masked_train_res, X_masked_test, MODEL_MASK)
]

for data_type, X_train_res, X_test, out_dir in run_configs:
    print(f"\n{'='*50}\nBẮT ĐẦU PIPELINE CHO DỮ LIỆU: {data_type}\n{'='*50}")

    # Đã undersample ở ngoài, chỉ cần fit_transform
    fe = get_feature_engineer()
    print(f"[{data_type}] Fitting Feature Engineer...")
    X_transformed = fe.fit_transform(X_train_res, y_resampled)

    # 3. Tuning & Training
    best_svc = tune_and_train("LinearSVC", X_transformed, y_resampled)
    best_lr  = tune_and_train("LogisticRegression", X_transformed, y_resampled)
    best_lgb = tune_and_train("LightGBM", X_transformed, y_resampled)

    del X_transformed
    gc.collect()

    # 4. ĐÁNH GIÁ TRÊN TẬP TEST CHO CẢ 3 MODEL
    print(f"\n[{data_type}] ĐÁNH GIÁ TRÊN TẬP TEST...")
    X_test_transformed = fe.transform(X_test)
    
    trained_models = {
        "LinearSVC": best_svc,
        "LogisticRegression": best_lr,
        "LightGBM": best_lgb
    }
    
    # Khởi tạo string để lưu vào file text
    report_text = f"EVALUATION REPORT - {data_type} DATA\n" + "="*50 + "\n"

    for model_name, model in trained_models.items():
        y_pred = model.predict(X_test_transformed)

        # In ra console
        print(f"\n--- Report {model_name} ({data_type}) ---")
        clf_report = classification_report(y_test, y_pred, target_names=le.classes_)
        print(clf_report)

        # Ghi vào biến lưu trữ
        report_text += f"\nModel: {model_name}\n{'-'*30}\n{clf_report}\n"

    # 5. Lưu trữ (Bao gồm Pipeline, Models và Report)
    print(f"[{data_type}] Xuất file vào {out_dir}...")
    with open(out_dir / "label_encoder.pkl", "wb") as f: pickle.dump(le, f)
    with open(out_dir / "feature_engineer.pkl", "wb") as f: pickle.dump(fe, f)
    with open(out_dir / "model_svc.pkl", "wb") as f: pickle.dump(best_svc, f)
    with open(out_dir / "model_lr.pkl", "wb") as f: pickle.dump(best_lr, f)
    with open(out_dir / "model_lgb.pkl", "wb") as f: pickle.dump(best_lgb, f)

    # Lưu báo cáo đánh giá ra file text để tiện xem lại sau
    with open(out_dir / "evaluation_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    inference_pipeline = Pipeline([("preprocessing", fe), ("classifier", best_lgb)])
    with open(out_dir / "inference_pipeline_lgb.pkl", "wb") as f: pickle.dump(inference_pipeline, f)

    # Dọn RAM trước khi sang vòng lặp kế tiếp
    del X_train_res, fe, inference_pipeline, X_test_transformed
    gc.collect()

print("\n[FINISHED] Toàn bộ quy trình đã xong và tối ưu RAM!")
