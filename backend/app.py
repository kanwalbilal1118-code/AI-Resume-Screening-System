import io
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

try:
    from tensorflow.keras.models import load_model
    tf_available = True
except ImportError:
    load_model = None
    tf_available = False

try:
    from PyPDF2 import PdfReader
    pdf_available = True
except ImportError:
    try:
        from pypdf import PdfReader
        pdf_available = True
    except ImportError:
        PdfReader = None
        pdf_available = False


BASE_DIR = Path(__file__).resolve().parent


# ==================================
# PAGE CONFIG
# ==================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="resume",
    layout="wide",
)


# ==================================
# SIDEBAR
# ==================================

st.sidebar.title("AI Resume Analyzer Features")

st.sidebar.success("Machine Learning & Deep Learning Project")

st.sidebar.info(
    """
Models Used

- Random Forest
- Gradient Boosting
- CNN
- LSTM

Features

- ATS Scoring
- Resume Classification
- Skill Gap Analysis
- Career Guidance
- Job Recommendation
"""
)


# ==================================
# LOAD FILES
# ==================================

@st.cache_resource
def load_models():
    rf_model = joblib.load(BASE_DIR / "rf_model.pkl")
    gb_model = joblib.load(BASE_DIR / "gb_model.pkl")

    cnn_model = None
    lstm_model = None
    if tf_available and load_model is not None:
        try:
            cnn_model = load_model(BASE_DIR / "cnn_model.h5")
        except Exception:
            cnn_model = None

        try:
            lstm_model = load_model(BASE_DIR / "lstm_model.h5")
        except Exception:
            lstm_model = None

    tfidf = joblib.load(BASE_DIR / "tfidf.pkl")
    label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")
    tokenizer = joblib.load(BASE_DIR / "tokenizer.pkl")
    job_tfidf = joblib.load(BASE_DIR / "job_tfidf.pkl")
    jobs_df = pd.read_csv(BASE_DIR / "jobs_small.csv")

    return (
        rf_model,
        gb_model,
        cnn_model,
        lstm_model,
        tfidf,
        label_encoder,
        tokenizer,
        job_tfidf,
        jobs_df,
    )


(
    rf_model,
    gb_model,
    cnn_model,
    lstm_model,
    tfidf,
    label_encoder,
    tokenizer,
    job_tfidf,
    jobs_df,
) = load_models()


# ==================================
# TEXT CLEANING
# ==================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==================================
# PREDICTIONS
# ==================================

def predict_category(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = rf_model.predict(vector)[0]
    return label_encoder.inverse_transform([prediction])[0]


def predict_category_gb(text):
    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])
    prediction = gb_model.predict(vector)[0]
    return label_encoder.inverse_transform([prediction])[0]


# ==================================
# ATS SCORE
# ==================================

def ats_score(resume_text, job_description):
    vectors = job_tfidf.transform(
        [
            clean_text(resume_text),
            clean_text(job_description),
        ]
    )
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return round(score * 100, 2)


# ==================================
# SKILL GAP
# ==================================

def skill_gap_analysis(resume_text, job_text):
    resume_words = set(clean_text(resume_text).split())
    job_words = set(clean_text(job_text).split())
    missing = job_words - resume_words
    return sorted(missing)[:15]


# ==================================
# CAREER GUIDANCE
# ==================================

def career_guidance(score):
    if score >= 80:
        return "Excellent Match"
    if score >= 60:
        return "Good Match"
    if score >= 40:
        return "Needs Skill Improvement"
    return "Strong Upskilling Recommended"


# ==================================
# JOB RECOMMENDATION
# ==================================

def recommend_jobs(resume_text, top_n=5):
    resume_vector = job_tfidf.transform([clean_text(resume_text)])
    jobs_vector = job_tfidf.transform(jobs_df["combined_text"].fillna(""))

    similarities = cosine_similarity(resume_vector, jobs_vector).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]

    return jobs_df.iloc[top_indices][["Job Title", "Role", "Company"]]


# ==================================
# HEADER
# ==================================

st.title("AI Resume Analyzer & Career Guidance System")

st.markdown(
    """
### Intelligent ATS Resume Screening System

This application uses Machine Learning and Deep Learning techniques to analyze resumes and recommend suitable careers.
"""
)


# ==================================
# RESUME INPUT
# ==================================

upload_types = ["txt"]
upload_label = "Upload Resume (.txt)"
if pdf_available and PdfReader is not None:
    upload_types.append("pdf")
    upload_label = "Upload Resume (.txt, .pdf)"

uploaded_file = st.file_uploader(upload_label, type=upload_types)

resume_text = ""

def extract_pdf_text(uploaded_file):
    if PdfReader is None:
        return ""
    try:
        file_bytes = uploaded_file.read()
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(
            page.extract_text() or "" for page in reader.pages
        )
        return text.strip()
    except Exception:
        return ""

if uploaded_file is not None:
    if uploaded_file.name.lower().endswith(".pdf"):
        resume_text = extract_pdf_text(uploaded_file)
        if resume_text:
            st.success("PDF Resume Uploaded Successfully")
        else:
            st.warning(
                "Could not extract text from the uploaded PDF. Please paste your resume text below."
            )
    else:
        resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.success("Resume Uploaded Successfully")

    if not resume_text:
        resume_text = st.text_area("Paste Resume Text", height=300)
else:
    resume_text = st.text_area("Paste Resume Text", height=300)


# ==================================
# ANALYZE
# ==================================

if st.button("Analyze Resume"):
    if not resume_text.strip():
        st.warning("Please upload or paste resume.")
    else:
        category = predict_category(resume_text)
        gb_category = predict_category_gb(resume_text)
        recommendations = recommend_jobs(resume_text)

        best_index = recommendations.index[0]
        best_job = jobs_df.loc[best_index, "combined_text"]

        score = ats_score(resume_text, best_job)
        missing_skills = skill_gap_analysis(resume_text, best_job)
        guidance = career_guidance(score)

        st.success("Analysis Completed Successfully")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("ATS Score", f"{score}%")

        with col2:
            st.metric("RF Prediction", category)

        with col3:
            st.metric("GB Prediction", gb_category)

        with col4:
            st.metric("Career Guidance", guidance)

        st.markdown("---")

        st.subheader("Loaded Deep Learning Models")
        if cnn_model is not None:
            st.success("CNN Model Loaded Successfully")
        else:
            st.warning("CNN Model unavailable or failed to load")

        if lstm_model is not None:
            st.success("LSTM Model Loaded Successfully")
        else:
            st.warning("LSTM Model unavailable or failed to load")

        tab1, tab2 = st.tabs(["Skill Gap Analysis", "Job Recommendations"])

        with tab1:
            st.write(missing_skills)

        with tab2:
            st.dataframe(recommendations, use_container_width=True)
