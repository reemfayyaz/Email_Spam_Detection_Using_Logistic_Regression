import os
from pathlib import Path

import joblib
import streamlit as st

# ---------------------------------------------------------
# Application configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("spam_model.pkl")
VECTORIZER_PATH = Path("tfidf_vectorizer.pkl")

# ---------------------------------------------------------
# Professional UI styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 900px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.8rem 1.5rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 55%, #2563eb 100%);
            color: white;
            margin-bottom: 1.25rem;
            box-shadow: 0 12px 30px rgba(37, 99, 235, 0.18);
        }
        .hero h1 { margin: 0; font-size: 2.1rem; }
        .hero p { margin: .55rem 0 0; opacity: .9; }
        .info-card {
            border: 1px solid rgba(128,128,128,.22);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: .5rem 0 1rem;
        }
        .result-spam {
            padding: 1.1rem;
            border-radius: 16px;
            background: rgba(239, 68, 68, 0.10);
            border: 1px solid rgba(239, 68, 68, 0.35);
        }
        .result-ham {
            padding: 1.1rem;
            border-radius: 16px;
            background: rgba(34, 197, 94, 0.10);
            border: 1px solid rgba(34, 197, 94, 0.35);
        }
        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            font-weight: 700;
            min-height: 3rem;
        }
        .small-note { font-size: .88rem; opacity: .8; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load the trained Logistic Regression model and TF-IDF vectorizer."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


st.markdown(
    """
    <div class="hero">
        <h1>📧 Email Spam Detection</h1>
        <p>Logistic Regression + TF-IDF text classification for Spam vs Ham email screening.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("About the Model")
    st.write("**Algorithm:** Logistic Regression")
    st.write("**Text features:** TF-IDF")
    st.write("**Classes:** Ham (0), Spam (1)")
    st.write("**Training setup:** 80% train / 20% test")
    st.divider()
    st.caption("Model files expected in the same folder as app.py:")
    st.code("spam_model.pkl\ntfidf_vectorizer.pkl", language=None)

# Check required files before attempting to load them.
missing = [str(p) for p in (MODEL_PATH, VECTORIZER_PATH) if not p.exists()]
if missing:
    st.error("Required model file(s) not found: " + ", ".join(missing))
    st.info(
        "Run the training notebook first and place `spam_model.pkl` and "
        "`tfidf_vectorizer.pkl` in the same directory as `app.py`."
    )
    st.stop()

try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error("The saved model files could not be loaded.")
    st.exception(exc)
    st.stop()

st.subheader("Check an Email")
st.markdown(
    '<div class="info-card">Enter the email subject and message body. '
    'The app combines both fields, applies the saved TF-IDF vectorizer, '
    'and uses the trained Logistic Regression model to predict Spam or Ham.</div>',
    unsafe_allow_html=True,
)

subject = st.text_input(
    "Email subject",
    placeholder="Example: Urgent Action Required",
)
email_body = st.text_area(
    "Email body",
    height=210,
    placeholder="Paste the email message here...",
)

col1, col2 = st.columns([2, 1])
with col1:
    analyze = st.button("🔎 Analyze Email", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.rerun()

if analyze:
    combined_text = f"{subject.strip()} {email_body.strip()}".strip()

    if not combined_text:
        st.warning("Please enter an email subject or email body before analyzing.")
    else:
        vector = vectorizer.transform([combined_text])
        prediction = int(model.predict(vector)[0])

        ham_probability = None
        spam_probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(vector)[0]
            classes = list(model.classes_)
            if 0 in classes:
                ham_probability = float(probabilities[classes.index(0)]) * 100
            if 1 in classes:
                spam_probability = float(probabilities[classes.index(1)]) * 100

        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown(
                '<div class="result-spam"><h3>🚨 SPAM EMAIL</h3>'
                '<p>This message was classified as spam.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-ham"><h3>✅ SAFE EMAIL (HAM)</h3>'
                '<p>This message was classified as a legitimate email.</p></div>',
                unsafe_allow_html=True,
            )

        if ham_probability is not None and spam_probability is not None:
            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Ham confidence", f"{ham_probability:.2f}%")
            m2.metric("Spam confidence", f"{spam_probability:.2f}%")

            st.progress(min(max(spam_probability / 100, 0.0), 1.0), text="Spam probability")

        with st.expander("How this prediction works"):
            st.write(
                "The saved TF-IDF vectorizer transforms the entered text into numerical "
                "features. The trained Logistic Regression classifier then predicts class "
                "0 (Ham) or class 1 (Spam)."
            )

st.divider()
st.caption(
    "Educational machine-learning project. Predictions depend on the training data and "
    "should not be treated as a complete email-security solution."
)
