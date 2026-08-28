import os
from pathlib import Path
from typing import Optional, Tuple, List

import joblib
import numpy as np
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
# Professional UI styling (kept + small visual upgrades)
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1000px;
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
        .hero p { margin: .55rem 0 0; opacity: .95; }
        .info-card {
            border: 1px solid rgba(128,128,128,.12);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: .5rem 0 1rem;
            background: rgba(250,250,250,0.02);
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
        .feature-tag { display:inline-block; margin:3px 6px; padding:6px 10px; border-radius:999px; background:#111827; color:#fff; font-weight:600; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts() -> Tuple[object, object]:
    """Load the trained Logistic Regression model and TF-IDF vectorizer."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def top_contributing_terms(vector, model, vectorizer, top_n: int = 8) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Return top positive (spam) and top negative (ham) contributing terms for a single sample.

    This computes the contribution of each feature as coef * tfidf_value and returns the
    top contributors for positive and negative signs.
    """
    # Feature names
    try:
        feature_names = vectorizer.get_feature_names_out()
    except Exception:
        # Fallback for older vectorizers
        feature_names = vectorizer.get_feature_names()

    coefs = None
    # For binary logistic regression in sklearn, coef_ shape is (1, n_features)
    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
    else:
        coefs = np.zeros(len(feature_names))

    arr = vector.toarray()[0]
    contributions = coefs * arr

    # Find non-zero contributions
    nz_idx = np.where(arr != 0)[0]
    if nz_idx.size == 0:
        return [], []

    contribs = [(int(i), float(contributions[i])) for i in nz_idx]
    contribs_sorted = sorted(contribs, key=lambda x: x[1], reverse=True)

    top_positive = [(feature_names[i], val) for i, val in contribs_sorted if val > 0][:top_n]
    top_negative = [(feature_names[i], val) for i, val in sorted(contribs, key=lambda x: x[1]) if val < 0][:top_n]
    return top_positive, top_negative


# Top hero area
st.markdown(
    """
    <div class="hero">
        <h1>📧 Email Spam Detection — Interactive</h1>
        <p>Logistic Regression + TF-IDF with interactive explanation, samples, and quick-file analysis.</p>
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

    st.markdown("---")
    st.subheader("Quick actions")
    st.write("Use examples to try different types of messages or upload a .txt email file.")
    st.caption("Made interactive with feature highlights and adjustable sensitivity.")

# Check required files before attempting to load them.
missing = [str(p) for p in (MODEL_PATH, VECTORIZER_PATH) if not p.exists()]
if missing:
    st.error("Required model file(s) not found: " + ", ".join(missing))
    st.info(
        "Run the training notebook first and place `spam_model.pkl` and "
        "`tfidf_vectorizer.pkl` in the same directory as `app.py`.")
    st.stop()

try:
    model, vectorizer = load_artifacts()
except Exception as exc:
    st.error("The saved model files could not be loaded.")
    st.exception(exc)
    st.stop()

# Sample messages
EXAMPLES = {
    "Friendly update": (
        "Team meeting update",
        "Hi team, just a reminder about our standup tomorrow at 10am. Please share updates before the meeting."
    ),
    "Phishing style": (
        "Urgent: Account Verification Required",
        "Dear user, we detected unusual activity. Click the link immediately to verify your account and avoid suspension."
    ),
    "Marketing promo": (
        "Huge Savings — Limited Time Offer!",
        "Congratulations! You have been selected for an exclusive discount. Buy now and save 70% on top brands."
    ),
}

# Input area with samples and file upload
st.subheader("Check an Email")
st.markdown(
    '<div class="info-card">Enter the email subject and message body, or pick a sample. You can also upload a plain text file.</div>',
    unsafe_allow_html=True,
)

colA, colB = st.columns([3, 1])
with colB:
    sample_choice = st.selectbox("Try a sample", ["(none)"] + list(EXAMPLES.keys()))
    uploaded_file = st.file_uploader("Upload .txt email (optional)", type=["txt"], label_visibility="visible")

with colA:
    subject = st.text_input(
        "Email subject",
        placeholder="Example: Urgent Action Required",
    )
    email_body = st.text_area(
        "Email body",
        height=210,
        placeholder="Paste the email message here...",
    )

if sample_choice and sample_choice != "(none)":
    s, b = EXAMPLES[sample_choice]
    if not subject and not email_body:
        subject = s
        email_body = b

if uploaded_file is not None:
    try:
        raw = uploaded_file.read().decode("utf-8", errors="ignore")
        # Heuristic split: first line as subject if short
        lines = [l for l in raw.splitlines() if l.strip()]
        if lines:
            if not subject:
                subject = lines[0][:200]
            if not email_body:
                email_body = "\n".join(lines[1:]) if len(lines) > 1 else ""
    except Exception:
        st.warning("Could not read uploaded file. Make sure it is a UTF-8 encoded text file.")

col1, col2 = st.columns([2, 1])
with col1:
    analyze = st.button("🔎 Analyze Email", type="primary", use_container_width=True)
with col2:
    clear = st.button("Clear", use_container_width=True)

if clear:
    st.experimental_rerun()

# Spam threshold slider (special interactive control)
threshold = st.slider("Spam threshold (flag as spam if model probability >=)", 0, 100, 50, step=1)

if analyze:
    combined_text = f"{(subject or '').strip()} {(email_body or '').strip()}".strip()

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

        # Interpret using user threshold if probabilities present
        flagged_spam = None
        if spam_probability is not None:
            flagged_spam = spam_probability >= threshold

        if flagged_spam is True:
            st.markdown(
                '<div class="result-spam"><h3>🚨 FLAGGED AS SPAM</h3>'
                '<p>This message is likely spam based on the configured threshold.</p></div>',
                unsafe_allow_html=True,
            )
        elif flagged_spam is False:
            st.markdown(
                '<div class="result-ham"><h3>✅ Not flagged (likely Ham)</h3>'
                '<p>This message appears legitimate.</p></div>',
                unsafe_allow_html=True,
            )
        else:
            # fallback to raw prediction if no probabilities
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

        # Animated celebration for ham and warning for spam
        if spam_probability is not None:
            st.write("")
            m1, m2 = st.columns(2)
            m1.metric("Ham confidence", f"{(ham_probability or 0):.2f}%")
            m2.metric("Spam confidence", f"{(spam_probability or 0):.2f}%")

            st.progress(min(max((spam_probability or 0) / 100.0, 0.0), 1.0), text="Spam probability")

            if spam_probability < 20:
                st.success("Low spam probability — looks good!")
                st.balloons()
            elif spam_probability > 80:
                st.warning("High spam probability — exercise caution.")

        # Show top contributing terms
        with st.expander("Show why the model made this prediction (top contributing words)"):
            pos_terms, neg_terms = top_contributing_terms(vector, model, vectorizer, top_n=10)
            if not pos_terms and not neg_terms:
                st.write("No high-impact terms found in this message (the text may be out-of-vocabulary or too short).")
            else:
                if pos_terms:
                    st.write("Top words pushing toward SPAM:")
                    for term, val in pos_terms:
                        st.markdown(f"<span class=\"feature-tag\">{term} ({val:.3f})</span>", unsafe_allow_html=True)
                if neg_terms:
                    st.write("Top words pushing toward HAM:")
                    for term, val in neg_terms:
                        st.markdown(f"<span class=\"feature-tag\">{term} ({val:.3f})</span>", unsafe_allow_html=True)

        # Show the combined text and allow easy copying
        with st.expander("Show combined email text (subject + body)"):
            st.code(combined_text)
            st.caption("Copy the text above to reproduce this input.")

        # Small tips and notes
        with st.expander("Tips to try"):
            st.write(
                "- Try short phishing-style subjects like 'Urgent' or 'Action Required' to see how the model reacts.\n"
                "- Use the threshold slider to tune sensitivity for your use case.\n"
                "- The top contributing words are a heuristic explanation based on model coefficients and TF-IDF values."
            )

st.divider()
st.caption(
    "Educational machine-learning project. Predictions depend on the training data and "
    "should not be treated as a complete email-security solution."
)
