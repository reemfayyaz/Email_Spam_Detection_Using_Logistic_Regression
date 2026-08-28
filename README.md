# Email Spam Detection - Streamlit App

A machine-learning web application that classifies email text as **Spam** or **Ham (safe email)** using a trained **Logistic Regression** model and **TF-IDF** text features.

## Project Summary

The supplied notebook trains the model on `us_email_dataset_10000_rows.csv`. It combines the `subject` and `email_body` fields into one text feature, converts the text to TF-IDF vectors with up to 5,000 features, and trains a Logistic Regression binary classifier.

The notebook maps:

- `Ham` -> `0`
- `Spam` -> `1`

The recorded notebook evaluation uses an 80/20 train-test split and reports **100.00% accuracy on 2,000 test samples**.

## Required Project Files

Place these files together in one folder:

```text
app.py
spam_model.pkl
tfidf_vectorizer.pkl
requirements.txt
README.md
```

The two `.pkl` files are produced by the training notebook with:

```python
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
```

## Run Locally

1. Open a terminal in the project folder.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start Streamlit:

```bash
streamlit run app.py
```

4. Open the local Streamlit URL shown in the terminal.

## App Features

- Professional Streamlit interface
- Separate email subject and body inputs
- Spam/Ham classification
- Ham and Spam confidence percentages when supported by the model
- Probability progress indicator
- Clear error message when model files are missing
- Cached model loading for improved app performance
- Responsive sidebar with model details

## Model Pipeline

```text
Email Subject + Email Body
          |
          v
     Combined Text
          |
          v
 TF-IDF Vectorizer
(max_features = 5000,
 stop_words = "english")
          |
          v
 Logistic Regression
(C = 5, solver = "liblinear",
 max_iter = 3000)
          |
          v
     Spam / Ham
```

## Notebook Evaluation

The supplied notebook records:

- Dataset rows: 10,000
- Dataset columns: 8
- Ham samples: 5,000
- Spam samples: 5,000
- Test samples: 2,000
- Accuracy: 100.00%
- Confusion matrix: `[[993, 0], [0, 1007]]`

A perfect test score should be validated carefully on new, real-world email data before treating the model as production-ready.

## Deployment on Streamlit Community Cloud

1. Upload `app.py`, `requirements.txt`, `spam_model.pkl`, and `tfidf_vectorizer.pkl` to a GitHub repository.
2. Sign in to Streamlit Community Cloud.
3. Select the repository and choose `app.py` as the entry point.
4. Deploy the application.

## Important Note

The app requires the exact saved model and TF-IDF vectorizer produced by the notebook. If either file is missing or was trained with an incompatible scikit-learn version, loading may fail.

## Author

Reem Fayyaz
