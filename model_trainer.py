# model_trainer.py
# Train 4 ML classifiers fresh on the current machine's sklearn version
# Fixes: sklearn version mismatch that caused prediction failures

import pandas as pd
import os, json, pickle, re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import sklearn

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))


def clean_text(text):
    text   = str(text).lower()
    text   = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 2]
    return " ".join(tokens)


def load_dataset():
    for path in [
        "dataset/resume_dataset.csv",
        "dataset/Resume.csv",
        "dataset/resume.csv",
        "dataset/UpdatedResumeDataSet.csv",
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path).dropna()
            df.columns = [c.strip() for c in df.columns]
            text_col = cat_col = None
            for col in df.columns:
                if 'resume' in col.lower(): text_col = col
                if 'category' in col.lower(): cat_col  = col
            if not text_col: text_col = df.columns[0]
            if not cat_col:  cat_col  = df.columns[1]
            df = df[[text_col, cat_col]].dropna()
            df.columns = ['text', 'category']
            return df['text'], df['category'], None

    return None, None, "Dataset CSV not found in dataset/ folder"


def train_and_save(progress_callback=None):

    def log(msg):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    log(f"info:Training on scikit-learn v{sklearn.__version__}")

    log("step:Loading dataset...")
    X, y, err = load_dataset()
    if err:
        log(f"error:{err}")
        return False

    log(f"info:Loaded {len(X)} resumes | {y.nunique()} categories")

    log("step:Cleaning text (tokenizing + removing stopwords)...")
    X_clean = X.apply(clean_text)

    log("step:Encoding labels...")
    le        = LabelEncoder()
    y_encoded = le.fit_transform(y)

    log("step:Vectorizing with TF-IDF (3000 features)...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_tfidf    = vectorizer.fit_transform(X_clean)

    log("step:Splitting train / test sets (80 / 20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y_encoded, test_size=0.2, random_state=42
    )

    models = {
        "Naive Bayes":         MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=500, C=5, random_state=42),
        "SVM":                 LinearSVC(max_iter=1000, C=1.0, random_state=42),
        "KNN":                 KNeighborsClassifier(n_neighbors=3, metric='euclidean', algorithm='brute'),
    }

    results    = {}
    best_model = best_name = None
    best_f1    = 0

    for name, model in models.items():
        log(f"training:{name}")
        model.fit(X_train, y_train)

        log(f"evaluating:{name}")
        y_pred = model.predict(X_test)

        acc  = round(accuracy_score(y_test, y_pred) * 100, 2)
        prec = round(precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        rec  = round(recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)
        f1   = round(f1_score(y_test, y_pred, average='weighted', zero_division=0) * 100, 2)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        log(f"result:{name}|{acc}|{prec}|{rec}|{f1}")

        if f1 > best_f1:
            best_f1, best_model, best_name = f1, model, name

    log(f"step:Saving best model ({best_name})...")
    os.makedirs("models", exist_ok=True)

    with open("models/best_model.pkl",       "wb") as f: pickle.dump(best_model,  f)
    with open("models/tfidf_vectorizer.pkl",  "wb") as f: pickle.dump(vectorizer,  f)
    with open("models/label_encoder.pkl",    "wb") as f: pickle.dump(le,           f)
    with open("models/model_results.json",   "w")  as f:
        json.dump({
            "best_model":    best_name,
            "sklearn_version": sklearn.__version__,
            "results":       results
        }, f, indent=2)

    log(f"done:{best_name}|{best_f1}")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print(f"  Resume Screening — ML Trainer (sklearn {sklearn.__version__})")
    print("=" * 60)
    train_and_save()
    print("\n✅ Done! Run: python app.py")
