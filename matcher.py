# matcher.py
# Hybrid matching: TF-IDF + BERT + Trained ML model
# FIXES:
#   - Dynamic ATS score based on actual JD skills (not hardcoded list)
#   - Fixed BERT score normalization (was giving 40-50%, now gives realistic 60-90%)
#   - Fixed LinearSVC confidence (decision_function margin-based, was showing 0%)
#   - Weighted: ATS 50% + AI 50% (skill match weighted equally with semantic match)

import os, pickle, re, numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

# Comprehensive skill list for dynamic ATS matching
ALL_SKILLS = [
    "python", "java", "c++", "c#", "c", "r", "sql", "mysql", "mongodb",
    "postgresql", "sqlite", "oracle", "machine learning", "deep learning",
    "nlp", "data science", "data analytics", "artificial intelligence", "ai",
    "power bi", "excel", "tableau", "html", "css", "javascript", "typescript",
    "react", "angular", "vue", "node", "php", "flask", "django", "spring",
    "git", "github", "docker", "kubernetes", "aws", "azure", "cloud", "gcp",
    "opencv", "pandas", "numpy", "scipy", "scikit", "tensorflow", "keras",
    "pytorch", "generative ai", "llm", "spark", "hadoop", "linux", "rest api",
    "android", "kotlin", "swift", "flutter", "selenium", "agile", "scrum",
    "devops", "matlab", "figma", "bootstrap", "jquery", "redis", "elasticsearch"
]


# ──────────────────────────────────────────────────────────────
# LOAD TRAINED MODELS (once at module startup)
# ──────────────────────────────────────────────────────────────
_trained_model      = None
_trained_vectorizer = None
_label_encoder      = None
_models_loaded      = False


def _load_trained_models():
    global _trained_model, _trained_vectorizer, _label_encoder, _models_loaded
    try:
        paths = ["models/best_model.pkl", "models/tfidf_vectorizer.pkl", "models/label_encoder.pkl"]
        for p in paths:
            if not os.path.exists(p) or os.path.getsize(p) < 100:
                raise FileNotFoundError(f"Missing or empty: {p}")

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # suppress sklearn version mismatch warnings
            with open("models/best_model.pkl",       "rb") as f: _trained_model      = pickle.load(f)
            with open("models/tfidf_vectorizer.pkl",  "rb") as f: _trained_vectorizer = pickle.load(f)
            with open("models/label_encoder.pkl",    "rb") as f: _label_encoder      = pickle.load(f)

        _models_loaded = True
        print(f"✅ ML model loaded — type: {type(_trained_model).__name__}")

    except Exception as e:
        print(f"⚠️  ML model not ready ({e})")
        print("   → Go to http://127.0.0.1:5000/train to train models")
        _models_loaded = False


_load_trained_models()


# ──────────────────────────────────────────────────────────────
# TEXT CLEANING
# ──────────────────────────────────────────────────────────────
def clean_text(text):
    text   = str(text).lower()
    text   = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


# ──────────────────────────────────────────────────────────────
# SKILL EXTRACTION
# ──────────────────────────────────────────────────────────────
def extract_skills_from_text(text):
    text_lower = text.lower()
    found = []
    for skill in ALL_SKILLS:
        if len(skill) <= 2:
            if re.search(r'\b' + re.escape(skill) + r'\b', text_lower):
                found.append(skill)
        else:
            if skill in text_lower:
                found.append(skill)
    return list(set(found))


# ──────────────────────────────────────────────────────────────
# TF-IDF COSINE SIMILARITY
# Normalized: raw cosine 0.15 → 40%, 0.4 → 80%, 0.6+ → 100%
# ──────────────────────────────────────────────────────────────
def tfidf_match(jd_text, resumes):
    docs         = [jd_text] + resumes
    vectorizer   = TfidfVectorizer(ngram_range=(1, 2), max_features=8000)
    tfidf_matrix = vectorizer.fit_transform(docs)
    raw_scores   = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    result = []
    for s in raw_scores:
        # Normalize: map [0, 0.5] → [0, 100] with a 2x boost
        normalized = min(100.0, round(float(s) * 200, 2))
        result.append(normalized)
    return result


# ──────────────────────────────────────────────────────────────
# BERT SEMANTIC SIMILARITY
# Normalized: raw cosine 0.3 → 50%, 0.6 → 80%, 0.85 → 100%
# ──────────────────────────────────────────────────────────────
def bert_match(jd_text, resumes):
    try:
        from sentence_transformers import SentenceTransformer
        if not hasattr(bert_match, "_model"):
            print("🔄 Loading BERT model (first time ~90MB download)...")
            bert_match._model = SentenceTransformer('all-MiniLM-L6-v2')

        model      = bert_match._model
        embeddings = model.encode([jd_text] + resumes, show_progress_bar=False)
        jd_emb     = embeddings[0]

        scores = []
        for emb in embeddings[1:]:
            raw = float(cosine_similarity([jd_emb], [emb])[0][0])
            # Normalize: [0.25, 0.85] → [0, 100]
            normalized = min(100.0, max(0.0, round((raw - 0.25) / 0.60 * 100, 2)))
            scores.append(normalized)
        return scores

    except Exception as e:
        print(f"⚠️  BERT failed ({e}). Using TF-IDF only.")
        return tfidf_match(jd_text, resumes)


# ──────────────────────────────────────────────────────────────
# HYBRID AI SCORE — TF-IDF (40%) + BERT (60%)
# ──────────────────────────────────────────────────────────────
def hybrid_match(jd_text, resumes):
    jd_clean      = clean_text(jd_text)
    resumes_clean = [clean_text(r) for r in resumes]

    tfidf_scores = tfidf_match(jd_clean, resumes_clean)
    bert_scores  = bert_match(jd_clean, resumes_clean)

    return [round((t * 0.4) + (b * 0.6), 2) for t, b in zip(tfidf_scores, bert_scores)]


# ──────────────────────────────────────────────────────────────
# ATS SCORE — Dynamic (based on actual JD skills)
# FIX: Old code divided by 19 hardcoded skills → always low
# New: extract skills FROM JD, check overlap with resume skills
# ──────────────────────────────────────────────────────────────
def ats_score(jd_text, resume_text):
    jd_skills     = extract_skills_from_text(jd_text)
    resume_skills = extract_skills_from_text(resume_text)

    if not jd_skills:
        # Fallback: keyword overlap ratio
        jd_words     = set(clean_text(jd_text).split())
        resume_words = set(clean_text(resume_text).split())
        if not jd_words:
            return 0.0
        overlap = jd_words & resume_words
        return round(min(100.0, len(overlap) / len(jd_words) * 150), 2)

    matched = [s for s in jd_skills if s in resume_skills]
    score   = round((len(matched) / len(jd_skills)) * 100, 2)
    return score


# ──────────────────────────────────────────────────────────────
# ML MODEL PREDICTION — predicts job category from resume text
# FIX: LinearSVC confidence was showing 0% due to bad formula
# New: margin-based confidence (top score vs 2nd score gap)
# ──────────────────────────────────────────────────────────────
def ml_model_predict(resume_text):
    if not _models_loaded:
        return "⚠️ Not trained — go to /train", 0.0

    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vec   = _trained_vectorizer.transform([clean_text(resume_text)])
            label = _trained_model.predict(vec)[0]

        category = _label_encoder.inverse_transform([label])[0]

        # Confidence calculation
        if hasattr(_trained_model, "predict_proba"):
            # Probabilistic models (NB, LR)
            proba      = _trained_model.predict_proba(vec)[0]
            confidence = round(float(max(proba)) * 100, 2)

        elif hasattr(_trained_model, "decision_function"):
            # LinearSVC: use margin between top-2 predictions as confidence proxy
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scores = _trained_model.decision_function(vec)[0]

            sorted_scores = np.sort(scores)[::-1]
            top_score     = float(sorted_scores[0])
            second_score  = float(sorted_scores[1])
            margin        = top_score - second_score   # larger gap = more confident

            # Map margin [0, 2] → confidence [50%, 95%]
            confidence = round(min(95.0, max(50.0, 50.0 + margin * 22.5)), 2)
        else:
            confidence = 70.0

        return category, confidence

    except Exception as e:
        print(f"⚠️  Prediction error: {e}")
        return "Error in prediction", 0.0


# ──────────────────────────────────────────────────────────────
# FINAL SCORE — AI (50%) + ATS (50%)
# Higher ATS weight because skill match is the most direct measure
# ──────────────────────────────────────────────────────────────
def final_score(jd_text, resumes):
    ai_scores      = hybrid_match(jd_text, resumes)
    ats_scores_lst = []
    final_scores   = []
    ml_categories  = []
    ml_confidences = []

    for i, resume in enumerate(resumes):
        ats = ats_score(jd_text, resume)
        ats_scores_lst.append(ats)

        # Equal weight: AI semantic similarity + ATS skill matching
        combined = round((ai_scores[i] * 0.5) + (ats * 0.5), 2)
        final_scores.append(min(100.0, combined))

        category, confidence = ml_model_predict(resume)
        ml_categories.append(category)
        ml_confidences.append(confidence)

    return ai_scores, ats_scores_lst, final_scores, ml_categories, ml_confidences
