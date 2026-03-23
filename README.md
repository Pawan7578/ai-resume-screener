# Resume Screening & Job Matching System Using Machine Learning

**Student:** GOLI NAVEEN KUMAR  
**Register No:** 2411022250045  
**Course:** MCA (Generative AI) – 4th Semester  
**Mentor:** S. Lakshmi Devi  

---

## Problem Statement

Recruiters receive a large number of resumes for each job opening, making manual screening slow and inconsistent. This system uses Machine Learning to automatically analyze resumes, match them with job requirements, and rank candidates — enabling faster, fair, and data-driven evaluation.

---

## Features

- Upload multiple resumes (PDF / DOCX) and a job description (TXT)
- Hybrid matching: TF-IDF cosine similarity + BERT semantic matching
- 4 trained ML classifiers: Naive Bayes, Logistic Regression, SVM, KNN
- Suitability score per candidate (AI Score + ATS Score)
- Skill gap identification and improvement suggestions
- Candidate ranking with hiring decision grading
- Model evaluation report (accuracy, precision, recall, F1)
- Excel report export
- Secure login with session management

---

## Project Structure

```
Matching resumes to job description/
├── app.py                  # Main Flask application
├── matcher.py              # TF-IDF + BERT + ML matching logic
├── resume_parser.py        # PDF and DOCX text extraction
├── skill_extractor.py      # Skill and experience detection
├── model_trainer.py        # Train 4 ML classifiers on dataset
├── model_evaluator.py      # Load and compare model results
├── eda.py                  # Exploratory Data Analysis charts
├── .env                    # Secret config (not committed to git)
├── requirements.txt        # Python dependencies
├── README.md
├── dataset/
│   └── resume_dataset.csv  # Kaggle resume dataset
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── model_results.json
├── templates/
│   ├── login.html
│   ├── dashboard.html
│   ├── result.html
│   └── model_report.html
├── resumes/                # Uploaded resumes (runtime)
├── job_descriptions/       # Uploaded JD (runtime)
└── reports/
    ├── final_report.xlsx
    └── eda/                # EDA chart images
```

---

## Setup & Installation

### 1. Clone / Download the project
```bash
cd "Matching resumes to job description"
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NLTK data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Download Dataset
- Go to: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
- Download `resume_dataset.csv`
- Place it in the `dataset/` folder

### 5. Configure credentials
Edit the `.env` file:
```
SECRET_KEY=your-random-secret-key
ADMIN_USERNAME=admin
ADMIN_PASSWORD=your_password
```

### 6. Train the ML models (run once)
```bash
python model_trainer.py
```
This trains Naive Bayes, Logistic Regression, SVM, and KNN, compares them, and saves the best model.

### 7. (Optional) Run EDA
```bash
python eda.py
```
Generates analysis charts in `reports/eda/`.

### 8. Start the application
```bash
python app.py
```
Open browser: http://127.0.0.1:5000

---

## How to Use

1. Login at http://127.0.0.1:5000 with your credentials
2. Upload a job description (.txt) and multiple resumes (.pdf or .docx)
3. Click **Analyze Resumes**
4. View ranked candidates with scores, skills, and recommendations
5. Visit **/model-report** to see the ML model comparison
6. Download the Excel report

---

## Methodology

1. **Data Collection** — Kaggle resume dataset (2400+ resumes, 25 categories)
2. **Text Preprocessing** — Tokenization, stop-word removal, stemming (NLTK)
3. **EDA** — Skill frequency, category distribution, word count analysis
4. **Feature Extraction** — TF-IDF vectorization (5000 features)
5. **Model Training** — Naive Bayes, Logistic Regression, SVM, KNN
6. **Model Evaluation** — Accuracy, Precision, Recall, F1-Score comparison
7. **Hybrid Matching** — Best ML model + BERT semantic similarity + ATS scoring
8. **Results** — Suitability scores, skill gaps, candidate ranking, Excel export

---

## Technologies Used

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Web Framework | Flask |
| ML Models | Naive Bayes, Logistic Regression, SVM, KNN |
| NLP | NLTK, Sentence-Transformers (BERT) |
| Features | TF-IDF, Cosine Similarity |
| Visualization | Matplotlib, Seaborn, Chart.js |
| Data | Pandas, NumPy |
| Resume Parsing | PyPDF2, python-docx |

---

## Expected Outcomes

- Automated resume screening and ranking
- Accurate job-resume matching using trained ML models
- Skill-gap identification for candidates
- Model performance comparison dashboard
- Visual analytics for recruiters

---

## License

Academic project — MCA, 4th Semester, 2024–25