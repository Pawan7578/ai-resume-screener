# 🧠 AI Resume Screener & ATS Checker

> An intelligent resume screening system that ranks candidates against job descriptions using hybrid AI scoring (TF-IDF + BERT), ATS keyword matching, and trained ML classifiers — with separate portals for recruiters and job seekers.

<br>

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

---

## 📌 Overview

**AI Resume Screener** is a full-stack web application that helps:

- 🏢 **Recruiters** — upload multiple resumes, paste a job description, and instantly get AI-ranked candidates with skill gap analysis and downloadable PDF/Excel reports
- 👤 **Job Seekers** — upload their resume against any job description to get a personal ATS score, missing skills, and improvement suggestions

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🤖 **Hybrid AI Scoring** | Combines TF-IDF (40%) + BERT semantic similarity (60%) for AI score |
| 📋 **ATS Score** | Dynamic keyword match between resume and actual JD skills |
| 🏆 **Candidate Ranking** | Multi-resume upload with ranked leaderboard and score chart |
| 🧪 **ML Classification** | Trains 4 classifiers (Naive Bayes, Logistic Regression, SVM, KNN) to predict resume job category |
| 📊 **Model Report** | Accuracy, Precision, Recall, F1-Score comparison across all models |
| 📄 **PDF Reports** | Per-candidate downloadable PDF reports via ReportLab |
| 📥 **Excel Export** | Full results exported to `.xlsx` for offline review |
| 👤 **User Portal** | Separate login for job seekers with personal history tracking |
| 🔐 **Admin Panel** | Secure admin login for recruiter workflows |
| 🔴 **Live Training** | Real-time ML training progress via Server-Sent Events (SSE) |

---

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **Flask** — web framework & routing
- **SQLite** — user authentication & history (via `sqlite3`)
- **Scikit-learn** — ML training (NB, LR, SVM, KNN)
- **Sentence-Transformers** — BERT semantic similarity (`all-MiniLM-L6-v2`)
- **ReportLab** — PDF report generation
- **pypdf / pdfplumber / pdfminer** — multi-strategy PDF extraction
- **python-docx** — DOCX text extraction
- **pandas / openpyxl** — data processing & Excel export

### Frontend
- **HTML5 / CSS3** — custom design system in `base.html`
- **Chart.js** — score comparison bar charts
- **Vanilla JavaScript** — tab switching, SSE streaming, PDF download

---

## 📂 Project Structure

```
ai-resume-screener/
│
├── app.py                      # Main Flask app — all routes & session logic
├── matcher.py                  # Hybrid AI scoring (TF-IDF + BERT + ATS)
├── resume_parser.py            # PDF / DOCX text extraction & artifact cleanup
├── skill_extractor.py          # Skill & experience detection from resume text
├── model_trainer.py            # Train 4 ML classifiers with live progress callback
├── model_evaluator.py          # Load & return saved model results for templates
├── database.py                 # SQLite — user registration, login, result history
├── pdf_report.py               # Generate per-candidate PDF reports (ReportLab)
│
├── templates/
│   ├── base.html               # Shared layout, design system, global styles
│   ├── auth.html               # Admin login + User login + Register (tabbed)
│   ├── dashboard.html          # Admin dashboard — upload JD & resumes
│   ├── result.html             # Ranked candidate table + score chart
│   ├── train.html              # ML training page with live SSE progress
│   ├── model_report.html       # Model evaluation metrics & visual chart
│   ├── user_dashboard.html     # User portal — analyze resume + history
│   └── user_result.html        # User's personal analysis result page
│
├── dataset/
│   └── resume_dataset.csv      # Training data — place your CSV here
│
├── models/                     # Auto-generated after training
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── label_encoder.pkl
│   └── model_results.json
│
├── resumes/                    # Temp storage for admin-uploaded resumes
├── job_descriptions/           # Stores the active job description file
├── user_uploads/               # Temp storage for user-uploaded resumes
│
├── reports/
│   ├── final_report.xlsx       # Latest admin analysis export
│   └── candidates/             # Per-candidate PDF report files
│
├── .env                        # Environment variables (do NOT commit)
├── requirements.txt            # All Python dependencies
├── database.db                 # SQLite DB — auto-created on first run
└── README.md
```

---

## 📊 How It Works

```
User uploads Resume(s) + Job Description
           │
           ▼
┌─────────────────────────────────┐
│        resume_parser.py         │  ← Extracts text from PDF / DOCX
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│          matcher.py             │
│  ┌─────────────┐ ┌───────────┐  │
│  │  TF-IDF 40% │ │ BERT  60% │  │  ← AI Score (semantic similarity)
│  └─────────────┘ └───────────┘  │
│  ┌──────────────────────────┐   │
│  │   ATS Keyword Match      │   │  ← ATS Score (JD skill overlap)
│  └──────────────────────────┘   │
│         Final = AI×50% + ATS×50%│
└─────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────┐
│       ML Classifier             │  ← Predicts job category + confidence
│  (Best of NB / LR / SVM / KNN) │
└─────────────────────────────────┘
           │
           ▼
   Ranked Results + PDF + Excel
```

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Pawan7578/ai-resume-screener.git
cd ai-resume-screener
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
Create a `.env` file in the root directory:
```env
SECRET_KEY=your-secret-key-here
ADMIN_USERNAME=admin
ADMIN_PASSWORD=admin123
FLASK_DEBUG=False
```

### 4. Add training dataset *(optional — needed for ML features)*
Place your resume CSV in the `dataset/` folder:
```
dataset/resume_dataset.csv
```
The CSV should have columns: `Resume` (text) and `Category` (job label).

### 5. Run the app
```bash
python app.py
```

Open your browser at → **http://127.0.0.1:5000**

---

## 🔐 Access

| Portal | URL | Default Credentials |
|--------|-----|---------------------|
| Admin Panel | `http://127.0.0.1:5000/` | `admin` / `admin123` |
| User Portal | `http://127.0.0.1:5000/user/login` | Register a new account |

> ⚠️ Change default admin credentials in `.env` before deploying.

---

## 🤖 Training ML Models

1. Log in as Admin
2. Navigate to **Train** → click **Start Training**
3. Watch the live progress — all 4 models train and are evaluated
4. The best model (highest F1-score) is saved to `models/` and used automatically

---

## 📸 Screenshots

> *(Add screenshots here)*

---

## 🔥 Future Improvements

- [ ] Resume auto-conversion to ATS-friendly format
- [ ] AI-powered resume rewriting suggestions
- [ ] Multi-language resume support
- [ ] Job portal API integration (LinkedIn, Naukri)
- [ ] Mobile-responsive UI overhaul
- [ ] Email notifications for candidates

---

## 📄 License

This project is licensed for educational and portfolio use.

---

## 👨‍💻 Author

**Pawan Kumar N**
Aspiring Data Analyst & AI Developer

[![GitHub](https://img.shields.io/badge/GitHub-Pawan7578-181717?style=flat&logo=github)](https://github.com/Pawan7578)

---

<p align="center">Made with ❤️ as an MCA Capstone Project</p>
