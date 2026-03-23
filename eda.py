# eda.py
# Exploratory Data Analysis on the Resume Dataset
# Run: python eda.py
# Charts are saved to reports/eda/

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nltk
from collections import Counter

nltk.download('punkt',     quiet=True)
nltk.download('stopwords', quiet=True)

OUTPUT_DIR = os.path.join("reports", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style="whitegrid", palette="muted")


# -------------------------------
# LOAD DATA
# -------------------------------
def load_data():
    candidates = [
        os.path.join("dataset", "resume_dataset.csv"),
        os.path.join("dataset", "Resume.csv"),
        os.path.join("dataset", "resume.csv"),
        os.path.join("dataset", "UpdatedResumeDataSet.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path).dropna()
            # Normalize column names
            df.columns = [c.strip() for c in df.columns]
            for col in df.columns:
                if 'resume' in col.lower():
                    df = df.rename(columns={col: 'Resume_str'})
                if 'category' in col.lower():
                    df = df.rename(columns={col: 'Category'})
            df['word_count']  = df['Resume_str'].apply(lambda x: len(str(x).split()))
            df['text_length'] = df['Resume_str'].apply(len)
            print(f"✅ Loaded {len(df)} resumes from {path}")
            return df
    print("❌ Dataset not found. Please place CSV in dataset/ folder.")
    return None


# -------------------------------
# CHART 1: Category Distribution
# -------------------------------
def plot_category_distribution(df):
    fig, ax = plt.subplots(figsize=(12, 7))
    counts = df['Category'].value_counts()
    bars = sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="Blues_d")
    ax.set_title("Number of Resumes per Job Category", fontsize=15, pad=14)
    ax.set_xlabel("Resume Count")
    ax.set_ylabel("Job Category")
    for i, v in enumerate(counts.values):
        ax.text(v + 1, i, str(v), va='center', fontsize=10)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_category_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# -------------------------------
# CHART 2: Word Count Distribution
# -------------------------------
def plot_word_count(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['word_count'], bins=40, kde=True, ax=ax, color="#4facfe")
    ax.axvline(df['word_count'].mean(), color='red', linestyle='--', label=f"Mean: {df['word_count'].mean():.0f}")
    ax.set_title("Distribution of Resume Word Counts", fontsize=15, pad=14)
    ax.set_xlabel("Word Count")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "2_word_count_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# -------------------------------
# CHART 3: Top 20 Skills
# -------------------------------
def plot_top_skills(df):
    skills_db = [
        "python", "java", "sql", "machine learning", "deep learning",
        "nlp", "data science", "data analytics", "power bi", "excel",
        "tableau", "html", "css", "javascript", "react", "node",
        "flask", "django", "git", "docker", "aws", "azure", "cloud",
        "opencv", "pandas", "numpy", "tensorflow", "keras",
        "scikit", "generative ai", "c++", "mongodb", "php", "r",
        "hadoop", "spark", "linux", "rest api"
    ]

    all_text   = " ".join(df['Resume_str'].str.lower().tolist())
    skill_freq = {s: all_text.count(s) for s in skills_db if all_text.count(s) > 0}
    top20      = dict(sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:20])

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=list(top20.values()), y=list(top20.keys()), ax=ax, palette="Greens_d")
    ax.set_title("Top 20 Most Frequent Skills in Dataset", fontsize=15, pad=14)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Skill")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "3_top_skills.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# -------------------------------
# CHART 4: Avg Word Count by Category
# -------------------------------
def plot_avg_length_by_category(df):
    avg = df.groupby('Category')['word_count'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=avg.values, y=avg.index, ax=ax, palette="Purples_d")
    ax.set_title("Average Resume Word Count by Job Category", fontsize=15, pad=14)
    ax.set_xlabel("Average Word Count")
    ax.set_ylabel("Category")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "4_avg_length_by_category.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# -------------------------------
# CHART 5: Skill Co-occurrence Heatmap
# -------------------------------
def plot_skill_heatmap(df):
    top_skills = [
        "python", "java", "sql", "machine learning", "html",
        "css", "javascript", "react", "aws", "docker",
        "excel", "tableau", "flask", "tensorflow", "nlp"
    ]
    sample = df.sample(min(500, len(df)), random_state=42)
    matrix = pd.DataFrame({
        skill: sample['Resume_str'].str.lower().str.contains(skill).astype(int)
        for skill in top_skills
    })

    fig, ax = plt.subplots(figsize=(13, 9))
    sns.heatmap(
        matrix.corr(), annot=True, fmt=".2f", cmap="coolwarm",
        ax=ax, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8}
    )
    ax.set_title("Skill Co-occurrence Correlation Heatmap", fontsize=15, pad=14)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "5_skill_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {path}")


# -------------------------------
# SUMMARY STATS
# -------------------------------
def print_summary(df):
    print("\n📊 DATASET SUMMARY")
    print("-" * 45)
    print(f"  Total resumes     : {len(df)}")
    print(f"  Job categories    : {df['Category'].nunique()}")
    print(f"  Avg word count    : {df['word_count'].mean():.0f}")
    print(f"  Max word count    : {df['word_count'].max()}")
    print(f"  Min word count    : {df['word_count'].min()}")
    print(f"\n  Top 5 categories:")
    for cat, cnt in df['Category'].value_counts().head(5).items():
        print(f"    {cat:<30} {cnt} resumes")
    print("-" * 45)


# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("  Resume Dataset — Exploratory Data Analysis")
    print("=" * 50)

    df = load_data()
    if df is None:
        exit(1)

    print_summary(df)

    print("\n🎨 Generating charts...")
    plot_category_distribution(df)
    plot_word_count(df)
    plot_top_skills(df)
    plot_avg_length_by_category(df)
    plot_skill_heatmap(df)

    print(f"\n✅ All 5 charts saved to: {OUTPUT_DIR}/")