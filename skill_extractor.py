# skill_extractor.py
# Extract skills and experience from resume text
# Handles compressed PDF artifacts (merged words, alternate spellings)

import re

# Each entry: (canonical_name, [patterns_to_match])
# Multiple patterns handle compressed PDF variants
SKILLS_WITH_VARIANTS = [
    ("python",           ["python"]),
    ("java",             ["java"]),
    ("c++",              ["c++", "c \+\+", "c\+\+"]),
    ("c#",               ["c#", "c sharp"]),
    ("c",                [r"\bc\b"]),
    ("r",                [r"\br\b"]),
    ("sql",              ["sql"]),
    ("mysql",            ["mysql", "my sql", "mys ql", "mysq l"]),
    ("mongodb",          ["mongodb", "mongo db", "mongod b"]),
    ("postgresql",       ["postgresql", "postgres"]),
    ("sqlite",           ["sqlite"]),
    ("machine learning", ["machine learning", "machinelearning", "ml\b"]),
    ("deep learning",    ["deep learning", "deeplearning"]),
    ("nlp",              ["nlp", "natural language"]),
    ("data science",     ["data science", "datascience"]),
    ("data analytics",   ["data analytics", "dataanalytics"]),
    ("artificial intelligence", ["artificial intelligence"]),
    ("ai",               [r"\bai\b"]),
    ("power bi",         ["power bi", "powerbi"]),
    ("excel",            ["excel"]),
    ("tableau",          ["tableau"]),
    ("html",             ["html"]),
    ("css",              ["css"]),
    ("javascript",       ["javascript", "java script", "javas cript"]),
    ("typescript",       ["typescript"]),
    ("react",            ["react"]),
    ("angular",          ["angular"]),
    ("vue",              ["vue"]),
    ("node",             ["node.js", "nodejs", "node js"]),
    ("php",              ["php"]),
    ("flask",            ["flask"]),
    ("django",           ["django"]),
    ("spring",           ["spring"]),
    ("git",              [r"\bgit\b"]),
    ("github",           ["github"]),
    ("docker",           ["docker"]),
    ("kubernetes",       ["kubernetes", "k8s"]),
    ("aws",              [r"\baws\b"]),
    ("azure",            ["azure"]),
    ("cloud",            ["cloud"]),
    ("gcp",              [r"\bgcp\b"]),
    ("opencv",           ["opencv", "open cv"]),
    ("pandas",           ["pandas"]),
    ("numpy",            ["numpy"]),
    ("scipy",            ["scipy"]),
    ("scikit",           ["scikit", "sklearn"]),
    ("tensorflow",       ["tensorflow"]),
    ("keras",            ["keras"]),
    ("pytorch",          ["pytorch"]),
    ("generative ai",    ["generative ai"]),
    ("llm",              [r"\bllm\b"]),
    ("spark",            ["spark"]),
    ("hadoop",           ["hadoop"]),
    ("linux",            ["linux"]),
    ("rest api",         ["rest api", "restapi"]),
    ("android",          ["android"]),
    ("kotlin",           ["kotlin"]),
    ("swift",            ["swift"]),
    ("flutter",          ["flutter"]),
    ("selenium",         ["selenium"]),
    ("agile",            ["agile"]),
    ("scrum",            ["scrum"]),
    ("devops",           ["devops"]),
    ("matlab",           ["matlab"]),
    ("figma",            ["figma"]),
    ("bootstrap",        ["bootstrap"]),
    ("jquery",           ["jquery"]),
    ("redis",            ["redis"]),
    ("xampp",            ["xampp", "wamp"]),
]


def extract_skills(text):
    text_lower = str(text).lower()
    found      = []

    for canonical, patterns in SKILLS_WITH_VARIANTS:
        for pattern in patterns:
            try:
                if re.search(pattern, text_lower):
                    found.append(canonical)
                    break   # match found, no need to check more patterns
            except re.error:
                if pattern in text_lower:
                    found.append(canonical)
                    break

    return list(set(found))


def extract_experience(text):
    text     = str(text).lower()
    patterns = [
        r'(\d+)\+?\s+years?\s+of\s+experience',
        r'(\d+)\+?\s+years?\s+experience',
        r'experience\s+of\s+(\d+)\+?\s+years?',
        r'(\d+)\+?\s+yrs?[\s\.]',
    ]
    all_years = []
    for pattern in patterns:
        for m in re.findall(pattern, text):
            try:
                all_years.append(int(m))
            except:
                pass

    if all_years:
        years = max(all_years)
        if years == 0: return "Fresher"
        if years == 1: return "1 year"
        return f"{years} years"
    return "Fresher"
