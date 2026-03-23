# database.py
# SQLite database for user registration and login

import sqlite3
import hashlib
import os

DB_PATH = "database.db"


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    UNIQUE NOT NULL,
            email      TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS user_results (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            resume_name  TEXT,
            jd_text      TEXT,
            ai_score     REAL,
            ats_score    REAL,
            final_score  REAL,
            ml_category  TEXT,
            skills_found TEXT,
            missing      TEXT,
            decision     TEXT,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()
    print("✅ Database initialized")


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, email, password):
    """Returns (True, None) on success or (False, error_message) on failure."""
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username.strip(), email.strip().lower(), hash_password(password))
        )
        conn.commit()
        conn.close()
        return True, None
    except sqlite3.IntegrityError as e:
        if "username" in str(e):
            return False, "Username already taken."
        if "email" in str(e):
            return False, "Email already registered."
        return False, "Registration failed."


def login_user(username, password):
    """Returns user row if valid, else None."""
    conn  = get_db()
    user  = conn.execute(
        "SELECT * FROM users WHERE username = ? AND password = ?",
        (username.strip(), hash_password(password))
    ).fetchone()
    conn.close()
    return user


def save_user_result(user_id, resume_name, jd_text, ai, ats, final, ml_cat, skills, missing, decision):
    conn = get_db()
    conn.execute("""
        INSERT INTO user_results
        (user_id, resume_name, jd_text, ai_score, ats_score, final_score,
         ml_category, skills_found, missing, decision)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (user_id, resume_name, jd_text[:500], ai, ats, final, ml_cat,
          ",".join(skills), ",".join(missing), decision))
    conn.commit()
    conn.close()


def get_user_history(user_id):
    conn    = get_db()
    results = conn.execute(
        "SELECT * FROM user_results WHERE user_id = ? ORDER BY created_at DESC",
        (user_id,)
    ).fetchall()
    conn.close()
    return results

# BUG FIX: Removed redundant module-level init_db() call.
# init_db() is explicitly called by app.py after imports, so calling
# it here caused double initialization and duplicate console output.
