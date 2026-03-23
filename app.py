# app.py
# AI Resume Matcher — Full Application
# Features: Admin module + User module + JD text input + PDF reports

import os, sys, shutil, json, queue, threading, tempfile
from flask import (Flask, render_template, request, redirect,
                   send_file, session, Response, stream_with_context, jsonify)
from dotenv import load_dotenv
import pandas as pd

# ── Clear pycache on startup ──────────────────────────────
pycache = os.path.join(os.path.dirname(__file__), '__pycache__')
if os.path.exists(pycache):
    shutil.rmtree(pycache)

load_dotenv()

from matcher         import final_score
from resume_parser   import load_resumes_from_folder, extract_text_from_pdf, extract_text_from_docx
from skill_extractor import extract_skills, extract_experience
from model_evaluator import get_results_for_template
from database        import init_db, register_user, login_user, save_user_result, get_user_history
from pdf_report      import generate_candidate_pdf

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "resume-secret-2024")

ADMIN_USER = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASS = os.getenv("ADMIN_PASSWORD", "admin123")

# Create required folders (EDA folder removed)
for folder in ["resumes", "job_descriptions", "reports",
               "reports/candidates", "models", "dataset", "user_uploads"]:
    os.makedirs(folder, exist_ok=True)

init_db()


# ────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────
def is_admin():
    return session.get("logged_in") is True and session.get("role") == "admin"

def is_user():
    return session.get("logged_in") is True and session.get("role") == "user"

def load_jd_from_folder():
    files = [f for f in os.listdir("job_descriptions") if not f.startswith('.')]
    if not files:
        return None
    with open(os.path.join("job_descriptions", files[0]), "r", encoding="utf-8") as f:
        return f.read()

def _clear_jd_folder():
    """Remove all files from job_descriptions/ to avoid stale JD accumulation."""
    for f in os.listdir("job_descriptions"):
        try:
            os.remove(os.path.join("job_descriptions", f))
        except OSError:
            pass

def build_result_entry(i, text, name, ai_scores, ats_scores, final_scores,
                        ml_categories, ml_confidences, jd_skills):
    skills  = extract_skills(text)
    exp     = extract_experience(text)
    score   = final_scores[i]
    missing = [s for s in jd_skills if s not in skills]

    if "machine learning" in skills or "data science" in skills or "python" in skills:
        summary = "Suitable for AI / Data Science roles"
    elif "sql" in skills or "power bi" in skills:
        summary = "Suitable for Data Analyst roles"
    elif "java" in skills or "spring" in skills:
        summary = "Suitable for Backend Developer roles"
    elif "html" in skills or "javascript" in skills:
        summary = "Suitable for Web Developer roles"
    else:
        summary = "Suitable for IT / Software roles"

    roles = []
    if "machine learning" in skills or "python" in skills: roles.append("AI Engineer")
    if "sql" in skills or "power bi" in skills:            roles.append("Data Analyst")
    if "java" in skills:                                    roles.append("Java Developer")
    if "html" in skills or "javascript" in skills:         roles.append("Web Developer")
    if "react" in skills or "node" in skills:              roles.append("Full-Stack Developer")
    if not roles:                                           roles.append("Software Engineer")

    decision = (
        "⭐ Outstanding"       if score >= 80 else
        "🟢 Very Good"         if score >= 65 else
        "🔵 Good"              if score >= 50 else
        "🟡 Average"           if score >= 35 else
        "🔴 Needs Improvement"
    )

    return {
        "name":          name,
        "ai":            ai_scores[i],
        "ats":           ats_scores[i],
        "final":         final_scores[i],
        "skills":        ", ".join(sorted(skills)) if skills else "—",
        "missing":       ", ".join(missing) if missing else "None",
        "suggestion":    "Excellent match!" if not missing else "Add: " + ", ".join(missing[:5]),
        "summary":       summary,
        "roles":         ", ".join(roles),
        "exp":           exp,
        "decision":      decision,
        "ml_category":   ml_categories[i],
        "ml_confidence": ml_confidences[i],
    }


# ────────────────────────────────────────────────────────────
# ADMIN LOGIN / LOGOUT
# ────────────────────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("username") == ADMIN_USER and \
           request.form.get("password") == ADMIN_PASS:
            session["logged_in"] = True
            session["role"]      = "admin"
            session["username"]  = "Admin"
            return redirect("/dashboard")
        error = "Invalid username or password."
    return render_template("auth.html", tab="admin", error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/")


# ────────────────────────────────────────────────────────────
# USER REGISTER / LOGIN
# ────────────────────────────────────────────────────────────
@app.route("/user/register", methods=["GET", "POST"])
def user_register():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")

        if not username or not email or not password:
            error = "All fields are required."
        elif password != confirm:
            error = "Passwords do not match."
        elif len(password) < 6:
            error = "Password must be at least 6 characters."
        else:
            ok, msg = register_user(username, email, password)
            if ok:
                return redirect("/user/login?registered=1")
            else:
                error = msg

    return render_template("auth.html", tab="register", reg_error=error)


@app.route("/user/login", methods=["GET", "POST"])
def user_login():
    error      = None
    registered = request.args.get("registered")
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        user     = login_user(username, password)
        if user:
            session["logged_in"] = True
            session["role"]      = "user"
            session["user_id"]   = user["id"]
            session["username"]  = user["username"]
            return redirect("/user/dashboard")
        error = "Invalid username or password."
    return render_template("auth.html", tab="user", error=error, registered=registered)


@app.route("/user/logout")
def user_logout():
    session.clear()
    return redirect("/user/login")


# ────────────────────────────────────────────────────────────
# USER DASHBOARD
# ────────────────────────────────────────────────────────────
@app.route("/user/dashboard")
def user_dashboard():
    if not is_user():
        return redirect("/user/login")
    history = get_user_history(session["user_id"])
    return render_template("user_dashboard.html",
                           username=session["username"],
                           history=history)


# ────────────────────────────────────────────────────────────
# USER — ANALYZE OWN RESUME
# ────────────────────────────────────────────────────────────
@app.route("/user/analyze", methods=["POST"])
def user_analyze():
    if not is_user():
        return redirect("/user/login")

    # Get JD — from textarea or file
    jd_text = ""
    jd_file = request.files.get("jd_file")
    jd_raw  = request.form.get("jd_text", "").strip()

    if jd_raw:
        jd_text = jd_raw
    elif jd_file and jd_file.filename and jd_file.filename.endswith(".txt"):
        jd_text = jd_file.read().decode("utf-8", errors="ignore")

    if not jd_text:
        return "❌ Please provide a Job Description.", 400

    # Get resume file
    resume_file = request.files.get("resume")
    if not resume_file or resume_file.filename == "":
        return "❌ Please upload a resume.", 400

    ext = os.path.splitext(resume_file.filename)[1].lower()
    if ext not in {".pdf", ".docx"}:
        return "❌ Only PDF or DOCX resumes accepted.", 400

    # Save resume temporarily
    tmp_path = os.path.join("user_uploads", resume_file.filename)
    resume_file.save(tmp_path)

    try:
        # Extract text
        if ext == ".pdf":
            resume_text = extract_text_from_pdf(tmp_path)
        else:
            resume_text = extract_text_from_docx(tmp_path)

        if not resume_text.strip():
            return "❌ Could not read resume text. Try a different file.", 400

        # Score
        ai_s, ats_s, final_s, ml_cats, ml_confs = final_score(jd_text, [resume_text])
        jd_skills = extract_skills(jd_text)

        entry = build_result_entry(0, resume_text, resume_file.filename,
                                   ai_s, ats_s, final_s, ml_cats, ml_confs, jd_skills)

        # Save to history
        skills_list  = extract_skills(resume_text)
        missing_list = [s for s in jd_skills if s not in skills_list]
        save_user_result(
            session["user_id"], resume_file.filename, jd_text,
            entry["ai"], entry["ats"], entry["final"],
            entry["ml_category"], skills_list, missing_list, entry["decision"]
        )

        return render_template("user_result.html",
                               result=entry,
                               username=session["username"],
                               jd_text=jd_text[:200] + "..." if len(jd_text) > 200 else jd_text)

    except Exception as e:
        import traceback; traceback.print_exc()
        return f"❌ Error: {str(e)}", 500
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ────────────────────────────────────────────────────────────
# ADMIN DASHBOARD
# ────────────────────────────────────────────────────────────
@app.route("/dashboard")
def dashboard():
    if not is_admin():
        return redirect("/")
    model_trained = (os.path.exists("models/best_model.pkl") and
                     os.path.getsize("models/best_model.pkl") > 0)
    return render_template("dashboard.html", model_trained=model_trained)


# ────────────────────────────────────────────────────────────
# ADMIN — UPLOAD & PROCESS (supports JD text + file)
# ────────────────────────────────────────────────────────────
@app.route("/upload", methods=["POST"])
def upload_files():
    if not is_admin():
        return redirect("/")

    # JD — textarea or file
    jd_text = ""
    jd_raw  = request.form.get("jd_text", "").strip()
    jd_file = request.files.get("jd")

    if jd_raw:
        jd_text = jd_raw
        # BUG FIX: Clear old JD files before saving pasted JD to avoid stale accumulation
        _clear_jd_folder()
        with open("job_descriptions/pasted_jd.txt", "w", encoding="utf-8") as f:
            f.write(jd_text)
    elif jd_file and jd_file.filename:
        _clear_jd_folder()
        jd_file.save(os.path.join("job_descriptions", jd_file.filename))
        jd_text = load_jd_from_folder()
    else:
        return "❌ Provide a Job Description (paste or upload .txt).", 400

    resume_files = request.files.getlist("resumes")
    if not resume_files or resume_files[0].filename == "":
        return "❌ No resume files uploaded.", 400

    for f in resume_files:
        if os.path.splitext(f.filename)[1].lower() not in {".pdf", ".docx"}:
            return f"❌ Invalid format: {f.filename}. Use PDF or DOCX.", 400

    for f in os.listdir("resumes"):
        os.remove(os.path.join("resumes", f))
    for file in resume_files:
        file.save(os.path.join("resumes", file.filename))

    try:
        resumes, names = load_resumes_from_folder("resumes")
        if not resumes:
            return "❌ No valid resumes found.", 400

        ai_scores, ats_scores, final_scores, ml_categories, ml_confidences = \
            final_score(jd_text, resumes)

    except Exception as e:
        import traceback; traceback.print_exc()
        return f"❌ Processing error: {str(e)}", 500

    jd_skills = extract_skills(jd_text)
    results   = [
        build_result_entry(i, text, names[i], ai_scores, ats_scores,
                           final_scores, ml_categories, ml_confidences, jd_skills)
        for i, text in enumerate(resumes)
    ]
    results = sorted(results, key=lambda x: x["final"], reverse=True)
    top     = results[0] if results else None

    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(results).to_excel("reports/final_report.xlsx", index=False)

    return render_template("result.html", results=results, top=top)


# ────────────────────────────────────────────────────────────
# CANDIDATE PDF REPORT DOWNLOAD
# ────────────────────────────────────────────────────────────
@app.route("/download/candidate-pdf", methods=["POST"])
def download_candidate_pdf():
    if not is_admin():
        return redirect("/")
    try:
        candidate = request.get_json()
        safe_name = candidate.get("name","report").replace(" ","_").replace("/","_")
        out_path  = os.path.join("reports", "candidates", f"{safe_name}_report.pdf")

        ok, result = generate_candidate_pdf(candidate, out_path)
        if ok:
            return send_file(out_path, as_attachment=True,
                             download_name=f"{safe_name}_ATS_Report.pdf",
                             mimetype="application/pdf")
        else:
            return f"❌ PDF error: {result}", 500
    except Exception as e:
        return f"❌ {str(e)}", 500


@app.route("/download/user-pdf", methods=["POST"])
def download_user_pdf():
    if not is_user():
        return redirect("/user/login")
    try:
        candidate = request.get_json()
        safe_name = candidate.get("name","report").replace(" ","_").replace("/","_")
        out_path  = os.path.join("reports", "candidates", f"user_{safe_name}_report.pdf")

        ok, result = generate_candidate_pdf(candidate, out_path)
        if ok:
            return send_file(out_path, as_attachment=True,
                             download_name=f"{safe_name}_ATS_Report.pdf",
                             mimetype="application/pdf")
        return f"❌ {result}", 500
    except Exception as e:
        return f"❌ {str(e)}", 500


# ────────────────────────────────────────────────────────────
# TRAIN PAGE
# ────────────────────────────────────────────────────────────
@app.route("/train")
def train_page():
    if not is_admin():
        return redirect("/")
    return render_template("train.html")


@app.route("/train/stream")
def train_stream():
    if not is_admin():
        return redirect("/")

    msg_queue = queue.Queue()

    def run_training():
        try:
            for f in ["models/best_model.pkl", "models/tfidf_vectorizer.pkl",
                      "models/label_encoder.pkl", "models/model_results.json"]:
                if os.path.exists(f): os.remove(f)

            from model_trainer import train_and_save
            train_and_save(progress_callback=lambda m: msg_queue.put(m))

            import matcher as m
            m._load_trained_models()
        except Exception as e:
            msg_queue.put(f"error:{str(e)}")
        finally:
            msg_queue.put("__END__")

    threading.Thread(target=run_training, daemon=True).start()

    def generate():
        while True:
            try:
                msg = msg_queue.get(timeout=180)
                if msg == "__END__":
                    yield "data: __END__\n\n"; break
                yield f"data: {msg}\n\n"
            except queue.Empty:
                yield "data: error:Training timed out\n\n"; break

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


# ────────────────────────────────────────────────────────────
# MODEL REPORT + DOWNLOADS
# ────────────────────────────────────────────────────────────
@app.route("/model-report")
def model_report():
    if not is_admin():
        return redirect("/")
    model_results, best_model = get_results_for_template()
    return render_template("model_report.html", results=model_results, best=best_model)


@app.route("/download")
def download():
    if not is_admin():
        return redirect("/")
    path = "reports/final_report.xlsx"
    if not os.path.exists(path):
        return "❌ No report found. Run an analysis first.", 404
    return send_file(path, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
