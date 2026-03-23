# pdf_report.py
# Generate a downloadable PDF report for a single candidate
# Uses reportlab (pure Python, no LaTeX needed)

import os
from datetime import datetime


def generate_candidate_pdf(candidate, output_path):
    """
    Generate a PDF report for a single candidate.
    candidate: dict with keys name, ai, ats, final, ml_category,
               ml_confidence, skills, missing, suggestion,
               summary, roles, exp, decision
    output_path: where to save the PDF
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                        Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_CENTER, TA_LEFT

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        doc    = SimpleDocTemplate(output_path, pagesize=A4,
                                   topMargin=2*cm, bottomMargin=2*cm,
                                   leftMargin=2*cm, rightMargin=2*cm)
        styles = getSampleStyleSheet()
        story  = []

        # ── Color palette ──────────────────────────────────
        BLUE    = colors.HexColor("#4facfe")
        GREEN   = colors.HexColor("#28a745")
        RED     = colors.HexColor("#dc3545")
        ORANGE  = colors.HexColor("#fd7e14")
        DARK    = colors.HexColor("#333333")
        LIGHT   = colors.HexColor("#f8f9fa")
        MUTED   = colors.HexColor("#6c757d")

        # ── Custom styles ───────────────────────────────────
        title_style = ParagraphStyle("title", parent=styles["Title"],
                                     fontSize=22, textColor=BLUE,
                                     spaceAfter=4, alignment=TA_CENTER)
        sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
                                     fontSize=11, textColor=MUTED,
                                     spaceAfter=2, alignment=TA_CENTER)
        h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
                                     fontSize=13, textColor=DARK,
                                     spaceBefore=14, spaceAfter=6,
                                     borderPad=4)
        body_style  = ParagraphStyle("body", parent=styles["Normal"],
                                     fontSize=10, textColor=DARK,
                                     leading=16)
        label_style = ParagraphStyle("label", parent=styles["Normal"],
                                     fontSize=9, textColor=MUTED)

        # ── HEADER ─────────────────────────────────────────
        story.append(Paragraph("AI Resume Matcher", title_style))
        story.append(Paragraph("Candidate Evaluation Report", sub_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
            ParagraphStyle("date", parent=styles["Normal"], fontSize=9,
                           textColor=MUTED, alignment=TA_CENTER)
        ))
        story.append(Spacer(1, 0.3*cm))
        story.append(HRFlowable(width="100%", thickness=2, color=BLUE))
        story.append(Spacer(1, 0.3*cm))

        # ── CANDIDATE NAME ──────────────────────────────────
        name = candidate.get("name", "Candidate").replace(".pdf","").replace(".docx","")
        story.append(Paragraph(f"<b>{name}</b>", ParagraphStyle(
            "cname", parent=styles["Normal"], fontSize=16, textColor=DARK,
            spaceAfter=4
        )))
        story.append(Paragraph(
            f"Experience: {candidate.get('exp','—')}  |  "
            f"ML Category: {candidate.get('ml_category','—')}  |  "
            f"Decision: {candidate.get('decision','—')}",
            body_style
        ))
        story.append(Spacer(1, 0.4*cm))

        # ── SCORE TABLE ─────────────────────────────────────
        story.append(Paragraph("Score Summary", h2_style))

        final = float(candidate.get("final", 0))
        ai    = float(candidate.get("ai",    0))
        ats   = float(candidate.get("ats",   0))
        conf  = float(candidate.get("ml_confidence", 0))

        def score_color(s):
            if s >= 75: return GREEN
            if s >= 50: return ORANGE
            return RED

        score_data = [
            ["Metric",          "Score",    "Rating"],
            ["AI Score",        f"{ai}%",   "Semantic match (TF-IDF + BERT)"],
            ["ATS Score",       f"{ats}%",  "Keyword match with JD"],
            ["Final Score",     f"{final}%","Combined evaluation"],
            ["ML Confidence",   f"{conf}%", "Model prediction confidence"],
        ]

        score_table = Table(score_data, colWidths=[4.5*cm, 2.5*cm, 9*cm])
        score_table.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), BLUE),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 10),
            ("ALIGN",       (1,0), (1,-1), "CENTER"),
            ("BACKGROUND",  (0,1), (-1,1), LIGHT),
            ("BACKGROUND",  (0,3), (-1,3), LIGHT),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, LIGHT]),
            ("TEXTCOLOR",   (1,1), (1,1), score_color(ai)),
            ("TEXTCOLOR",   (1,2), (1,2), score_color(ats)),
            ("TEXTCOLOR",   (1,3), (1,3), score_color(final)),
            ("FONTNAME",    (1,3), (1,3), "Helvetica-Bold"),
            ("FONTSIZE",    (1,3), (1,3), 12),
            ("TOPPADDING",  (0,0), (-1,-1), 6),
            ("BOTTOMPADDING",(0,0),(-1,-1), 6),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 0.4*cm))

        # ── SKILLS FOUND ────────────────────────────────────
        story.append(Paragraph("Skills Detected", h2_style))
        skills_text = candidate.get("skills", "—")
        story.append(Paragraph(skills_text, body_style))
        story.append(Spacer(1, 0.3*cm))

        # ── MISSING SKILLS ───────────────────────────────────
        story.append(Paragraph("Missing Skills", h2_style))
        # BUG FIX: removed unused `color` variable (was assigned but never referenced)
        missing = candidate.get("missing", "None")
        story.append(Paragraph(
            f'<font color="{"#dc3545" if missing != "None" else "#28a745"}">{missing}</font>',
            body_style
        ))
        story.append(Spacer(1, 0.3*cm))

        # ── SUGGESTION ───────────────────────────────────────
        story.append(Paragraph("AI Suggestion", h2_style))
        story.append(Paragraph(candidate.get("suggestion", "—"), body_style))
        story.append(Spacer(1, 0.3*cm))

        # ── SUMMARY + ROLES ──────────────────────────────────
        story.append(Paragraph("Profile Summary", h2_style))
        summary_data = [
            ["Resume Summary", candidate.get("summary", "—")],
            ["Recommended Roles", candidate.get("roles", "—")],
        ]
        summary_table = Table(summary_data, colWidths=[4.5*cm, 11.5*cm])
        summary_table.setStyle(TableStyle([
            ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 10),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#dee2e6")),
            ("BACKGROUND",  (0,0), (0,-1), LIGHT),
            ("TOPPADDING",  (0,0), (-1,-1), 8),
            ("BOTTOMPADDING",(0,0),(-1,-1), 8),
            ("LEFTPADDING", (0,0), (-1,-1), 8),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 0.5*cm))

        # ── FOOTER ───────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#dee2e6")))
        story.append(Spacer(1, 0.2*cm))
        story.append(Paragraph(
            "Generated by AI Resume Matcher | MCA Capstone Project",
            ParagraphStyle("footer", parent=styles["Normal"], fontSize=8,
                           textColor=MUTED, alignment=TA_CENTER)
        ))

        doc.build(story)
        return True, output_path

    except ImportError:
        return False, "reportlab not installed. Run: pip install reportlab"
    except Exception as e:
        return False, str(e)
