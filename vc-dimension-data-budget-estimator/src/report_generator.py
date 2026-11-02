# src/report_generator.py
# ─────────────────────────────────────────────────────────────────
# Report Generator supporting:
# 1. Downloadable PDF Report (via ReportLab)
# 2. Downloadable Excel Workbook (via OpenPyXL)
# 3. Standalone Interactive HTML Report (with embedded base64 charts)
# ─────────────────────────────────────────────────────────────────

import os
import io
import base64
import datetime
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ReportLab imports
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

# OpenPyXL imports
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from src.evaluation import build_comparison_dataframe


# ── Figure Conversion Helper ──────────────────────────────────────

def figure_to_base64(fig: Figure, dpi: int = 150) -> str:
    """Convert a Matplotlib figure into a base64 encoded PNG data URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# ── 1. PDF Report Generator (ReportLab) ───────────────────────────

def generate_pdf_report(
    results: List[Dict[str, Any]],
    dataset_name: str,
    dataset_info: Optional[Dict[str, Any]] = None,
    epsilon: float = 0.10,
    delta: float = 0.05,
    bound_setting: str = "realizable",
    figures: Optional[Dict[str, Figure]] = None,
    output_path: Optional[str] = None,
    cost_per_sample: float = 1.0,
) -> bytes:
    """
    Generate an executive-grade PDF research and budget report using ReportLab.
    Includes title, metadata, VC dimensions, PAC theory, learning curve evaluations,
    theoretical vs empirical comparison, and data budget recommendations.
    """
    buf = io.BytesIO()
    target_stream = output_path if output_path else buf

    doc = SimpleDocTemplate(
        target_stream,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1E293B"),
        alignment=0,
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=11,
        leading=14,
        textColor=colors.HexColor("#64748B"),
        spaceAfter=15,
    )
    h2_style = ParagraphStyle(
        "ReportH2",
        parent=styles["Heading2"],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
        spaceBefore=14,
        spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6,
    )
    callout_style = ParagraphStyle(
        "ReportCallout",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E3A8A"),
        backColor=colors.HexColor("#EFF6FF"),
        borderPadding=8,
        spaceBefore=8,
        spaceAfter=10,
    )

    story = []

    # 1. Header
    story.append(Paragraph("Machine Learning Data Budget Report", title_style))
    now_str = datetime.datetime.now().strftime("%B %d, %Y - %H:%M UTC")
    story.append(Paragraph(
        f"<b>VC-Dimension-Guided Data Budget Estimator</b> · Generated: {now_str}",
        subtitle_style
    ))
    story.append(Spacer(1, 8))

    # 2. Executive Summary Box
    ds_info = dataset_info or {}
    n_samples = ds_info.get("n_samples", "N/A")
    n_features = ds_info.get("n_features", "N/A")
    task_type = ds_info.get("task_type", "classification").capitalize()

    best_model = "N/A"
    best_acc = 0.0
    for r in results:
        v_acc = r.get("final_val_acc") or 0.0
        if v_acc > best_acc:
            best_acc = v_acc
            best_model = r.get("model_name", "N/A")

    exec_summary = (
        f"<b>Executive Summary:</b> Analysis conducted on dataset <b>{dataset_name}</b> "
        f"({n_samples:,} records, {n_features} features, Task: {task_type}). "
        f"Target error tolerance set at ε = {epsilon:.2f} with confidence 1 - δ = {(1-delta)*100:.0f}%. "
        f"Top performing model: <b>{best_model}</b> (Validation Accuracy: {best_acc*100:.1f}%). "
        "Theoretical distribution-free PAC bounds significantly exceed empirical requirements, "
        "confirming that real-world sample complexity is dominated by data regularity rather than worst-case limits."
    )
    story.append(Paragraph(exec_summary, callout_style))
    story.append(Spacer(1, 10))

    # 3. Dataset Profile Table
    story.append(Paragraph("1. Dataset Characteristics & PAC Parameters", h2_style))
    param_data = [
        ["Parameter", "Configured Value", "Specification / Meaning"],
        ["Dataset Name", str(dataset_name), "Evaluated dataset"],
        ["Samples & Features", f"{n_samples} samples, {n_features} features", "Dimensionality"],
        ["Target Error Tolerance (ε)", f"{epsilon:.3f}", "Maximum acceptable generalization error"],
        ["Failure Probability (δ)", f"{delta:.3f}", f"Confidence: {(1-delta)*100:.0f}%"],
        ["PAC Setting", bound_setting.capitalize(), "Zero-noise (Realizable) or Noisy (Agnostic)"],
    ]
    t_params = Table(param_data, colWidths=[160, 140, 240])
    t_params.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F1F5F9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
    ]))
    story.append(t_params)
    story.append(Spacer(1, 12))

    # 4. Model Comparison Table
    story.append(Paragraph("2. Model Evaluation & Sample Requirements", h2_style))
    if results:
        comp_df = build_comparison_dataframe(results)
        table_headers = ["Model", "VC Dim", "Val Acc", "Theory (m)", "Empirical", "Predicted", "Ratio"]
        table_rows = [table_headers]

        for _, row in comp_df.iterrows():
            table_rows.append([
                str(row.get("Model", ""))[:20],
                str(row.get("VC-Dim Estimate", "")),
                str(row.get("Final Val Accuracy (%)", "")) + "%",
                f"{row.get('Theoretical Bound (m)', ''):,}" if isinstance(row.get('Theoretical Bound (m)'), (int, float)) else str(row.get('Theoretical Bound (m)', '')),
                str(row.get("Empirical Requirement (m)", "")),
                str(row.get("Predicted Requirement (m)", "")),
                str(row.get("Theory/Empirical Ratio", "")),
            ])

        t_comp = Table(table_rows, colWidths=[105, 55, 60, 80, 75, 80, 65])
        t_comp.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#FFFFFF")),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        story.append(t_comp)
    else:
        story.append(Paragraph("No experiment models evaluated yet.", body_style))

    story.append(Spacer(1, 12))

    # 5. Data Budget Recommendation Table
    story.append(Paragraph("3. Recommended Data Budget & Cost Analysis", h2_style))
    budget_headers = ["Model", "Current N", "Empirical Need", "Additional Needed", f"Cost (@ ${cost_per_sample:.2f})"]
    budget_rows = [budget_headers]

    curr_n = int(n_samples) if isinstance(n_samples, (int, float)) else 0
    for r in results:
        emp = r.get("empirical", {})
        m_req = emp.get("sample_count")
        if m_req is None and r.get("prediction"):
            m_req = r.get("prediction", {}).get("sample_count")

        add_needed = max(0, m_req - curr_n) if m_req else "N/A"
        cost_str = f"${add_needed * cost_per_sample:,.2f}" if isinstance(add_needed, (int, float)) else "N/A"
        m_req_str = f"{m_req:,}" if isinstance(m_req, (int, float)) else "Not achieved"

        budget_rows.append([
            r.get("model_name", "")[:22],
            f"{curr_n:,}",
            m_req_str,
            f"{add_needed:,}" if isinstance(add_needed, (int, float)) else add_needed,
            cost_str,
        ])

    t_budget = Table(budget_rows, colWidths=[130, 90, 100, 100, 100])
    t_budget.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#065F46")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#FFFFFF")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0FDF4")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(t_budget)
    story.append(Spacer(1, 14))

    # 6. Embedded Charts (if provided)
    if figures:
        story.append(Paragraph("4. Empirical Learning Curves & Visualizations", h2_style))
        for fig_name, fig in figures.items():
            if isinstance(fig, Figure):
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", dpi=130, bbox_inches="tight")
                img_buf.seek(0)
                img = RLImage(img_buf, width=6.5 * inch, height=3.2 * inch)
                story.append(Paragraph(f"<b>Figure:</b> {fig_name}", body_style))
                story.append(img)
                story.append(Spacer(1, 10))

    # 7. Theoretical Notes & Limitations
    story.append(Paragraph("5. Methodology, PAC Theory & Limitations", h2_style))
    notes_text = (
        "<b>Theoretical Foundations:</b> The sample complexity bounds reported are computed using "
        "the Blumer et al. (1989) realizable formula and Anthony & Bartlett (1999) agnostic bound. "
        "Because PAC bounds are distribution-free and account for the worst possible adversarial distribution, "
        "they are conservative upper bounds. Practical requirements observed empirically through learning curves "
        "benefit from structured, benign data manifolds and margin separation.<br/><br/>"
        "<b>Limitations:</b> Power-law extrapolations are indicative heuristics and become uncertain "
        "when predicting beyond 5× the observed training set range."
    )
    story.append(Paragraph(notes_text, body_style))

    doc.build(story)

    if output_path:
        with open(output_path, "rb") as f:
            return f.read()
    return buf.getvalue()


# ── 2. Excel Report Generator (OpenPyXL) ───────────────────────────

def generate_excel_report(
    results: List[Dict[str, Any]],
    dataset_name: str,
    dataset_info: Optional[Dict[str, Any]] = None,
    epsilon: float = 0.10,
    delta: float = 0.05,
    output_path: Optional[str] = None,
    cost_per_sample: float = 1.0,
) -> bytes:
    """
    Generate an executive multi-tab Excel workbook using OpenPyXL.
    Includes formatted tabs:
      - Sheet 1: Executive Summary & Project Specs
      - Sheet 2: Model Comparison & PAC Bounds
      - Sheet 3: Learning Curve Data Points
      - Sheet 4: Data Budget & Cost Recommendations
    """
    wb = openpyxl.Workbook()

    # Styling elements
    font_header = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    fill_navy = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
    fill_green = PatternFill(start_color="065F46", end_color="065F46", fill_type="solid")
    fill_blue = PatternFill(start_color="1E40AF", end_color="1E40AF", fill_type="solid")
    font_title = Font(name="Calibri", size=14, bold=True, color="0F172A")
    font_bold = Font(name="Calibri", size=11, bold=True)
    border_thin = Border(
        left=Side(style="thin", color="CBD5E1"),
        right=Side(style="thin", color="CBD5E1"),
        top=Side(style="thin", color="CBD5E1"),
        bottom=Side(style="thin", color="CBD5E1")
    )
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")

    # ── Sheet 1: Summary ──────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "Executive Summary"
    ws1.views.sheetView[0].showGridLines = True

    ws1.append(["VC-Dimension Data Budget Estimator — Project Report"])
    ws1["A1"].font = font_title
    ws1.append([f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
    ws1.append([])

    ds_info = dataset_info or {}
    ws1.append(["Project Configuration", "Value"])
    ws1.append(["Dataset Name", str(dataset_name)])
    ws1.append(["Dataset Total Samples", ds_info.get("n_samples", "N/A")])
    ws1.append(["Input Feature Count", ds_info.get("n_features", "N/A")])
    ws1.append(["Target Error Tolerance (ε)", epsilon])
    ws1.append(["Failure Risk (δ)", delta])
    ws1.append(["Confidence Level (1 - δ)", f"{(1-delta)*100:.1f}%"])
    ws1.append(["Unit Data Cost ($ / sample)", cost_per_sample])

    for col in ("A", "B"):
        ws1[f"{col}4"].fill = fill_navy
        ws1[f"{col}4"].font = font_header

    # ── Sheet 2: Model Comparison ─────────────────────────────────
    ws2 = wb.create_sheet(title="Model Comparison")
    ws2.views.sheetView[0].showGridLines = True

    comp_df = build_comparison_dataframe(results)
    headers2 = [
        "Model", "VC Dimension", "VC Dim Type", "Theoretical PAC (m)",
        "Empirical Needed (m)", "Power-Law Pred (m)", "Val Accuracy (%)",
        "Final Val Error", "Training Time (s)", "Theory/Empirical Ratio"
    ]
    ws2.append(headers2)
    for col_idx in range(1, len(headers2) + 1):
        cell = ws2.cell(row=1, column=col_idx)
        cell.fill = fill_blue
        cell.font = font_header
        cell.alignment = align_center

    for _, row in comp_df.iterrows():
        ws2.append([
            row.get("Model"),
            row.get("VC-Dim Estimate"),
            row.get("VC-Dim Label"),
            row.get("Theoretical Bound (m)"),
            row.get("Empirical Requirement (m)"),
            row.get("Predicted Requirement (m)"),
            row.get("Final Val Accuracy (%)"),
            row.get("Final Val Error"),
            row.get("Training Time (s)"),
            row.get("Theory/Empirical Ratio"),
        ])

    # ── Sheet 3: Learning Curve Data ──────────────────────────────
    ws3 = wb.create_sheet(title="Learning Curve Data")
    ws3.views.sheetView[0].showGridLines = True

    headers3 = ["Model", "Sample Size (m)", "Train Error", "Train Error Std", "Val Error", "Val Error Std", "Generalization Gap"]
    ws3.append(headers3)
    for col_idx in range(1, len(headers3) + 1):
        cell = ws3.cell(row=1, column=col_idx)
        cell.fill = fill_navy
        cell.font = font_header
        cell.alignment = align_center

    for r in results:
        mname = r.get("model_name", "")
        lc = r.get("learning_curve", {})
        scs = lc.get("sample_counts", [])
        tes = lc.get("train_errors", [])
        te_stds = lc.get("train_errors_std", [0]*len(scs))
        ves = lc.get("val_errors", [])
        ve_stds = lc.get("val_errors_std", [0]*len(scs))
        gaps = lc.get("generalization_gaps", [0]*len(scs))

        for sc, te, te_s, ve, ve_s, gap in zip(scs, tes, te_stds, ves, ve_stds, gaps):
            ws3.append([mname, sc, te, te_s, ve, ve_s, gap])

    # ── Sheet 4: Data Budget Estimates ────────────────────────────
    ws4 = wb.create_sheet(title="Data Budget Recommendation")
    ws4.views.sheetView[0].showGridLines = True

    headers4 = ["Model", "Current Dataset Size", "Empirical Requirement", "Additional Samples Needed", "Unit Cost ($)", "Estimated Additional Budget ($)"]
    ws4.append(headers4)
    for col_idx in range(1, len(headers4) + 1):
        cell = ws4.cell(row=1, column=col_idx)
        cell.fill = fill_green
        cell.font = font_header
        cell.alignment = align_center

    curr_n = int(ds_info.get("n_samples", 0)) if isinstance(ds_info.get("n_samples"), (int, float)) else 0

    for r in results:
        m_req = r.get("empirical", {}).get("sample_count")
        if m_req is None and r.get("prediction"):
            m_req = r.get("prediction", {}).get("sample_count")

        add_needed = max(0, m_req - curr_n) if m_req else "N/A"
        cost_val = add_needed * cost_per_sample if isinstance(add_needed, (int, float)) else "N/A"

        ws4.append([
            r.get("model_name"),
            curr_n,
            m_req if m_req is not None else "Not achieved",
            add_needed,
            cost_per_sample,
            cost_val,
        ])

    # Auto-adjust column widths across all sheets
    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            col_letter = get_column_letter(col[0].column)
            sheet.column_dimensions[col_letter].width = max(max_len + 4, 12)

    buf = io.BytesIO()
    wb.save(buf)
    wb.close()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(buf.getvalue())
        return buf.getvalue()

    return buf.getvalue()


# ── 3. Standalone HTML Report ─────────────────────────────────────

def generate_html_report(
    results: List[Dict[str, Any]],
    dataset_name: str,
    dataset_info: Optional[Dict[str, Any]] = None,
    epsilon: float = 0.10,
    delta: float = 0.05,
    bound_setting: str = "realizable",
    figures: Optional[Dict[str, Figure]] = None,
) -> str:
    """
    Generate an executive-grade, standalone HTML experiment report.
    All figures are embedded as base64 images so the report is completely
    portable and print-to-PDF ready.
    """
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dataset_info = dataset_info or {}
    n_samples = dataset_info.get("n_samples", "N/A")
    n_features = dataset_info.get("n_features", "N/A")
    n_classes = dataset_info.get("n_classes", len(dataset_info.get("class_proportions", {})) or "N/A")

    df_comp = build_comparison_dataframe(results)

    # Encode figures if provided
    encoded_figures = {}
    if figures:
        for name, fig in figures.items():
            if isinstance(fig, Figure):
                encoded_figures[name] = figure_to_base64(fig)

    # Compute key highlights
    best_acc = 0.0
    best_model = "N/A"
    max_ratio = 0.0
    min_theory = None
    min_empirical = None

    for r in results:
        val_acc = r.get("final_val_acc") or 0.0
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = r.get("model_name", "N/A")

        p_m = r.get("pac_bound", {}).get("m")
        if p_m is not None:
            min_theory = p_m if min_theory is None else min(min_theory, p_m)

        e_m = r.get("empirical", {}).get("sample_count")
        if e_m is not None:
            min_empirical = e_m if min_empirical is None else min(min_empirical, e_m)
            if p_m is not None and e_m > 0:
                max_ratio = max(max_ratio, p_m / e_m)

    # Build Comparison Table HTML
    if not df_comp.empty:
        rows_html = ""
        for _, row in df_comp.iterrows():
            ratio_val = row.get("Theory/Empirical Ratio", "N/A")
            rows_html += f"""
            <tr>
                <td style="font-weight: 600;">{row.get('Model', '')}</td>
                <td>{row.get('VC-Dim Estimate', '')} <small style="color: #64748b;">({row.get('VC-Dim Label', '')})</small></td>
                <td><strong>{row.get('Theoretical Bound (m)', ''):,}</strong></td>
                <td style="color: #059669; font-weight: 600;">{row.get('Empirical Requirement (m)', '')}</td>
                <td style="color: #dc2626;">{row.get('Predicted Requirement (m)', '')}</td>
                <td>{row.get('Final Val Accuracy (%)', '')}%</td>
                <td><span class="badge badge-purple">{ratio_val}</span></td>
            </tr>
            """
        comp_table_html = f"""
        <table class="data-table">
            <thead>
                <tr>
                    <th>Model Family</th>
                    <th>VC Dimension</th>
                    <th>PAC Bound (m)</th>
                    <th>Empirical Requirement</th>
                    <th>Power-Law Predicted</th>
                    <th>Val Accuracy</th>
                    <th>Theory / Empirical Ratio</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
        """
    else:
        comp_table_html = "<p style='color: #64748b;'>No experiment results recorded.</p>"

    # Build Figures HTML
    figures_html = ""
    if encoded_figures:
        figures_html += "<h2 class='section-title'>Experimental Figures & Visualizations</h2><div class='figures-grid'>"
        for title, uri in encoded_figures.items():
            figures_html += f"""
            <div class="figure-card">
                <h3>{title}</h3>
                <img src="{uri}" alt="{title}" />
            </div>
            """
        figures_html += "</div>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>VC Dimension Data Budget Report — {dataset_name}</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    body {{
        font-family: 'Inter', -apple-system, sans-serif;
        background-color: #f8fafc;
        color: #0f172a;
        margin: 0;
        padding: 40px 20px;
    }}
    .container {{
        max-width: 1060px;
        margin: 0 auto;
        background: #ffffff;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    }}
    .header {{
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 20px;
        margin-bottom: 30px;
    }}
    .header h1 {{
        font-size: 1.85rem;
        margin: 0 0 8px 0;
        color: #0f172a;
    }}
    .header p {{
        margin: 0;
        color: #64748b;
        font-size: 0.95rem;
    }}
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 35px;
    }}
    .kpi-card {{
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
    }}
    .kpi-label {{
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 6px;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: #0f172a;
    }}
    .section-title {{
        font-size: 1.25rem;
        color: #1e293b;
        border-bottom: 1px solid #e2e8f0;
        padding-bottom: 8px;
        margin-top: 35px;
        margin-bottom: 18px;
    }}
    .data-table {{
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        font-size: 0.9rem;
    }}
    .data-table th, .data-table td {{
        padding: 12px 14px;
        text-align: left;
        border-bottom: 1px solid #e2e8f0;
    }}
    .data-table th {{
        background-color: #f1f5f9;
        font-weight: 600;
        color: #334155;
    }}
    .badge {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }}
    .badge-purple {{
        background-color: #f3e8ff;
        color: #7e22ce;
    }}
    .figures-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(460px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }}
    .figure-card {{
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 15px;
        background: #ffffff;
    }}
    .figure-card img {{
        width: 100%;
        height: auto;
        border-radius: 4px;
    }}
    .print-btn {{
        background-color: #2563eb;
        color: #ffffff;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        font-size: 0.9rem;
        font-weight: 600;
        cursor: pointer;
        float: right;
    }}
    @media print {{
        body {{ background: #ffffff; padding: 0; }}
        .container {{ box-shadow: none; padding: 0; max-width: 100%; }}
        .print-btn {{ display: none; }}
    }}
</style>
</head>
<body>
<div class="container">
    <button class="print-btn" onclick="window.print()">🖨️ Print / Save to PDF</button>
    <div class="header">
        <h1>How Much Data Does Your Model Really Need?</h1>
        <p>A VC-Dimension-Guided Data Budget Estimator · Report Generated: {now_str}</p>
    </div>

    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Dataset</div>
            <div class="kpi-value" style="font-size: 1.25rem;">{dataset_name}</div>
            <small style="color: #64748b;">{n_samples} rows · {n_features} features</small>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">PAC Specs</div>
            <div class="kpi-value">ε={epsilon:.2f}</div>
            <small style="color: #64748b;">Confidence: {(1-delta)*100:.0f}% (δ={delta:.2f})</small>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Top Performer</div>
            <div class="kpi-value" style="color: #2563eb;">{best_acc*100:.1f}%</div>
            <small style="color: #64748b;">{best_model}</small>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Max Conservatism</div>
            <div class="kpi-value" style="color: #7e22ce;">{max_ratio:.1f}×</div>
            <small style="color: #64748b;">Theory vs Empirical Need</small>
        </div>
    </div>

    <h2 class="section-title">Theoretical vs Empirical Comparison Table</h2>
    {comp_table_html}

    {figures_html}

    <h2 class="section-title">Academic References & Foundational Theory</h2>
    <ol style="font-size: 0.85rem; color: #475569; line-height: 1.6;">
        <li>Blumer, A., Ehrenfeucht, A., Haussler, D., & Warmuth, M. K. (1989). Learnability and the Vapnik-Chervonenkis dimension. <em>Journal of the ACM, 36</em>(4), 929–965.</li>
        <li>Vapnik, V. N. (1998). <em>Statistical Learning Theory</em>. John Wiley & Sons.</li>
        <li>Anthony, M., & Bartlett, P. L. (1999). <em>Neural Network Learning: Theoretical Foundations</em>. Cambridge University Press.</li>
        <li>Devroye, L., Györfi, L., & Lugosi, G. (1996). <em>A Probabilistic Theory of Pattern Recognition</em>. Springer.</li>
    </ol>
</div>
</body>
</html>
"""
    return html


def save_report_to_disk(report_content: Union[str, bytes], output_dir: str = "reports", filename: str = "report.html") -> str:
    """Save the report (HTML, PDF, or Excel) to disk."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, filename)
    mode = "wb" if isinstance(report_content, bytes) else "w"
    encoding = None if isinstance(report_content, bytes) else "utf-8"

    with open(file_path, mode, encoding=encoding) as f:
        f.write(report_content)
    return file_path
