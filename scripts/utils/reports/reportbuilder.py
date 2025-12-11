

import os
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from scripts.utils.logging.logger import *

# ---------- Utilities ----------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64

def safe_float(val, default=0.0):
    try:
        return float(val)
    except (TypeError, ValueError):
        return default

def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SectionHeader", fontSize=16, leading=20, spaceAfter=10,
                              textColor=colors.HexColor("#800020"), underlineWidth=1))
    styles.add(ParagraphStyle(name="NormalJustify", fontSize=10, leading=13, alignment=4))
    styles.add(ParagraphStyle(name="Small", fontSize=8, leading=10, alignment=4))
    return styles

def make_table(title, rows, styles, col_widths=None):
    header = [Paragraph(f"<b>{title}</b>", styles["SectionHeader"])]
    table_data = [[Paragraph("<b>"+str(c)+"</b>", styles["NormalJustify"]) for c in rows[0]]]
    for r in rows[1:]:
        table_data.append([Paragraph(str(c), styles["NormalJustify"]) for c in r])
    table = Table(table_data, colWidths=col_widths)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#800020")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "TOP"),
    ]))
    return [header, table, Spacer(1, 12)]

def make_image_section(title, img_b64, styles, text=None):
    section = [Paragraph(title, styles["SectionHeader"])]
    if img_b64:
        img_bytes = base64.b64decode(img_b64)
        img = Image(io.BytesIO(img_bytes))
        img._restrictSize(14*cm, 8*cm)
        section.append(img)
        section.append(Spacer(1, 6))
    if text:
        section.append(Paragraph(text.replace("\n","<br/>"), styles["NormalJustify"]))
        section.append(Spacer(1, 12))
    return section

def plot_gp_health(gp_health_history):
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(gp_health_history, lw=2, color="#800080")
    ax.set_title("GP Health Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("GP Health")
    img_b64 = fig_to_base64(fig)
    plt.close(fig)
    return img_b64

def plot_gp_sigma(history_sigma):
    if history_sigma is None or len(history_sigma)==0:
        return None
    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(history_sigma, lw=2, color="#FF8000")
    ax.set_title("GP Prediction StdDev (Sigma) Over Iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sigma")
    img_b64 = fig_to_base64(fig)
    plt.close(fig)
    return img_b64

def plot_pca_candidates(history_X):
    X_arr = np.array(history_X)
    if X_arr.ndim == 2 and X_arr.shape[1] > 1:
        pca = PCA(n_components=2)
        Xp = pca.fit_transform(X_arr)
        fig, ax = plt.subplots(figsize=(5,4))
        sc = ax.scatter(Xp[:,0], Xp[:,1], c=np.arange(len(Xp)), cmap="plasma", s=40)
        ax.set_title("PCA Candidate Cloud")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    else:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.plot(X_arr, np.arange(len(X_arr)), color="#800020")
        ax.set_title("1D Input Exploration")
        ax.set_xlabel("Input")
        ax.set_ylabel("Iteration")
    img_b64 = fig_to_base64(fig)
    plt.close(fig)
    return img_b64

# ---------- Main Report Function ----------
def generate_pro_report1(func_id, gp_health_history, acq_history, history_X, history_y,
                        best_results, save_dir, history_sigma=None):
    """
    Generates a professional PDF report summarizing the Bayesian Optimization run.

    Args:
        func_id: Function / experiment ID.
        gp_health_history: list of GP health metrics over iterations.
        acq_history: list of dicts, each containing acquisition function evaluations per iteration.
        history_X: list of explored candidate inputs.
        history_y: list of corresponding outputs.
        best_results: list of dicts (as returned from adaptive BBO), each with iteration info.
        save_dir: output folder.
        history_sigma: optional list of GP sigma values per iteration.
    """
    folder_name = make_date_folder()
    save_dir =    os.path.join(
        save_dir,  folder_name,"reports"
    )
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%b%d_%y_%H%M%S").lower()

    pdf_path = os.path.join(save_dir, f"function_{func_id}_pro_report{timestamp}.pdf")
    styles = build_styles()
    story = []
    if history_y is None or len(history_y) == 0:
        print(f"Warning: No evaluations for function {func_id}, skipping report generation")
        return



    # ---------------- Cover Page ----------------
    story.append(Paragraph(f"Function {func_id} — Weekly Optimization Report", styles["SectionHeader"]))
    
    # Determine best result from history_y
    if len(history_y) > 0:
        best_idx = np.argmax(history_y)
        best_output = history_y[best_idx]
        best_input = history_X[best_idx]
        best_iteration = best_idx + 1
    else:
        best_output, best_input, best_iteration = "N/A", "N/A", "N/A"

    story.append(Paragraph(
        f"<b>Best Output:</b> {np.round(best_output,7) if isinstance(best_output,float) else best_output}<br/>"
        f"<b>Best Input:</b> {np.round(best_input,7) if hasattr(best_input,'__iter__') else best_input}<br/>"
        f"<b>Achieved at Iteration:</b> {best_iteration}<br/>",
        styles["NormalJustify"]
    ))
    story.append(PageBreak())

    # ---------------- GP Health Section ----------------
    gp_img = plot_gp_health(gp_health_history)
    story += make_image_section("GP Health Evolution", gp_img, styles,
                                "Higher GP health indicates a more stable model. Drops may indicate noise, kernel mismatch, or extrapolation.")

    rows = [["Metric","Value"],
            ["Min GP Health", f"{np.min(gp_health_history):.3f}" if len(gp_health_history)>0 else "N/A"],
            ["Max GP Health", f"{np.max(gp_health_history):.3f}" if len(gp_health_history)>0 else "N/A"],
            ["Final GP Health", f"{gp_health_history[-1]:.3f}" if len(gp_health_history)>0 else "N/A"]]
    story += make_table("GP Diagnostic Summary", rows, styles, col_widths=[5*cm,5*cm])

    # ---------------- GP Sigma Section ----------------
    sigma_img = plot_gp_sigma(history_sigma)
    if sigma_img:
        story += make_image_section("GP Prediction Uncertainty (Sigma)", sigma_img, styles,
                                    "Higher sigma indicates more uncertainty in predictions, guiding exploration vs exploitation.")

    story.append(PageBreak())

    # ---------------- Acquisition Section ----------------
    if acq_history and isinstance(acq_history,list):
        last_acq = acq_history[-1]
        acq_values = last_acq.get("acq_values",{})
        rows = [["Acquisition","Max Value Last Iter"]]
        for acq_name, vals in acq_values.items():
            rows.append([acq_name, f"{np.max(vals):.5f}" if len(vals)>0 else "N/A"])
        story += make_table("Acquisition Final Snapshot", rows, styles)
        best_acq_name = max(acq_values, key=lambda k: np.max(acq_values[k]) if len(acq_values[k])>0 else -np.inf)
        story.append(Paragraph(f"<b>Best Acquisition Function:</b> {best_acq_name}", styles["NormalJustify"]))

    story.append(PageBreak())

    # ---------------- PCA Candidate Plot ----------------
    pca_img = plot_pca_candidates(history_X)
    story += make_image_section("PCA Candidate Map", pca_img, styles,
                                "Shows clustering of explored regions. Dense clusters = exploitation; spread = exploration.")

    story.append(PageBreak())

    # ---------------- Best Results Summary ----------------
    rows = [["Iteration","Best Input","Best Output","Kernel","GP Health","Acquisition"]]
    for res in best_results:
        iteration = res.get("iteration","N/A")
        inp = res.get("best_input","N/A")
        outp = res.get("best_output","N/A")
        kernel = res.get("kernel","N/A")
        gp_h = np.round(res.get("gp_health",-1),3) if res.get("gp_health") is not None else "N/A"
        acq = res.get("acquisition","N/A")
        rows.append([iteration, np.round(inp,4) if hasattr(inp,"__iter__") else inp, 
                     np.round(outp,5) if isinstance(outp,float) else outp, kernel, gp_h, acq])
    story += make_table("All Iterations Summary", rows, styles, col_widths=[2*cm,4*cm,3*cm,3*cm,2*cm,3*cm])

    # ---------------- Explanations ----------------
    explanation = (
        "Interpretation:\n"
        "- GP Health close to 1.0 indicates a well-conditioned model; values below 0.6 suggest kernel mismatch or high noise.\n"
        "- Sigma shows prediction uncertainty; high values indicate regions worth exploring.\n"
        "- Acquisition function guides the next candidate selection; highest value indicates best balance between exploration and exploitation.\n"
        "- Best output shows the current optimum found; compare across iterations to assess improvement.\n\n"
        "Recommended next steps:\n"
        "- Consider exploring high-sigma regions to reduce uncertainty.\n"
        "- Review iterations with low GP health for potential kernel adjustments.\n"
        "- Track acquisition function trends to ensure diverse exploration."
    )
    story.append(Paragraph(explanation.replace("\n","<br/>"), styles["NormalJustify"]))

    # ---------------- Build PDF with flattening ----------------
    flat_story = []
    for item in story:
        if isinstance(item, list):
            flat_story.extend(item)
        else:
            flat_story.append(item)

    pdf = SimpleDocTemplate(pdf_path, pagesize=A4)
    pdf.build(flat_story)
    print("Professional report generated:", pdf_path)
    return pdf_path
# working one above testing new one below


def generate_pro_report(function_id,
                        gp_health_history,
                        iteration_acq_history,
                        history_X,
                        history_y,
                        best_results,
                        save_dir="reports",
                        history_sigma=None):

    # ---------- Setup ----------
    folder_name = make_date_folder()
    save_dir = os.path.join(save_dir, folder_name, "reports")
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%b%d_%y_%H%M%S").lower()
    report_file = os.path.join(save_dir, f"function_{function_id}_report{timestamp}.pdf")

    styles = getSampleStyleSheet()
    normal = styles["BodyText"]

    # Wrapped text style
    wrap_style = ParagraphStyle(
        "wrap",
        parent=styles["BodyText"],
        fontSize=9,
        leading=11
    )

    doc = SimpleDocTemplate(report_file, pagesize=A4)
    elements = []

    # ---------- Title ----------
    elements.append(Paragraph(f"Bayesian Optimization Report: Function {function_id}", styles['Title']))
    elements.append(Spacer(1, 18))

    # ======================================================================
    #   GP HEALTH SUMMARY
    # ======================================================================
    elements.append(Paragraph("GP Health Summary per Iteration", styles['Heading2']))

    table_data = [["Iteration", "GP Health", "Status"]]

    for i, gh in enumerate(gp_health_history):
        status = "Good" if gh > 0.7 else "Medium" if gh > 0.5 else "Low"

        table_data.append([
            Paragraph(str(i+1), wrap_style),
            Paragraph(f"{gh:.3f}", wrap_style),
            Paragraph(status, wrap_style)
        ])

    table = Table(table_data, colWidths=[2*cm, 3*cm, 3*cm])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 18))

    # ======================================================================
    #   GP SIGMA SUMMARY
    # ======================================================================
    if history_sigma:
        elements.append(Paragraph("GP Sigma Summary", styles['Heading2']))

        sigma_table = [["Iteration", "Mean Sigma", "Min Sigma", "Max Sigma"]]

        for i, s in enumerate(history_sigma):
            s = np.array(s)
            sigma_table.append([
                Paragraph(str(i+1), wrap_style),
                Paragraph(f"{s.mean():.4f}", wrap_style),
                Paragraph(f"{s.min():.4f}", wrap_style),
                Paragraph(f"{s.max():.4f}", wrap_style),
            ])

        table = Table(sigma_table, colWidths=[2*cm, 3*cm, 3*cm, 3*cm])
        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 18))

    # ======================================================================
    #   BEST RESULTS
    # ======================================================================
    elements.append(Paragraph("Best Results Per Iteration", styles['Heading2']))

    best_table = [["Iteration", "Best Input", "Best Output",
                   "Kernel", "Acquisition", "GP Health", "Anomalies"]]

    for br, iah in zip(best_results, iteration_acq_history):

        anomalies = iah.get("anomalies", [])
        anomalies_str = ", ".join([str(a) for a in anomalies]) if anomalies else "None"

        best_table.append([
            Paragraph(str(br["iteration"]), wrap_style),
            Paragraph(str(br["best_input"]), wrap_style),
            Paragraph(str(br["best_output"]), wrap_style),
            Paragraph(str(br.get("kernel", "N/A")), wrap_style),
            Paragraph(str(br.get("acquisition", "N/A")), wrap_style),
            Paragraph(f"{br.get('gp_health', 0):.3f}", wrap_style),
            Paragraph(anomalies_str, wrap_style),
        ])

    table = Table(best_table,
                  colWidths=[1.8*cm, 4.2*cm, 2.7*cm, 4.2*cm, 3*cm, 2*cm, 3.5*cm])

    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,0), "CENTER"),
        ("LEFTPADDING", (0,0), (-1,-1), 4),
        ("RIGHTPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 18))

    # ======================================================================
    #   ACQUISITION VALUES
    # ======================================================================
    elements.append(Paragraph("Acquisition Values Per Iteration", styles['Heading2']))

    for iah in iteration_acq_history:
        elements.append(Paragraph(f"Iteration {iah['iteration']}", styles['Heading3']))

        acq_table = [["Method", "Max Value"]]

        for acq_name, values in iah["acq_values"].items():
            acq_table.append([
                Paragraph(acq_name, wrap_style),
                Paragraph(f"{np.max(values):.6f}", wrap_style)
            ])

        table = Table(acq_table, colWidths=[4*cm, 4*cm])

        table.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("LEFTPADDING", (0,0), (-1,-1), 4),
            ("RIGHTPADDING", (0,0), (-1,-1), 4),
            ("TOPPADDING", (0,0), (-1,-1), 4),
            ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ]))

        elements.append(table)
        elements.append(Spacer(1, 10))

    # ---------- Save PDF ----------
    doc.build(elements)
    print(f"Report saved: {report_file}")
