
from pathlib import Path
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_fraud_report(transactions):

    report_path = OUTPUT_DIR / "fraud_detection_report.pdf"

    total = len(transactions)
    fraud_count = sum(
        1 for row in transactions
        if str(row[5]).lower() == "fraud"
    )
    normal_count = total - fraud_count

    high_risk = sum(
        1 for row in transactions
        if str(row[6]).lower() == "high risk"
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#123B68"),
        fontSize=22,
        spaceAfter=15
    )

    subtitle_style = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        textColor=colors.HexColor("#555555"),
        fontSize=10,
        spaceAfter=12
    )

    document = SimpleDocTemplate(
        str(report_path),
        pagesize=A4,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch
    )

    content = []

    content.append(
        Paragraph("SENTINEL AI", title_style)
    )

    content.append(
        Paragraph(
            "Intelligent Financial Fraud Detection and Prevention System",
            subtitle_style
        )
    )

    content.append(
        Paragraph(
            f"Report Generated: {datetime.now():%d-%m-%Y %H:%M:%S}",
            styles["Normal"]
        )
    )

    content.append(Spacer(1, 25))

    content.append(
        Paragraph("Transaction Summary", styles["Heading2"])
    )

    summary_data = [
        ["Metric", "Result"],
        ["Total Transactions", str(total)],
        ["Fraudulent Transactions", str(fraud_count)],
        ["Normal Transactions", str(normal_count)],
        ["High Risk Transactions", str(high_risk)]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[3.5 * inch, 2 * inch]
    )

    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#123B68")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")
    ]))

    content.append(summary_table)
    content.append(Spacer(1, 25))

    content.append(
        Paragraph("Recent Transaction Records", styles["Heading2"])
    )

    table_data = [[
        "ID", "Amount", "Probability", "Prediction", "Risk"
    ]]

    for row in transactions[:30]:
        table_data.append([
            str(row[0]),
            f"{float(row[2]):.2f}",
            f"{float(row[3]) * 100:.2f}%",
            str(row[5]),
            str(row[6])
        ])

    records_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            0.45 * inch,
            1.0 * inch,
            1.0 * inch,
            1.1 * inch,
            1.1 * inch
        ]
    )

    records_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0),
         colors.HexColor("#123B68")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE")
    ]))

    content.append(records_table)
    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            "Note: Fraud predictions are risk indicators and should be "
            "reviewed before making financial decisions.",
            styles["Italic"]
        )
    )

    document.build(content)

    return report_path
