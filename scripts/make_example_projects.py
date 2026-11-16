"""Generate two fictitious AI project briefs (PDF) to test EU AI Compliance Auditor."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "examples")
SS = getSampleStyleSheet()
SS.add(ParagraphStyle("Cover", parent=SS["Title"], fontSize=26, leading=32, spaceAfter=10))
SS.add(ParagraphStyle("Sub", parent=SS["Heading2"], textColor=colors.HexColor("#555555"), spaceAfter=24))
SS.add(ParagraphStyle("B", parent=SS["BodyText"], fontSize=10, leading=14, spaceAfter=6))
SS.add(ParagraphStyle("Cell", parent=SS["BodyText"], fontSize=8.5, leading=11))
SS.add(
    ParagraphStyle(
        "CellH",
        parent=SS["BodyText"],
        fontSize=8.5,
        leading=11,
        textColor=colors.white,
        fontName="Helvetica-Bold",
    )
)
WIDTH = A4[0] - 4 * cm
FOOTER = "Fictitious project - created to test EU AI Compliance Auditor. Any resemblance to real companies is coincidental."


def table(rows: list[list[str]], widths: list[float], accent: str) -> Table:
    data = [[Paragraph(c, SS["CellH"] if i == 0 else SS["Cell"]) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(accent)),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0bec5")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
            ]
        )
    )
    return t


def bullets(items: list[str]) -> list[Paragraph]:
    return [Paragraph(f"&bull; {i}", SS["B"]) for i in items]


def build(path: Path, spec: dict[str, Any]) -> None:
    accent = spec["accent"]
    h1 = ParagraphStyle("H1x", parent=SS["Heading1"], textColor=colors.HexColor(accent))

    def footer(canvas: Any, doc: Any) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(2 * cm, 1.2 * cm, FOOTER)
        canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"{spec['short']} - page {doc.page}")
        canvas.restoreState()

    story: list[Any] = [
        Spacer(1, 3 * cm),
        Paragraph(spec["title"], SS["Cover"]),
        Paragraph(spec["subtitle"], SS["Sub"]),
        table([["Item", "Value"], *spec["cover"]], [5, 12], accent),
        PageBreak(),
    ]
    for heading, blocks in spec["sections"]:
        story.append(Paragraph(heading, h1))
        for kind, content in blocks:
            if kind == "p":
                story.append(Paragraph(content, SS["B"]))
            elif kind == "ul":
                story += bullets(content)
            elif kind == "table":
                rows, widths = content
                story.append(table(rows, widths, accent))
                story.append(Spacer(1, 0.3 * cm))
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=spec["title"],
        author=spec["company"],
        subject="AI project brief",
    )
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


# ---------------------------------------------------------------------------- high-risk / prohibited
WORKPULSE = {
    "short": "WorkPulse 360",
    "accent": "#8e1b1b",
    "company": "Nexora Analytics SAS",
    "title": "WorkPulse 360",
    "subtitle": "Workforce Intelligence Platform - Product brief for the EU launch",
    "cover": [
        ["Company", "Nexora Analytics SAS (Lyon, France) - software vendor"],
        ["Our role", "We develop WorkPulse 360 and sell it under our own name to employers (SaaS)"],
        [
            "Target customers",
            "Employers in France, Germany, Spain and Italy (logistics, call centres, retail)",
        ],
        ["Document version", "v2.3 - Go-to-market review, November 2026"],
        ["Planned EU launch", "January 2027"],
    ],
    "sections": [
        (
            "1. Executive summary",
            [
                (
                    "p",
                    "WorkPulse 360 gives HR departments a continuous, AI-driven view of every employee's engagement, "
                    "loyalty and performance. By combining webcam and microphone analytics, device monitoring and "
                    "external data sources, the platform computes a single <b>Trust Score</b> for each employee and "
                    "automatically recommends promotions, warnings and dismissals.",
                ),
                (
                    "p",
                    "Pilot customers reduced 'low-engagement headcount' by 18% in six months by acting directly on the "
                    "automatic recommendations.",
                ),
            ],
        ),
        (
            "2. Key features",
            [
                (
                    "ul",
                    [
                        "<b>Real-time emotion recognition</b>: during working hours, the employee's webcam and microphone are "
                        "analysed every 30 seconds to detect stress, frustration, boredom and enthusiasm. Emotional profiles are "
                        "shown to managers and stored in the employee's file.",
                        "<b>Biometric categorisation</b>: from face images, the model infers ethnic origin, age bracket and a "
                        "'likelihood of trade-union membership' used in the 'Labour Relations Risk' dashboard.",
                        "<b>Trust Score</b>: a 0-1000 score per employee aggregating productivity, emotional state, sick-leave "
                        "history, private social-media activity and off-work behaviour (gym attendance, nightlife check-ins).",
                        "<b>Automated HR decisions</b>: employees under 400 points are automatically flagged as 'dismissal "
                        "candidates'; the termination letter template is generated and sent to the employee after 10 days "
                        "unless a manager intervenes. Promotions are only offered to employees above 750 points.",
                        "<b>Productivity surveillance</b>: keystroke logging, screenshot every 5 minutes, and 24/7 location "
                        "tracking of the employee's personal smartphone through the mandatory WorkPulse app.",
                        "<b>Performance narratives</b>: a third-party general-purpose LLM (accessed via API) writes a monthly "
                        "'personality and loyalty assessment' for each employee.",
                    ],
                ),
            ],
        ),
        (
            "3. Technical architecture",
            [
                (
                    "table",
                    (
                        [
                            ["Component", "Technology", "Hosting"],
                            [
                                "Emotion & face analytics",
                                "Proprietary CNN trained on 2M scraped faces",
                                "US public cloud (us-east)",
                            ],
                            [
                                "Trust Score engine",
                                "Gradient boosting, retrained weekly",
                                "US public cloud (us-east)",
                            ],
                            [
                                "Narrative generation",
                                "Third-party general-purpose LLM via API",
                                "Provider's US data centres",
                            ],
                            [
                                "Data lake",
                                "All raw video, audio and screenshots",
                                "US public cloud, no encryption at rest",
                            ],
                            [
                                "Mobile agent",
                                "Android/iOS app (location every 2 min)",
                                "Employees' personal phones",
                            ],
                        ],
                        [4.5, 7, 5.5],
                    ),
                ),
            ],
        ),
        (
            "4. Data inventory",
            [
                (
                    "table",
                    (
                        [
                            ["Data", "Source", "Purpose", "Retention", "Shared with"],
                            [
                                "Webcam video and voice",
                                "Workstation",
                                "Emotion recognition",
                                "Indefinite (model training)",
                                "-",
                            ],
                            [
                                "Face images / biometric templates",
                                "Webcam, ID badge photo",
                                "Categorisation (ethnicity, union)",
                                "Indefinite",
                                "-",
                            ],
                            [
                                "Sick-leave reasons and medical certificates",
                                "HR system",
                                "Trust Score",
                                "Indefinite",
                                "Partner insurer",
                            ],
                            [
                                "Private social-media posts",
                                "Web scraping of public profiles",
                                "Trust Score (loyalty)",
                                "Indefinite",
                                "-",
                            ],
                            [
                                "24/7 GPS location",
                                "Personal smartphone",
                                "Off-work behaviour analysis",
                                "5 years",
                                "-",
                            ],
                            ["Keystrokes, screenshots", "Workstation agent", "Productivity", "5 years", "-"],
                            [
                                "Trust Score and HR flags",
                                "Computed",
                                "Promotion / dismissal",
                                "Indefinite",
                                "Partner insurer, recruiters",
                            ],
                        ],
                        [3.6, 3.2, 3.4, 3.2, 3.6],
                    ),
                ),
                (
                    "p",
                    "Aggregated Trust Scores are sold to a partner health insurer to adjust group-insurance premiums, "
                    "and to partner recruitment agencies as 'reference checks'.",
                ),
            ],
        ),
        (
            "5. Governance and legal position",
            [
                (
                    "ul",
                    [
                        "Legal basis: a consent clause is included in the employment contract template we provide to customers. "
                        "Employees who refuse cannot be hired.",
                        "No data protection impact assessment has been carried out: management considers it unnecessary "
                        "because employees have consented.",
                        "Employees are not informed of the emotion analysis 'to avoid biasing their behaviour'.",
                        "No Data Protection Officer has been appointed.",
                        "No standard contractual clauses are in place for transfers to the United States.",
                        "No human review of the automatic dismissal flags; managers can only see the score, not the reasons.",
                        "Requests for access or erasure are not supported: the Trust Score is 'proprietary information'.",
                        "No technical documentation, risk management or bias testing has been produced; the face model's error "
                        "rate on darker skin tones is unknown.",
                        "No AI literacy training for HR staff using the platform.",
                    ],
                ),
            ],
        ),
        (
            "6. Roadmap",
            [
                (
                    "ul",
                    [
                        "Q1 2027: EU launch with 40 customers (about 25,000 monitored employees).",
                        "Q2 2027: 'Pre-crime' module predicting which employees are likely to commit theft or fraud based on "
                        "their profile and emotional patterns.",
                        "Q3 2027: real-time voice-stress analysis in customer calls to rank call-centre agents.",
                    ],
                ),
            ],
        ),
    ],
}

# ---------------------------------------------------------------------------- limited risk / well governed
ATLAS = {
    "short": "Atlas Assist",
    "accent": "#1b5e20",
    "company": "Velocita Parts SARL",
    "title": "Atlas Assist",
    "subtitle": "AI customer-support assistant for an online bicycle-parts shop - Project brief",
    "cover": [
        ["Company", "Velocita Parts SARL (Bordeaux, France) - e-commerce, 45 employees"],
        ["Our role", "We build and operate the assistant ourselves on top of an open-weights language model"],
        ["Users", "Visitors and customers of velocita-parts.example (France, Belgium, Spain)"],
        ["Document version", "v1.4 - Pre-launch review, November 2026"],
        ["Planned launch", "February 2027"],
    ],
    "sections": [
        (
            "1. Executive summary",
            [
                (
                    "p",
                    "Atlas Assist is a chat assistant on our website that answers questions about products, "
                    "compatibility of parts, delivery times and the status of an order. It relies on retrieval-augmented "
                    "generation over our public catalogue and help-centre articles. It does not take any decision about "
                    "customers: refunds, returns and complaints are always handled by our human support team.",
                ),
                (
                    "p",
                    "Goal: answer 60% of the 1,200 weekly support requests instantly, 24/7, while improving the quality "
                    "of answers for the remaining requests handled by staff.",
                ),
            ],
        ),
        (
            "2. Key features",
            [
                (
                    "ul",
                    [
                        "Product and compatibility questions answered from the public catalogue (no personal data needed).",
                        "Order status lookup: the customer enters the order number and the e-mail address used for the order; "
                        "the assistant only displays the delivery status and tracking link.",
                        "Clear disclosure: the chat window opens with 'You are chatting with Atlas, an AI assistant. A human "
                        "advisor is available at any time.' and every answer carries an 'AI-generated' label.",
                        "'Talk to a human' button always visible; automatic hand-off when the assistant is not confident or "
                        "when the request concerns a refund, a complaint or a warranty claim.",
                        "No profiling, no product recommendations based on personal history, no emotion analysis, no "
                        "biometric data, no images or voice.",
                    ],
                ),
            ],
        ),
        (
            "3. Technical architecture",
            [
                (
                    "table",
                    (
                        [
                            ["Component", "Technology", "Hosting"],
                            [
                                "Language model",
                                "Open-weights 8B instruction model, self-hosted (no fine-tuning on customer data)",
                                "EU cloud provider, data centre in France",
                            ],
                            [
                                "Retrieval",
                                "Vector index of public catalogue and help-centre articles",
                                "Same EU data centre",
                            ],
                            [
                                "Order lookup",
                                "Read-only API to the order system (status and tracking only)",
                                "On-premise, Bordeaux",
                            ],
                            [
                                "Transcripts store",
                                "Encrypted database (AES-256), role-based access",
                                "EU cloud provider, France",
                            ],
                        ],
                        [4, 8, 5],
                    ),
                ),
                (
                    "p",
                    "Prompt-injection and data-leak tests were run before launch (120 adversarial scenarios). The model "
                    "has no access to payment data or to any customer data other than the order being looked up.",
                ),
            ],
        ),
        (
            "4. Data inventory",
            [
                (
                    "table",
                    (
                        [
                            ["Data", "Purpose", "Lawful basis", "Retention", "Recipients"],
                            [
                                "Chat messages",
                                "Answer the request",
                                "Contract (order questions) / legitimate interest (product questions)",
                                "30 days, then deleted",
                                "Support team",
                            ],
                            [
                                "Order number + e-mail (for lookup)",
                                "Verify the requester",
                                "Contract",
                                "Not stored after the session",
                                "-",
                            ],
                            [
                                "Pseudonymised transcripts",
                                "Improve help-centre articles",
                                "Legitimate interest (balancing test documented)",
                                "90 days",
                                "Support team leads",
                            ],
                            [
                                "Aggregated statistics",
                                "Service quality reporting",
                                "Not personal data",
                                "2 years",
                                "Management",
                            ],
                        ],
                        [3.3, 3.3, 4.2, 3, 3.2],
                    ),
                ),
                (
                    "p",
                    "No special categories of personal data are requested; a filter masks e-mail addresses, phone numbers "
                    "and card numbers that users type spontaneously before storage.",
                ),
            ],
        ),
        (
            "5. Governance and compliance",
            [
                (
                    "ul",
                    [
                        "Record of processing activities updated with the 'Atlas Assist' processing.",
                        "Privacy notice updated and linked in the chat window (purposes, lawful bases, retention, rights).",
                        "Data subject rights handled by the existing procedure (privacy@ address, answer within one month).",
                        "DPIA screening performed against the CNIL list and EDPB criteria: no high-risk criterion met, DPIA not "
                        "required - decision documented and reviewed by our external privacy counsel.",
                        "Data processing agreement (GDPR Art. 28) signed with the EU hosting provider; no transfer outside the EU.",
                        "Security: encryption in transit and at rest, access logging, quarterly access reviews, breach "
                        "notification procedure (72 hours) tested in October 2026.",
                        "Human oversight: support team leads review a random sample of 50 conversations per week; answers can "
                        "be corrected and the knowledge base updated.",
                        "AI literacy: the 12 support staff completed a half-day training on the assistant's capabilities, "
                        "limits and escalation rules.",
                        "AI Act self-assessment: the assistant interacts with natural persons (transparency obligation of "
                        "Article 50(1) applies); it does not fall under the Annex III high-risk areas and uses no prohibited practice.",
                    ],
                ),
            ],
        ),
        (
            "6. Roadmap and KPIs",
            [
                (
                    "ul",
                    [
                        "Launch February 2027 on the French site, then Belgium and Spain (French and Spanish languages).",
                        "KPIs: automatic resolution rate, hand-off rate, customer satisfaction (CSAT), share of wrong answers "
                        "found in the weekly review (target below 2%).",
                        "Annual review of the DPIA screening and of the AI Act self-assessment, or earlier if a new feature "
                        "(e.g. personalised recommendations) is added.",
                    ],
                ),
            ],
        ),
    ],
}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    build(OUT / "workpulse360_high_risk_project.pdf", WORKPULSE)
    build(OUT / "atlas_assist_compliant_project.pdf", ATLAS)
    print(f"Example projects written to {OUT}/")  # noqa: T201 - CLI output
