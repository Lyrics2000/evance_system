"""Shared ReportLab toolkit for the KSL analysis PDFs — consistent typography,
tables, figure blocks, callouts. Palette matches the charts."""
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, HRFlowable,
                                KeepTogether)
from reportlab.pdfgen import canvas as canvaslib
import os

INK = colors.HexColor("#0b0b0b")
SUB = colors.HexColor("#52514e")
BLUE = colors.HexColor("#2a78d6")
ORANGE = colors.HexColor("#eb6834")
AQUA = colors.HexColor("#1baf7a")
GREEN = colors.HexColor("#008300")
RED = colors.HexColor("#e34948")
VIOLET = colors.HexColor("#4a3aa7")
LIGHT = colors.HexColor("#f2f5fa")
LIGHTER = colors.HexColor("#fafbfc")
GRID = colors.HexColor("#d9dee6")
SURF = colors.HexColor("#fcfcfb")
FIG = "/home/claude/ksl/figs"

S = getSampleStyleSheet()
styles = {
    "title": ParagraphStyle("t", parent=S["Title"], fontName="Helvetica-Bold",
                            fontSize=21, leading=25, textColor=INK, spaceAfter=4),
    "subtitle": ParagraphStyle("st", fontName="Helvetica", fontSize=11.5,
                               leading=15, textColor=SUB, spaceAfter=2),
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=15, leading=19,
                         textColor=BLUE, spaceBefore=15, spaceAfter=7),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=12, leading=15,
                         textColor=INK, spaceBefore=10, spaceAfter=4),
    "body": ParagraphStyle("b", fontName="Helvetica", fontSize=9.9, leading=14.5,
                           textColor=INK, alignment=TA_JUSTIFY, spaceAfter=6),
    "bullet": ParagraphStyle("bl", fontName="Helvetica", fontSize=9.9, leading=14,
                             textColor=INK, leftIndent=13, bulletIndent=3, spaceAfter=3),
    "caption": ParagraphStyle("c", fontName="Helvetica-Oblique", fontSize=8.6,
                              leading=11.5, textColor=SUB, alignment=TA_CENTER,
                              spaceBefore=3, spaceAfter=10),
    "small": ParagraphStyle("sm", fontName="Helvetica", fontSize=8.6, leading=11.5,
                            textColor=SUB, spaceAfter=4),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=8.6, leading=11, textColor=INK),
    "cellb": ParagraphStyle("cellb", fontName="Helvetica-Bold", fontSize=8.6, leading=11, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="Helvetica-Bold", fontSize=8.6, leading=11, textColor=colors.white),
    "mono": ParagraphStyle("m", fontName="Courier", fontSize=8.4, leading=12,
                           textColor=INK, backColor=LIGHT, borderPadding=6, spaceAfter=6),
    "kicker": ParagraphStyle("k", fontName="Helvetica-Bold", fontSize=8.5, leading=11,
                             textColor=BLUE, spaceAfter=2),
}


def P(t, s="body"): return Paragraph(t, styles[s])
def H1(t): return Paragraph(t, styles["h1"])
def H2(t): return Paragraph(t, styles["h2"])
def sp(h=6): return Spacer(1, h)
def rule(c=GRID, w=0.8): return HRFlowable(width="100%", thickness=w, color=c,
                                           spaceBefore=6, spaceAfter=6)

def bullets(items, style="bullet"):
    return [Paragraph(f"•&nbsp;&nbsp;{it}", styles[style]) for it in items]


def caption(t): return Paragraph(t, styles["caption"])


def figure(name, cap, width=165*mm):
    path = os.path.join(FIG, name)
    from PIL import Image as PILImage
    iw, ih = PILImage.open(path).size
    h = width * ih / iw
    return KeepTogether([Image(path, width=width, height=h), caption(cap)])


def table(rows, col_widths, header=True, zebra=True, highlight_rows=None,
          align="LEFT", font=8.6, header_color=BLUE):
    """rows: list of list of strings (plain). First row header if header=True."""
    highlight_rows = highlight_rows or {}
    data = []
    for r, row in enumerate(rows):
        line = []
        for c, cell in enumerate(row):
            if header and r == 0:
                st = "cellh"
            elif c in highlight_rows.get(r, []):
                st = "cellb"
            else:
                st = "cell"
            line.append(Paragraph(str(cell), styles[st]))
        data.append(line)
    t = Table(data, colWidths=col_widths, repeatRows=1 if header else 0)
    ts = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
          ("ALIGN", (1, 0), (-1, -1), align),
          ("ALIGN", (0, 0), (0, -1), "LEFT"),
          ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
          ("TOPPADDING", (0, 0), (-1, -1), 3.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
          ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID)]
    if header:
        ts += [("BACKGROUND", (0, 0), (-1, 0), header_color),
               ("ALIGN", (0, 0), (-1, 0), "LEFT")]
    if zebra:
        for r in range(1, len(rows)):
            if r % 2 == 0:
                ts.append(("BACKGROUND", (0, r), (-1, r), LIGHTER))
    for r, cols in (highlight_rows or {}).items():
        ts.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#fff5ec")))
    t.setStyle(TableStyle(ts))
    return t


def callout(title, body_html, accent=BLUE, bg=LIGHT):
    inner = [Paragraph(title, ParagraphStyle("ct", fontName="Helvetica-Bold",
                        fontSize=9.4, leading=12, textColor=accent, spaceAfter=3)),
             Paragraph(body_html, ParagraphStyle("cb", fontName="Helvetica",
                        fontSize=9.2, leading=13.5, textColor=INK))]
    t = Table([[inner]], colWidths=[165*mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), bg),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10),
                           ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                           ("TOPPADDING", (0, 0), (-1, -1), 8),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                           ("LINEBEFORE", (0, 0), (0, -1), 3, accent)]))
    return KeepTogether([t, sp(4)])


def code(text):
    return Table([[Paragraph(text.replace(" ", "&nbsp;").replace("\n", "<br/>"),
                   styles["mono"])]], colWidths=[165*mm],
                 style=TableStyle([("BACKGROUND", (0,0),(-1,-1), LIGHT),
                                   ("LEFTPADDING",(0,0),(-1,-1),8),
                                   ("TOPPADDING",(0,0),(-1,-1),6),
                                   ("BOTTOMPADDING",(0,0),(-1,-1),6),
                                   ("LINEBEFORE",(0,0),(0,-1),3,GRID)]))


class NumberedDoc(SimpleDocTemplate):
    def __init__(self, path, footer="", **kw):
        super().__init__(path, pagesize=A4, topMargin=20*mm, bottomMargin=18*mm,
                         leftMargin=22*mm, rightMargin=22*mm, **kw)
        self.footer = footer

    def afterPage(self):
        c = self.canv
        c.saveState()
        c.setStrokeColor(GRID); c.setLineWidth(0.5)
        c.line(22*mm, 14*mm, A4[0]-22*mm, 14*mm)
        c.setFont("Helvetica", 7.5); c.setFillColor(SUB)
        c.drawString(22*mm, 9*mm, self.footer)
        c.drawRightString(A4[0]-22*mm, 9*mm, f"Page {c.getPageNumber()}")
        c.restoreState()


def titleblock(title, subtitle, meta_lines, accent=BLUE):
    els = []
    bar = Table([[""]], colWidths=[165*mm], rowHeights=[4])
    bar.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), accent)]))
    els += [bar, sp(8), Paragraph(title, styles["title"]),
            Paragraph(subtitle, styles["subtitle"]), sp(6)]
    mt = Table([[Paragraph(m, styles["small"])] for m in meta_lines], colWidths=[165*mm])
    mt.setStyle(TableStyle([("LEFTPADDING",(0,0),(-1,-1),0),("TOPPADDING",(0,0),(-1,-1),0),
                            ("BOTTOMPADDING",(0,0),(-1,-1),1)]))
    els += [mt, rule(accent, 1.4), sp(4)]
    return els
