"""Shared GLDB (Green Link Digital Bank) template helpers for python-pptx.

Reproduces the corporate template: navy + orange brand, white content slides
with a navy title top-left and logo top-right, light-grey rounded cards, numbered
circles, orange section dividers, and a navy footer rule.

Import this from the deck builders:
    from gldb_theme import *
"""
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

# ---- GLDB brand palette ----
NAVY   = RGBColor(0x17, 0x45, 0x8F)   # deep brand blue
NAVY_D = RGBColor(0x0F, 0x3A, 0x7A)
ORANGE = RGBColor(0xE8, 0x82, 0x0E)   # brand orange
CARD   = RGBColor(0xEE, 0xF1, 0xF7)   # light grey-blue card fill
INK    = RGBColor(0x33, 0x33, 0x33)   # body text
GREY   = RGBColor(0x8A, 0x90, 0x9C)   # footer / muted
WHITE  = RGBColor(0xFF, 0xFF, 0xFF)
GREEN  = RGBColor(0x2E, 0x8B, 0x57)   # occasional positive accent
RED    = RGBColor(0xC0, 0x39, 0x2B)   # occasional risk accent
LINE   = RGBColor(0xD9, 0xDE, 0xE7)

HEAD_FONT = "Arial"
BODY_FONT = "Calibri"

FOOTER = "GLDB  |  Green Link Digital Bank"


def new_prs():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    return prs


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _bg(s, prs, color):
    r = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    r.fill.solid(); r.fill.fore_color.rgb = color; r.line.fill.background()
    r.shadow.inherit = False
    sp = r._element; sp.getparent().remove(sp); s.shapes._spTree.insert(2, sp)
    return r


def R(txt, size=14, color=INK, bold=False, italic=False, font=BODY_FONT):
    return (txt, size, color, bold, italic, font)


def text(s, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
         space_after=4, line_spacing=1.0, wrap=True):
    tb = s.shapes.add_textbox(x, y, w, h); tf = tb.text_frame
    tf.word_wrap = wrap; tf.vertical_anchor = anchor
    tf.margin_left = Pt(2); tf.margin_right = Pt(2)
    tf.margin_top = Pt(1); tf.margin_bottom = Pt(1)
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after); p.line_spacing = line_spacing
        for (t, size, color, bold, italic, font) in para:
            rr = p.add_run(); rr.text = t
            rr.font.size = Pt(size); rr.font.color.rgb = color
            rr.font.bold = bold; rr.font.italic = italic; rr.font.name = font
    return tb


def box(s, x, y, w, h, fill=None, line=None, line_w=0.75, radius=False):
    shp = s.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None: shp.fill.background()
    else: shp.fill.solid(); shp.fill.fore_color.rgb = fill
    if line is None: shp.line.fill.background()
    else: shp.line.color.rgb = line; shp.line.width = Pt(line_w)
    shp.shadow.inherit = False
    return shp


def circle(s, x, y, d, fill, number=None, num_color=WHITE, size=18):
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, x, y, d, d)
    c.fill.solid(); c.fill.fore_color.rgb = fill; c.line.fill.background()
    c.shadow.inherit = False
    if number is not None:
        text(s, x, y, d, d, [[R(str(number), size, num_color, True, font=HEAD_FONT)]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    return c


def _logo(s, x, y, scale=1.0):
    """Simplified GLDB 'G' mark: orange ring with a gap + inner orange shape."""
    d = Inches(0.62 * scale)
    ring = s.shapes.add_shape(MSO_SHAPE.BLOCK_ARC, x, y, d, d)
    try:
        ring.fill.solid(); ring.fill.fore_color.rgb = ORANGE; ring.line.fill.background()
        ring.shadow.inherit = False
    except Exception:
        pass
    # inner wedge
    inner = s.shapes.add_shape(MSO_SHAPE.OVAL, x + d*0.28, y + d*0.28, d*0.44, d*0.44)
    inner.fill.solid(); inner.fill.fore_color.rgb = ORANGE; inner.line.fill.background()
    inner.shadow.inherit = False


def logo_mark(s, prs):
    """Top-right brand mark used on content slides."""
    _logo(s, prs.slide_width - Inches(1.15), Inches(0.35), scale=1.1)


def title_slide(prs, line1, line2=None, subtitle=None):
    s = _blank(prs); _bg(s, prs, NAVY)
    # GLDB wordmark top-left
    text(s, Inches(0.7), Inches(0.5), Inches(4), Inches(0.9),
         [[R("GLDB", 34, WHITE, True, font=HEAD_FONT)]])
    # big orange logo right
    big = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(8.6), Inches(1.4), Inches(3.6), Inches(3.6))
    big.fill.background(); big.line.color.rgb = ORANGE; big.line.width = Pt(26)
    big.shadow.inherit = False
    wedge = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(9.55), Inches(2.35), Inches(1.7), Inches(1.7))
    wedge.fill.solid(); wedge.fill.fore_color.rgb = ORANGE; wedge.line.fill.background()
    wedge.shadow.inherit = False
    # title
    runs = [[R(line1, 44, WHITE, True, font=HEAD_FONT)]]
    if line2:
        runs.append([R(line2, 44, WHITE, True, font=HEAD_FONT)])
    text(s, Inches(0.7), Inches(2.7), Inches(7.3), Inches(1.8), runs, line_spacing=1.0)
    if subtitle:
        text(s, Inches(0.72), Inches(4.7), Inches(7.3), Inches(1.0),
             [[R(subtitle, 16, RGBColor(0xC9, 0xD3, 0xE6), False)]])
    return s


def section_divider(prs, number, title):
    s = _blank(prs); _bg(s, prs, ORANGE)
    text(s, prs.slide_width - Inches(3.0), Inches(0.5), Inches(2.3), Inches(0.9),
         [[R("GLDB", 32, WHITE, True, font=HEAD_FONT)]], align=PP_ALIGN.RIGHT)
    text(s, Inches(0.75), Inches(2.3), Inches(6), Inches(1.4),
         [[R(number, 60, NAVY, True, font=HEAD_FONT)]])
    text(s, Inches(0.78), Inches(4.0), Inches(11), Inches(1.2),
         [[R(title, 40, WHITE, True, font=HEAD_FONT)]])
    return s


def content_slide(prs, title, two_line=False):
    s = _blank(prs); _bg(s, prs, WHITE)
    # title (navy, top-left) - supports 1 or 2 lines
    th = Inches(1.35) if two_line else Inches(0.95)
    text(s, Inches(0.7), Inches(0.45), Inches(10.3), th,
         [[R(title, 30, NAVY, True, font=HEAD_FONT)]], line_spacing=1.0)
    logo_mark(s, prs)
    return s


def footer(s, prs, page):
    text(s, Inches(0.7), prs.slide_height - Inches(0.5), Inches(7), Inches(0.35),
         [[R(FOOTER, 10, GREY, False)]])
    text(s, prs.slide_width - Inches(1.2), prs.slide_height - Inches(0.5), Inches(0.6), Inches(0.35),
         [[R(str(page), 10, GREY, False)]], align=PP_ALIGN.RIGHT)


def closing_slide(prs, big="Thank you", contact=None):
    s = _blank(prs); _bg(s, prs, NAVY)
    ring = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.7), Inches(0.9), Inches(1.5), Inches(1.5))
    ring.fill.background(); ring.line.color.rgb = ORANGE; ring.line.width = Pt(12)
    ring.shadow.inherit = False
    wedge = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(1.1), Inches(1.3), Inches(0.72), Inches(0.72))
    wedge.fill.solid(); wedge.fill.fore_color.rgb = ORANGE; wedge.line.fill.background()
    wedge.shadow.inherit = False
    text(s, Inches(0.7), Inches(3.0), Inches(10), Inches(1.4),
         [[R(big, 46, WHITE, True, font=HEAD_FONT)]])
    if contact:
        text(s, Inches(0.72), Inches(4.55), Inches(10), Inches(0.6),
             [[R(contact, 15, RGBColor(0xC9, 0xD3, 0xE6), False)]])
    return s


# ---- reusable content components in GLDB style ----

def card(s, x, y, w, h, heading=None, body=None, body_size=12, heading_color=NAVY,
         fill=CARD, head_size=16):
    box(s, x, y, w, h, fill=fill, radius=True)
    pad = Inches(0.25); cy = y + pad
    if heading:
        text(s, x + pad, cy, w - 2*pad, Inches(0.5), [[R(heading, head_size, heading_color, True, font=HEAD_FONT)]])
        cy += Inches(0.62)
    if body:
        paras = body if isinstance(body, list) else [body]
        runs = [[R(p, body_size, INK, False)] for p in paras]
        text(s, x + pad, cy, w - 2*pad, y + h - cy - pad, runs, space_after=5, line_spacing=1.1)


def numbered_card(s, x, y, w, h, number, number_fill, heading, body, body_size=12):
    box(s, x, y, w, h, fill=CARD, radius=True)
    d = Inches(0.8)
    circle(s, x + Inches(0.3), y + Inches(0.35), d, number_fill, number=number, size=18)
    text(s, x + Inches(0.3), y + Inches(1.35), w - Inches(0.6), Inches(0.5),
         [[R(heading, 17, NAVY, True, font=HEAD_FONT)]])
    paras = body if isinstance(body, list) else [body]
    text(s, x + Inches(0.3), y + Inches(2.0), w - Inches(0.6), h - Inches(2.2),
         [[R(p, body_size, INK, False)] for p in paras], space_after=4, line_spacing=1.12)


def stat_card(s, x, y, w, h, value, label, highlight=False):
    fill = NAVY if highlight else CARD
    box(s, x, y, w, h, fill=fill, radius=True)
    vcolor = ORANGE if highlight else NAVY
    lcolor = WHITE if highlight else GREY
    text(s, x, y + Inches(0.5), w, Inches(1.0), [[R(value, 40, vcolor, True, font=HEAD_FONT)]],
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    text(s, x, y + h - Inches(1.0), w, Inches(0.6), [[R(label, 13, lcolor, False)]],
         align=PP_ALIGN.CENTER)


def pill_header(s, x, y, w, txt, fill=NAVY, color=WHITE):
    box(s, x, y, w, Inches(0.62), fill=fill, radius=True)
    text(s, x + Inches(0.25), y, w - Inches(0.4), Inches(0.62),
         [[R(txt, 16, color, True, font=HEAD_FONT)]], anchor=MSO_ANCHOR.MIDDLE)


def table_row(s, x, y, widths, cells, header=False, zebra=False):
    h = Inches(0.72)
    if header:
        box(s, x, y, sum(widths, Emu(0)), h, fill=NAVY)
    elif zebra:
        box(s, x, y, sum(widths, Emu(0)), h, fill=CARD)
    cx = x
    for i, (txt, w) in enumerate(zip(cells, widths)):
        col = WHITE if header else INK
        bold = header
        text(s, cx + Inches(0.2), y, w - Inches(0.3), h,
             [[R(txt, 12.5 if not header else 13, col, bold, font=HEAD_FONT if header else BODY_FONT)]],
             anchor=MSO_ANCHOR.MIDDLE)
        cx += w
    return h
