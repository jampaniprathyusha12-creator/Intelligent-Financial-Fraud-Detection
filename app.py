from src.report_generator import generate_fraud_report
import base64
import html
import inspect
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ==========================================================
# 1. PROJECT CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "src"))

from predict import predict_transaction  # noqa: E402
from database.database import (  # noqa: E402
    initialize_database, save_transaction, get_all_transactions
)

st.set_page_config(
    page_title="Sentinel AI | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Prepare the local SQLite database for transaction history.
initialize_database()

# Works on both old and new Streamlit versions
_BTN = {"width": "stretch"} if "width" in inspect.signature(st.button).parameters else {"use_container_width": True}
_PLOT = {"width": "stretch"} if "width" in inspect.signature(st.plotly_chart).parameters else {"use_container_width": True}

# ==========================================================
# 2. THEME (LIGHT / DARK)
# ==========================================================

PALETTES = {
    "dark": dict(
        bg="linear-gradient(135deg, #0A0E1F 0%, #12163A 48%, #2B1F5C 100%)",
        side="#0B0F23", card="rgba(20,24,52,0.78)", card_solid="#14183A", card2="#1B2047",
        line="rgba(255,255,255,0.08)", line2="rgba(255,255,255,0.22)",
        text="#E9ECFA", muted="#8F96BD", grid="rgba(255,255,255,0.07)",
        acc="#4F8CFF", acc2="#8B5CF6",
        high="#FF5C7A", medium="#F7B23B", low="#34D399",
        shadow="0 10px 34px rgba(0,0,0,0.35)", hover="rgba(255,255,255,0.05)",
    ),
    "light": dict(
        bg="linear-gradient(135deg, #F6F7FE 0%, #EEF0FC 52%, #E7E3FB 100%)",
        side="#FFFFFF", card="#FFFFFF", card_solid="#FFFFFF", card2="#F3F4FC",
        line="#E2E5F4", line2="#B9BFE0",
        text="#1B1F3B", muted="#6A7096", grid="#E8EAF5",
        acc="#3B6DF0", acc2="#7C4DEB",
        high="#E0364F", medium="#D98A06", low="#12A06C",
        shadow="0 8px 26px rgba(60,70,140,0.09)", hover="rgba(59,109,240,0.06)",
    ),
}

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

DARK = st.session_state.dark_mode
P = PALETTES["dark" if DARK else "light"]

BG, SIDE, CARD, CARD_SOLID, CARD2 = P["bg"], P["side"], P["card"], P["card_solid"], P["card2"]
LINE, LINE2, TEXT, MUTED, GRID = P["line"], P["line2"], P["text"], P["muted"], P["grid"]
ACC, ACC2, HIGH, MEDIUM, LOW = P["acc"], P["acc2"], P["high"], P["medium"], P["low"]
SHADOW, HOVER = P["shadow"], P["hover"]

RISK_COLORS = {"High Risk": HIGH, "Medium Risk": MEDIUM, "Low Risk": LOW}
RISK_ORDER = ["High Risk", "Medium Risk", "Low Risk"]


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def blend(c1, c2, t):
    a, b = c1.lstrip("#"), c2.lstrip("#")
    mix = [round(int(a[i:i + 2], 16) * (1 - t) + int(b[i:i + 2], 16) * t) for i in (0, 2, 4)]
    return "#" + "".join(f"{v:02x}" for v in mix)


# ==========================================================
# 3. ILLUSTRATIONS (inline SVG, no image files needed)
# ==========================================================

SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'


def svg_img(svg, width=None, alt=""):
    encoded = base64.b64encode(svg.strip().encode("utf-8")).decode("ascii")
    style = f' style="width:{width};"' if width else ""
    return f'<img src="data:image/svg+xml;base64,{encoded}" alt="{alt}"{style}/>'


def _tx_card(x, y, accent, amount, label):
    return (
        f'<g transform="translate({x} {y})">'
        f'<rect width="160" height="84" rx="14" fill="#171C46" stroke="{accent}" stroke-width="2"/>'
        f'<circle cx="26" cy="26" r="8" fill="{accent}"/>'
        '<rect x="42" y="21" width="58" height="9" rx="4.5" fill="#3B4580"/>'
        f'<text x="16" y="64" fill="#FFFFFF" font-size="18" font-weight="700">{amount}</text>'
        f'<rect x="88" y="48" width="58" height="22" rx="11" fill="{accent}"/>'
        f'<text x="117" y="63" fill="#0A0E1F" font-size="11" font-weight="700" text-anchor="middle">{label}</text>'
        '</g>'
    )


HERO_SVG = (
    f'<svg {SVG_NS} viewBox="0 0 520 380" font-family="Arial, Helvetica, sans-serif">'
    '<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="#4F8CFF"/><stop offset="1" stop-color="#8B5CF6"/></linearGradient></defs>'
    '<circle cx="260" cy="190" r="150" fill="#8B5CF6" opacity="0.12"/>'
    '<circle cx="260" cy="190" r="105" fill="#4F8CFF" opacity="0.16"/>'
    '<g stroke="#4A5599" stroke-width="2" stroke-dasharray="4 7" fill="none">'
    '<path d="M174 95 L204 140"/><path d="M346 90 L322 140"/>'
    '<path d="M166 290 L206 245"/><path d="M354 300 L316 250"/></g>'
    '<path d="M260 66 L346 98 V188 C346 252 306 292 260 314 C214 292 174 252 174 188 V98 Z" fill="url(#g)"/>'
    '<path d="M260 84 L330 109 V188 C330 241 298 276 260 295 C222 276 190 241 190 188 V109 Z" '
    'fill="#0F1438" opacity="0.35"/>'
    '<path d="M222 192 L250 220 L300 160" fill="none" stroke="#FFFFFF" stroke-width="14" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    + _tx_card(14, 38, "#34D399", "$42.10", "Normal")
    + _tx_card(346, 28, "#FF5C7A", "$9,870", "High risk")
    + _tx_card(6, 268, "#34D399", "$128.50", "Normal")
    + _tx_card(354, 278, "#F7B23B", "$560.00", "Review")
    + '</svg>'
)


def _icon(body):
    return (
        f'<svg {SVG_NS} viewBox="0 0 48 48" fill="none" stroke="{ACC}" stroke-width="3.5" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )


ICON_SCORE = _icon(
    '<path d="M7 33 A17 17 0 0 1 41 33"/><path d="M24 33 L32 21"/>'
    f'<circle cx="24" cy="33" r="2.5" fill="{ACC}"/>'
)
ICON_RISK = _icon(
    f'<rect x="8" y="9" width="32" height="8" rx="3" fill="{HIGH}" stroke="none"/>'
    f'<rect x="8" y="21" width="24" height="8" rx="3" fill="{MEDIUM}" stroke="none"/>'
    f'<rect x="8" y="33" width="16" height="8" rx="3" fill="{LOW}" stroke="none"/>'
)
ICON_ALERT = _icon(
    '<path d="M24 7 C17 7 13 12 13 19 V26 L9 32 H39 L35 26 V19 C35 12 31 7 24 7 Z"/>'
    '<path d="M20 38 C20 40.5 21.8 42 24 42 C26.2 42 28 40.5 28 38"/>'
)
ICON_REPORT = _icon(
    '<rect x="9" y="6" width="30" height="36" rx="4"/>'
    f'<rect x="15" y="26" width="4" height="10" rx="1" fill="{ACC}" stroke="none"/>'
    f'<rect x="22" y="20" width="4" height="16" rx="1" fill="{ACC2}" stroke="none"/>'
    f'<rect x="29" y="14" width="4" height="22" rx="1" fill="{ACC}" stroke="none"/>'
)
ICON_UPLOAD = _icon(
    '<path d="M14 32 A8 8 0 0 1 15 16 A10 10 0 0 1 34 18 A7 7 0 0 1 34 32"/>'
    '<path d="M24 41 V26"/><path d="M18 31 L24 25 L30 31"/>'
)


def _state_svg(color, inner):
    return (
        f'<svg {SVG_NS} viewBox="0 0 240 170">'
        f'<circle cx="120" cy="85" r="70" fill="{color}" opacity="0.14"/>'
        f'<circle cx="46" cy="52" r="5" fill="{color}" opacity="0.45"/>'
        f'<circle cx="198" cy="120" r="4" fill="{color}" opacity="0.45"/>'
        f'{inner}</svg>'
    )


EMPTY_SVG = _state_svg(
    ACC,
    f'<rect x="78" y="38" width="76" height="96" rx="8" fill="{CARD_SOLID}" stroke="{LINE2}" stroke-width="2"/>'
    f'<rect x="90" y="54" width="40" height="7" rx="3.5" fill="{LINE2}"/>'
    f'<rect x="90" y="70" width="52" height="7" rx="3.5" fill="{LINE2}"/>'
    f'<rect x="90" y="86" width="30" height="7" rx="3.5" fill="{LINE2}"/>'
    f'<circle cx="150" cy="104" r="22" fill="{CARD_SOLID}" fill-opacity="0.8" stroke="{ACC2}" stroke-width="5"/>'
    f'<path d="M166 120 L186 140" stroke="{ACC2}" stroke-width="7" stroke-linecap="round"/>',
)

CLEAR_SVG = _state_svg(
    LOW,
    f'<path d="M120 36 L160 51 V88 C160 118 142 138 120 148 C98 138 80 118 80 88 V51 Z" fill="{LOW}"/>'
    '<path d="M102 90 L116 104 L140 74" fill="none" stroke="#FFFFFF" stroke-width="8" '
    'stroke-linecap="round" stroke-linejoin="round"/>',
)


def _node(x, y):
    return f'<circle cx="{x}" cy="{y}" r="9" fill="{ACC2}"/>'


ABOUT_SVG = (
    f'<svg {SVG_NS} viewBox="0 0 760 170" font-family="Arial, Helvetica, sans-serif">'
    f'<rect x="40" y="25" width="70" height="90" rx="10" fill="{CARD_SOLID}" stroke="{LINE2}" stroke-width="2"/>'
    f'<rect x="54" y="42" width="30" height="7" rx="3.5" fill="{LINE2}"/>'
    f'<rect x="54" y="58" width="42" height="7" rx="3.5" fill="{LINE2}"/>'
    f'<rect x="54" y="74" width="26" height="7" rx="3.5" fill="{LINE2}"/>'
    f'<text x="75" y="104" fill="{ACC}" font-size="13" font-weight="700" text-anchor="middle">CSV</text>'
    f'<g stroke="{MUTED}" stroke-width="2.5" fill="none" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M130 70 H200"/><path d="M192 62 L200 70 L192 78"/>'
    '<path d="M400 70 H470"/><path d="M462 62 L470 70 L462 78"/>'
    '<path d="M570 70 H640"/><path d="M632 62 L640 70 L632 78"/></g>'
    f'<g stroke="{ACC}" stroke-width="2" opacity="0.55">'
    '<path d="M240 40 L300 55"/><path d="M240 40 L300 85"/><path d="M240 70 L300 55"/>'
    '<path d="M240 70 L300 85"/><path d="M240 100 L300 55"/><path d="M240 100 L300 85"/>'
    '<path d="M300 55 L360 70"/><path d="M300 85 L360 70"/></g>'
    + _node(240, 40) + _node(240, 70) + _node(240, 100)
    + _node(300, 55) + _node(300, 85) + _node(360, 70)
    + f'<rect x="490" y="35" width="70" height="14" rx="5" fill="{HIGH}"/>'
    f'<rect x="490" y="63" width="52" height="14" rx="5" fill="{MEDIUM}"/>'
    f'<rect x="490" y="91" width="34" height="14" rx="5" fill="{LOW}"/>'
    '<defs><linearGradient id="s" x1="0" y1="0" x2="1" y2="1">'
    f'<stop offset="0" stop-color="{ACC}"/><stop offset="1" stop-color="{ACC2}"/></linearGradient></defs>'
    '<path d="M690 28 L722 40 V74 C722 98 708 112 690 120 C672 112 658 98 658 74 V40 Z" fill="url(#s)"/>'
    '<path d="M677 74 L688 85 L706 62" fill="none" stroke="#FFFFFF" stroke-width="6" '
    'stroke-linecap="round" stroke-linejoin="round"/>'
    f'<g fill="{MUTED}" font-size="13" font-weight="600" text-anchor="middle">'
    '<text x="75" y="145">Transactions</text><text x="300" y="145">ML model</text>'
    '<text x="525" y="145">Risk levels</text><text x="690" y="145">Alerts</text></g>'
    '</svg>'
)

# ==========================================================
# 4. STYLES
# ==========================================================

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {{ font-family: 'Manrope', sans-serif; }}

.stApp {{ background: {BG}; background-attachment: fixed; color: {TEXT}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1320px; }}

h1, h2, h3, h4, label, .stApp li,
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"] p {{ color: {TEXT}; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: {MUTED}; }}
hr {{ border-color: {LINE}; }}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {{ background: {SIDE}; border-right: 1px solid {LINE}; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 4px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
    padding: 11px 14px; border-radius: 12px; width: 100%;
    transition: background 0.15s ease;
}}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: {HOVER}; }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
    background: linear-gradient(90deg, {rgba(ACC, 0.22)}, {rgba(ACC2, 0.22)});
    box-shadow: inset 0 0 0 1px {rgba(ACC, 0.45)};
}}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-weight: 600; color: {TEXT}; }}
.side-label {{ font-size: 12px; font-weight: 700; color: {MUTED}; margin: 6px 0 8px 2px; }}

.brand {{ display: flex; align-items: center; gap: 12px; margin: 4px 0 2px 0; }}
.brand-mark {{
    width: 40px; height: 40px; border-radius: 12px;
    background: linear-gradient(135deg, {ACC}, {ACC2});
    color: #FFFFFF !important; display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 20px;
}}
.brand-name {{ font-size: 19px; font-weight: 800; color: {TEXT}; line-height: 1.1; }}
.brand-sub {{ font-size: 12px; color: {MUTED}; }}

.status-row {{ display: flex; align-items: center; gap: 10px; font-size: 13px; padding: 5px 0; color: {TEXT}; }}
.dot {{ width: 8px; height: 8px; border-radius: 50%; background: {LOW}; flex: none; box-shadow: 0 0 8px {rgba(LOW, 0.7)}; }}
.dot.idle {{ background: {MUTED}; box-shadow: none; }}

/* ---------- Page header ---------- */
.page-head {{
    display: flex; justify-content: space-between; align-items: center;
    gap: 16px; flex-wrap: wrap; margin-bottom: 26px;
}}
.page-title {{ font-size: 32px; font-weight: 800; color: {TEXT}; letter-spacing: -0.02em; line-height: 1.15; }}
.page-desc {{ font-size: 14px; color: {MUTED}; margin-top: 6px; }}
.head-right {{ display: flex; gap: 10px; }}
.chip {{
    display: inline-flex; align-items: center; gap: 8px;
    background: {CARD}; border: 1px solid {LINE}; color: {TEXT};
    font-size: 13px; font-weight: 600; padding: 8px 14px; border-radius: 12px;
}}
.section-title {{ font-size: 17px; font-weight: 700; color: {TEXT}; margin: 28px 0 12px 0; }}

/* ---------- Cards (st.container(border=True)) ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {CARD};
    border: 1px solid {LINE} !important;
    border-radius: 18px;
    box-shadow: {SHADOW};
    padding: 8px 10px;
    backdrop-filter: blur(10px);
}}
.card-title {{ font-size: 16px; font-weight: 700; color: {TEXT}; }}
.card-note {{ font-size: 12.5px; color: {MUTED}; margin: 2px 0 6px 0; }}

.panel {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 18px;
    padding: 20px 24px; box-shadow: {SHADOW};
}}
.panel-title {{ font-size: 15px; font-weight: 700; color: {TEXT}; }}
.panel-note {{ font-size: 13px; color: {MUTED}; margin-top: 2px; line-height: 1.6; }}
.panel-head {{ display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }}

/* ---------- Hero (always dark) ---------- */
.hero {{
    display: flex; align-items: center; justify-content: space-between; gap: 32px;
    padding: 36px 44px; border-radius: 22px; margin-bottom: 26px;
    background-color: #12163F;
    background-image:
        linear-gradient(rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.04) 1px, transparent 1px),
        linear-gradient(120deg, #141B52 0%, #2A1F6B 60%, #4B2A8F 100%);
    background-size: 32px 32px, 32px 32px, 100% 100%;
    box-shadow: {SHADOW};
}}
.hero-copy {{ flex: 1 1 52%; }}
.hero-art {{ flex: 1 1 44%; text-align: right; }}
.hero-art img {{ width: 100%; max-width: 440px; }}
.hero h1 {{
    color: #FFFFFF !important; font-size: 40px; font-weight: 800;
    letter-spacing: -0.03em; line-height: 1.1; max-width: 620px; margin: 0 0 14px 0; padding: 0;
}}
.hero p {{ color: #BCC6E6 !important; font-size: 16px; max-width: 560px; margin: 0; line-height: 1.6; }}
.hero-badges {{ display: flex; gap: 10px; margin-top: 22px; flex-wrap: wrap; }}
.badge {{
    background: rgba(255,255,255,0.10); border: 1px solid rgba(255,255,255,0.20);
    color: #E5ECF7 !important; font-size: 12.5px; font-weight: 600;
    padding: 6px 13px; border-radius: 999px;
}}

/* ---------- KPI cards ---------- */
.kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 16px; margin-bottom: 20px; }}
.kpi {{
    background: {CARD}; border: 1px solid {LINE}; border-radius: 18px;
    padding: 18px 22px; box-shadow: {SHADOW}; position: relative; overflow: hidden;
}}
.kpi::after {{
    content: ""; position: absolute; right: -30px; top: -30px; width: 100px; height: 100px;
    border-radius: 50%; background: var(--tone); opacity: 0.16;
}}
.kpi-label {{ font-size: 13px; font-weight: 600; color: {MUTED}; display: flex; align-items: center; gap: 8px; }}
.kpi-label i {{ width: 8px; height: 8px; border-radius: 50%; background: var(--tone); display: inline-block; }}
.kpi-value {{ font-size: 32px; font-weight: 800; color: {TEXT}; letter-spacing: -0.02em; margin-top: 8px; line-height: 1.1; }}
.kpi-note {{ font-size: 12.5px; color: {MUTED}; margin-top: 6px; }}

/* ---------- Steps ---------- */
.steps {{ display: flex; margin: 4px 0 8px 0; }}
.step {{ flex: 1; padding: 14px 18px; background: {CARD}; border: 1px solid {LINE}; border-left: none; }}
.step:first-child {{ border-left: 1px solid {LINE}; border-radius: 14px 0 0 14px; }}
.step:last-child {{ border-radius: 0 14px 14px 0; }}
.step b {{ color: {ACC}; font-size: 13px; }}
.step div {{ font-size: 14px; font-weight: 700; margin-top: 2px; color: {TEXT}; }}
.step span {{ font-size: 12.5px; color: {MUTED}; }}

/* ---------- Feature tiles ---------- */
.feature-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.feature {{
    display: flex; gap: 16px; align-items: flex-start;
    background: {CARD}; border: 1px solid {LINE}; border-radius: 18px; padding: 18px 20px;
    box-shadow: {SHADOW};
}}
.feature-icon {{
    flex: none; width: 52px; height: 52px; border-radius: 14px;
    background: {rgba(ACC, 0.14)}; display: flex; align-items: center; justify-content: center;
}}
.panel-head .feature-icon {{ width: 44px; height: 44px; }}
.feature h4 {{ font-size: 15px; font-weight: 700; margin: 0 0 4px 0; color: {TEXT}; }}
.feature p {{ font-size: 13.5px; color: {MUTED}; margin: 0; line-height: 1.55; }}

/* ---------- Empty state & callouts ---------- */
.empty {{
    text-align: center; background: {CARD}; border: 1px solid {LINE};
    border-radius: 18px; padding: 34px 24px 30px 24px; box-shadow: {SHADOW};
}}
.empty img {{ width: 220px; max-width: 60%; }}
.empty h3 {{ font-size: 18px; margin: 6px 0 6px 0; color: {TEXT}; }}
.empty p {{ color: {MUTED}; font-size: 14px; line-height: 1.6; max-width: 460px; margin: 0 auto; }}

.callout {{
    display: flex; gap: 12px; align-items: flex-start;
    border-radius: 14px; padding: 13px 16px; margin: 10px 0;
    background: var(--bgc); border: 1px solid var(--bdc); color: {TEXT}; font-size: 14px; line-height: 1.5;
}}
.callout b.ic {{
    flex: none; width: 22px; height: 22px; border-radius: 50%;
    background: var(--tone); color: #FFFFFF; font-size: 12px;
    display: flex; align-items: center; justify-content: center;
}}
.callout.success {{ --tone: {LOW}; --bgc: {rgba(LOW, 0.12)}; --bdc: {rgba(LOW, 0.35)}; }}
.callout.error {{ --tone: {HIGH}; --bgc: {rgba(HIGH, 0.12)}; --bdc: {rgba(HIGH, 0.35)}; }}
.callout.warning {{ --tone: {MEDIUM}; --bgc: {rgba(MEDIUM, 0.12)}; --bdc: {rgba(MEDIUM, 0.35)}; }}
.callout.info {{ --tone: {ACC}; --bgc: {rgba(ACC, 0.12)}; --bdc: {rgba(ACC, 0.35)}; }}

/* ---------- Pills, tables ---------- */
.pill {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 12px; font-weight: 700; }}
.pill.high {{ background: {rgba(HIGH, 0.16)}; color: {HIGH}; }}
.pill.medium {{ background: {rgba(MEDIUM, 0.16)}; color: {MEDIUM}; }}
.pill.low {{ background: {rgba(LOW, 0.16)}; color: {LOW}; }}
.pill.muted {{ background: {HOVER}; color: {MUTED}; }}

.tbl-wrap {{
    overflow: auto; border: 1px solid {LINE}; border-radius: 14px;
    background: {CARD}; margin-top: 6px;
}}
table.tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
.tbl th {{
    position: sticky; top: 0; z-index: 1; background: {CARD2}; color: {MUTED};
    font-weight: 700; text-align: left; padding: 12px 16px; white-space: nowrap;
    border-bottom: 1px solid {LINE};
}}
.tbl td {{ padding: 11px 16px; border-bottom: 1px solid {LINE}; color: {TEXT}; white-space: nowrap; }}
.tbl td.txt {{ white-space: normal; min-width: 260px; color: {MUTED}; }}
.tbl tr:last-child td {{ border-bottom: none; }}
.tbl tbody tr:hover td {{ background: {HOVER}; }}
.pbar {{ display: flex; align-items: center; gap: 12px; }}
.pbar span {{ min-width: 58px; font-weight: 700; }}
.pbar div {{ width: 110px; height: 6px; background: {GRID}; border-radius: 3px; overflow: hidden; }}
.pbar i {{ display: block; height: 100%; background: linear-gradient(90deg, {ACC}, {ACC2}); }}

/* ---------- Alerts / mini list / stats ---------- */
.alert-row {{
    display: flex; align-items: center; gap: 18px;
    background: {CARD}; border: 1px solid {LINE}; border-left: 5px solid {HIGH};
    border-radius: 12px; padding: 12px 18px; margin-bottom: 8px;
}}
.alert-row.medium {{ border-left-color: {MEDIUM}; }}
.alert-id {{ font-weight: 800; font-size: 14px; min-width: 130px; color: {TEXT}; }}
.alert-score {{ min-width: 150px; color: {TEXT}; }}
.alert-score .num {{ font-weight: 800; font-size: 15px; }}
.bar {{ height: 6px; background: {GRID}; border-radius: 3px; margin-top: 4px; overflow: hidden; }}
.bar > div {{ height: 100%; background: {HIGH}; }}
.alert-row.medium .bar > div {{ background: {MEDIUM}; }}
.alert-text {{ font-size: 13.5px; color: {MUTED}; flex: 1; }}

.mini {{
    display: flex; justify-content: space-between; align-items: center;
    background: {CARD2}; border: 1px solid {LINE}; border-radius: 14px;
    padding: 10px 14px; margin-bottom: 10px;
}}
.mini-id {{ font-size: 13px; font-weight: 700; color: {TEXT}; margin-bottom: 4px; }}
.mini-score {{ font-size: 17px; font-weight: 800; color: {TEXT}; }}
.stat {{ padding: 12px 0; border-bottom: 1px solid {LINE}; }}
.stat:last-child {{ border-bottom: none; }}
.stat-label {{ font-size: 12.5px; color: {MUTED}; }}
.stat-value {{ font-size: 24px; font-weight: 800; color: {TEXT}; letter-spacing: -0.02em; }}

.cap {{ border-left: 3px solid {ACC}; padding: 2px 0 2px 16px; margin-bottom: 18px; }}
.cap h4 {{ font-size: 15px; font-weight: 700; margin: 0 0 4px 0; color: {TEXT}; }}
.cap p {{ font-size: 14px; color: {MUTED}; margin: 0; line-height: 1.55; }}

.footer {{ text-align: center; color: {MUTED}; font-size: 12.5px; padding: 12px 0 4px 0; }}

/* ---------- Native widgets ---------- */
div.stButton > button {{
    background: linear-gradient(90deg, {ACC}, {ACC2}); border: none; border-radius: 12px;
    min-height: 46px; font-weight: 700; box-shadow: 0 6px 18px {rgba(ACC2, 0.35)};
}}
div.stButton > button, div.stButton > button * {{ color: #FFFFFF !important; }}
div.stButton > button:hover {{ filter: brightness(1.1); border: none; }}
div[data-testid="stDownloadButton"] button {{
    background: transparent; border: 1.5px solid {ACC}; border-radius: 12px;
    min-height: 46px; font-weight: 700;
}}
div[data-testid="stDownloadButton"] button, div[data-testid="stDownloadButton"] button * {{ color: {ACC} !important; }}
div[data-testid="stDownloadButton"] button:hover {{ background: {rgba(ACC, 0.12)}; }}
div.stButton > button:focus-visible, div[data-testid="stDownloadButton"] button:focus-visible {{
    outline: 3px solid {rgba(ACC, 0.45)}; outline-offset: 2px;
}}

[data-testid="stFileUploaderDropzone"] {{
    background: {CARD}; border: 1.5px dashed {LINE2}; border-radius: 18px; padding: 22px;
}}
[data-testid="stFileUploaderDropzone"] * {{ color: {MUTED}; }}
[data-testid="stFileUploaderDropzone"] button {{ background: {rgba(ACC, 0.14)}; border: 1px solid {ACC}; }}
[data-testid="stFileUploaderDropzone"] button * {{ color: {ACC} !important; }}
[data-testid="stFileUploaderFile"] * {{ color: {TEXT}; }}

div[data-baseweb="select"] > div {{ background: {CARD2}; border-color: {LINE}; border-radius: 12px; }}
div[data-baseweb="select"] * {{ color: {TEXT}; }}
span[data-baseweb="tag"] {{ background: {rgba(ACC, 0.18)}; }}
span[data-baseweb="tag"] * {{ color: {ACC}; }}
div[data-baseweb="popover"] ul, div[data-baseweb="popover"] [role="listbox"],
div[data-baseweb="popover"] [data-baseweb="menu"] {{ background: {CARD_SOLID} !important; }}
div[data-baseweb="popover"] li, div[data-baseweb="popover"] li * {{ color: {TEXT}; }}
div[data-baseweb="popover"] li:hover {{ background: {HOVER}; }}

[data-testid="stSlider"] [role="slider"] {{ background: {ACC}; box-shadow: 0 0 0 4px {rgba(ACC, 0.25)}; }}
[data-testid="stSlider"] * {{ color: {TEXT}; }}
[data-testid="stSlider"] [data-testid="stTickBarMin"], [data-testid="stSlider"] [data-testid="stTickBarMax"] {{ color: {MUTED}; }}

[data-baseweb="tab-list"] {{ gap: 6px; }}
[data-baseweb="tab-border"] {{ background: {LINE} !important; }}
[data-baseweb="tab-highlight"] {{ background: {ACC} !important; }}
button[role="tab"] p {{ color: {MUTED}; font-weight: 600; font-size: 14px; }}
button[role="tab"][aria-selected="true"] p {{ color: {ACC}; }}

[data-testid="stExpander"] details {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 14px; }}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {{ color: {TEXT}; }}

[data-testid="stProgress"] [role="progressbar"] > div {{ background: linear-gradient(90deg, {ACC}, {ACC2}); }}
[data-testid="stSpinner"] * {{ color: {MUTED}; }}

@media (max-width: 800px) {{
    .hero {{ flex-direction: column; padding: 28px 24px; }}
    .hero h1 {{ font-size: 30px; }}
    .hero-art {{ display: none; }}
    .feature-grid {{ grid-template-columns: 1fr; }}
    .steps {{ flex-direction: column; }}
    .step {{ border-left: 1px solid {LINE}; border-radius: 0 !important; }}
    .alert-row {{ flex-wrap: wrap; }}
}}
</style>
""",
    unsafe_allow_html=True,
)

# ==========================================================
# 5. CONSTANTS AND SESSION STATE
# ==========================================================

REQUIRED_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MAX_ROWS = 5000
TABLE_ROWS = 300

if "fraud_results" not in st.session_state:
    st.session_state.fraud_results = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

if "last_transaction_count" not in st.session_state:
    st.session_state.last_transaction_count = 0

# ==========================================================
# 6. REUSABLE FUNCTIONS
# ==========================================================


def page_heading(title, description):
    today = datetime.now().strftime("%d %b %Y")
    st.markdown(
        f"""
        <div class="page-head">
            <div>
                <div class="page-title">{title}</div>
                <div class="page-desc">{description}</div>
            </div>
            <div class="head-right">
                <span class="chip"><span class="dot"></span>Model online</span>
                <span class="chip">{today}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def card_title(title, note=""):
    st.markdown(
        f'<div class="card-title">{title}</div><div class="card-note">{note}</div>',
        unsafe_allow_html=True,
    )


def kpi_grid(items):
    """items: list of (label, value, note, tone_colour)."""
    cells = "".join(
        f'<div class="kpi" style="--tone:{tone};">'
        f'<div class="kpi-label"><i></i>{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-note">{note}</div></div>'
        for label, value, note, tone in items
    )
    st.markdown(f'<div class="kpi-grid">{cells}</div>', unsafe_allow_html=True)


def callout(kind, text):
    icons = {"success": "✓", "error": "!", "warning": "!", "info": "i"}
    st.markdown(
        f'<div class="callout {kind}"><b class="ic">{icons[kind]}</b><div>{text}</div></div>',
        unsafe_allow_html=True,
    )


def feature_tile(icon_svg, title, text):
    return (
        f'<div class="feature"><div class="feature-icon">{svg_img(icon_svg, "30px", title)}</div>'
        f'<div><h4>{title}</h4><p>{text}</p></div></div>'
    )


def empty_state(svg, title, text):
    st.markdown(
        f'<div class="empty">{svg_img(svg, None, title)}<h3>{title}</h3><p>{text}</p></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height=290, legend=True):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope, sans-serif", color=MUTED, size=12),
        margin=dict(l=6, r=6, t=8, b=6),
        showlegend=legend,
        legend=dict(orientation="h", y=-0.2, x=0, font=dict(color=MUTED)),
        hoverlabel=dict(bgcolor=CARD_SOLID, font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False, tickfont=dict(color=MUTED))
    return fig


def _cell(col, value):
    if col == "Transaction ID":
        return f'<span style="font-weight:700;">#{int(value)}</span>'
    if col == "Fraud Probability (%)":
        score = float(value)
        width = min(max(score, 0), 100)
        return f'<div class="pbar"><span>{score:.2f}%</span><div><i style="width:{width}%"></i></div></div>'
    if col == "Risk Level":
        cls = {"High Risk": "high", "Medium Risk": "medium", "Low Risk": "low"}.get(value, "muted")
        return f'<span class="pill {cls}">{html.escape(str(value))}</span>'
    if col == "Prediction":
        cls = "high" if value == "Potential Fraud" else "low"
        return f'<span class="pill {cls}">{html.escape(str(value))}</span>'
    if isinstance(value, (float, np.floating)):
        return f"{value:,.4f}"
    return html.escape(str(value))


def html_table(df, max_rows=TABLE_ROWS, height=420):
    shown = df.head(max_rows)
    text_cols = {"Alert", "Recommendation"}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in shown.columns)
    body = []
    for record in shown.to_dict("records"):
        cells = "".join(
            f'<td class="{"txt" if col in text_cols else ""}">{_cell(col, val)}</td>'
            for col, val in record.items()
        )
        body.append(f"<tr>{cells}</tr>")
    st.markdown(
        f'<div class="tbl-wrap" style="max-height:{height}px;">'
        f'<table class="tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )
    if len(df) > max_rows:
        st.caption(f"Showing the first {max_rows:,} of {len(df):,} rows. Download the report for the full list.")


def alert_rows(df, limit=15):
    rows = []
    for _, row in df.head(limit).iterrows():
        score = float(row["Fraud Probability (%)"])
        tone = "" if row["Risk Level"] == "High Risk" else "medium"
        rows.append(
            f'<div class="alert-row {tone}">'
            f'<div class="alert-id">Transaction #{int(row["Transaction ID"])}</div>'
            f'<div class="alert-score"><div class="num">{score:.2f}%</div>'
            f'<div class="bar"><div style="width:{min(max(score, 0), 100)}%"></div></div></div>'
            f'<div class="alert-text">{html.escape(str(row["Alert"]))}</div></div>'
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def mini_alerts(df, n=4):
    items = []
    for _, row in df.head(n).iterrows():
        cls = {"High Risk": "high", "Medium Risk": "medium"}.get(row["Risk Level"], "low")
        items.append(
            f'<div class="mini"><div><div class="mini-id">Transaction #{int(row["Transaction ID"])}</div>'
            f'<span class="pill {cls}">{html.escape(str(row["Risk Level"]))}</span></div>'
            f'<div class="mini-score">{float(row["Fraud Probability (%)"]):.1f}%</div></div>'
        )
    st.markdown(
        '<div class="card-note" style="margin-top:8px;">Top alerts</div>' + "".join(items),
        unsafe_allow_html=True,
    )


def validate_transactions(data):
    if data.empty:
        return False, "The uploaded CSV file is empty."

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        return False, "Missing columns: " + ", ".join(missing)

    clean_data = data[REQUIRED_COLUMNS].copy()

    for column in REQUIRED_COLUMNS:
        clean_data[column] = pd.to_numeric(clean_data[column], errors="coerce")

    if clean_data.isnull().values.any():
        return False, "The file contains missing or non-numeric values."

    if not np.isfinite(clean_data.to_numpy()).all():
        return False, "The file contains infinite or invalid numeric values."

    if (clean_data["Amount"] < 0).any():
        return False, "Transaction amounts cannot be negative."

    return True, clean_data


def analyze_transactions(data):
    results = []
    total = len(data)

    progress_bar = st.progress(0)
    status_text = st.empty()

    for position, (_, row) in enumerate(data.iterrows(), start=1):
        prediction = predict_transaction(row.to_dict())

        result_row = {
            "Transaction ID": position,
            "Fraud Probability (%)": prediction["fraud_probability"],
            "Prediction": prediction["prediction"],
            "Risk Level": prediction["risk_level"],
            "Recommendation": prediction["recommendation"],
            "Alert": prediction["alert"],
        }
        results.append(result_row)

        # Persist each scored transaction in SQLite.
        save_transaction(
            transaction_time=row["Time"],
            transaction_amount=row["Amount"],
            fraud_probability=float(prediction["fraud_probability"]) / 100.0,
            prediction=prediction["prediction"],
            risk_level=prediction["risk_level"],
            recommendation=prediction["recommendation"],
            alert=prediction["alert"],
        )

        if position % 10 == 0 or position == total:
            progress_bar.progress(position / total)
            status_text.caption(f"Scoring transaction {position:,} of {total:,}")

    progress_bar.empty()
    status_text.empty()

    return pd.DataFrame(results)


def sample_template():
    return pd.DataFrame(columns=REQUIRED_COLUMNS).to_csv(index=False).encode("utf-8")


# ==========================================================
# 7. SIDEBAR
# ==========================================================

NAV_ICONS = {
    "Executive Dashboard": "📊",
    "Transaction Analysis": "🔍",
    "Fraud Intelligence": "🛡️",
    "Model Performance": "📈",
    "Transaction History": "🗄️",
    "About the Project": "ℹ️",
}

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">S</div>
            <div>
                <div class="brand-name">Sentinel AI</div>
                <div class="brand-sub">Financial fraud intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    selected_page = st.radio(
        "Navigation",
        list(NAV_ICONS.keys()),
        format_func=lambda name: f"{NAV_ICONS[name]}   {name}",
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown('<div class="side-label">Appearance</div>', unsafe_allow_html=True)
    st.toggle("Dark mode", key="dark_mode")

    st.divider()

    loaded = st.session_state.fraud_results is not None
    st.markdown(
        f"""
        <div class="status-row"><span class="dot"></span>Prediction service online</div>
        <div class="status-row"><span class="dot"></span>Model loaded</div>
        <div class="status-row"><span class="dot {'' if loaded else 'idle'}"></span>
            {'Analysis in session' if loaded else 'No analysis yet'}</div>
        """,
        unsafe_allow_html=True,
    )

    if loaded and st.session_state.uploaded_name:
        st.caption(f"Source: {st.session_state.uploaded_name}")

    st.divider()
    st.caption("Intelligent Financial Fraud Detection")
    st.caption("Academic Project | 2026")


# ==========================================================
# 8. EXECUTIVE DASHBOARD
# ==========================================================

if selected_page == "Executive Dashboard":

    page_heading("Fraud Detection", "Portfolio-level view of transaction risk and model alerts.")

    results = st.session_state.fraud_results

    if results is None:

        st.markdown(
            f"""
            <div class="hero">
                <div class="hero-copy">
                    <h1>Catch suspicious transactions before they cost you.</h1>
                    <p>Upload a transaction file and Sentinel AI scores every record,
                    ranks it by risk, and tells your team which ones to review first.</p>
                    <div class="hero-badges">
                        <span class="badge">Fraud scoring</span>
                        <span class="badge">Risk levels</span>
                        <span class="badge">Alert queue</span>
                        <span class="badge">CSV export</span>
                    </div>
                </div>
                <div class="hero-art">{svg_img(HERO_SVG, None, "Shield protecting transactions")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section_title("How it works")
        st.markdown(
            """
            <div class="steps">
                <div class="step"><b>1</b><div>Upload</div><span>CSV of transactions</span></div>
                <div class="step"><b>2</b><div>Validate</div><span>Columns and values checked</span></div>
                <div class="step"><b>3</b><div>Score</div><span>Model estimates fraud probability</span></div>
                <div class="step"><b>4</b><div>Review</div><span>Alerts ranked by risk</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        section_title("What you get")
        tiles = "".join(
            [
                feature_tile(ICON_SCORE, "Fraud probability for every transaction",
                             "A machine learning model scores each record based on its transaction pattern."),
                feature_tile(ICON_RISK, "Low, medium and high risk levels",
                             "Scores are grouped so analysts can focus on what matters."),
                feature_tile(ICON_ALERT, "Alerts with recommended actions",
                             "Suspicious transactions are flagged with a next step for verification."),
                feature_tile(ICON_REPORT, "Charts and downloadable reports",
                             "Export the full results as CSV for audits and sharing."),
            ]
        )
        st.markdown(f'<div class="feature-grid">{tiles}</div>', unsafe_allow_html=True)

        callout("info", "Open <b>Transaction Analysis</b> in the sidebar to upload your first file.")

    else:

        total = len(results)
        scores = results["Fraud Probability (%)"].astype(float)
        fraud = int((results["Prediction"] == "Potential Fraud").sum())
        normal = total - fraud
        high = int((results["Risk Level"] == "High Risk").sum())
        medium = int((results["Risk Level"] == "Medium Risk").sum())
        fraud_rate = fraud / total * 100 if total else 0

        kpi_grid(
            [
                ("Transactions analyzed", f"{total:,}", "Processed in this session", ACC),
                ("Potential fraud", f"{fraud:,}", f"{fraud_rate:.2f}% of all transactions", HIGH),
                ("Predicted normal", f"{normal:,}", "No action needed", LOW),
                ("High risk", f"{high:,}", f"{medium:,} more at medium risk", MEDIUM),
            ]
        )

        ranked = results.sort_values("Fraud Probability (%)", ascending=False)
        ordered_ids = results.sort_values("Transaction ID")

        # ---- Row 1 ----
        left, right = st.columns([1.05, 1.3])

        with left:
            with st.container(border=True):
                card_title("Fraud detection overview", "Risk level split and key scores")
                a, b = st.columns([1.25, 1])

                with a:
                    counts = results["Risk Level"].value_counts()
                    labels = [r for r in RISK_ORDER if r in counts.index] + [
                        r for r in counts.index if r not in RISK_ORDER
                    ]
                    donut = go.Figure(
                        go.Pie(
                            labels=labels,
                            values=[int(counts[r]) for r in labels],
                            hole=0.7,
                            sort=False,
                            marker=dict(
                                colors=[RISK_COLORS.get(r, MUTED) for r in labels],
                                line=dict(color=CARD_SOLID, width=3),
                            ),
                            textinfo="none",
                        )
                    )
                    donut.add_annotation(
                        text=f"<b>{fraud_rate:.2f}%</b><br><span style='font-size:11px'>flagged</span>",
                        showarrow=False,
                        font=dict(size=22, color=TEXT),
                    )
                    st.plotly_chart(style_fig(donut, 270), **_PLOT)

                with b:
                    st.markdown(
                        f"""
                        <div class="stat"><div class="stat-label">Average score</div>
                            <div class="stat-value">{scores.mean():.2f}%</div></div>
                        <div class="stat"><div class="stat-label">Peak score</div>
                            <div class="stat-value">{scores.max():.2f}%</div></div>
                        <div class="stat"><div class="stat-label">Medium risk</div>
                            <div class="stat-value">{medium:,}</div></div>
                        """,
                        unsafe_allow_html=True,
                    )

        with right:
            with st.container(border=True):
                card_title("Fraud trend", "Cumulative flagged transactions in upload order")
                a, b = st.columns([2.2, 1])

                with a:
                    cum_fraud = (ordered_ids["Prediction"] == "Potential Fraud").cumsum()
                    cum_high = (ordered_ids["Risk Level"] == "High Risk").cumsum()
                    trend = go.Figure()
                    trend.add_trace(
                        go.Scatter(
                            x=ordered_ids["Transaction ID"], y=cum_fraud, name="Potential fraud",
                            mode="lines", line=dict(color=ACC2, width=3),
                            fill="tozeroy", fillcolor=rgba(ACC2, 0.25),
                        )
                    )
                    trend.add_trace(
                        go.Scatter(
                            x=ordered_ids["Transaction ID"], y=cum_high, name="High risk",
                            mode="lines", line=dict(color=ACC, width=2.5),
                        )
                    )
                    trend.update_xaxes(title_text="Transaction ID")
                    st.plotly_chart(style_fig(trend, 270), **_PLOT)

                with b:
                    mini_alerts(ranked, 4)

        # ---- Row 2 ----
        left, right = st.columns(2)

        with left:
            with st.container(border=True):
                card_title("Risk score by transaction", "Average and peak score across the upload")
                n_bins = max(1, min(30, total))
                chunks = np.array_split(np.arange(total), n_bins)
                sorted_scores = ordered_ids["Fraud Probability (%)"].astype(float).to_numpy()
                xs = [int(ordered_ids["Transaction ID"].iloc[c[0]]) for c in chunks]
                avg_line = [float(sorted_scores[c].mean()) for c in chunks]
                max_line = [float(sorted_scores[c].max()) for c in chunks]

                score_fig = go.Figure()
                score_fig.add_trace(
                    go.Scatter(
                        x=xs, y=max_line, name="Peak score", mode="lines+markers",
                        line=dict(color=ACC2, width=2.5), marker=dict(size=6),
                        fill="tozeroy", fillcolor=rgba(ACC2, 0.2),
                    )
                )
                score_fig.add_trace(
                    go.Scatter(
                        x=xs, y=avg_line, name="Average score", mode="lines+markers",
                        line=dict(color=ACC, width=2.5), marker=dict(size=6),
                    )
                )
                score_fig.update_yaxes(title_text="Fraud probability (%)")
                st.plotly_chart(style_fig(score_fig, 300), **_PLOT)

        with right:
            with st.container(border=True):
                card_title("Score distribution", "Transactions per probability range (log scale)")
                counts_hist, edges = np.histogram(scores, bins=10, range=(0, 100))
                labels_hist = [f"{int(edges[i])}–{int(edges[i + 1])}" for i in range(10)]
                bar_colors = [blend(ACC, ACC2, i / 9) for i in range(10)]
                hist = go.Figure(
                    go.Bar(x=labels_hist, y=counts_hist, marker=dict(color=bar_colors))
                )
                hist.update_yaxes(type="log", title_text="Transactions")
                hist.update_xaxes(title_text="Fraud probability (%)")
                st.plotly_chart(style_fig(hist, 300, legend=False), **_PLOT)

        with st.container(border=True):
            card_title("Highest-risk transactions", "Top 10 by fraud probability")
            html_table(ranked.head(10), height=420)


# ==========================================================
# 9. TRANSACTION ANALYSIS
# ==========================================================

elif selected_page == "Transaction Analysis":

    page_heading("Transaction analysis", "Upload transaction records and run fraud predictions.")

    section_title("Upload transactions")

    up_col, help_col = st.columns([2.2, 1])

    with help_col:
        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-head">
                    <div class="feature-icon">{svg_img(ICON_UPLOAD, "28px", "Upload")}</div>
                    <div class="panel-title">File requirements</div>
                </div>
                <div class="panel-note">
                    CSV format<br>
                    Columns: Time, V1 to V28, Amount<br>
                    Numeric values only<br>
                    Up to {MAX_ROWS:,} rows<br>
                    A Class column is ignored
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download CSV template",
            data=sample_template(),
            file_name="transaction_template.csv",
            mime="text/csv",
            **_BTN,
        )

    with up_col:
        uploaded_file = st.file_uploader(
            "Choose a transaction CSV file", type=["csv"], key="transaction_upload"
        )

    if uploaded_file is not None:

        try:
            raw_data = pd.read_csv(uploaded_file)

            if "Class" in raw_data.columns:
                raw_data = raw_data.drop(columns=["Class"])

            valid, validated_data = validate_transactions(raw_data)

            if not valid:
                callout("error", f"{validated_data} Fix the file and upload it again.")

            else:
                callout("success", f"File is valid. <b>{len(validated_data):,}</b> transactions found.")

                with st.expander("Preview first 8 rows"):
                    html_table(validated_data.head(8), height=320)
                    st.caption(f"{len(REQUIRED_COLUMNS)} features | {len(validated_data):,} records")

                if len(validated_data) > MAX_ROWS:
                    callout(
                        "warning",
                        f"This file has more than {MAX_ROWS:,} rows. "
                        "Split it into smaller files and upload one at a time.",
                    )

                elif st.button("Run fraud detection", **_BTN):
                    with st.spinner("Scoring transactions..."):
                        try:
                            output = analyze_transactions(validated_data)
                            st.session_state.fraud_results = output
                            st.session_state.uploaded_name = uploaded_file.name
                            callout("success", "Analysis complete. Results are below.")
                        except Exception as error:
                            callout("error", f"Prediction failed: {html.escape(str(error))}")

        except Exception as error:
            callout("error", f"Could not read the CSV file: {html.escape(str(error))}")

    if st.session_state.fraud_results is not None:

        results = st.session_state.fraud_results

        st.divider()

        fraud_count = int((results["Prediction"] == "Potential Fraud").sum())
        normal_count = len(results) - fraud_count
        high_count = int((results["Risk Level"] == "High Risk").sum())

        kpi_grid(
            [
                ("Analyzed", f"{len(results):,}", "Transactions scored", ACC),
                ("Potential fraud", f"{fraud_count:,}", "Flagged by the model", HIGH),
                ("Normal", f"{normal_count:,}", "Predicted legitimate", LOW),
                ("High risk", f"{high_count:,}", "Review first", MEDIUM),
            ]
        )

        tab_results, tab_export = st.tabs(["Results", "Export"])

        with tab_results:
            f1, f2 = st.columns(2)
            risk_filter = f1.multiselect("Risk level", RISK_ORDER, default=RISK_ORDER)
            predictions = sorted(results["Prediction"].unique())
            pred_filter = f2.multiselect("Prediction", predictions, default=predictions)

            filtered = results[
                results["Risk Level"].isin(risk_filter) & results["Prediction"].isin(pred_filter)
            ]

            st.caption(f"{len(filtered):,} of {len(results):,} transactions match the filters")
            html_table(filtered, height=460)

        with tab_export:
            st.markdown(
                '<div class="panel-note" style="margin-bottom:12px;">'
                "Download every scored transaction with its risk level, recommendation and alert.</div>",
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download fraud detection report",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="financial_fraud_analysis.csv",
                mime="text/csv",
                **_BTN,
            )
            st.caption(f"Source file: {st.session_state.uploaded_name or 'CSV upload'}")


# ==========================================================
# 10. FRAUD INTELLIGENCE
# ==========================================================

elif selected_page == "Fraud Intelligence":

    page_heading("Fraud intelligence", "Review suspicious activity and decide what to investigate.")

    results = st.session_state.fraud_results

    if results is None:
        empty_state(
            EMPTY_SVG,
            "No results to review yet",
            "Open Transaction Analysis, upload a CSV file and run fraud detection. "
            "Flagged transactions will appear here.",
        )

    else:

        fraud_results = results[results["Prediction"] == "Potential Fraud"].copy()
        high_risk = results[results["Risk Level"] == "High Risk"].copy()
        medium_risk = results[results["Risk Level"] == "Medium Risk"].copy()

        kpi_grid(
            [
                ("Potential fraud cases", f"{len(fraud_results):,}", "Classified by the model", HIGH),
                ("High-risk cases", f"{len(high_risk):,}", "Investigate now", HIGH),
                ("Medium-risk cases", f"{len(medium_risk):,}", "Verify with the customer", MEDIUM),
            ]
        )

        tab_alerts, tab_fraud, tab_search = st.tabs(
            ["Priority alerts", "Potential fraud", "Score threshold"]
        )

        with tab_alerts:
            if high_risk.empty:
                empty_state(
                    CLEAR_SVG,
                    "No high-risk transactions",
                    "The model did not flag any transaction as high risk in this file.",
                )
            else:
                ranked = high_risk.sort_values("Fraud Probability (%)", ascending=False)
                st.caption(
                    f"Top {min(15, len(ranked))} of {len(ranked):,} high-risk transactions, "
                    "highest score first"
                )
                alert_rows(ranked, limit=15)

                if len(ranked) > 15:
                    with st.expander(f"Show all {len(ranked):,} high-risk transactions"):
                        html_table(ranked)

        with tab_fraud:
            if fraud_results.empty:
                empty_state(
                    CLEAR_SVG,
                    "No potential fraud found",
                    "Every transaction in this file was predicted as normal.",
                )
            else:
                html_table(
                    fraud_results.sort_values("Fraud Probability (%)", ascending=False),
                    height=460,
                )

        with tab_search:
            threshold = st.slider(
                "Show transactions with a fraud probability of at least (%)",
                min_value=0.0, max_value=100.0, value=50.0, step=0.5,
            )
            above = results[results["Fraud Probability (%)"] >= threshold].sort_values(
                "Fraud Probability (%)", ascending=False
            )
            st.caption(f"{len(above):,} transactions at or above {threshold:.1f}%")
            if above.empty:
                callout("info", "No transactions reach this score. Lower the threshold.")
            else:
                html_table(above, height=420)


# ==========================================================
# 11. MODEL PERFORMANCE
# ==========================================================

elif selected_page == "Model Performance":

    page_heading(
        "Model performance",
        "Evaluation metrics, classification results and confusion matrix for the trained model.",
    )

    evaluation_file = BASE_DIR / "outputs" / "model_evaluation.json"
    confusion_image = BASE_DIR / "outputs" / "fraud_confusion_matrix.png"

    if not evaluation_file.exists():
        empty_state(
            EMPTY_SVG,
            "Evaluation report not found",
            "Run python src/model_evaluation.py in your project terminal, "
            "then refresh this page.",
        )
    else:
        try:
            with evaluation_file.open("r", encoding="utf-8") as file:
                metrics = json.load(file)

            section_title("Key evaluation metrics")
            kpi_grid(
                [
                    ("Accuracy", f"{metrics['accuracy']:.2f}%", "Overall correct predictions", ACC),
                    ("Precision", f"{metrics['precision']:.2f}%", "Fraud predictions that were correct", LOW),
                    ("Recall", f"{metrics['recall']:.2f}%", "Actual fraud cases detected", MEDIUM),
                    ("F1 score", f"{metrics['f1_score']:.2f}%", "Balance of precision and recall", ACC2),
                ]
            )

            section_title("Evaluation dataset")
            c1, c2, c3 = st.columns(3)
            c1.metric("Total transactions", f"{metrics['total_transactions']:,}")
            c2.metric("Actual fraud", f"{metrics['actual_fraud']:,}")
            c3.metric("Actual normal", f"{metrics['actual_normal']:,}")

            section_title("Confusion matrix")
            matrix_values = [
                [metrics["true_negative"], metrics["false_positive"]],
                [metrics["false_negative"], metrics["true_positive"]],
            ]
            matrix_fig = go.Figure(
                data=go.Heatmap(
                    z=matrix_values,
                    x=["Predicted normal", "Predicted fraud"],
                    y=["Actual normal", "Actual fraud"],
                    text=matrix_values,
                    texttemplate="%{text}",
                    textfont={"size": 20},
                    colorscale=[[0, CARD_SOLID], [1, ACC]],
                    showscale=False,
                    xgap=3,
                    ygap=3,
                )
            )
            matrix_fig.update_layout(
                height=300,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT, family="Manrope, sans-serif"),
                margin=dict(l=8, r=8, t=12, b=8),
            )
            st.plotly_chart(matrix_fig, **_PLOT)

            st.caption(
                f"True negatives: {metrics['true_negative']:,}  |  "
                f"False positives: {metrics['false_positive']:,}  |  "
                f"False negatives: {metrics['false_negative']:,}  |  "
                f"True positives: {metrics['true_positive']:,}"
            )

            section_title("Classification report")
            report = metrics.get("classification_report", {})
            report_rows = []
            for class_name in ("Normal", "Fraud"):
                if class_name in report:
                    item = report[class_name]
                    report_rows.append(
                        {
                            "Class": class_name,
                            "Precision": round(item["precision"], 4),
                            "Recall": round(item["recall"], 4),
                            "F1 score": round(item["f1-score"], 4),
                            "Support": int(item["support"]),
                        }
                    )
            if report_rows:
                st.dataframe(pd.DataFrame(report_rows), hide_index=True, use_container_width=True)

            if confusion_image.exists():
                with st.expander("View saved confusion matrix image"):
                    st.image(str(confusion_image), use_container_width=True)

            callout(
                "info",
                "These metrics describe the saved evaluation sample. "
                "The current sample should be checked for overlap with training data "
                "and tested on an independent holdout set before reporting the scores "
                "as final model performance.",
            )

        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            callout("error", f"The evaluation report could not be read: {html.escape(str(error))}")


# ==========================================================
# 12. TRANSACTION HISTORY
# ==========================================================

elif selected_page == "Transaction History":

    page_heading(
        "Transaction history",
        "Previously analysed transactions stored in the local SQLite database.",
    )

    records = get_all_transactions()
    history_columns = [
        "Database ID", "Transaction Time", "Amount", "Fraud Probability",
        "Prediction", "Risk Level", "Recommendation", "Alert", "Saved At"
    ]

    if not records:
        empty_state(
            EMPTY_SVG,
            "No saved transactions yet",
            "Run fraud detection from Transaction Analysis. The results will appear here automatically.",
        )
    else:
        history_df = pd.DataFrame(records, columns=history_columns)
        history_df["Fraud Probability"] = history_df["Fraud Probability"].astype(float) * 100
        history_df = history_df.rename(columns={"Fraud Probability": "Fraud Probability (%)"})
        section_title("Stored transaction overview")
        total_saved = len(history_df)
        fraud_saved = int((history_df["Prediction"] == "Potential Fraud").sum())
        high_saved = int((history_df["Risk Level"] == "High Risk").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Total saved", f"{total_saved:,}")
        c2.metric("Potential fraud", f"{fraud_saved:,}")
        c3.metric("High-risk records", f"{high_saved:,}")

        section_title("Saved transactions")
        st.dataframe(history_df, hide_index=True, use_container_width=True)

        csv_bytes = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download transaction history (CSV)",
            data=csv_bytes,
            file_name="sentinel_ai_transaction_history.csv",
            mime="text/csv",
            **_BTN,
        )
                # Generate and download PDF report
        pdf_path = generate_fraud_report(records)

        with open(pdf_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()

        st.download_button(
            "Download fraud detection report (PDF)",
            data=pdf_bytes,
            file_name="sentinel_ai_fraud_detection_report.pdf",
            mime="application/pdf",
            **_BTN,
        )
        callout(
            "info",
            "Records are stored locally in database/fraud_detection.db. "
            "This academic prototype flags transactions for review; it does not automatically block payments.",
        )


# ==========================================================
# 13. ABOUT THE PROJECT
# ==========================================================

elif selected_page == "About the Project":

    page_heading("About Sentinel AI", "Intelligent financial fraud detection and prevention system.")

    st.markdown(
        """
        <div class="panel">
            <div class="panel-title">Project overview</div>
            <div class="panel-note" style="font-size:14.5px;max-width:780px;margin-top:8px;">
            This project is a machine learning-based fraud detection system.
            It analyzes transaction patterns, estimates the likelihood of
            suspicious activity, classifies each transaction by risk and
            generates alerts for further review.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("Platform at a glance")
    st.markdown(
        f'<div class="panel" style="text-align:center;">'
        f'{svg_img(ABOUT_SVG, "100%", "Transactions flow through the model to risk levels and alerts")}'
        f"</div>",
        unsafe_allow_html=True,
    )

    section_title("Technology stack")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="cap"><h4>Programming</h4><p>Python, Pandas, NumPy</p></div>',
                    unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="cap"><h4>Machine learning</h4><p>Scikit-learn, Random Forest, XGBoost</p></div>',
                    unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="cap"><h4>Application</h4><p>Streamlit, Plotly, SQLite transaction storage</p></div>',
                    unsafe_allow_html=True)

    section_title("System workflow")
    st.markdown(
        """
        <div class="steps">
            <div class="step"><b>1</b><div>Transaction input</div><span>CSV upload</span></div>
            <div class="step"><b>2</b><div>Data validation</div><span>Schema and value checks</span></div>
            <div class="step"><b>3</b><div>Feature scaling</div><span>Prepare model inputs</span></div>
            <div class="step"><b>4</b><div>ML prediction</div><span>Fraud probability</span></div>
            <div class="step"><b>5</b><div>Risk classification</div><span>Low, medium, high</span></div>
            <div class="step"><b>6</b><div>Fraud alert</div><span>Review queue</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    section_title("How to read risk levels")
    callout("success", "<b>Low risk:</b> the transaction looks normal based on the model score.")
    callout("warning", "<b>Medium risk:</b> extra verification is recommended.")
    callout("error", "<b>High risk:</b> possible fraud that needs investigation.")

    st.caption(
        "Model predictions support decisions and are not proof of fraud. "
        "Risk thresholds need validation against appropriate evaluation data."
    )


# ==========================================================
# 14. FOOTER
# ==========================================================

st.divider()
st.markdown(
    '<div class="footer">Sentinel AI | Intelligent Financial Fraud Detection System</div>',
    unsafe_allow_html=True,
)