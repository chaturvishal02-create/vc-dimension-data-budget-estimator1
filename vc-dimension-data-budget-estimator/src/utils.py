# src/utils.py
# ─────────────────────────────────────────────────────────────────
# UI/UX helpers, theme stylers, and numerical formatting functions.
# ─────────────────────────────────────────────────────────────────

from typing import Union
import streamlit as st


def format_number(val: Union[int, float, None], decimals: int = 2) -> str:
    """Format numbers cleanly with comma separators."""
    if val is None or (isinstance(val, float) and (np_isnan := (val != val))):
        return "N/A"
    if isinstance(val, (int, float)):
        if abs(val) >= 1000:
            return f"{val:,.0f}"
        if isinstance(val, int) or val.is_integer():
            return f"{int(val):,}"
        return f"{val:.{decimals}f}"
    return str(val)


def format_percent(val: Union[float, None], decimals: int = 1) -> str:
    """Format float into clean percentage string."""
    if val is None:
        return "N/A"
    return f"{val * 100:.{decimals}f}%"


def format_currency(val: Union[int, float, None]) -> str:
    """Format float into standard USD currency string."""
    if val is None:
        return "N/A"
    return f"${val:,.2f}"


def inject_custom_css(is_dark: bool = False):
    """Inject responsive CSS styles supporting Light and Dark modes."""
    bg_card = "#1E293B" if is_dark else "#FFFFFF"
    border_color = "#334155" if is_dark else "#E2E8F0"
    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#64748B"
    bg_subtle = "#0F172A" if is_dark else "#F8FAFC"
    badge_bg = "#312E81" if is_dark else "#EEF2FF"
    badge_text = "#A5B4FC" if is_dark else "#4338CA"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-top: 12px;
        margin-bottom: 24px;
    }}
    .kpi-card {{
        background: {bg_card};
        border: 1px solid {border_color};
        border-radius: 10px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }}
    .kpi-label {{
        font-size: 0.76rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {text_secondary};
        margin-bottom: 6px;
    }}
    .kpi-value {{
        font-size: 1.55rem;
        font-weight: 700;
        color: {text_primary};
        line-height: 1.2;
    }}
    .kpi-sub {{
        font-size: 0.8rem;
        color: {text_secondary};
        margin-top: 4px;
    }}

    .formula-box {{
        background: {bg_subtle};
        border: 1px solid {border_color};
        border-left: 4px solid #6366F1;
        border-radius: 8px;
        padding: 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.88rem;
        color: {text_primary};
        margin: 12px 0;
        white-space: pre-wrap;
    }}

    .info-card {{
        background: {bg_subtle};
        border: 1px solid {border_color};
        border-left: 4px solid #0EA5E9;
        border-radius: 8px;
        padding: 14px 16px;
        color: {text_primary};
        font-size: 0.88rem;
        line-height: 1.5;
        margin-bottom: 14px;
    }}

    .success-card {{
        background: {'#064E3B' if is_dark else '#F0FDF4'};
        border: 1px solid {'#047857' if is_dark else '#BBF7D0'};
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 14px 16px;
        color: {'#ECFDF5' if is_dark else '#14532D'};
        font-size: 0.88rem;
        margin-bottom: 14px;
    }}

    .warning-card {{
        background: {'#78350F' if is_dark else '#FFFBEB'};
        border: 1px solid {'#B45309' if is_dark else '#FDE68A'};
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 14px 16px;
        color: {'#FEF3C7' if is_dark else '#78350F'};
        font-size: 0.88rem;
        margin-bottom: 14px;
    }}

    .badge {{
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        background: {badge_bg};
        color: {badge_text};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

