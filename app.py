import base64
import io
import math
from html import escape
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="M-Guide | Maternal Diabetes Prevention",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# DESIGN
# =========================================================
CUSTOM_CSS = """
<style>
    :root {
        --ink: #132238;
        --muted: #5b677a;
        --line: #dce3ea;
        --soft: #f6f8fb;
        --card: #ffffff;
        --blue: #1769aa;
        --teal: #0f8f83;
        --rose: #c85268;
        --green-bg: #e9f7ef;
        --green: #17633a;
        --amber-bg: #fff6dc;
        --amber: #7b5608;
        --red-bg: #fdecef;
        --red: #9d1f33;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-size: 16px;
        color: var(--ink);
    }

    .stApp {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 1rem;
        padding-bottom: 2.2rem;
    }

    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--line);
    }

    .app-header {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 1rem;
        align-items: center;
        padding: 1rem 0 0.7rem 0;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1rem;
    }

    .brand-kicker {
        color: var(--blue);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .brand-title {
        color: var(--ink);
        font-size: 2rem;
        line-height: 1.08;
        font-weight: 850;
        margin: 0;
    }

    .brand-subtitle {
        max-width: 780px;
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.55;
        margin-top: 0.42rem;
    }

    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 0.7rem 0 1rem 0;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        min-height: 2rem;
        padding: 0.3rem 0.7rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: white;
        color: var(--ink);
        font-size: 0.86rem;
        font-weight: 700;
    }

    .notice {
        padding: 0.85rem 1rem;
        background: #fff9e8;
        border: 1px solid #f0dfaa;
        border-radius: 8px;
        color: #4d3804;
        line-height: 1.55;
        margin-bottom: 1rem;
    }

    .panel {
        background: white;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        height: 100%;
    }

    .panel-title {
        color: var(--ink);
        font-size: 1.16rem;
        line-height: 1.2;
        font-weight: 850;
        margin-bottom: 0.25rem;
    }

    .panel-text {
        color: var(--muted);
        font-size: 0.93rem;
        line-height: 1.55;
        margin-bottom: 0.9rem;
    }

    .result-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.8rem;
    }

    .result-kicker {
        color: var(--muted);
        font-size: 0.82rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }

    .result-value {
        color: var(--ink);
        font-size: 2.5rem;
        line-height: 1;
        font-weight: 900;
        margin: 0.2rem 0 0.5rem 0;
    }

    .risk-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 0.32rem 0.7rem;
        font-size: 0.84rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .risk-low { background: var(--green-bg); color: var(--green); border-color: #b8dec8; }
    .risk-mod { background: var(--amber-bg); color: var(--amber); border-color: #ead28a; }
    .risk-high { background: var(--red-bg); color: var(--red); border-color: #efbac3; }

    .meter {
        height: 0.65rem;
        border-radius: 999px;
        background: #e7edf3;
        overflow: hidden;
        margin: 0.8rem 0 0.45rem 0;
    }

    .meter-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--teal), var(--blue), var(--rose));
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.88rem;
        line-height: 1.5;
    }

    .list-clean {
        margin: 0.4rem 0 0 0;
        padding-left: 1.1rem;
    }

    .list-clean li {
        margin-bottom: 0.38rem;
        line-height: 1.5;
    }

    .report-card {
        background: white;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.8rem;
    }

    div[data-testid="stForm"] {
        border: 0;
        padding: 0;
        background: transparent;
    }

    div[data-testid="stMetric"] {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.85rem;
    }

    @media (max-width: 760px) {
        .block-container {
            padding-left: 0.9rem;
            padding-right: 0.9rem;
        }
        .app-header {
            display: block;
        }
        .brand-title {
            font-size: 1.55rem;
        }
        .result-value {
            font-size: 2rem;
        }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

ADVANCED_CSS = """
<style>
    .block-container {
        max-width: 1320px;
    }

    p, li, label, label p,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: var(--ink) !important;
    }

    [data-testid="stWidgetLabel"] p {
        font-weight: 760 !important;
        font-size: 0.9rem !important;
    }

    section[data-testid="stSidebar"] {
        background: #fbfcfe;
    }

    section[data-testid="stSidebar"] button {
        border-radius: 8px !important;
        font-weight: 800 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input {
        border-radius: 8px !important;
    }

    .hero-banner {
        display: grid;
        grid-template-columns: minmax(0, 1.2fr) minmax(280px, 0.8fr);
        gap: 1.2rem;
        align-items: stretch;
        margin: 0 0 1rem 0;
    }

    .hero-main {
        background: linear-gradient(135deg, #10284f 0%, #1769aa 52%, #21a8b6 100%);
        border-radius: 16px;
        padding: 1.35rem 1.45rem;
        color: #fff;
        min-height: 230px;
        box-shadow: 0 18px 44px rgba(19, 34, 56, 0.16);
    }

    .hero-main p,
    .hero-main span,
    .hero-main div {
        color: #fff !important;
    }

    .hero-kicker {
        display: inline-flex;
        border: 1px solid rgba(255, 255, 255, 0.3);
        border-radius: 999px;
        padding: 0.28rem 0.68rem;
        font-size: 0.78rem;
        font-weight: 850;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
        background: rgba(255, 255, 255, 0.12);
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.02;
        font-weight: 900;
        margin: 0 0 0.75rem 0;
        color: #fff;
    }

    .hero-copy {
        max-width: 880px;
        font-size: 1.03rem;
        line-height: 1.62;
        opacity: 0.96;
    }

    .hero-side {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 1.1rem;
        min-height: 230px;
        box-shadow: 0 12px 32px rgba(19, 34, 56, 0.08);
    }

    .side-title {
        color: var(--ink);
        font-size: 1.2rem;
        font-weight: 850;
        margin-bottom: 0.75rem;
    }

    .pathway-row {
        display: grid;
        grid-template-columns: 44px 1fr;
        gap: 0.75rem;
        align-items: start;
        padding: 0.66rem 0;
        border-top: 1px solid #edf1f5;
    }

    .pathway-row:first-of-type {
        border-top: 0;
        padding-top: 0;
    }

    .pathway-num {
        width: 36px;
        height: 36px;
        display: inline-grid;
        place-items: center;
        border-radius: 10px;
        background: #edf5ff;
        color: var(--blue);
        font-weight: 900;
    }

    .pathway-title {
        color: var(--ink);
        font-size: 0.96rem;
        font-weight: 850;
        margin-bottom: 0.12rem;
    }

    .pathway-text {
        color: var(--muted) !important;
        font-size: 0.86rem;
        line-height: 1.42;
    }

    .stage-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.95rem 0 1.1rem 0;
    }

    .stage-card {
        background: white;
        border: 1px solid var(--line);
        border-top: 5px solid var(--blue);
        border-radius: 12px;
        padding: 1rem;
        min-height: 260px;
        box-shadow: 0 10px 30px rgba(19, 34, 56, 0.06);
    }

    .stage-card.active {
        border-color: #8ab8dc;
        border-top-color: var(--blue);
        box-shadow: 0 16px 40px rgba(23, 105, 170, 0.16);
    }

    .stage-card.teal { border-top-color: var(--teal); }
    .stage-card.rose { border-top-color: var(--rose); }

    .stage-chip {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        background: #f1f6fb;
        color: var(--ink);
        padding: 0.28rem 0.58rem;
        font-size: 0.78rem;
        font-weight: 850;
        margin-bottom: 0.58rem;
    }

    .stage-name {
        color: var(--ink);
        font-size: 1.34rem;
        line-height: 1.12;
        font-weight: 900;
        margin-bottom: 0.4rem;
    }

    .stage-target {
        color: var(--blue);
        font-size: 0.92rem;
        font-weight: 850;
        line-height: 1.45;
        margin-bottom: 0.55rem;
    }

    .stage-meta {
        display: grid;
        gap: 0.42rem;
        margin-top: 0.65rem;
    }

    .meta-line {
        color: var(--muted) !important;
        font-size: 0.85rem;
        line-height: 1.4;
    }

    .meta-line strong {
        color: var(--ink);
    }

    .module-detail {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
    }

    .module-detail-head {
        display: flex;
        justify-content: space-between;
        gap: 0.8rem;
        align-items: flex-start;
        margin-bottom: 0.85rem;
    }

    .module-detail-title {
        color: var(--ink);
        font-size: 1.42rem;
        line-height: 1.15;
        font-weight: 900;
    }

    .module-detail-text {
        color: var(--muted) !important;
        font-size: 0.94rem;
        line-height: 1.55;
        margin-top: 0.3rem;
    }

    .model-badge {
        border-radius: 999px;
        background: #eef6ff;
        color: var(--blue);
        padding: 0.34rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 850;
        white-space: nowrap;
    }

    .info-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.6rem;
    }

    .info-tile {
        background: #f8fafc;
        border: 1px solid #e5ebf2;
        border-radius: 10px;
        padding: 0.78rem;
    }

    .info-label {
        color: var(--muted) !important;
        font-size: 0.74rem;
        font-weight: 850;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .info-value {
        color: var(--ink);
        font-size: 0.9rem;
        line-height: 1.42;
        font-weight: 720;
    }

    .result-card.primary-result {
        border-top: 5px solid var(--blue);
    }

    .result-stage {
        color: var(--blue);
        font-size: 0.82rem;
        font-weight: 900;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }

    .form-caption {
        color: var(--muted) !important;
        font-size: 0.86rem;
        line-height: 1.45;
        margin: -0.25rem 0 0.85rem 0;
    }

    @media (max-width: 980px) {
        .hero-banner,
        .stage-grid,
        .info-grid {
            grid-template-columns: 1fr;
        }
        .module-detail-head {
            display: block;
        }
        .model-badge {
            display: inline-flex;
            margin-top: 0.65rem;
        }
    }
</style>
"""
st.markdown(ADVANCED_CSS, unsafe_allow_html=True)


# =========================================================
# PATHS / CONSTANTS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "All_data_Catboost_GDM_ML_model.bin"
SCALER_FILE = BASE_DIR / "All_data_scaler.pkl"
ASSETS_DIR = BASE_DIR / "assets"
LOGO_CANDIDATES = [
    ASSETS_DIR / "custom_logo.png",
    ASSETS_DIR / "custom_logo.jpg",
    ASSETS_DIR / "custom_logo.jpeg",
    ASSETS_DIR / "custom_logo.svg",
    ASSETS_DIR / "logo_placeholder.svg",
]

FEATURE_COLUMNS = [
    "Height",
    "Weight",
    "Parity",
    "Age",
    "Caucasian",
    "Oceanian_not_WANZ",
    "ME_NA_SSA",
    "South_Central_Asia",
    "SouthEast_NorthEast_Asia",
    "Other NEC",
    "Family_Hist_DM",
    "Past_Hist_GDM",
    "Past_Hist_Obs_Complica",
]

RECODE_DICT = {
    "Caucasian": "Caucasian",
    "Oceanian (not white-Australian or white-New Zealander)": "Oceanian_not_WANZ",
    "Middle-Eastern, North African, or Sub-Saharan African": "ME_NA_SSA",
    "Southern and Central Asian": "South_Central_Asia",
    "South-East and North-East Asian": "SouthEast_NorthEast_Asia",
    "Other NEC": "Other NEC",
}

DEFAULTS = {
    "active_module": "Booking visit",
    "patient_name": "Demo patient",
    "gdm_status": "Not diagnosed / screening stage",
    "age": 30,
    "height": 165,
    "weight": 70.0,
    "parity": 1,
    "ethnicity_group": "Southern and Central Asian",
    "family_hist_dm": 1,
    "past_hist_gdm": 0,
    "past_hist_obs_complica": 0,
    "anc_threshold": 0.50,
    "antenatal_fpg": 5.5,
    "antenatal_2h_ogtt": 8.6,
    "recurrent_gdm": 0,
    "insulin_treatment": 0,
    "irregular_menses": 0,
    "post_view_antenatal_2h_ogtt": 8.6,
    "postnatal_fpg": 5.2,
    "postnatal_2h_ogtt": 7.4,
    "postnatal_bmi": 27.0,
    "anc_prob": None,
    "anc_pred": None,
    "ant_prob": None,
    "post_prob": None,
    "booking_feature_frame": pd.DataFrame(),
    "institution_name": "MCHRI",
    "contact_email": "yitayeh.mengistu@monash.edu",
    "public_app_url": "https://m-guide-maternal-diabetes-prevention-platform-na7gro2wekpgin3k.streamlit.app/",
    "github_url": "https://github.com/YitayehMengistu/M-Guide-Maternal-Diabetes-Prevention-Platform",
    "publication_url": "https://doi.org/10.1016/j.ijmedinf.2023.105228",
    "publication_url_secondary": "https://doi.org/10.1016/j.clnu.2024.06.006",
    "app_tagline": "Pregnancy-to-postpartum diabetes risk support",
    "model_version": "Prototype v3.1",
    "report_note": "Research demo only. Decision support and presentation use only.",
}

YES_NO = {0: "No", 1: "Yes"}
MODULES = ["Booking visit", "Pregnancy after GDM", "Postnatal review"]
STAGE_DETAILS = {
    "Booking visit": {
        "number": "01",
        "theme": "blue",
        "short_name": "Booking GDM risk",
        "card_title": "Booking visit",
        "prediction": "Predicts risk of developing GDM by routine 24-28 week screening.",
        "model": "Saved CatBoost model + scaler",
        "population": "Women at first antenatal booking",
        "inputs": "Age, height, weight, parity, ethnicity, diabetes family history, past GDM, obstetric history",
        "output": "GDM risk percentage, risk band, and booking-stage action recommendation",
        "threshold": "Adjustable demo threshold; default 0.50",
        "button": "Use booking predictor",
        "intro": "Use routine booking information to support earlier prevention conversations before standard GDM screening.",
        "result_label": "Prediction 1 | Booking-stage GDM risk",
    },
    "Pregnancy after GDM": {
        "number": "02",
        "theme": "teal",
        "short_name": "After-GDM T2DM risk",
        "card_title": "After GDM in pregnancy",
        "prediction": "Predicts future type 2 diabetes risk after delivery among women with GDM.",
        "model": "Published antenatal logistic equation",
        "population": "Women with GDM during pregnancy",
        "inputs": "Antenatal FPG, antenatal 2-hour OGTT, recurrent GDM, insulin treatment, irregular menses, parity, family history",
        "output": "Future T2DM risk percentage and postpartum prevention plan",
        "threshold": "Published action threshold 0.096",
        "button": "Use antenatal predictor",
        "intro": "Estimate post-delivery type 2 diabetes risk before birth, when postpartum follow-up planning can still be arranged.",
        "result_label": "Prediction 2 | Future T2DM risk after GDM",
    },
    "Postnatal review": {
        "number": "03",
        "theme": "rose",
        "short_name": "Postnatal T2DM update",
        "card_title": "Postnatal follow-up",
        "prediction": "Updates long-term future type 2 diabetes risk using postpartum test results.",
        "model": "Published postnatal logistic equation",
        "population": "Women after a pregnancy affected by GDM",
        "inputs": "Linked antenatal 2-hour OGTT, postnatal FPG, postnatal 2-hour OGTT, postnatal BMI",
        "output": "Updated future T2DM risk percentage and longer-term follow-up recommendation",
        "threshold": "Published action threshold 0.086",
        "button": "Use postnatal predictor",
        "intro": "Refresh risk after delivery using postpartum glucose and BMI information, then document a longer-term prevention pathway.",
        "result_label": "Prediction 3 | Postnatal future T2DM update",
    },
}

EXPERIENCE_CSS = """
<style>
    .platform-summary {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.85rem 0 1rem 0;
    }

    .summary-metric {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.86rem 0.9rem;
        box-shadow: 0 8px 24px rgba(19, 34, 56, 0.05);
    }

    .summary-label {
        color: var(--muted) !important;
        font-size: 0.76rem;
        font-weight: 850;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.22rem;
    }

    .summary-value {
        color: var(--ink);
        font-size: 1.08rem;
        line-height: 1.25;
        font-weight: 900;
    }

    .summary-note {
        color: var(--muted) !important;
        font-size: 0.78rem;
        line-height: 1.35;
        margin-top: 0.25rem;
    }

    .stage-card {
        padding: 0 !important;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        min-height: 338px;
    }

    .stage-top {
        padding: 1rem 1rem 0.95rem 1rem;
        background: linear-gradient(135deg, #1b4f86 0%, #2274b7 100%);
    }

    .stage-card.teal .stage-top {
        background: linear-gradient(135deg, #0a756f 0%, #20a98d 100%);
    }

    .stage-card.rose .stage-top {
        background: linear-gradient(135deg, #b1445a 0%, #e596a1 100%);
    }

    .stage-top .stage-chip,
    .stage-top .stage-name,
    .stage-top .stage-target {
        color: #fff !important;
    }

    .stage-top .stage-chip {
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.24);
    }

    .stage-top .stage-target {
        opacity: 0.96;
        margin-bottom: 0;
    }

    .stage-body {
        padding: 0.9rem 1rem 1rem 1rem;
        display: flex;
        flex: 1;
        flex-direction: column;
    }

    .stage-status {
        margin-top: auto;
        border-top: 1px solid #edf1f5;
        padding-top: 0.7rem;
        display: flex;
        justify-content: space-between;
        gap: 0.75rem;
        align-items: center;
    }

    .stage-status-label {
        color: var(--muted) !important;
        font-size: 0.75rem;
        font-weight: 850;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    .stage-status-value {
        color: var(--ink);
        font-size: 0.92rem;
        font-weight: 900;
        text-align: right;
    }

    .interpret-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 0.75rem 0 0.2rem 0;
    }

    .interpret-box {
        background: #f8fafc;
        border: 1px solid #e5ebf2;
        border-radius: 10px;
        padding: 0.75rem;
    }

    @media (max-width: 980px) {
        .platform-summary,
        .interpret-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
"""
st.markdown(EXPERIENCE_CSS, unsafe_allow_html=True)


# =========================================================
# STATE
# =========================================================
def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.active_module not in MODULES:
        legacy_modules = {
            "booking": "Booking visit",
            "antenatal": "Pregnancy after GDM",
            "postnatal": "Postnatal review",
        }
        st.session_state.active_module = legacy_modules.get(str(st.session_state.active_module), DEFAULTS["active_module"])


def load_demo_patient() -> None:
    demo = {
        "active_module": "Booking visit",
        "patient_name": "Demo patient",
        "gdm_status": "GDM confirmed",
        "age": 33,
        "height": 160,
        "weight": 78.5,
        "parity": 2,
        "ethnicity_group": "Southern and Central Asian",
        "family_hist_dm": 1,
        "past_hist_gdm": 1,
        "past_hist_obs_complica": 1,
        "antenatal_fpg": 5.8,
        "antenatal_2h_ogtt": 9.6,
        "recurrent_gdm": 1,
        "insulin_treatment": 1,
        "irregular_menses": 1,
        "post_view_antenatal_2h_ogtt": 9.6,
        "postnatal_fpg": 5.2,
        "postnatal_2h_ogtt": 9.6,
        "postnatal_bmi": 32.2,
    }
    for key, value in demo.items():
        st.session_state[key] = value


def reset_all() -> None:
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


init_state()


# =========================================================
# HELPERS
# =========================================================
def yes_no(value: int) -> str:
    return YES_NO[int(value)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def first_existing_logo() -> Optional[Path]:
    for path in LOGO_CANDIDATES:
        if path.exists():
            return path
    return None


def recode_ethnicity(group_label: str) -> str:
    return RECODE_DICT.get(group_label, "Other NEC")


def safe_link(url: str, label: str) -> str:
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return escape(label)
    return f'<a href="{escape(url)}" target="_blank">{escape(label)}</a>'


def booking_risk_band(prob: float) -> str:
    if prob < 0.10:
        return "Low"
    if prob < 0.20:
        return "Moderate"
    if prob < 0.35:
        return "High"
    return "Very high"


def published_model_band(prob: float, threshold: float) -> str:
    if prob < threshold:
        return "Low"
    if prob < 0.20:
        return "Moderate"
    return "High"


def action_label(prob: float, threshold: float) -> str:
    return "Above action threshold" if prob >= threshold else "Below action threshold"


def action_intensity(prob: float, threshold: float) -> str:
    if prob < threshold:
        return "Low"
    if prob < 0.20:
        return "Medium"
    return "High"


def risk_css_class(label: str) -> str:
    return "risk-low" if label == "Low" else "risk-mod" if label == "Moderate" else "risk-high"


def risk_pill(label: str) -> str:
    return f'<span class="risk-pill {risk_css_class(label)}">{escape(label)}</span>'


@st.cache_data(show_spinner=False)
def qr_image(url: str) -> Optional[bytes]:
    payload = (url or "").strip()
    if not payload.startswith(("http://", "https://")):
        return None
    import qrcode

    qr = qrcode.QRCode(version=1, box_size=6, border=2)
    qr.add_data(payload)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white")
    if hasattr(qr_img, "get_image"):
        qr_img = qr_img.get_image()
    buf = io.BytesIO()
    qr_img.save(buf, format="PNG")
    return buf.getvalue()


def qr_data_uri(url: str) -> str:
    img_bytes = qr_image(url)
    if img_bytes is None:
        return ""
    encoded = base64.b64encode(img_bytes).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


# =========================================================
# MODEL LOADING / PREDICTION
# =========================================================
@st.cache_resource(show_spinner=False)
def load_booking_assets():
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        return None, None, "Booking model or scaler file was not found in the app folder."
    import joblib
    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(MODEL_FILE))
    scaler = joblib.load(str(SCALER_FILE))
    return model, scaler, None


def build_booking_features() -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLUMNS}
    row["Height"] = int(st.session_state.height)
    row["Weight"] = float(st.session_state.weight)
    row["Parity"] = int(st.session_state.parity)
    row["Age"] = int(st.session_state.age)
    row["Family_Hist_DM"] = int(st.session_state.family_hist_dm)
    row["Past_Hist_GDM"] = int(st.session_state.past_hist_gdm)
    row["Past_Hist_Obs_Complica"] = int(st.session_state.past_hist_obs_complica)
    row[recode_ethnicity(str(st.session_state.ethnicity_group))] = 1

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    df["Height"] = df["Height"].astype(int)
    df["Weight"] = df["Weight"].astype(float)
    df["Parity"] = df["Parity"].astype("int8")
    df["Age"] = df["Age"].astype(int)
    for col in FEATURE_COLUMNS[4:]:
        df[col] = df[col].astype(int)
    return df


def predict_booking_risk():
    model, scaler, error = load_booking_assets()
    if model is None or scaler is None:
        raise RuntimeError(error or "Booking model is not available.")
    features = build_booking_features()
    scaled_features = scaler.transform(features)
    prob = float(model.predict_proba(scaled_features)[0, 1])
    pred = int(prob >= float(st.session_state.anc_threshold))
    return prob, pred, features


def predict_antenatal_t2dm_after_gdm() -> float:
    logit = (
        -10.0757
        + 0.7086 * float(st.session_state.antenatal_fpg)
        + 0.3656 * float(st.session_state.antenatal_2h_ogtt)
        + 0.3190 * int(st.session_state.recurrent_gdm)
        + 0.5100 * int(st.session_state.insulin_treatment)
        + 0.3526 * int(st.session_state.parity)
        + 1.0922 * int(st.session_state.irregular_menses)
        + 0.0972 * int(st.session_state.family_hist_dm)
    )
    return sigmoid(logit)


def predict_postnatal_t2dm_after_gdm() -> float:
    logit = (
        -15.3625
        + 0.3008 * float(st.session_state.post_view_antenatal_2h_ogtt)
        + 1.0033 * float(st.session_state.postnatal_fpg)
        + 0.5581 * float(st.session_state.postnatal_2h_ogtt)
        + 0.0359 * float(st.session_state.postnatal_bmi)
    )
    return sigmoid(logit)


# =========================================================
# RECOMMENDATIONS
# =========================================================
def booking_action_payload(prob: float, threshold: float) -> dict:
    intensity = action_intensity(prob, threshold)
    reasons = []
    if int(st.session_state.family_hist_dm) == 1:
        reasons.append("family history of diabetes")
    if int(st.session_state.past_hist_gdm) == 1:
        reasons.append("past history of GDM")
    if int(st.session_state.past_hist_obs_complica) == 1:
        reasons.append("past obstetric complications")
    if int(st.session_state.age) >= 35:
        reasons.append("maternal age 35 years or above")
    if int(st.session_state.parity) >= 2:
        reasons.append("multiparity")

    if intensity == "Low":
        actions = [
            "Continue routine antenatal care and routine GDM screening at 24-28 weeks.",
            "Reinforce healthy nutrition, activity, and weight monitoring early in pregnancy.",
        ]
        follow_up = "Routine GDM screening at 24-28 weeks."
    elif intensity == "Medium":
        actions = [
            "Provide targeted lifestyle counselling and reinforce attendance for screening.",
            "Document the elevated booking risk for timely care-team review.",
        ]
        follow_up = "Enhanced antenatal review; consider earlier testing if local policy supports it."
    else:
        actions = [
            "Flag as high risk for GDM and consider earlier glucose assessment according to local policy.",
            "Provide intensified lifestyle counselling and closer antenatal follow-up.",
        ]
        follow_up = "Earlier antenatal review and prevention-focused care planning."

    note = None
    if str(st.session_state.ethnicity_group) in {
        "Southern and Central Asian",
        "South-East and North-East Asian",
        "Middle-Eastern, North African, or Sub-Saharan African",
    }:
        note = "Offer culturally appropriate dietary counselling and communication support where available."

    return {
        "intensity": intensity,
        "actions": actions,
        "reasons": reasons or ["current predicted risk level"],
        "follow_up": follow_up,
        "tailored_note": note,
    }


def antenatal_action_payload(prob: float, threshold: float) -> dict:
    intensity = action_intensity(prob, threshold)
    reasons = []
    if float(st.session_state.antenatal_fpg) >= 5.6:
        reasons.append("higher antenatal fasting plasma glucose")
    if float(st.session_state.antenatal_2h_ogtt) >= 8.5:
        reasons.append("higher antenatal 2-hour OGTT")
    if int(st.session_state.recurrent_gdm) == 1:
        reasons.append("recurrent GDM")
    if int(st.session_state.insulin_treatment) == 1:
        reasons.append("insulin treatment during pregnancy")
    if int(st.session_state.irregular_menses) == 1:
        reasons.append("history of irregular menstrual cycles")
    if int(st.session_state.family_hist_dm) == 1:
        reasons.append("family history of diabetes")

    if intensity == "Low":
        actions = [
            "Plan routine postpartum glucose follow-up.",
            "Document postpartum OGTT timing in the discharge plan.",
        ]
        follow_up = "Routine postpartum OGTT and usual primary-care follow-up."
    elif intensity == "Medium":
        actions = [
            "Create a structured postpartum follow-up plan before delivery.",
            "Provide targeted counselling about future type 2 diabetes prevention.",
        ]
        follow_up = "Structured postpartum plan with a clear testing date."
    else:
        actions = [
            "Arrange enhanced postpartum follow-up and active recall for testing.",
            "Consider referral to lifestyle, weight-management, or diabetes-prevention services.",
        ]
        follow_up = "Enhanced postpartum follow-up and ongoing diabetes-prevention review."

    return {
        "intensity": intensity,
        "actions": actions,
        "reasons": reasons or ["current predicted risk level"],
        "follow_up": follow_up,
        "tailored_note": None,
    }


def postnatal_action_payload(prob: float, threshold: float) -> dict:
    intensity = action_intensity(prob, threshold)
    reasons = []
    if float(st.session_state.postnatal_fpg) >= 5.6:
        reasons.append("elevated postnatal fasting glucose")
    if float(st.session_state.postnatal_2h_ogtt) >= 7.8:
        reasons.append("elevated postnatal 2-hour glucose")
    if float(st.session_state.postnatal_bmi) >= 30:
        reasons.append("BMI in the obesity range")
    elif float(st.session_state.postnatal_bmi) >= 25:
        reasons.append("BMI above the healthy range")
    if float(st.session_state.post_view_antenatal_2h_ogtt) >= 8.5:
        reasons.append("higher antenatal 2-hour OGTT")

    if intensity == "Low":
        actions = [
            "Continue routine diabetes-prevention advice and periodic glycaemic surveillance.",
            "Reinforce healthy eating, activity, and weight-management goals.",
        ]
        follow_up = "Routine repeat glycaemic surveillance in primary care."
    elif intensity == "Medium":
        actions = [
            "Provide targeted diabetes-prevention counselling and closer primary-care follow-up.",
            "Encourage repeat glucose testing and weight-management support.",
        ]
        follow_up = "Closer primary-care follow-up with repeat glucose testing."
    else:
        actions = [
            "Escalate diabetes-prevention follow-up and regular glycaemic monitoring.",
            "Consider referral for structured lifestyle intervention or specialist review.",
        ]
        follow_up = "Enhanced diabetes-prevention follow-up and active recall."

    return {
        "intensity": intensity,
        "actions": actions,
        "reasons": reasons or ["current predicted risk level"],
        "follow_up": follow_up,
        "tailored_note": None,
    }


# =========================================================
# RENDER HELPERS
# =========================================================
def render_header() -> None:
    logo_path = first_existing_logo()
    st.markdown(
        f"""
        <div class="hero-banner">
            <div class="hero-main">
                <div class="hero-kicker">{escape(st.session_state.model_version)} | Research demo</div>
                <h1 class="hero-title">M-Guide Maternal Diabetes Prevention Platform</h1>
                <div class="hero-copy">
                    {escape(st.session_state.app_tagline)} across the pregnancy continuum: booking risk of GDM,
                    future type 2 diabetes risk after GDM, and postnatal risk update using postpartum results.
                </div>
            </div>
            <div class="hero-side">
                <div class="side-title">Three linked prediction stages</div>
                <div class="pathway-row">
                    <div class="pathway-num">01</div>
                    <div><div class="pathway-title">Booking visit</div><div class="pathway-text">Predict developing GDM before routine 24-28 week screening.</div></div>
                </div>
                <div class="pathway-row">
                    <div class="pathway-num">02</div>
                    <div><div class="pathway-title">Pregnancy after GDM</div><div class="pathway-text">Estimate future T2DM risk and plan postpartum prevention before delivery.</div></div>
                </div>
                <div class="pathway-row">
                    <div class="pathway-num">03</div>
                    <div><div class="pathway-title">Postnatal review</div><div class="pathway-text">Update long-term T2DM risk using postpartum glucose and BMI.</div></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if logo_path is not None:
        st.sidebar.image(str(logo_path), width=160)

    st.markdown(
        f"""
        <div class="status-row">
            <span class="chip">Patient: {escape(str(st.session_state.patient_name))}</span>
            <span class="chip">Status: {escape(str(st.session_state.gdm_status))}</span>
            <span class="chip">Completed: {modules_completed()}/3</span>
            <span class="chip">Active prediction: {escape(STAGE_DETAILS[str(st.session_state.active_module)]["short_name"])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice"><strong>Research demo only.</strong> This platform is not a stand-alone diagnostic or clinical deployment tool without governance, local validation, security review, and approval.</div>',
        unsafe_allow_html=True,
    )


def stage_result_status(module_name: str) -> str:
    if module_name == "Booking visit" and st.session_state.anc_prob is not None:
        prob = float(st.session_state.anc_prob)
        return f"{prob:.1%} | {booking_risk_band(prob)}"
    if module_name == "Pregnancy after GDM" and st.session_state.ant_prob is not None:
        prob = float(st.session_state.ant_prob)
        return f"{prob:.1%} | {published_model_band(prob, 0.096)}"
    if module_name == "Postnatal review" and st.session_state.post_prob is not None:
        prob = float(st.session_state.post_prob)
        return f"{prob:.1%} | {published_model_band(prob, 0.086)}"
    return "Not run yet"


def render_platform_summary() -> None:
    active_details = STAGE_DETAILS[str(st.session_state.active_module)]
    report_state = "Ready for download" if modules_completed() > 0 else "Awaiting prediction"
    st.markdown(
        f"""
        <div class="platform-summary">
            <div class="summary-metric">
                <div class="summary-label">Pathway progress</div>
                <div class="summary-value">{modules_completed()}/3 predictions complete</div>
                <div class="summary-note">Booking, after-GDM pregnancy, and postnatal review.</div>
            </div>
            <div class="summary-metric">
                <div class="summary-label">Active module</div>
                <div class="summary-value">{escape(active_details["short_name"])}</div>
                <div class="summary-note">{escape(active_details["population"])}</div>
            </div>
            <div class="summary-metric">
                <div class="summary-label">Thresholds</div>
                <div class="summary-value">Booking {float(st.session_state.anc_threshold):.2f}</div>
                <div class="summary-note">Published T2DM thresholds: 0.096 and 0.086.</div>
            </div>
            <div class="summary-metric">
                <div class="summary-label">Summary report</div>
                <div class="summary-value">{escape(report_state)}</div>
                <div class="summary-note">CSV and HTML export remain available below.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stage_cards() -> None:
    st.markdown('<div class="brand-kicker" style="margin-top:0.2rem;">Choose a prediction module</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for col, module_name in zip(cols, MODULES):
        details = STAGE_DETAILS[module_name]
        active = "active" if st.session_state.active_module == module_name else ""
        theme = details["theme"]
        status = stage_result_status(module_name)
        with col:
            st.markdown(
                f"""
                <div class="stage-card {theme} {active}">
                    <div class="stage-top">
                        <div class="stage-chip">Prediction {escape(details["number"])}</div>
                        <div class="stage-name">{escape(details["card_title"])}</div>
                        <div class="stage-target">{escape(details["prediction"])}</div>
                    </div>
                    <div class="stage-body">
                        <div class="stage-meta">
                            <div class="meta-line"><strong>Model:</strong> {escape(details["model"])}</div>
                            <div class="meta-line"><strong>For:</strong> {escape(details["population"])}</div>
                            <div class="meta-line"><strong>Output:</strong> {escape(details["output"])}</div>
                        </div>
                        <div class="stage-status">
                            <div class="stage-status-label">Current result</div>
                            <div class="stage-status-value">{escape(status)}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(details["button"], key=f"select_{module_name}", use_container_width=True):
                st.session_state.active_module = module_name
                st.rerun()


def render_module_intro(module_name: str) -> None:
    details = STAGE_DETAILS[module_name]
    st.markdown(
        f"""
        <div class="module-detail">
            <div class="module-detail-head">
                <div>
                    <div class="brand-kicker">Prediction {escape(details["number"])}</div>
                    <div class="module-detail-title">{escape(details["short_name"])}</div>
                    <div class="module-detail-text">{escape(details["intro"])}</div>
                </div>
                <div class="model-badge">{escape(details["model"])}</div>
            </div>
            <div class="info-grid">
                <div class="info-tile">
                    <div class="info-label">Prediction target</div>
                    <div class="info-value">{escape(details["prediction"])}</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Intended patient point</div>
                    <div class="info-value">{escape(details["population"])}</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Inputs used</div>
                    <div class="info-value">{escape(details["inputs"])}</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Action threshold</div>
                    <div class="info-value">{escape(details["threshold"])}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(prob: float, band: str, threshold: float, title: str, subtitle: str, payload: dict, module_name: str) -> None:
    pct = max(0.0, min(100.0, prob * 100.0))
    details = STAGE_DETAILS[module_name]
    actions = "".join(f"<li>{escape(item)}</li>" for item in payload["actions"])
    reasons = "".join(f"<li>{escape(item)}</li>" for item in payload["reasons"])
    note = ""
    if payload.get("tailored_note"):
        note = f'<p class="small-muted"><strong>Tailored support:</strong> {escape(payload["tailored_note"])}</p>'

    st.markdown(
        f"""
        <div class="result-card primary-result">
            <div class="result-stage">{escape(details["result_label"])}</div>
            <div class="result-kicker">{escape(title)}</div>
            <div class="result-value">{pct:.1f}%</div>
            <div>{risk_pill(band)} <span class="small-muted"> {escape(action_label(prob, threshold))}</span></div>
            <div class="meter"><div class="meter-fill" style="width:{pct:.1f}%"></div></div>
            <div class="small-muted">{escape(subtitle)} {escape(details["threshold"])}.</div>
            <div class="interpret-grid">
                <div class="interpret-box">
                    <div class="info-label">What this estimates</div>
                    <div class="info-value">{escape(details["prediction"])}</div>
                </div>
                <div class="interpret-box">
                    <div class="info-label">How to use it</div>
                    <div class="info-value">{escape(details["output"])}</div>
                </div>
            </div>
        </div>
        <div class="result-card">
            <div class="result-kicker">Suggested next action</div>
            <ul class="list-clean">{actions}</ul>
        </div>
        <div class="result-card">
            <div class="result-kicker">Why this was suggested</div>
            <p class="small-muted">Action intensity: <strong>{escape(payload["intensity"])}</strong></p>
            <ul class="list-clean">{reasons}</ul>
            <p class="small-muted"><strong>Follow-up:</strong> {escape(payload["follow_up"])}</p>
            {note}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_result(module_name: str) -> None:
    details = STAGE_DETAILS[module_name]
    st.markdown(
        f"""
        <div class="result-card primary-result">
            <div class="result-stage">{escape(details["result_label"])}</div>
            <div class="result-kicker">Awaiting calculation</div>
            <div class="result-value" style="font-size:1.45rem;">Ready when you are</div>
            <div class="small-muted">{escape(details["output"])}. Enter the predictors and calculate risk; results stay on screen while you move between modules.</div>
            <div class="interpret-grid">
                <div class="interpret-box">
                    <div class="info-label">Prediction target</div>
                    <div class="info-value">{escape(details["prediction"])}</div>
                </div>
                <div class="interpret-box">
                    <div class="info-label">Input set</div>
                    <div class="info-value">{escape(details["inputs"])}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_current_result(module_name: str) -> None:
    if module_name == "Booking visit" and st.session_state.anc_prob is not None:
        prob = float(st.session_state.anc_prob)
        render_result(
            prob,
            booking_risk_band(prob),
            float(st.session_state.anc_threshold),
            "Risk of developing GDM",
            "Booking-stage output from the saved CatBoost model.",
            booking_action_payload(prob, float(st.session_state.anc_threshold)),
            "Booking visit",
        )
    elif module_name == "Pregnancy after GDM" and st.session_state.ant_prob is not None:
        prob = float(st.session_state.ant_prob)
        render_result(
            prob,
            published_model_band(prob, 0.096),
            0.096,
            "Future T2DM risk after delivery",
            "Published antenatal logistic model for women with GDM.",
            antenatal_action_payload(prob, 0.096),
            "Pregnancy after GDM",
        )
    elif module_name == "Postnatal review" and st.session_state.post_prob is not None:
        prob = float(st.session_state.post_prob)
        render_result(
            prob,
            published_model_band(prob, 0.086),
            0.086,
            "Updated future T2DM risk",
            "Published postnatal model using postpartum glucose values and BMI.",
            postnatal_action_payload(prob, 0.086),
            "Postnatal review",
        )
    else:
        render_empty_result(module_name)


# =========================================================
# MODULES
# =========================================================
def render_booking_module() -> None:
    left, right = st.columns([1.15, 0.85])
    with left:
        render_module_intro("Booking visit")
        st.markdown('<div class="form-caption">Enter the booking predictors below. The model predicts the probability of developing GDM later in pregnancy.</div>', unsafe_allow_html=True)
        with st.form("booking_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("Age (years)", min_value=10, max_value=60, step=1, key="age")
                st.number_input("Height (cm)", min_value=100, max_value=220, step=1, key="height")
                st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, step=0.1, key="weight")
            with c2:
                st.number_input("Parity", min_value=0, max_value=15, step=1, key="parity")
                st.selectbox("Family history of diabetes", [0, 1], format_func=yes_no, key="family_hist_dm")
                st.selectbox("Ethnicity group", list(RECODE_DICT.keys()), key="ethnicity_group")
            with c3:
                st.selectbox("Past history of GDM", [0, 1], format_func=yes_no, key="past_hist_gdm")
                st.selectbox(
                    "Past obstetric complications",
                    [0, 1],
                    format_func=yes_no,
                    key="past_hist_obs_complica",
                )
                st.slider("Action threshold", 0.05, 0.90, step=0.01, key="anc_threshold")
            submitted = st.form_submit_button("Calculate booking risk", type="primary", use_container_width=True)

        if submitted:
            try:
                prob, pred, feature_frame = predict_booking_risk()
                st.session_state.anc_prob = prob
                st.session_state.anc_pred = pred
                st.session_state.booking_feature_frame = feature_frame
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

        if st.session_state.anc_prob is not None:
            with st.expander("Model inputs sent to scaler and CatBoost"):
                st.dataframe(st.session_state.booking_feature_frame, use_container_width=True)

    with right:
        render_current_result("Booking visit")


def render_antenatal_module() -> None:
    left, right = st.columns([1.15, 0.85])
    with left:
        render_module_intro("Pregnancy after GDM")
        st.markdown('<div class="form-caption">Use this when GDM has been diagnosed in pregnancy and you want to plan postpartum diabetes prevention before delivery.</div>', unsafe_allow_html=True)
        with st.form("antenatal_form"):
            c1, c2 = st.columns(2)
            with c1:
                st.number_input("Antenatal FPG (mmol/L)", min_value=0.0, max_value=30.0, step=0.1, key="antenatal_fpg")
                st.number_input("Antenatal 2-hour OGTT (mmol/L)", min_value=0.0, max_value=40.0, step=0.1, key="antenatal_2h_ogtt")
                st.number_input("Parity", min_value=0, max_value=15, step=1, key="parity")
            with c2:
                st.selectbox("Recurrent GDM", [0, 1], format_func=yes_no, key="recurrent_gdm")
                st.selectbox("Insulin treatment in pregnancy", [0, 1], format_func=yes_no, key="insulin_treatment")
                st.selectbox("Irregular menstrual cycles", [0, 1], format_func=yes_no, key="irregular_menses")
                st.selectbox("Family history of diabetes", [0, 1], format_func=yes_no, key="family_hist_dm")
            submitted = st.form_submit_button("Calculate future T2DM risk", type="primary", use_container_width=True)

        if submitted:
            st.session_state.ant_prob = predict_antenatal_t2dm_after_gdm()
            st.session_state.post_view_antenatal_2h_ogtt = float(st.session_state.antenatal_2h_ogtt)

    with right:
        render_current_result("Pregnancy after GDM")


def render_postnatal_module() -> None:
    left, right = st.columns([1.15, 0.85])
    with left:
        render_module_intro("Postnatal review")
        st.markdown('<div class="form-caption">Use this after postpartum testing to update future type 2 diabetes risk and longer-term follow-up planning.</div>', unsafe_allow_html=True)
        with st.form("postnatal_form"):
            c1, c2 = st.columns(2)
            with c1:
                st.number_input(
                    "Antenatal 2-hour OGTT (linked, mmol/L)",
                    min_value=0.0,
                    max_value=40.0,
                    step=0.1,
                    key="post_view_antenatal_2h_ogtt",
                )
                st.number_input("Postnatal FPG (mmol/L)", min_value=0.0, max_value=30.0, step=0.1, key="postnatal_fpg")
            with c2:
                st.number_input("Postnatal 2-hour OGTT (mmol/L)", min_value=0.0, max_value=40.0, step=0.1, key="postnatal_2h_ogtt")
                st.number_input("Postnatal BMI (kg/m2)", min_value=10.0, max_value=80.0, step=0.1, key="postnatal_bmi")
            submitted = st.form_submit_button("Calculate postnatal risk", type="primary", use_container_width=True)

        if submitted:
            st.session_state.antenatal_2h_ogtt = float(st.session_state.post_view_antenatal_2h_ogtt)
            st.session_state.post_prob = predict_postnatal_t2dm_after_gdm()

    with right:
        render_current_result("Postnatal review")


# =========================================================
# REPORTING
# =========================================================
def modules_completed() -> int:
    return int(st.session_state.anc_prob is not None) + int(st.session_state.ant_prob is not None) + int(st.session_state.post_prob is not None)


def patient_context_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Patient label", str(st.session_state.patient_name)],
            ["Current pathway status", str(st.session_state.gdm_status)],
            ["Age", str(st.session_state.age)],
            ["Parity", str(st.session_state.parity)],
            ["Ethnicity group", str(st.session_state.ethnicity_group)],
            ["Family history of diabetes", yes_no(st.session_state.family_hist_dm)],
        ],
        columns=["Field", "Value"],
    )


def summary_dataframe() -> pd.DataFrame:
    rows = []
    if st.session_state.anc_prob is not None:
        rows.append(
            {
                "Module": "Booking - risk of GDM",
                "Probability_pct": float(st.session_state.anc_prob) * 100,
                "Risk band": booking_risk_band(float(st.session_state.anc_prob)),
                "Action label": action_label(float(st.session_state.anc_prob), float(st.session_state.anc_threshold)),
            }
        )
    if st.session_state.ant_prob is not None:
        rows.append(
            {
                "Module": "Pregnancy after GDM - future T2DM risk",
                "Probability_pct": float(st.session_state.ant_prob) * 100,
                "Risk band": published_model_band(float(st.session_state.ant_prob), 0.096),
                "Action label": action_label(float(st.session_state.ant_prob), 0.096),
            }
        )
    if st.session_state.post_prob is not None:
        rows.append(
            {
                "Module": "Postnatal review - future T2DM risk",
                "Probability_pct": float(st.session_state.post_prob) * 100,
                "Risk band": published_model_band(float(st.session_state.post_prob), 0.086),
                "Action label": action_label(float(st.session_state.post_prob), 0.086),
            }
        )
    return pd.DataFrame(rows)


def report_html(summary_df: pd.DataFrame) -> str:
    patient_rows = "".join(
        f"<tr><th>{escape(str(field))}</th><td>{escape(str(value))}</td></tr>"
        for field, value in patient_context_table().values.tolist()
    )
    if summary_df.empty:
        summary_rows = "<tr><td colspan='4'>No module has been run yet.</td></tr>"
    else:
        summary_rows = "".join(
            f"<tr><td>{escape(str(row['Module']))}</td><td>{row['Probability_pct']:.1f}%</td><td>{escape(str(row['Risk band']))}</td><td>{escape(str(row['Action label']))}</td></tr>"
            for _, row in summary_df.iterrows()
        )
    qr_uri = qr_data_uri(st.session_state.public_app_url)
    qr_html = (
        f"<img src='{qr_uri}' style='width:112px;height:112px;border:1px solid #dce3ea;border-radius:8px;padding:4px;background:white;'>"
        if qr_uri
        else ""
    )
    return f"""
    <div style="background:white;border:1px solid #dce3ea;border-radius:10px;padding:18px;max-width:960px;">
        <div style="display:flex;justify-content:space-between;gap:16px;border-bottom:2px solid #1769aa;padding-bottom:12px;margin-bottom:14px;">
            <div>
                <h2 style="margin:0;color:#132238;">M-Guide Maternal Diabetes Prevention Platform</h2>
                <p style="margin:6px 0 0 0;color:#5b677a;">{escape(st.session_state.app_tagline)} | {escape(st.session_state.institution_name)} | {escape(st.session_state.model_version)}</p>
            </div>
            {qr_html}
        </div>
        <h3 style="color:#132238;">Patient context</h3>
        <table style="width:100%;border-collapse:collapse;">{patient_rows}</table>
        <h3 style="color:#132238;">Risk journey summary</h3>
        <table style="width:100%;border-collapse:collapse;">
            <thead><tr><th>Module</th><th>Probability</th><th>Risk band</th><th>Action label</th></tr></thead>
            <tbody>{summary_rows}</tbody>
        </table>
        <p style="border-top:1px solid #dce3ea;margin-top:14px;padding-top:10px;color:#5b677a;font-size:13px;">
            <strong>{escape(st.session_state.report_note)}</strong><br>
            Contact: {escape(st.session_state.contact_email)}<br>
            Public app: {escape(st.session_state.public_app_url)}
        </p>
    </div>
    """


def render_report_section() -> None:
    with st.expander("Summary report and downloads"):
        summary_df = summary_dataframe()
        if summary_df.empty:
            st.info("Run one or more modules to populate the report.")
        else:
            st.dataframe(summary_df, use_container_width=True)

        html_report = report_html(summary_df)
        csv_bytes = summary_df.to_csv(index=False).encode("utf-8") if not summary_df.empty else b"Module,Probability_pct,Risk band,Action label\n"
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Download summary CSV",
                data=csv_bytes,
                file_name="maternal_diabetes_risk_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "Download HTML report",
                data=html_report,
                file_name="maternal_diabetes_report.html",
                mime="text/html",
                use_container_width=True,
            )
        st.markdown(html_report, unsafe_allow_html=True)


def render_share_section() -> None:
    with st.expander("Links and sharing"):
        cols = st.columns(3)
        share_items = [
            ("Public app", st.session_state.public_app_url),
            ("GitHub repository", st.session_state.github_url),
            ("Primary publication", st.session_state.publication_url),
        ]
        for col, (title, url) in zip(cols, share_items):
            with col:
                st.markdown(f"**{title}**")
                st.caption(url or "URL not set")
                img_bytes = qr_image(url)
                if img_bytes:
                    st.image(img_bytes, width=145)


def render_model_library_section() -> None:
    with st.expander("Model library and interpretation guide"):
        cols = st.columns(3)
        for col, module_name in zip(cols, MODULES):
            details = STAGE_DETAILS[module_name]
            with col:
                st.markdown(
                    f"""
                    <div class="module-detail">
                        <div class="brand-kicker">Prediction {escape(details["number"])}</div>
                        <div class="module-detail-title" style="font-size:1.12rem;">{escape(details["short_name"])}</div>
                        <div class="module-detail-text"><strong>Prediction:</strong> {escape(details["prediction"])}</div>
                        <div class="module-detail-text"><strong>Model:</strong> {escape(details["model"])}</div>
                        <div class="module-detail-text"><strong>Inputs:</strong> {escape(details["inputs"])}</div>
                        <div class="module-detail-text"><strong>Output:</strong> {escape(details["output"])}</div>
                        <div class="module-detail-text"><strong>Threshold:</strong> {escape(details["threshold"])}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Demo", use_container_width=True):
            load_demo_patient()
            st.rerun()
    with c2:
        if st.button("Reset", use_container_width=True):
            reset_all()
            st.rerun()

    st.text_input("Patient label", key="patient_name")
    st.selectbox(
        "Current pathway status",
        ["Not diagnosed / screening stage", "GDM confirmed", "Postnatal follow-up"],
        key="gdm_status",
    )

    with st.expander("Branding and links"):
        st.text_input("Institution name", key="institution_name")
        st.text_input("Contact email", key="contact_email")
        st.text_input("Public app URL", key="public_app_url")
        st.text_input("GitHub repo URL", key="github_url")
        st.text_input("Primary publication URL / DOI", key="publication_url")
        st.text_input("Secondary publication URL / DOI", key="publication_url_secondary")
        st.text_input("App tagline", key="app_tagline")
        st.text_input("Model version label", key="model_version")


# =========================================================
# PAGE
# =========================================================
render_header()
render_platform_summary()

render_stage_cards()
selected_module = st.session_state.active_module

if selected_module == "Booking visit":
    render_booking_module()
elif selected_module == "Pregnancy after GDM":
    render_antenatal_module()
else:
    render_postnatal_module()

render_model_library_section()
render_report_section()
render_share_section()

st.markdown(
    f"""
    <div class="small-muted" style="border-top:1px solid #dce3ea;margin-top:1.2rem;padding-top:0.8rem;">
        {safe_link(st.session_state.github_url, "GitHub")} | {safe_link(st.session_state.publication_url, "Publication 1")} | {safe_link(st.session_state.publication_url_secondary, "Publication 2")}<br>
        <strong>M-Guide Maternal Diabetes Prevention Platform</strong>. Contact: {escape(st.session_state.contact_email)}. {escape(st.session_state.report_note)}
    </div>
    """,
    unsafe_allow_html=True,
)
