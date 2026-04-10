import base64
import io
import math
from html import escape
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import plotly.graph_objects as go
import qrcode
import streamlit as st
from catboost import CatBoostClassifier

st.set_page_config(
    page_title="M-Guide | Maternal Diabetes Prevention Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLING
# =========================================================
CUSTOM_CSS = """
<style>
    :root {
        --navy: #10284f;
        --blue: #005eb8;
        --cyan: #1ea8c7;
        --teal: #25b49f;
        --rose: #e89da5;
        --bg: #f5f8fc;
        --card: #ffffff;
        --border: #d7e2ef;
        --text: #132a4e;
        --muted: #5d7190;
        --success-bg: #eaf7ee;
        --success-text: #17603a;
        --warn-bg: #fff5d9;
        --warn-text: #855400;
        --danger-bg: #fde8ea;
        --danger-text: #991b1b;
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-size: 17px;
    }

    .stApp {
        background:
            radial-gradient(circle at top right, rgba(0, 94, 184, 0.07), transparent 24%),
            linear-gradient(180deg, #fbfdff 0%, var(--bg) 100%);
    }

    .block-container {
        max-width: 1600px !important;
        padding-top: 1.2rem !important;
        padding-bottom: 2.8rem !important;
        padding-left: 1.6rem !important;
        padding-right: 1.6rem !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08172f 0%, #10284f 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
        min-width: 320px !important;
        max-width: 320px !important;
    }

    section[data-testid="stSidebar"] * {
        color: #f7fbff;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        font-size: 1.28rem !important;
        font-weight: 900 !important;
        color: white !important;
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] div[data-baseweb="select"] * {
        font-size: 0.98rem !important;
    }

    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stSlider"] label {
        display: block !important;
        color: var(--text) !important;
        font-weight: 800 !important;
        margin-bottom: 0.22rem !important;
    }

    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stSlider"] label p {
        color: inherit !important;
        font-weight: inherit !important;
        font-size: 1rem !important;
    }

    div[data-testid="stNumberInput"] input {
        font-size: 1rem !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="select"] > div {
        min-height: 3rem !important;
        font-size: 1rem !important;
    }

    div.stButton > button {
        min-height: 3rem !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
    }

    .hero {
        background: linear-gradient(135deg, #103365 0%, #1d5d94 50%, #22b4c9 100%);
        border-radius: 28px;
        padding: 1.7rem 1.8rem;
        color: white;
        box-shadow: 0 18px 40px rgba(16, 40, 79, 0.16);
        position: relative;
        overflow: hidden;
        margin-bottom: 1rem;
    }

    .hero:before {
        content: "";
        position: absolute;
        right: -40px;
        top: -40px;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(255,255,255,0.18) 0%, rgba(255,255,255,0.04) 60%, transparent 72%);
        border-radius: 50%;
    }

    .hero-chip {
        display: inline-block;
        padding: 0.32rem 0.9rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.18);
        font-size: 0.84rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }

    .hero-title {
        margin: 0;
        font-size: 2.8rem;
        line-height: 1.02;
        font-weight: 900;
        color: white;
        max-width: 980px;
    }

    .hero-text {
        margin-top: 0.7rem;
        font-size: 1.16rem;
        line-height: 1.6;
        max-width: 980px;
        color: rgba(255,255,255,0.98);
    }

    .summary-tile {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        min-height: 100px;
        box-shadow: 0 10px 26px rgba(16, 40, 79, 0.05);
    }

    .summary-tile .label {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 0.35rem;
    }

    .summary-tile .value {
        color: var(--text);
        font-size: 1.55rem;
        line-height: 1.18;
        font-weight: 900;
    }

    .journey {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .journey-step {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem;
        min-height: 156px;
        box-shadow: 0 10px 26px rgba(16, 40, 79, 0.05);
    }

    .journey-step strong {
        display: block;
        color: var(--text);
        font-size: 1.18rem;
        margin-bottom: 0.45rem;
    }

    .journey-step span {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.6;
    }

    .notice {
        background: linear-gradient(90deg, #f6d15a 0%, #f3e38c 100%);
        border: 1px solid rgba(136,95,0,0.18);
        color: #3f2c00;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        font-weight: 800;
        font-size: 1rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    .section-title {
        color: var(--text);
        font-size: 2rem;
        line-height: 1.1;
        font-weight: 900;
        margin: 0.3rem 0 0.35rem 0;
    }

    .section-text {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        max-width: 980px;
        margin-bottom: 0.9rem;
    }

    .application-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(16, 40, 79, 0.05);
        height: 100%;
    }

    .application-top {
        color: white;
        padding: 1.2rem 1.25rem 1.15rem 1.25rem;
        min-height: 132px;
    }

    .application-top.blue { background: #2c5a90; }
    .application-top.teal { background: #23b98d; }
    .application-top.rose { background: #e29aa1; }

    .application-kicker {
        font-size: 0.82rem;
        opacity: 0.96;
        margin-bottom: 0.35rem;
    }

    .application-title {
        font-size: 2rem;
        line-height: 1.04;
        font-weight: 900;
    }

    .application-body {
        padding: 1.2rem 1.25rem 1.35rem 1.25rem;
        color: #294466;
        font-size: 0.98rem;
        line-height: 1.7;
    }

    .pill-note {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        width: fit-content;
        border-radius: 999px;
        background: #28489c;
        color: white;
        padding: 0.64rem 1.05rem;
        font-size: 0.94rem;
        font-weight: 800;
        margin-top: 0.85rem;
    }

    .module-shell {
        background: white;
        border: 1px solid var(--border);
        border-radius: 26px;
        padding: 1.35rem;
        box-shadow: 0 14px 30px rgba(16, 40, 79, 0.06);
        margin-top: 1.15rem;
        margin-bottom: 1rem;
    }

    .module-kicker {
        color: var(--blue);
        font-size: 0.84rem;
        font-weight: 900;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }

    .module-title {
        color: var(--text);
        font-size: 2.15rem;
        line-height: 1.1;
        font-weight: 900;
        margin: 0 0 0.45rem 0;
    }

    .module-description {
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 1rem;
    }

    .form-section-title {
        color: var(--text);
        font-size: 1.22rem;
        font-weight: 900;
        margin: 0.65rem 0 0.32rem 0;
        padding-bottom: 0.2rem;
        border-bottom: 2px solid #dbe5f2;
    }

    .form-section-note {
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.6;
        margin-bottom: 0.7rem;
    }

    .callout {
        background: #edf5ff;
        border: 1px solid #cfe0f7;
        border-left: 5px solid var(--blue);
        border-radius: 18px;
        padding: 1rem 1.05rem;
        color: #173860;
        font-size: 0.98rem;
        line-height: 1.7;
        margin-top: 0.15rem;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-left: 5px solid var(--blue);
        border-radius: 20px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 24px rgba(16, 40, 79, 0.05);
        height: 100%;
    }

    .metric-card .small-label {
        color: var(--muted);
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .metric-card .big-value {
        color: var(--text);
        font-size: 2rem;
        line-height: 1.08;
        font-weight: 900;
    }

    .metric-card .subtext {
        color: var(--muted);
        font-size: 0.95rem;
        line-height: 1.55;
        margin-top: 0.35rem;
    }

    .risk-pill {
        display: inline-block;
        padding: 0.36rem 0.85rem;
        border-radius: 999px;
        font-size: 0.9rem;
        font-weight: 800;
        border: 1px solid transparent;
    }

    .risk-low { background: var(--success-bg); color: var(--success-text); border-color: #bde3ca; }
    .risk-mod { background: var(--warn-bg); color: var(--warn-text); border-color: #f1d27a; }
    .risk-high { background: var(--danger-bg); color: var(--danger-text); border-color: #f2b8bd; }

    .action-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 1rem 1.05rem;
        box-shadow: 0 10px 24px rgba(16, 40, 79, 0.05);
        height: 100%;
    }

    .action-kicker {
        color: var(--blue);
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-weight: 900;
        margin-bottom: 0.3rem;
    }

    .action-title {
        color: var(--text);
        font-size: 1.12rem;
        font-weight: 900;
        margin-bottom: 0.6rem;
    }

    .action-list {
        margin: 0;
        padding-left: 1.1rem;
        display: flex;
        flex-direction: column;
        gap: 0.45rem;
    }

    .action-list li,
    .reason-list li {
        color: var(--text);
        font-size: 0.96rem;
        line-height: 1.6;
    }

    .reason-list {
        margin: 0;
        padding-left: 1rem;
        display: flex;
        flex-direction: column;
        gap: 0.38rem;
    }

    .intensity-pill {
        display: inline-block;
        padding: 0.28rem 0.75rem;
        border-radius: 999px;
        background: #f4f8fc;
        border: 1px solid var(--border);
        color: var(--text);
        font-size: 0.82rem;
        font-weight: 900;
        margin-bottom: 0.6rem;
    }

    .note-box {
        margin-top: 0.7rem;
        padding: 0.75rem 0.9rem;
        border-radius: 14px;
        background: #edf5ff;
        border: 1px solid #cfe0f7;
        color: #173860;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .light-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 22px;
        padding: 1.2rem 1.25rem;
        box-shadow: 0 10px 24px rgba(16, 40, 79, 0.05);
        height: 100%;
    }

    .light-card h4 {
        margin: 0 0 0.4rem 0;
        color: var(--text);
        font-size: 1.3rem;
        font-weight: 900;
        line-height: 1.2;
    }

    .light-card p {
        color: #294466;
        font-size: 1rem;
        line-height: 1.7;
        margin: 0;
    }

    .accent-card {
        background: linear-gradient(180deg, #ea507a 0%, #e34774 100%);
        color: white;
        border-radius: 22px;
        padding: 1.35rem 1.35rem;
        min-height: 220px;
        box-shadow: 0 12px 28px rgba(16, 40, 79, 0.08);
    }

    .accent-card .badge {
        display: inline-block;
        background: rgba(255,255,255,0.94);
        color: #d94c73;
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 900;
        margin-bottom: 0.8rem;
    }

    .accent-card .title {
        font-size: 2.05rem;
        line-height: 1.1;
        font-weight: 900;
        margin: 0;
    }

    .dark-solution {
        background: #234f81;
        color: white;
        border-radius: 22px;
        padding: 1.35rem;
        min-height: 220px;
        box-shadow: 0 12px 28px rgba(16, 40, 79, 0.08);
    }

    .dark-solution .badge {
        display: inline-block;
        background: rgba(255,255,255,0.92);
        color: #234f81;
        border-radius: 999px;
        padding: 0.28rem 0.75rem;
        font-size: 0.82rem;
        font-weight: 900;
        margin-bottom: 0.8rem;
    }

    .dark-solution .title {
        font-size: 2rem;
        line-height: 1.12;
        font-weight: 900;
        margin-bottom: 0.55rem;
    }

    .dark-solution .text {
        font-size: 1rem;
        line-height: 1.7;
        margin: 0;
    }

    .footer-wrap {
        margin-top: 1.4rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 0.94rem;
        line-height: 1.65;
    }

    .footer-wrap a {
        color: var(--blue);
        text-decoration: none;
        font-weight: 800;
    }

    .footer-pill {
        display: inline-block;
        padding: 0.2rem 0.58rem;
        border-radius: 999px;
        background: #eaf1fb;
        color: var(--blue);
        font-size: 0.74rem;
        font-weight: 900;
        margin-right: 0.28rem;
        margin-bottom: 0.3rem;
    }

    @media (max-width: 1200px) {
        .journey { grid-template-columns: 1fr 1fr; }
        .hero-title { font-size: 2.3rem; }
        .application-title { font-size: 1.6rem; }
    }

    @media (max-width: 760px) {
        .journey { grid-template-columns: 1fr; }
        .hero-title { font-size: 2rem; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
    "Oceanian (not white-Australian or white-New Zealander)": "Oceanian_not_WANZ",
    "Southern and Central Asian": "South_Central_Asia",
    "South-East and North-East Asian": "SouthEast_NorthEast_Asia",
    "Caucasian": "Caucasian",
    "Middle-Eastern, North African, or Sub-Saharan African": "ME_NA_SSA",
    "Other NEC": "Other NEC",
}

DEFAULTS = {
    "active_module": "booking",
    "patient_name": "Demo patient",
    "age": 30,
    "height": 165,
    "weight": 70.0,
    "parity": 1,
    "ethnicity_group": "Southern and Central Asian",
    "family_hist_dm": 1,
    "past_hist_gdm": 0,
    "past_hist_obs_complica": 0,
    "anc_threshold": 0.50,
    "gdm_status": "Not diagnosed / screening stage",
    # Antenatal published model
    "antenatal_fpg": 5.5,
    "antenatal_2h_ogtt": 8.6,
    "recurrent_gdm": 0,
    "insulin_treatment": 0,
    "irregular_menses": 0,
    "antenatal_parity": 1,
    "antenatal_family_hist_dm": 1,
    # Postnatal model
    "post_view_antenatal_2h_ogtt": 8.6,
    "postnatal_fpg": 5.2,
    "postnatal_2h_ogtt": 7.4,
    "postnatal_bmi": 27.0,
    # Results
    "anc_prob": None,
    "anc_pred": None,
    "ant_prob": None,
    "post_prob": None,
    "booking_feature_frame": pd.DataFrame(),
    # Branding / sharing
    "institution_name": "MCHRI",
    "institution_unit": "",
    "contact_email": "yitayeh.mengistu@monash.edu",
    "public_app_url": "https://m-guide-maternal-diabetes-prevention-platform-na7gro2wekpgin3k.streamlit.app/",
    "github_url": "https://github.com/YitayehMengistu/M-Guide-Maternal-Diabetes-Prevention-Platform",
    "publication_url": "https://doi.org/10.1016/j.ijmedinf.2023.105228",
    "publication_url_secondary": "https://doi.org/10.1016/j.clnu.2024.06.006",
    "app_tagline": "Pregnancy-to-postpartum diabetes risk platform",
    "model_version": "Prototype v3.0",
    "report_note": "Research demo only. Decision support and presentation use only.",
}

YES_NO = {0: "No", 1: "Yes"}

# =========================================================
# STATE
# =========================================================
def init_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sync_shared_to_antenatal():
    st.session_state.antenatal_parity = int(st.session_state.parity)
    st.session_state.antenatal_family_hist_dm = int(st.session_state.family_hist_dm)


def sync_antenatal_to_shared():
    st.session_state.parity = int(st.session_state.antenatal_parity)
    st.session_state.family_hist_dm = int(st.session_state.antenatal_family_hist_dm)


def sync_antenatal_to_postnatal_link():
    st.session_state.post_view_antenatal_2h_ogtt = float(st.session_state.antenatal_2h_ogtt)


def sync_postnatal_link_to_antenatal():
    st.session_state.antenatal_2h_ogtt = float(st.session_state.post_view_antenatal_2h_ogtt)


def load_demo_patient():
    demo = {
        "active_module": "booking",
        "patient_name": "Demo patient",
        "age": 33,
        "height": 160,
        "weight": 78.5,
        "parity": 2,
        "ethnicity_group": "Southern and Central Asian",
        "family_hist_dm": 1,
        "past_hist_gdm": 1,
        "past_hist_obs_complica": 1,
        "gdm_status": "GDM confirmed",
        "antenatal_fpg": 5.8,
        "antenatal_2h_ogtt": 9.6,
        "recurrent_gdm": 1,
        "insulin_treatment": 1,
        "irregular_menses": 1,
        "antenatal_parity": 2,
        "antenatal_family_hist_dm": 1,
        "post_view_antenatal_2h_ogtt": 9.6,
        "postnatal_fpg": 5.2,
        "postnatal_2h_ogtt": 9.6,
        "postnatal_bmi": 32.2,
    }
    for key, value in demo.items():
        st.session_state[key] = value


def reset_all():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


init_state()
sync_shared_to_antenatal()

# =========================================================
# HELPERS
# =========================================================
def yes_no(value: int) -> str:
    return YES_NO[int(value)]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


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
    label = label.lower()
    if label == "low":
        return "risk-low"
    if label == "moderate":
        return "risk-mod"
    return "risk-high"


def risk_pill(label: str) -> str:
    return f'<span class="risk-pill {risk_css_class(label)}">{escape(label)}</span>'


def recode_ethnicity(group_label: str) -> str:
    return RECODE_DICT.get(group_label, "Other NEC")


def first_existing_logo() -> Optional[Path]:
    for path in LOGO_CANDIDATES:
        if path.exists():
            return path
    return None


def qr_image(url: str):
    payload = (url or "").strip()
    if not payload.startswith(("http://", "https://")):
        return None
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


def safe_link(url: str, label: str) -> str:
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return escape(label)
    return f'<a href="{escape(url)}" target="_blank">{escape(label)}</a>'


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_booking_model():
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        return None, None, "Booking model or scaler file not found in the app folder."
    model = CatBoostClassifier()
    model.load_model(str(MODEL_FILE))
    scaler = joblib.load(str(SCALER_FILE))
    return model, scaler, None


booking_model, booking_scaler, booking_model_error = load_booking_model()

# =========================================================
# PREDICTION FUNCTIONS
# =========================================================
def build_booking_features() -> pd.DataFrame:
    row = {col: 0 for col in FEATURE_COLUMNS}
    row["Height"] = int(st.session_state.height)
    row["Weight"] = float(st.session_state.weight)
    row["Parity"] = int(st.session_state.parity)
    row["Age"] = int(st.session_state.age)
    row["Family_Hist_DM"] = int(st.session_state.family_hist_dm)
    row["Past_Hist_GDM"] = int(st.session_state.past_hist_gdm)
    row["Past_Hist_Obs_Complica"] = int(st.session_state.past_hist_obs_complica)
    ethnicity_col = recode_ethnicity(str(st.session_state.ethnicity_group))
    row[ethnicity_col] = 1

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    df["Height"] = df["Height"].astype(int)
    df["Weight"] = df["Weight"].astype(float)
    df["Parity"] = df["Parity"].astype("int8")
    df["Age"] = df["Age"].astype(int)
    for col in [
        "Caucasian",
        "Oceanian_not_WANZ",
        "ME_NA_SSA",
        "South_Central_Asia",
        "SouthEast_NorthEast_Asia",
        "Other NEC",
        "Family_Hist_DM",
        "Past_Hist_GDM",
        "Past_Hist_Obs_Complica",
    ]:
        df[col] = df[col].astype(int)
    return df


def predict_booking_risk():
    if booking_model is None or booking_scaler is None:
        raise RuntimeError(booking_model_error or "Booking model is not available.")
    X = build_booking_features()
    X_scaled = booking_scaler.transform(X)
    prob = float(booking_model.predict_proba(X_scaled)[0, 1])
    pred = int(prob >= float(st.session_state.anc_threshold))
    return prob, pred, X


def predict_antenatal_t2dm_after_gdm() -> float:
    # 7-variable antenatal published model
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
        + 0.3008 * float(st.session_state.antenatal_2h_ogtt)
        + 1.0033 * float(st.session_state.postnatal_fpg)
        + 0.5581 * float(st.session_state.postnatal_2h_ogtt)
        + 0.0359 * float(st.session_state.postnatal_bmi)
    )
    return sigmoid(logit)


# =========================================================
# CHARTS / RESULT HELPERS
# =========================================================
def make_gauge(prob: float, title: str, threshold: Optional[float] = None):
    gauge = {
        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#5d7190"},
        "bar": {"color": "#005eb8"},
        "bgcolor": "white",
        "borderwidth": 1,
        "bordercolor": "#d7e2ef",
        "steps": [
            {"range": [0, 10], "color": "#eaf7ee"},
            {"range": [10, 20], "color": "#fff5d9"},
            {"range": [20, 100], "color": "#fde8ea"},
        ],
    }
    if threshold is not None:
        gauge["threshold"] = {
            "line": {"color": "#10284f", "width": 3},
            "thickness": 0.75,
            "value": threshold * 100,
        }

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": title, "font": {"size": 16}},
            gauge=gauge,
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=18, r=18, t=55, b=10),
        paper_bgcolor="white",
        font={"color": "#132a4e"},
    )
    return fig


def make_completion_donut():
    completed = modules_completed()
    fig = go.Figure(
        data=[
            go.Pie(
                values=[completed, max(0, 3 - completed)],
                labels=["Completed", "Pending"],
                hole=0.68,
                marker=dict(colors=["#005eb8", "#dce6f4"]),
                textinfo="label+percent",
            )
        ]
    )
    fig.update_layout(
        title="Journey completion",
        height=310,
        margin=dict(l=10, r=10, t=48, b=10),
        paper_bgcolor="white",
    )
    return fig


def render_result_cards(prob: float, band: str, title: str, threshold: float, subtitle: str):
    left, right = st.columns([1.05, 1])
    with left:
        st.plotly_chart(make_gauge(prob, title, threshold), width="stretch")
    with right:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="small-label">Result summary</div>
                <div class="big-value">{prob:.1%}</div>
                <div class="subtext">{escape(subtitle)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="small-label">Risk band</div>
                    <div class="big-value" style="font-size:1.2rem;">{risk_pill(band)}</div>
                    <div class="subtext">Band shown for visual triage.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="small-label">Action label</div>
                    <div class="big-value" style="font-size:1rem;">{escape(action_label(prob, threshold))}</div>
                    <div class="subtext">Threshold {threshold:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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
            "Continue routine antenatal care and routine GDM screening at 24–28 weeks.",
            "Reinforce healthy nutrition, physical activity, and weight monitoring early in pregnancy.",
        ]
        follow_up = "Routine GDM screening at 24–28 weeks with standard antenatal review."
    elif intensity == "Medium":
        actions = [
            "Provide targeted antenatal lifestyle counselling and reinforce attendance for routine GDM screening.",
            "Document the elevated booking risk so the care team can plan timely review.",
            "Consider closer follow-up before routine 24–28 week screening, depending on local protocol.",
        ]
        follow_up = "Enhanced antenatal review before routine screening; consider earlier testing if local policy supports it."
    else:
        actions = [
            "Flag as high risk for GDM and consider earlier glucose assessment or earlier review according to local policy.",
            "Provide intensified lifestyle counselling and closer antenatal follow-up from booking.",
            "Discuss the result within the care team so a prevention-focused plan is documented early.",
        ]
        follow_up = "Earlier antenatal review and consideration of earlier glucose testing according to local protocol."

    tailored_note = None
    if str(st.session_state.ethnicity_group) in {
        "Southern and Central Asian",
        "South-East and North-East Asian",
        "Middle-Eastern, North African, or Sub-Saharan African",
    }:
        tailored_note = "Offer culturally appropriate dietary counselling and communication support where available."

    return {
        "intensity": intensity,
        "actions": actions,
        "reasons": reasons or ["Based primarily on the current predicted risk level."],
        "follow_up": follow_up,
        "tailored_note": tailored_note,
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
            "Plan routine postpartum glucose follow-up and reinforce long-term diabetes prevention advice before delivery.",
            "Ensure postpartum OGTT timing is documented in the discharge plan.",
        ]
        follow_up = "Routine postpartum OGTT and usual primary-care follow-up."
    elif intensity == "Medium":
        actions = [
            "Create a structured postpartum follow-up plan before delivery.",
            "Provide targeted counselling about future type 2 diabetes prevention and the importance of postpartum testing.",
            "Consider referral to dietetic or lifestyle support where available.",
        ]
        follow_up = "Structured postpartum plan with a clear testing date and prevention counselling before discharge."
    else:
        actions = [
            "Arrange enhanced postpartum follow-up and clearly document the need for early postpartum testing.",
            "Emphasise postpartum OGTT attendance and long-term diabetes surveillance in primary care.",
            "Consider referral to structured lifestyle, weight-management, or diabetes-prevention services where available.",
        ]
        follow_up = "Enhanced postpartum follow-up with active recall for OGTT and ongoing diabetes-prevention review."

    return {
        "intensity": intensity,
        "actions": actions,
        "reasons": reasons or ["Based primarily on the current predicted risk level."],
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
            "Continue routine long-term diabetes-prevention advice and periodic glycaemic surveillance as per local guideline.",
            "Reinforce healthy eating, physical activity, and weight-management goals.",
        ]
        follow_up = "Routine repeat glycaemic surveillance in primary care according to guideline."
    elif intensity == "Medium":
        actions = [
            "Provide targeted diabetes-prevention counselling and closer primary-care follow-up.",
            "Encourage repeat glucose testing and weight-management support.",
            "Document the need for longer-term surveillance after a pregnancy affected by GDM.",
        ]
        follow_up = "Closer primary-care follow-up with repeat glucose testing and weight-management support."
    else:
        actions = [
            "Escalate follow-up for diabetes prevention and regular glycaemic monitoring.",
            "Consider referral for structured lifestyle intervention or specialist review depending on local pathway.",
            "Use the result to support shared decision-making about longer-term diabetes surveillance.",
        ]
        follow_up = "Enhanced diabetes-prevention follow-up and active recall for repeat glycaemic review."

    return {
        "intensity": intensity,
        "actions": actions,
        "reasons": reasons or ["Based primarily on the current predicted risk level."],
        "follow_up": follow_up,
        "tailored_note": None,
    }


def render_recommendation_panel(payload: dict):
    actions_html = "".join(f"<li>{escape(item)}</li>" for item in payload["actions"])
    reasons_html = "".join(f"<li>{escape(item)}</li>" for item in payload["reasons"])
    note_html = ""
    if payload.get("tailored_note"):
        note_html = f"<div class='note-box'><strong>Tailored support:</strong> {escape(payload['tailored_note'])}</div>"

    c1, c2, c3 = st.columns([1.15, 0.92, 0.9])
    with c1:
        st.markdown(
            f"""
            <div class="action-card">
                <div class="action-kicker">Care recommendation</div>
                <div class="action-title">Suggested next action</div>
                <ul class="action-list">{actions_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="action-card">
                <div class="action-kicker">Reasoning</div>
                <div class="action-title">Why this was suggested</div>
                <div class="intensity-pill">Action intensity: {escape(payload['intensity'])}</div>
                <ul class="reason-list">{reasons_html}</ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="action-card">
                <div class="action-kicker">Follow-up</div>
                <div class="action-title">Recommended follow-up</div>
                <div style="color:#132a4e; font-size:0.98rem; line-height:1.65;">{escape(payload['follow_up'])}</div>
                {note_html}
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# REPORT HELPERS
# =========================================================
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


def modules_completed() -> int:
    return int(st.session_state.anc_prob is not None) + int(st.session_state.ant_prob is not None) + int(st.session_state.post_prob is not None)


def summary_dataframe() -> pd.DataFrame:
    rows = []
    if st.session_state.anc_prob is not None:
        rows.append(
            {
                "Module": "ANC booking - risk of GDM",
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
                "Module": "Postnatal follow-up - future T2DM risk",
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
        interpretation = "Run one or more modules to populate the report page."
    else:
        summary_rows = "".join(
            f"<tr><td>{escape(str(row['Module']))}</td><td>{row['Probability_pct']:.1f}%</td><td>{escape(str(row['Risk band']))}</td><td>{escape(str(row['Action label']))}</td></tr>"
            for _, row in summary_df.iterrows()
        )
        interpretation = "This prototype combines booking risk of GDM with antenatal and postnatal future T2DM risk after GDM in one prevention-oriented platform."

    qr_uri = qr_data_uri(st.session_state.public_app_url)
    qr_html = (
        f"<div style='border:1px dashed #d7e2ef;border-radius:14px;padding:0.8rem;text-align:center;background:#fbfdff;'><img src='{qr_uri}' style='width:135px;height:135px;border:1px solid #d7e2ef;border-radius:10px;padding:0.25rem;background:white;'><div style='margin-top:0.45rem;color:#5b6f8a;font-size:0.82rem;'>Scan to open public app</div></div>"
        if qr_uri
        else ""
    )

    return f"""
    <div style="background:white;border:1px solid #d5e0ee;border-radius:14px;padding:1.25rem 1.35rem;box-shadow:0 14px 36px rgba(16,34,61,0.08);max-width:980px;margin:0 auto 1rem auto;">
        <div style="display:grid;grid-template-columns:1.55fr 0.65fr;gap:1rem;border-bottom:2px solid #005eb8;padding-bottom:0.8rem;margin-bottom:0.9rem;">
            <div>
                <div style="color:#132a4e;font-size:1.55rem;font-weight:800;margin-bottom:0.2rem;">M-Guide | Maternal Diabetes Prevention Platform</div>
                <div style="color:#5b6f8a;font-size:0.92rem;line-height:1.4;">{escape(st.session_state.app_tagline)}<br>{escape(st.session_state.institution_name)} | {escape(st.session_state.model_version)}</div>
            </div>
            {qr_html}
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.9rem;margin:0.9rem 0;">
            <div style="border:1px solid #d7e2ef;border-radius:14px;padding:0.85rem 0.95rem;background:#fbfdff;">
                <h5 style="margin:0 0 0.35rem 0;color:#132a4e;font-size:0.95rem;">Patient context</h5>
                <table style="width:100%;border-collapse:collapse;font-size:0.92rem;">{patient_rows}</table>
            </div>
            <div style="border:1px solid #d7e2ef;border-radius:14px;padding:0.85rem 0.95rem;background:#fbfdff;">
                <h5 style="margin:0 0 0.35rem 0;color:#132a4e;font-size:0.95rem;">Interpretation</h5>
                <p style="margin:0;color:#173860;line-height:1.55;">{escape(interpretation)}</p>
            </div>
        </div>
        <div style="border:1px solid #d7e2ef;border-radius:14px;padding:0.85rem 0.95rem;background:#fbfdff;">
            <h5 style="margin:0 0 0.35rem 0;color:#132a4e;font-size:0.95rem;">Risk journey table</h5>
            <table style="width:100%;border-collapse:collapse;font-size:0.92rem;">
                <thead>
                    <tr><th style="text-align:left;border:1px solid #dbe5f0;padding:0.5rem 0.55rem;background:#f3f7fb;">Module</th><th style="text-align:left;border:1px solid #dbe5f0;padding:0.5rem 0.55rem;background:#f3f7fb;">Probability</th><th style="text-align:left;border:1px solid #dbe5f0;padding:0.5rem 0.55rem;background:#f3f7fb;">Risk band</th><th style="text-align:left;border:1px solid #dbe5f0;padding:0.5rem 0.55rem;background:#f3f7fb;">Action label</th></tr>
                </thead>
                <tbody>{summary_rows}</tbody>
            </table>
        </div>
        <div style="margin-top:0.9rem;color:#5b6f8a;font-size:0.82rem;border-top:1px solid #e4ebf3;padding-top:0.7rem;line-height:1.45;">
            <strong>{escape(st.session_state.report_note)}</strong><br>
            Contact: {escape(st.session_state.contact_email)}<br>
            Public app: {escape(st.session_state.public_app_url)}
        </div>
    </div>
    """


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🩺 Risk journey controls")

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Load demo patient", width="stretch"):
        load_demo_patient()
        st.rerun()
with c2:
    if st.button("Reset all inputs", width="stretch"):
        reset_all()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.text_input("Patient label", key="patient_name")
st.sidebar.number_input(
    "Parity",
    min_value=0,
    max_value=15,
    step=1,
    key="parity",
    on_change=sync_shared_to_antenatal,
)
st.sidebar.selectbox(
    "Family history of diabetes",
    options=[0, 1],
    format_func=yes_no,
    key="family_hist_dm",
    on_change=sync_shared_to_antenatal,
)
st.sidebar.selectbox(
    "Current pathway status",
    options=["Not diagnosed / screening stage", "GDM confirmed", "Postnatal follow-up"],
    key="gdm_status",
)
st.sidebar.slider(
    "ANC booking classification threshold",
    min_value=0.05,
    max_value=0.90,
    step=0.01,
    key="anc_threshold",
)
st.sidebar.info(
    "For the published post-GDM models, the paper-reported thresholds are fixed at 0.096 (antenatal) and 0.086 (postnatal)."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Branding & public share")
st.sidebar.text_input("Institution name", key="institution_name")
st.sidebar.text_input("Institution unit", key="institution_unit")
st.sidebar.text_input("Contact email", key="contact_email")
st.sidebar.text_input("Public app URL", key="public_app_url")
st.sidebar.text_input("GitHub repo URL", key="github_url")
st.sidebar.text_input("Primary publication URL / DOI", key="publication_url")
st.sidebar.text_input("Secondary publication URL / DOI", key="publication_url_secondary")
st.sidebar.text_input("App tagline", key="app_tagline")
st.sidebar.text_input("Model version label", key="model_version")

# =========================================================
# HEADER
# =========================================================
hero_left, hero_right = st.columns([4.6, 1])
with hero_left:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-chip">MONASH-STYLE PUBLIC DEMO</div>
            <h1 class="hero-title">M-Guide | Maternal Diabetes Prevention Platform</h1>
            <div class="hero-text">{escape(st.session_state.app_tagline)}. A longitudinal risk journey from ANC booking to postnatal prevention, linking early GDM risk estimation with future type 2 diabetes risk after GDM during pregnancy and again after delivery.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    logo_path = first_existing_logo()
    if logo_path is not None:
        st.image(str(logo_path), width=180)

sum_cols = st.columns(4)
summary_items = [
    ("Patient label", str(st.session_state.patient_name)),
    ("Pathway status", str(st.session_state.gdm_status)),
    ("Modules completed", f"{modules_completed()}/3"),
    ("Deployment mode", "Public demo ready"),
]
for col, (label, value) in zip(sum_cols, summary_items):
    with col:
        st.markdown(
            f"""
            <div class="summary-tile">
                <div class="label">{escape(label)}</div>
                <div class="value">{escape(value)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown(
    """
    <div class="journey">
        <div class="journey-step"><strong>1. Booking visit</strong><span>Estimate risk of developing GDM at 24–28 weeks from the saved CatBoost model.</span></div>
        <div class="journey-step"><strong>2. Routine screening</strong><span>Screening takes place at 24–28 weeks and women with GDM move into future diabetes prevention planning.</span></div>
        <div class="journey-step"><strong>3. Pregnancy after GDM</strong><span>Use the antenatal logistic equation to estimate future T2DM risk after delivery.</span></div>
        <div class="journey-step"><strong>4. Postnatal follow-up</strong><span>Use postpartum glucose values and BMI to update long-term future T2DM risk.</span></div>
        <div class="journey-step"><strong>5. Prevention summary</strong><span>Combine outputs into one presentation-ready patient prevention passport.</span></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice">Research demo only. This platform is for presentation and decision-support demonstration. It is not stand-alone diagnosis or real-world clinical deployment without governance, local validation, security review, and approval.</div>',
    unsafe_allow_html=True,
)

# =========================================================
# LANDING / APPLICATIONS
# =========================================================
st.markdown("<div class='section-title'>Applications</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-text'>Choose a module depending on where the woman is in the pathway. The organisation below is intentionally simple: clear purpose, grouped predictors, and one obvious action.</div>",
    unsafe_allow_html=True,
)

app_cols = st.columns(3)
app_defs = [
    ("blue", "Maternal diabetes platform", "Booking visit", "Estimate the probability of developing gestational diabetes during pregnancy using the saved CatBoost model and scaler.", "Open booking screen →", "booking"),
    ("teal", "Future diabetes prevention", "After GDM in pregnancy", "Use the published antenatal equation to estimate future type 2 diabetes risk after delivery among women with GDM.", "Open antenatal model →", "antenatal"),
    ("rose", "Postpartum review", "Postnatal follow-up", "Update long-term future diabetes risk using linked antenatal OGTT, postnatal fasting glucose, postnatal 2-hour OGTT, and BMI.", "Open postnatal model →", "postnatal"),
]
for col, (color, kicker, title, body, button_text, module_name) in zip(app_cols, app_defs):
    with col:
        st.markdown(
            f"""
            <div class="application-card">
                <div class="application-top {color}">
                    <div class="application-kicker">{escape(kicker)}</div>
                    <div class="application-title">{escape(title)}</div>
                </div>
                <div class="application-body">{escape(body)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(button_text, key=f"open_{module_name}", width="stretch"):
            st.session_state.active_module = module_name
            st.rerun()

# =========================================================
# MODULE RENDERERS
# =========================================================
def render_booking_module():
    st.markdown(
        """
        <div class="module-shell">
            <div class="module-kicker">Stage 1</div>
            <div class="module-title">ANC booking risk of developing GDM</div>
            <div class="module-description">This module uses the saved CatBoost model and preprocessing scaler already stored in the app folder. Predictors are grouped in the same order clinicians would usually gather them at the first booking visit.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if booking_model_error:
        st.error(booking_model_error)

    st.markdown('<div class="form-section-title">Shared background predictors</div><div class="form-section-note">These two variables are shared across the platform and are entered first for clarity.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Parity", min_value=0, max_value=15, step=1, key="parity", on_change=sync_shared_to_antenatal)
    with c2:
        st.selectbox("Family history of diabetes", options=[0, 1], format_func=yes_no, key="family_hist_dm", on_change=sync_shared_to_antenatal)

    st.markdown('<div class="form-section-title">Maternal characteristics</div><div class="form-section-note">Core booking predictors available at the first antenatal visit.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Age (years)", min_value=10, max_value=60, step=1, key="age")
        st.number_input("Height (cm)", min_value=100, max_value=220, step=1, key="height")
        st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, step=0.1, key="weight")
    with c2:
        st.selectbox("Ethnicity group", options=list(RECODE_DICT.keys()), key="ethnicity_group")
    with c3:
        st.markdown(
            '<div class="callout"><strong>Presentation tip:</strong> use this screen to show how the app turns routine booking information into an early prevention conversation.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="form-section-title">Previous obstetric history</div><div class="form-section-note">The booking model uses two history variables: past history of GDM and past obstetric complications.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox("Past history of GDM", options=[0, 1], format_func=yes_no, key="past_hist_gdm")
    with c2:
        st.selectbox(
            "Past obstetric complications",
            options=[0, 1],
            format_func=yes_no,
            key="past_hist_obs_complica",
            help="Yes if there is any relevant past obstetric complication included in the original model definition.",
        )
    with c3:
        st.markdown(
            '<div class="callout"><strong>Model predictors:</strong> age, height, weight, ethnicity group, parity, family history of diabetes, past history of GDM, and past obstetric complications.</div>',
            unsafe_allow_html=True,
        )

    if st.button("Run booking prediction", type="primary", width="stretch", key="run_booking"):
        try:
            prob, pred, feature_frame = predict_booking_risk()
            st.session_state.anc_prob = prob
            st.session_state.anc_pred = pred
            st.session_state.booking_feature_frame = feature_frame
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if st.session_state.anc_prob is not None:
        prob = float(st.session_state.anc_prob)
        band = booking_risk_band(prob)
        render_result_cards(
            prob,
            band,
            "Risk of developing GDM at 24–28 weeks",
            float(st.session_state.anc_threshold),
            "Booking-stage risk output from the saved CatBoost model.",
        )
        st.markdown("### Suggested next action")
        render_recommendation_panel(booking_action_payload(prob, float(st.session_state.anc_threshold)))
        with st.expander("Model inputs sent to scaler and CatBoost model"):
            st.dataframe(st.session_state.booking_feature_frame, width="stretch")


def render_antenatal_module():
    st.markdown(
        """
        <div class="module-shell">
            <div class="module-kicker">Stage 2</div>
            <div class="module-title">Pregnancy after GDM: future T2DM risk after delivery</div>
            <div class="module-description">This published antenatal model uses exactly seven predictors. They are grouped below as glucose markers, pregnancy history and treatment, and shared background variables so the form is easier to follow in a live demonstration.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="form-section-title">1) Antenatal glucose markers</div><div class="form-section-note">Enter the two pregnancy glucose measurements used in the published model.</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Antenatal fasting plasma glucose (FPG, mmol/L)", min_value=0.0, max_value=30.0, step=0.1, key="antenatal_fpg")
    with c2:
        st.number_input("Antenatal 2-hour OGTT (mmol/L)", min_value=0.0, max_value=40.0, step=0.1, key="antenatal_2h_ogtt", on_change=sync_antenatal_to_postnatal_link)

    st.markdown('<div class="form-section-title">2) Pregnancy history and treatment</div><div class="form-section-note">These three yes/no variables capture recurrence, treatment intensity, and menstrual history.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.selectbox("Recurrent GDM", options=[0, 1], format_func=yes_no, key="recurrent_gdm")
    with c2:
        st.selectbox("Insulin treatment in pregnancy", options=[0, 1], format_func=yes_no, key="insulin_treatment")
    with c3:
        st.selectbox("History of irregular menstrual cycles", options=[0, 1], format_func=yes_no, key="irregular_menses")

    st.markdown('<div class="form-section-title">3) Shared background variables</div><div class="form-section-note">These two predictors are part of the published antenatal model and can be edited here directly for demonstration clarity.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1.1])
    with c1:
        st.number_input("Parity", min_value=0, max_value=15, step=1, key="antenatal_parity", on_change=sync_antenatal_to_shared)
    with c2:
        st.selectbox("Family history of diabetes", options=[0, 1], format_func=yes_no, key="antenatal_family_hist_dm", on_change=sync_antenatal_to_shared)
    with c3:
        st.markdown(
            '<div class="callout"><strong>Seven predictors in total:</strong> antenatal FPG, antenatal 2-hour OGTT, recurrent GDM, insulin treatment in pregnancy, history of irregular menstrual cycles, parity, and family history of diabetes.</div>',
            unsafe_allow_html=True,
        )

    if st.button("Run antenatal future T2DM prediction", type="primary", width="stretch", key="run_antenatal"):
        st.session_state.ant_prob = predict_antenatal_t2dm_after_gdm()

    if st.session_state.ant_prob is not None:
        prob = float(st.session_state.ant_prob)
        band = published_model_band(prob, 0.096)
        render_result_cards(
            prob,
            band,
            "Future T2DM risk after delivery - antenatal model",
            0.096,
            "Published antenatal logistic model for women with GDM.",
        )
        st.markdown("### Suggested next action")
        render_recommendation_panel(antenatal_action_payload(prob, 0.096))
        st.info("Paper-reported action threshold used in this demo: 0.096.")


def render_postnatal_module():
    st.markdown(
        """
        <div class="module-shell">
            <div class="module-kicker">Stage 3</div>
            <div class="module-title">Postnatal follow-up: updated future T2DM risk</div>
            <div class="module-description">This postnatal model combines the linked antenatal 2-hour OGTT with postpartum fasting glucose, postpartum 2-hour OGTT, and postpartum BMI.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="form-section-title">Postnatal model variables</div><div class="form-section-note">The linked antenatal 2-hour OGTT is shown alongside the three postpartum predictors used in the published model.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input(
            "Antenatal 2-hour OGTT (linked value, mmol/L)",
            min_value=0.0,
            max_value=40.0,
            step=0.1,
            key="post_view_antenatal_2h_ogtt",
            on_change=sync_postnatal_link_to_antenatal,
        )
        st.number_input("Postnatal fasting plasma glucose (FPG, mmol/L)", min_value=0.0, max_value=30.0, step=0.1, key="postnatal_fpg")
    with c2:
        st.number_input("Postnatal 2-hour OGTT (mmol/L)", min_value=0.0, max_value=40.0, step=0.1, key="postnatal_2h_ogtt")
        st.number_input("Postnatal body mass index (BMI, kg/m²)", min_value=10.0, max_value=80.0, step=0.1, key="postnatal_bmi")
    with c3:
        st.markdown(
            '<div class="callout"><strong>Practical note:</strong> this module is strongest when postpartum testing has actually been completed.</div>',
            unsafe_allow_html=True,
        )

    if st.button("Run postnatal future T2DM prediction", type="primary", width="stretch", key="run_postnatal"):
        st.session_state.post_prob = predict_postnatal_t2dm_after_gdm()

    if st.session_state.post_prob is not None:
        prob = float(st.session_state.post_prob)
        band = published_model_band(prob, 0.086)
        render_result_cards(
            prob,
            band,
            "Future T2DM risk after delivery - postnatal model",
            0.086,
            "Published postnatal logistic model using postpartum glucose values and BMI.",
        )
        st.markdown("### Suggested next action")
        render_recommendation_panel(postnatal_action_payload(prob, 0.086))
        st.info("Paper-reported action threshold used in this demo: 0.086.")


if st.session_state.active_module == "booking":
    render_booking_module()
elif st.session_state.active_module == "antenatal":
    render_antenatal_module()
else:
    render_postnatal_module()

# =========================================================
# SUPPORTING SECTIONS
# =========================================================
st.markdown("<div class='section-title'>About this approach</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-text'>The goal is not simply to predict risk, but to support clearer conversations, proportionate follow-up, and better allocation of care across the maternal pathway.</div>",
    unsafe_allow_html=True,
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="accent-card">
            <div class="badge">Why consider risk stratification?</div>
            <div class="title">Use risk to tailor care rather than treat every woman as needing the same intensity of follow-up.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="light-card">
            <h4>Earlier, clearer conversations</h4>
            <p>Earlier identification of risk can support <strong>timely lifestyle advice</strong>, <strong>appropriate testing</strong>, and more focused follow-up.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="light-card">
            <h4>Simple language and visuals</h4>
            <p>A clean web tool can communicate an <strong>individualised estimate of risk</strong> in language that is easier to discuss in clinic and postpartum review.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
        <div class="light-card">
            <h4>Accessible on one platform</h4>
            <p>A single interface keeps booking, pregnancy-after-GDM, and postnatal models together, so clinicians and women can move through the pathway without changing tools.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        """
        <div class="light-card">
            <h4>Readable outputs with clear next steps</h4>
            <p>Each model returns a percentage risk, a visual band, and a structured practical next-action section to support shared decision-making.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        """
        <div class="dark-solution">
            <div class="badge">Our solution</div>
            <div class="title">A cleaner, demonstration-ready prediction experience</div>
            <div class="text">This version is intentionally polished for public demonstration: grouped predictors, stronger visual hierarchy, obvious actions, and no duplicated sections.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<div class='section-title'>Research and sharing</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='section-text'>Use the public app, code repository, and publications below for demonstrations, posters, slide decks, and collaborator sharing.</div>",
    unsafe_allow_html=True,
)

share_cols = st.columns(3)
share_items = [
    ("Public app", st.session_state.public_app_url, "Scan to open the deployed Streamlit app."),
    ("GitHub repository", st.session_state.github_url, "Scan to view source code and deployment files."),
    ("Publication / DOI", st.session_state.publication_url, "Scan to open the publication or project page."),
]
for col, (title, url, caption) in zip(share_cols, share_items):
    with col:
        st.markdown(
            f"""
            <div class="light-card">
                <h4>{escape(title)}</h4>
                <p>{escape(caption)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        img_bytes = qr_image(url)
        if img_bytes is not None:
            st.image(img_bytes, width=170)
        else:
            st.info(f"Add a valid URL for {title.lower()} in the sidebar to generate a QR code.")
        st.caption(url or "URL not set")

with st.expander("Clinical report and downloads"):
    summary_df = summary_dataframe()
    html_report = report_html(summary_df)
    st.markdown(html_report, unsafe_allow_html=True)
    csv_bytes = summary_df.to_csv(index=False).encode("utf-8") if not summary_df.empty else b"Module,Probability\n"
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download summary CSV",
            data=csv_bytes,
            file_name="maternal_diabetes_risk_summary.csv",
            mime="text/csv",
        )
    with c2:
        st.download_button(
            "Download HTML report",
            data=html_report,
            file_name="maternal_diabetes_report.html",
            mime="text/html",
        )

with st.expander("Platform details"):
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown(
            """
            <div class="light-card">
                <h4>Deployment checklist</h4>
                <p>Repository contents should include <code>app.py</code>, <code>requirements.txt</code>, <code>.streamlit/config.toml</code>, the saved booking model, the saved scaler, and optional branding assets.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""
            <div class="light-card">
                <h4>Current links</h4>
                <p><strong>Contact:</strong> {escape(st.session_state.contact_email)}<br>
                <strong>GitHub:</strong> {safe_link(st.session_state.github_url, 'Open repository')}<br>
                <strong>Publication 1:</strong> {safe_link(st.session_state.publication_url, 'Primary DOI')}<br>
                <strong>Publication 2:</strong> {safe_link(st.session_state.publication_url_secondary, 'Secondary DOI')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    f"""
    <div class="footer-wrap">
        <div>
            <span class="footer-pill">{escape(st.session_state.institution_name)}</span>
            <span class="footer-pill">{escape(st.session_state.model_version)}</span>
        </div>
        <div style="margin-top:0.35rem;">{safe_link(st.session_state.github_url, 'GitHub')} · {safe_link(st.session_state.publication_url, 'Publication 1')} · {safe_link(st.session_state.publication_url_secondary, 'Publication 2')} · {safe_link(st.session_state.public_app_url, 'Public app')}</div>
        <div style="margin-top:0.35rem;"><strong>M-Guide | Maternal Diabetes Prevention Platform</strong> — a clean, applications-first prototype for research demonstration and public sharing.</div>
        <div>Contact: {escape(st.session_state.contact_email)} | {escape(st.session_state.report_note)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
