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
# BRANDING + STYLING
# =========================================================
CUSTOM_CSS = """
<style>
    :root {
        --mono-navy: #0b1f41;
        --mono-blue: #005eb8;
        --mono-cyan: #14b7c6;
        --mono-soft: #f4f8fc;
        --mono-border: #d8e1ee;
        --mono-text: #10223d;
        --mono-muted: #5b6f8a;
        --success-bg: #e8f7ef;
        --success-text: #17603a;
        --warn-bg: #fff5d9;
        --warn-text: #8a5a00;
        --danger-bg: #fde8ea;
        --danger-text: #991b1b;
    }
    .stApp {
        background:
            radial-gradient(circle at top right, rgba(0, 94, 184, 0.08), transparent 24%),
            linear-gradient(180deg, #fbfdff 0%, #f4f8fc 100%);
    }
    .block-container {
        padding-top: 1.05rem;
        padding-bottom: 2.2rem;
        max-width: 1420px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08172f 0%, #10284f 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    section[data-testid="stSidebar"] * {
        color: #f7fbff;
    }
    div[data-testid="stTabs"] {
        margin-top: 0.15rem;
        margin-bottom: 0.35rem;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        color: #4a5f7a !important;
        font-weight: 700;
        font-size: 0.98rem;
        padding-top: 0.55rem;
        padding-bottom: 0.55rem;
        border-bottom: 3px solid transparent;
        transition: color 0.15s ease, border-color 0.15s ease;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: var(--mono-blue) !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: var(--mono-navy) !important;
        border-bottom-color: var(--mono-blue) !important;
    }
    div[data-testid="stTabs"] button[role="tab"] p {
        color: inherit !important;
        font-weight: inherit !important;
    }
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label,
    div[data-testid="stTextInput"] label,
    div[data-testid="stTextArea"] label,
    div[data-testid="stDateInput"] label,
    div[data-testid="stTimeInput"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stRadio"] label,
    div[data-testid="stCheckbox"] label {
        display: block !important;
        color: var(--mono-text) !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        margin-bottom: 0.18rem !important;
    }
    div[data-testid="stNumberInput"] label p,
    div[data-testid="stSelectbox"] label p,
    div[data-testid="stTextInput"] label p,
    div[data-testid="stTextArea"] label p,
    div[data-testid="stDateInput"] label p,
    div[data-testid="stTimeInput"] label p,
    div[data-testid="stSlider"] label p,
    div[data-testid="stRadio"] label p,
    div[data-testid="stCheckbox"] label p {
        color: inherit !important;
        font-weight: inherit !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
        margin-bottom: 0 !important;
    }
    .hero-wrap {
        background: linear-gradient(120deg, var(--mono-navy) 0%, #153a72 52%, var(--mono-cyan) 100%);
        border-radius: 24px;
        padding: 1.4rem 1.55rem 1.15rem 1.55rem;
        color: white;
        box-shadow: 0 18px 40px rgba(12, 31, 65, 0.18);
        margin-bottom: 1rem;
        overflow: hidden;
        position: relative;
    }
    .hero-wrap:before {
        content: "";
        position: absolute;
        top: -30px;
        right: -10px;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(255,255,255,0.16) 0%, rgba(255,255,255,0.04) 58%, transparent 70%);
        border-radius: 50%;
    }
    .brand-row {
        display: grid;
        grid-template-columns: 1.7fr 0.7fr;
        gap: 1rem;
        align-items: stretch;
    }
    .brand-chip {
        display: inline-block;
        padding: 0.25rem 0.72rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.18);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin-bottom: 0.35rem;
    }
    .hero-wrap h1 {
        margin: 0.1rem 0 0.35rem 0;
        font-size: 2.05rem;
        line-height: 1.08;
        font-weight: 800;
    }
    .hero-wrap p {
        margin: 0;
        opacity: 0.97;
        max-width: 980px;
        font-size: 0.99rem;
        line-height: 1.45;
    }
    .logo-panel {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 18px;
        padding: 0.95rem;
        min-height: 160px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        text-align: center;
    }
    .logo-panel img {
        max-width: 100%;
        max-height: 72px;
        margin-bottom: 0.55rem;
        object-fit: contain;
        border-radius: 8px;
    }
    .logo-label {
        font-size: 0.78rem;
        font-weight: 700;
        opacity: 0.9;
        margin-bottom: 0.15rem;
    }
    .logo-caption {
        font-size: 0.82rem;
        opacity: 0.95;
    }
    .summary-strip {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 0.7rem;
        margin-top: 1rem;
    }
    .summary-tile {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.16);
        border-radius: 16px;
        padding: 0.8rem 0.95rem;
        min-height: 82px;
    }
    .summary-tile .label {
        font-size: 0.8rem;
        opacity: 0.88;
        margin-bottom: 0.25rem;
    }
    .summary-tile .value {
        font-size: 1.05rem;
        font-weight: 800;
        line-height: 1.22;
    }
    .journey {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.7rem;
        margin: 0.95rem 0 1rem 0;
    }
    .journey-step {
        background: white;
        border: 1px solid var(--mono-border);
        border-radius: 18px;
        padding: 0.95rem;
        min-height: 108px;
        box-shadow: 0 8px 24px rgba(16, 34, 61, 0.05);
    }
    .journey-step strong {
        display: block;
        color: var(--mono-navy);
        margin-bottom: 0.35rem;
        font-size: 0.95rem;
    }
    .journey-step span {
        color: var(--mono-muted);
        font-size: 0.88rem;
        line-height: 1.38;
    }
    .notice-banner {
        background: linear-gradient(90deg, #f9c846 0%, #f3e58a 100%);
        color: #3f2c00;
        border-radius: 16px;
        padding: 0.9rem 1rem;
        font-weight: 700;
        margin-bottom: 0.95rem;
        border: 1px solid rgba(136, 95, 0, 0.15);
    }
    .panel {
        background: white;
        border: 1px solid var(--mono-border);
        border-radius: 20px;
        padding: 1.05rem 1.1rem;
        box-shadow: 0 10px 28px rgba(16, 34, 61, 0.05);
        margin-bottom: 1rem;
    }
    .panel h3, .panel h4 {
        color: var(--mono-navy);
        margin-top: 0;
        margin-bottom: 0.35rem;
    }
    .muted {
        color: var(--mono-muted);
        font-size: 0.93rem;
    }
    .section-kicker {
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.4px;
        text-transform: uppercase;
        color: var(--mono-blue);
        margin-bottom: 0.45rem;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid var(--mono-border);
        border-left: 5px solid var(--mono-blue);
        border-radius: 18px;
        padding: 0.95rem 1rem;
        box-shadow: 0 8px 22px rgba(16, 34, 61, 0.05);
        height: 100%;
    }
    .metric-card .small-label {
        color: var(--mono-muted);
        font-size: 0.8rem;
        margin-bottom: 0.28rem;
    }
    .metric-card .big-value {
        color: var(--mono-navy);
        font-size: 1.7rem;
        font-weight: 800;
        line-height: 1.05;
    }
    .metric-card .subtext {
        color: var(--mono-muted);
        font-size: 0.86rem;
        margin-top: 0.4rem;
    }
    .risk-pill {
        display: inline-block;
        padding: 0.34rem 0.8rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 800;
        margin-top: 0.15rem;
        border: 1px solid transparent;
    }
    .risk-low {
        background: var(--success-bg);
        color: var(--success-text);
        border-color: #bde3ca;
    }
    .risk-mod {
        background: var(--warn-bg);
        color: var(--warn-text);
        border-color: #f1d27a;
    }
    .risk-high {
        background: var(--danger-bg);
        color: var(--danger-text);
        border-color: #f2b8bd;
    }
    .callout {
        background: #edf5ff;
        border: 1px solid #cfe0f7;
        border-left: 5px solid var(--mono-blue);
        border-radius: 16px;
        padding: 0.95rem 1rem;
        color: #173860;
        margin: 0.55rem 0 0.75rem 0;
    }
    .module-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.8rem;
        margin-top: 0.75rem;
    }
    .module-card {
        background: #fbfdff;
        border: 1px solid var(--mono-border);
        border-radius: 18px;
        padding: 1rem;
        min-height: 170px;
    }
    .module-card h4 {
        margin: 0.15rem 0 0.45rem 0;
        color: var(--mono-navy);
    }
    .mini-tag {
        display: inline-block;
        border-radius: 999px;
        padding: 0.18rem 0.6rem;
        background: #eaf1fb;
        color: var(--mono-blue);
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
    }
    .share-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 0.85rem;
    }
    .share-card {
        background: #fbfdff;
        border: 1px solid var(--mono-border);
        border-radius: 18px;
        padding: 1rem;
        min-height: 240px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .share-card h4 {
        margin: 0.2rem 0 0.25rem 0;
        color: var(--mono-navy);
    }
    .qr-wrap {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0.4rem 0 0.2rem 0;
    }
    .qr-wrap img {
        width: 150px;
        height: 150px;
        border-radius: 12px;
        border: 1px solid var(--mono-border);
        background: white;
        padding: 0.35rem;
    }
    .link-box {
        background: #f4f8fc;
        border: 1px solid var(--mono-border);
        border-radius: 12px;
        padding: 0.6rem 0.7rem;
        font-size: 0.84rem;
        word-break: break-word;
        color: #1d3153;
        margin-top: 0.5rem;
    }
    .report-page {
        background: white;
        border: 1px solid #d5e0ee;
        border-radius: 12px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 14px 36px rgba(16, 34, 61, 0.08);
        width: 100%;
        max-width: 980px;
        margin: 0 auto 1rem auto;
    }
    .report-head {
        display: grid;
        grid-template-columns: 1.55fr 0.65fr;
        gap: 1rem;
        border-bottom: 2px solid var(--mono-blue);
        padding-bottom: 0.8rem;
        margin-bottom: 0.9rem;
    }
    .report-title {
        color: var(--mono-navy);
        font-size: 1.55rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .report-subtitle {
        color: var(--mono-muted);
        font-size: 0.92rem;
        line-height: 1.4;
    }
    .report-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.9rem;
        margin: 0.9rem 0;
    }
    .report-box {
        border: 1px solid var(--mono-border);
        border-radius: 14px;
        padding: 0.85rem 0.95rem;
        background: #fbfdff;
    }
    .report-box h5 {
        margin: 0 0 0.35rem 0;
        color: var(--mono-navy);
        font-size: 0.95rem;
    }
    .report-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.75rem;
        font-size: 0.92rem;
    }
    .report-table th, .report-table td {
        border: 1px solid #dbe5f0;
        padding: 0.5rem 0.55rem;
        text-align: left;
        vertical-align: top;
    }
    .report-table th {
        background: #f3f7fb;
        color: var(--mono-navy);
    }
    .report-qr {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        border: 1px dashed var(--mono-border);
        border-radius: 14px;
        min-height: 190px;
        padding: 0.7rem;
        background: #fbfdff;
        text-align: center;
    }
    .report-qr img {
        width: 135px;
        height: 135px;
        border: 1px solid var(--mono-border);
        border-radius: 10px;
        padding: 0.25rem;
        background: white;
    }
    .report-footer {
        margin-top: 0.9rem;
        color: var(--mono-muted);
        font-size: 0.82rem;
        border-top: 1px solid #e4ebf3;
        padding-top: 0.7rem;
        line-height: 1.45;
    }
    .deploy-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.85rem;
        margin-top: 0.75rem;
    }
    .footer-wrap {
        margin-top: 1.25rem;
        padding: 1rem 1.1rem;
        border-top: 1px solid var(--mono-border);
        color: var(--mono-muted);
        font-size: 0.88rem;
        line-height: 1.5;
    }
    .footer-wrap a {
        color: var(--mono-blue);
        text-decoration: none;
        font-weight: 700;
    }
    .footer-top {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
        margin-bottom: 0.3rem;
    }
    .footer-pill {
        display: inline-block;
        padding: 0.18rem 0.55rem;
        border-radius: 999px;
        background: #eaf1fb;
        color: var(--mono-blue);
        font-size: 0.72rem;
        font-weight: 800;
        margin-right: 0.35rem;
        margin-bottom: 0.3rem;
    }
    @media (max-width: 1180px) {
        .brand-row, .report-head { grid-template-columns: 1fr; }
        .journey { grid-template-columns: 1fr 1fr; }
        .summary-strip { grid-template-columns: 1fr 1fr; }
        .module-grid, .share-grid, .deploy-grid, .report-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 760px) {
        .journey, .summary-strip { grid-template-columns: 1fr; }
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================================================
# CONSTANTS
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
    "patient_name": "Demo patient",
    "age": 30,
    "height": 165,
    "weight": 70.0,
    "parity": 1,
    "ethnicity_group": "Southern and Central Asian",
    "family_hist_dm": 1,
    "past_hist_gdm": 0,
    "past_shoulder_d": 0,
    "previous_preeclampsia": 0,
    "previous_macrosomia": 0,
    "anc_threshold": 0.50,
    "gdm_status": "Not diagnosed / screening stage",
    "antenatal_fpg": 5.5,
    "antenatal_2h_ogtt": 8.6,
    "post_view_antenatal_2h_ogtt": 8.6,
    "recurrent_gdm": 0,
    "insulin_treatment": 0,
    "irregular_menses": 0,
    "postnatal_fpg": 5.2,
    "postnatal_2h_ogtt": 7.4,
    "postnatal_bmi": 27.0,
    "anc_prob": None,
    "anc_pred": None,
    "ant_prob": None,
    "post_prob": None,
    "booking_feature_frame": pd.DataFrame(),
    "institution_name": "Monash University",
    "institution_unit": "School / Department / Research Group",
    "contact_email": "your.name@monash.edu",
    "public_app_url": "https://your-app-name.streamlit.app",
    "github_url": "https://github.com/your-username/maternal-diabetes-platform",
    "publication_url": "https://doi.org/your-paper-doi",
    "app_tagline": "Pregnancy-to-postpartum diabetes risk platform",
    "model_version": "Prototype v3.0",
    "report_note": "Research demo only. Decision support and presentation use only.",
}

# =========================================================
# STATE
# =========================================================
def init_state():
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_demo_patient():
    demo = {
        "patient_name": "Demo patient",
        "age": 33,
        "height": 160,
        "weight": 78.5,
        "parity": 2,
        "ethnicity_group": "Southern and Central Asian",
        "family_hist_dm": 1,
        "past_hist_gdm": 1,
        "past_shoulder_d": 0,
        "previous_preeclampsia": 1,
        "previous_macrosomia": 0,
        "gdm_status": "GDM confirmed",
        "antenatal_fpg": 5.8,
        "antenatal_2h_ogtt": 9.4,
        "post_view_antenatal_2h_ogtt": 9.4,
        "recurrent_gdm": 1,
        "insulin_treatment": 1,
        "irregular_menses": 1,
        "postnatal_fpg": 5.9,
        "postnatal_2h_ogtt": 8.9,
        "postnatal_bmi": 31.4,
    }
    for key, value in demo.items():
        st.session_state[key] = value


def reset_everything():
    for key, value in DEFAULTS.items():
        st.session_state[key] = value


init_state()


def sync_antenatal_to_post_view():
    st.session_state.post_view_antenatal_2h_ogtt = float(st.session_state.antenatal_2h_ogtt)


def sync_post_view_to_antenatal():
    st.session_state.antenatal_2h_ogtt = float(st.session_state.post_view_antenatal_2h_ogtt)


# =========================================================
# GENERIC HELPERS
# =========================================================
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


def risk_css_class(label: str) -> str:
    key = label.lower()
    if key == "low":
        return "risk-low"
    if key == "moderate":
        return "risk-mod"
    return "risk-high"


def risk_pill(label: str) -> str:
    return f'<span class="risk-pill {risk_css_class(label)}">{escape(label)}</span>'


def classification_text(prob: float, threshold: float) -> str:
    return "Above action threshold" if prob >= threshold else "Below action threshold"


def recode_ethnicity(group_label: str) -> str:
    return RECODE_DICT.get(group_label, "Other NEC")


def first_existing_logo() -> Optional[Path]:
    for path in LOGO_CANDIDATES:
        if path.exists():
            return path
    return None


def file_to_data_uri(path: Path) -> str:
    suffix = path.suffix.lower().lstrip('.')
    mime = "image/png"
    if suffix in {"jpg", "jpeg"}:
        mime = "image/jpeg"
    elif suffix == "svg":
        mime = "image/svg+xml"
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def logo_html() -> str:
    logo_path = first_existing_logo()
    if logo_path is None:
        return "<div class='logo-panel'><div class='logo-label'>Custom logo area</div><div class='logo-caption'>Add assets/custom_logo.png before deployment</div></div>"
    uri = file_to_data_uri(logo_path)
    return f"""
    <div class="logo-panel">
        <img src="{uri}" alt="Institution logo" />
        <div class="logo-label">Custom logo area</div>
        <div class="logo-caption">Replace <strong>assets/custom_logo.*</strong> with your approved branding.</div>
    </div>
    """


def qr_image(url: str):
    payload = (url or "").strip()
    if not payload or not payload.startswith(("http://", "https://")):
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
    model_path = MODEL_FILE
    scaler_path = SCALER_FILE
    if not model_path.exists() or not scaler_path.exists():
        return None, None, "Booking model or scaler file not found in the app folder."

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    scaler = joblib.load(str(scaler_path))
    return model, scaler, None


booking_model, booking_scaler, booking_model_error = load_booking_model()

# =========================================================
# PREDICTION HELPERS
# =========================================================
def build_booking_features() -> pd.DataFrame:
    past_hist_obs_complica = int(
        st.session_state.past_shoulder_d == 1
        or st.session_state.previous_preeclampsia == 1
        or st.session_state.previous_macrosomia == 1
    )

    row = {col: 0 for col in FEATURE_COLUMNS}
    row["Height"] = int(st.session_state.height)
    row["Weight"] = float(st.session_state.weight)
    row["Parity"] = int(st.session_state.parity)
    row["Age"] = int(st.session_state.age)
    row["Family_Hist_DM"] = int(st.session_state.family_hist_dm)
    row["Past_Hist_GDM"] = int(st.session_state.past_hist_gdm)
    row["Past_Hist_Obs_Complica"] = int(past_hist_obs_complica)

    ethnicity_col = recode_ethnicity(st.session_state.ethnicity_group)
    row[ethnicity_col] = 1

    df = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    df.loc[:, "Height"] = df["Height"].astype(int)
    df.loc[:, "Weight"] = df["Weight"].astype(float)
    df.loc[:, "Parity"] = df["Parity"].astype("int8")
    df.loc[:, "Age"] = df["Age"].astype(int)
    for col in [
        "Caucasian",
        "Oceanian_not_WANZ",
        "ME_NA_SSA",
        "South_Central_Asia",
        "SouthEast_NorthEast_Asia",
        "Other NEC",
    ]:
        df.loc[:, col] = df[col].astype("int64")
    for col in ["Family_Hist_DM", "Past_Hist_GDM", "Past_Hist_Obs_Complica"]:
        df.loc[:, col] = df[col].astype("int32")
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


def make_gauge(prob: float, title: str, threshold: Optional[float] = None, bar_color: str = "#005eb8"):
    steps = [
        {"range": [0, 10], "color": "#e8f7ef"},
        {"range": [10, 20], "color": "#fff5d9"},
        {"range": [20, 100], "color": "#fde8ea"},
    ]
    gauge = {
        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#57708c"},
        "bar": {"color": bar_color},
        "bgcolor": "white",
        "borderwidth": 1,
        "bordercolor": "#d8e1ee",
        "steps": steps,
    }
    if threshold is not None:
        gauge["threshold"] = {"line": {"color": "#10223d", "width": 3}, "thickness": 0.75, "value": threshold * 100}

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": title, "font": {"size": 16}},
            gauge=gauge,
        )
    )
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=55, b=10), paper_bgcolor="white", font={"color": "#10223d"})
    return fig


def make_comparison_chart(summary_df: pd.DataFrame):
    if summary_df.empty:
        return None
    fig = go.Figure(
        data=[
            go.Bar(
                x=summary_df["Short module"],
                y=summary_df["Probability_pct"],
                text=[f"{x:.1f}%" for x in summary_df["Probability_pct"]],
                textposition="outside",
                marker_color=["#005eb8", "#0d7d9d", "#14b7c6"][: len(summary_df)],
            )
        ]
    )
    fig.update_layout(
        title="Risk snapshot across completed modules",
        xaxis_title="Module",
        yaxis_title="Predicted probability (%)",
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=25, r=10, t=50, b=35),
    )
    return fig


def make_journey_line_chart(summary_df: pd.DataFrame):
    if summary_df.empty:
        return None
    fig = go.Figure(
        data=[
            go.Scatter(
                x=summary_df["Journey stage"],
                y=summary_df["Probability_pct"],
                mode="lines+markers+text",
                text=[f"{x:.1f}%" for x in summary_df["Probability_pct"]],
                textposition="top center",
                line={"width": 3, "color": "#005eb8"},
                marker={"size": 12, "color": "#14b7c6"},
            )
        ]
    )
    fig.update_layout(
        title="Risk journey view",
        xaxis_title="Journey stage",
        yaxis_title="Predicted probability (%)",
        height=350,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=25, r=10, t=50, b=35),
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
    fig.update_layout(title="Journey completion", height=320, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="white")
    return fig


def next_best_action(stage: str, prob: float, threshold: Optional[float] = None) -> list[str]:
    if stage == "booking":
        if threshold is not None and prob < threshold:
            return [
                "Continue routine antenatal pathway and standard 24-28 week GDM screening.",
                "Reinforce healthy nutrition, weight, and physical activity advice early in pregnancy.",
                "Reassess if new risk factors emerge before routine screening.",
            ]
        if prob < 0.20:
            return [
                "Discuss modifiable risk factors early in pregnancy.",
                "Consider targeted preventive counselling and reinforce routine screening attendance.",
                "Document the elevated risk so the care team remembers routine or earlier glucose assessment.",
            ]
        return [
            "Flag as high-risk for GDM and consider earlier testing per local protocol.",
            "Provide targeted nutrition and activity counselling at booking.",
            "Plan closer review before routine 24-28 week screening.",
        ]

    if stage == "antenatal_after_gdm":
        if threshold is not None and prob < threshold:
            return [
                "Discuss postpartum diabetes prevention before discharge from maternity care.",
                "Ensure postpartum glucose testing is scheduled and documented.",
            ]
        return [
            "Create an enhanced postpartum prevention plan before delivery.",
            "Emphasise postpartum OGTT follow-up and long-term diabetes screening.",
            "Offer structured lifestyle or weight-management support where available.",
        ]

    if threshold is not None and prob < threshold:
        return [
            "Continue routine diabetes-prevention counselling and periodic screening.",
            "Reinforce healthy weight, physical activity, and repeat glycaemic follow-up.",
        ]
    return [
        "Escalate diabetes-prevention follow-up in primary care or endocrinology.",
        "Prioritise weight-management, nutrition, and annual glycaemic surveillance.",
        "Use the result to support shared decision-making, not as a stand-alone diagnosis.",
    ]


def render_actions(actions: list[str]):
    for item in actions:
        st.markdown(f"- {item}")


def render_result_cards(prob: float, band: str, title: str, threshold: Optional[float], prediction_label: str, subtitle: str):
    left, right = st.columns([1.05, 1])
    with left:
        st.plotly_chart(make_gauge(prob, title, threshold), use_container_width=True)
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
                    <div class="big-value" style="font-size:1.22rem;">{risk_pill(band)}</div>
                    <div class="subtext">Band shown for visual triage.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            threshold_text = f"Threshold {threshold:.3f}" if threshold is not None else "Threshold not shown"
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="small-label">Action label</div>
                    <div class="big-value" style="font-size:1.05rem;">{escape(prediction_label)}</div>
                    <div class="subtext">{escape(threshold_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def patient_context_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ["Patient label", str(st.session_state.patient_name)],
            ["Current pathway status", str(st.session_state.gdm_status)],
            ["Age", str(st.session_state.age)],
            ["Parity", str(st.session_state.parity)],
            ["Ethnicity group", str(st.session_state.ethnicity_group)],
            ["Family history of diabetes", "Yes" if st.session_state.family_hist_dm == 1 else "No"],
        ],
        columns=["Field", "Value"],
    )


def summary_dataframe() -> pd.DataFrame:
    rows = []
    if st.session_state.anc_prob is not None:
        rows.append(
            {
                "Module": "ANC booking - risk of GDM",
                "Short module": "Booking risk of GDM",
                "Journey stage": "Booking visit",
                "Probability": round(float(st.session_state.anc_prob), 4),
                "Probability_pct": float(st.session_state.anc_prob) * 100,
                "Risk band": booking_risk_band(float(st.session_state.anc_prob)),
                "Threshold": float(st.session_state.anc_threshold),
                "Action label": classification_text(float(st.session_state.anc_prob), float(st.session_state.anc_threshold)),
            }
        )
    if st.session_state.ant_prob is not None:
        rows.append(
            {
                "Module": "Pregnancy after GDM - future T2DM risk",
                "Short module": "Pregnancy after GDM",
                "Journey stage": "Pregnancy after GDM",
                "Probability": round(float(st.session_state.ant_prob), 4),
                "Probability_pct": float(st.session_state.ant_prob) * 100,
                "Risk band": published_model_band(float(st.session_state.ant_prob), 0.096),
                "Threshold": 0.096,
                "Action label": classification_text(float(st.session_state.ant_prob), 0.096),
            }
        )
    if st.session_state.post_prob is not None:
        rows.append(
            {
                "Module": "Postnatal after GDM - future T2DM risk",
                "Short module": "Postnatal follow-up",
                "Journey stage": "Postnatal follow-up",
                "Probability": round(float(st.session_state.post_prob), 4),
                "Probability_pct": float(st.session_state.post_prob) * 100,
                "Risk band": published_model_band(float(st.session_state.post_prob), 0.086),
                "Threshold": 0.086,
                "Action label": classification_text(float(st.session_state.post_prob), 0.086),
            }
        )
    return pd.DataFrame(rows)


def modules_completed() -> int:
    return int(st.session_state.anc_prob is not None) + int(st.session_state.ant_prob is not None) + int(st.session_state.post_prob is not None)


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
        phrases = []
        if st.session_state.anc_prob is not None:
            phrases.append(
                f"Booking GDM risk {float(st.session_state.anc_prob):.1%} ({booking_risk_band(float(st.session_state.anc_prob))})."
            )
        if st.session_state.ant_prob is not None:
            phrases.append(
                f"Pregnancy-after-GDM future T2DM risk {float(st.session_state.ant_prob):.1%} ({published_model_band(float(st.session_state.ant_prob), 0.096)})."
            )
        if st.session_state.post_prob is not None:
            phrases.append(
                f"Postnatal future T2DM risk {float(st.session_state.post_prob):.1%} ({published_model_band(float(st.session_state.post_prob), 0.086)})."
            )
        interpretation = " ".join(phrases)

    qr_uri = qr_data_uri(st.session_state.public_app_url)
    qr_html = (
        f"<div class='report-qr'><img src='{qr_uri}' alt='QR code'/><div style='margin-top:0.45rem;color:#5b6f8a;font-size:0.82rem;'>Scan to open public app</div></div>"
        if qr_uri
        else "<div class='report-qr'><div style='color:#5b6f8a;font-size:0.86rem;'>Add a public app URL in the sidebar to generate a QR code.</div></div>"
    )

    return f"""
    <div class="report-page">
        <div class="report-head">
            <div>
                <div class="report-title">M-Guide | Maternal Diabetes Prevention Platform</div>
                <div class="report-subtitle">{escape(st.session_state.app_tagline)}<br>{escape(st.session_state.institution_name)} | {escape(st.session_state.institution_unit)} | {escape(st.session_state.model_version)}</div>
            </div>
            {qr_html}
        </div>
        <div class="report-grid">
            <div class="report-box">
                <h5>Patient context</h5>
                <table class="report-table">{patient_rows}</table>
            </div>
            <div class="report-box">
                <h5>Interpretation</h5>
                <p style="margin:0; color:#173860; line-height:1.55;">{escape(interpretation)}</p>
                <div class="callout" style="margin-top:0.8rem;">This prototype combines booking risk of GDM with antenatal and postnatal future T2DM risk after GDM in one prevention-oriented platform.</div>
            </div>
        </div>
        <div class="report-box">
            <h5>Risk journey table</h5>
            <table class="report-table">
                <thead>
                    <tr><th>Module</th><th>Probability</th><th>Risk band</th><th>Action label</th></tr>
                </thead>
                <tbody>{summary_rows}</tbody>
            </table>
        </div>
        <div class="report-footer">
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
st.sidebar.caption("One platform for ANC booking, pregnancy after GDM, and postnatal prevention follow-up.")

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Load demo patient", use_container_width=True):
        load_demo_patient()
        st.rerun()
with c2:
    if st.button("Reset all inputs", use_container_width=True):
        reset_everything()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.text_input("Patient label", key="patient_name")
st.sidebar.number_input("Parity", min_value=0, max_value=15, step=1, key="parity")
st.sidebar.selectbox(
    "Family history of diabetes",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
    key="family_hist_dm",
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
st.sidebar.text_input("Publication URL / DOI", key="publication_url")
st.sidebar.text_input("App tagline", key="app_tagline")
st.sidebar.text_input("Model version label", key="model_version")

# =========================================================
# HEADER
# =========================================================
completed_count = modules_completed()
hero_left, hero_right = st.columns([4, 1])
with hero_left:
    st.markdown(
        f"""
        <div class="hero-wrap">
            <span class="brand-chip">MONASH-STYLE PUBLIC DEMO</span>
            <h1>M-Guide | Maternal Diabetes Prevention Platform</h1>
            <p>{escape(st.session_state.app_tagline)}. A longitudinal risk journey linking ANC booking risk of gestational diabetes with future type 2 diabetes risk during pregnancy after GDM and again after delivery.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_right:
    logo_path = first_existing_logo()
    if logo_path is not None:
        st.image(str(logo_path), width=180)
    else:
        st.markdown(
            "<div class='logo-panel'><div class='logo-label'>Custom logo area</div><div class='logo-caption'>Add assets/custom_logo.png before deployment</div></div>",
            unsafe_allow_html=True,
        )

summary_cols = st.columns(4)
summary_items = [
    ("Patient label", str(st.session_state.patient_name)),
    ("Pathway status", str(st.session_state.gdm_status)),
    ("Modules completed", f"{completed_count}/3"),
    ("Deployment mode", "Public demo ready"),
]
for col, (label, value) in zip(summary_cols, summary_items):
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
  <div class="journey-step"><strong>1. Booking visit</strong><span>Estimate risk of developing GDM at 24-28 weeks from the saved CatBoost model.</span></div>
  <div class="journey-step"><strong>2. Routine screening</strong><span>Screening takes place at 24-28 weeks and women with GDM move into future diabetes prevention planning.</span></div>
  <div class="journey-step"><strong>3. Pregnancy after GDM</strong><span>Use the antenatal logistic equation to estimate future T2DM risk after delivery.</span></div>
  <div class="journey-step"><strong>4. Postnatal follow-up</strong><span>Use postpartum glucose values and BMI to update long-term future T2DM risk.</span></div>
  <div class="journey-step"><strong>5. Prevention summary</strong><span>Combine all outputs into one presentation-ready patient prevention passport.</span></div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="notice-banner">Research demo only. This platform is for presentation and decision-support demonstration. It is not stand-alone diagnosis or real-world clinical deployment without governance, local validation, security review, and approval.</div>',
    unsafe_allow_html=True,
)

# =========================================================
# TABS
# =========================================================
(
    tab_landing,
    tab_booking,
    tab_antenatal,
    tab_postnatal,
    tab_report,
    tab_about,
) = st.tabs(
    [
        "Public landing page",
        "Booking screen",
        "After GDM in pregnancy",
        "Postnatal follow-up",
        "Clinical report",
        "About & deploy",
    ]
)

with tab_landing:
    left, right = st.columns([1.2, 0.9])
    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">Public-facing overview</div>
                <h3>One platform, three prediction stages, one prevention story</h3>
                <p class="muted">Use this landing page when you share the app publicly. It explains the journey, the intended use, and the quickest path to a live demo.</p>
                <div class="callout"><strong>Simple positioning:</strong> describe the tool as a <strong>maternal metabolic risk journey</strong> rather than three disconnected calculators.</div>
                <div class="module-grid">
                    <div class="module-card"><span class="mini-tag">Module 1</span><h4>Booking visit</h4><p class="muted">Saved CatBoost model plus saved scaler to estimate risk of developing GDM at 24-28 weeks.</p></div>
                    <div class="module-card"><span class="mini-tag">Module 2</span><h4>After GDM in pregnancy</h4><p class="muted">Published antenatal logistic equation to estimate future T2DM risk after delivery among women with GDM.</p></div>
                    <div class="module-card"><span class="mini-tag">Module 3</span><h4>Postnatal follow-up</h4><p class="muted">Published postnatal logistic equation to update future T2DM risk using postpartum glucose values and BMI.</p></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">Use in 30 seconds</div>
                <h4>Fast demo sequence</h4>
                <ol>
                    <li>Click <strong>Load demo patient</strong> in the sidebar.</li>
                    <li>Run the <strong>Booking screen</strong> module.</li>
                    <li>Run <strong>After GDM in pregnancy</strong> and <strong>Postnatal follow-up</strong>.</li>
                    <li>Open <strong>Clinical report</strong> for a screenshot or export.</li>
                </ol>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">Live patient context</div>
                <h4>Shared patient summary</h4>
                <p class="muted">All modules use the same patient context so the tool feels like one longitudinal pathway.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.dataframe(patient_context_table(), width="stretch", hide_index=True)
        st.plotly_chart(make_completion_donut(), use_container_width=True)

    st.markdown(
        """
        <div class="panel">
            <div class="section-kicker">Public sharing section</div>
            <h3>Share the app, paper, and code with QR-enabled cards</h3>
            <p class="muted">Populate the URLs in the sidebar, then use these cards for posters, presentations, or the app landing page.</p>
        </div>
        """,
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
            st.markdown(f"<div class='panel'><div class='section-kicker'>{escape(title)}</div><h4>{escape(title)}</h4><p class='muted'>{escape(caption)}</p></div>", unsafe_allow_html=True)
            img_bytes = qr_image(url)
            if img_bytes is not None:
                st.image(img_bytes, width=170)
            else:
                st.info(f"Add a valid URL for {title.lower()} in the sidebar to generate a QR code.")
            st.caption(url or "URL not set")

    summary_df = summary_dataframe()
    if summary_df.empty:
        st.info("Run one or more modules to populate the journey visualizations.")
    else:
        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(make_comparison_chart(summary_df), use_container_width=True)
        with v2:
            st.plotly_chart(make_journey_line_chart(summary_df), use_container_width=True)

with tab_booking:
    st.markdown(
        """
        <div class="panel">
            <div class="section-kicker">Stage 1</div>
            <h3>ANC booking risk of developing GDM</h3>
            <p class="muted">This module uses the saved CatBoost model and preprocessing scaler already stored in the app folder.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if booking_model_error:
        st.error(booking_model_error)

    st.markdown("#### Predictors used in the booking model")
    g1, g2 = st.columns(2)
    with g1:
        booking_parity = st.number_input(
            "Parity",
            min_value=0,
            max_value=15,
            step=1,
            value=int(st.session_state.parity),
            key="booking_parity",
        )
    with g2:
        booking_family_hist_dm = st.selectbox(
            "Family history of diabetes",
            options=[0, 1],
            index=int(st.session_state.family_hist_dm),
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="booking_family_hist_dm",
        )
    st.session_state.parity = int(booking_parity)
    st.session_state.family_hist_dm = int(booking_family_hist_dm)

    st.markdown("##### Maternal characteristics")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("Age (years)", min_value=10, max_value=60, step=1, key="age")
        st.number_input("Height (cm)", min_value=100, max_value=220, step=1, key="height")
        st.number_input("Weight (kg)", min_value=30.0, max_value=250.0, step=0.1, key="weight")
    with c2:
        st.selectbox("Ethnicity group", options=list(RECODE_DICT.keys()), key="ethnicity_group")
    with c3:
        st.markdown(
            '<div class="callout"><strong>Presentation tip:</strong> position this module as the entry point for early prevention and targeted antenatal follow-up.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("##### Previous obstetric history")
    o1, o2, o3 = st.columns(3)
    with o1:
        st.selectbox(
            "Previous GDM",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="past_hist_gdm",
        )
        st.selectbox(
            "Previous shoulder dystocia",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="past_shoulder_d",
        )
    with o2:
        st.selectbox(
            "Previous pre-eclampsia",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="previous_preeclampsia",
        )
        st.selectbox(
            "Previous macrosomia",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="previous_macrosomia",
        )
    with o3:
        st.info(
            "The booking model uses age, height, weight, ethnicity, parity, family history of diabetes, previous GDM, and previous obstetric complications."
        )

    if st.button("Run booking prediction", type="primary", use_container_width=True):
        try:
            prob, pred, feature_frame = predict_booking_risk()
            st.session_state.anc_prob = prob
            st.session_state.anc_pred = pred
            st.session_state.booking_feature_frame = feature_frame
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if st.session_state.anc_prob is not None:
        anc_prob = float(st.session_state.anc_prob)
        anc_band = booking_risk_band(anc_prob)
        pred_label = classification_text(anc_prob, float(st.session_state.anc_threshold))
        render_result_cards(
            anc_prob,
            anc_band,
            "Risk of developing GDM at 24-28 weeks",
            float(st.session_state.anc_threshold),
            pred_label,
            "Booking-stage risk output from the saved CatBoost model.",
        )
        st.markdown("#### Suggested next action")
        render_actions(next_best_action("booking", anc_prob, float(st.session_state.anc_threshold)))
        with st.expander("Model inputs sent to scaler and CatBoost model"):
            st.dataframe(st.session_state.booking_feature_frame, use_container_width=True)

with tab_antenatal:
    st.markdown(
        """
        <div class="panel">
            <div class="section-kicker">Stage 2</div>
            <h3>Pregnancy after GDM: future T2DM risk after delivery</h3>
            <p class="muted">Use this module for women who already have GDM. It applies the published antenatal logistic equation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        st.number_input("Antenatal fasting plasma glucose (FPG)", min_value=0.0, max_value=30.0, step=0.1, key="antenatal_fpg")
        st.number_input("Antenatal 2h-OGTT", min_value=0.0, max_value=40.0, step=0.1, key="antenatal_2h_ogtt", on_change=sync_antenatal_to_post_view)
    with a2:
        st.selectbox(
            "History of recurrent GDM",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="recurrent_gdm",
        )
        st.selectbox(
            "Insulin treatment during pregnancy",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="insulin_treatment",
        )
    with a3:
        st.selectbox(
            "History of irregular menstrual cycle",
            options=[0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="irregular_menses",
        )
        st.metric("Parity", st.session_state.parity)
        st.metric("Family history of diabetes", "Yes" if st.session_state.family_hist_dm == 1 else "No")

    if st.button("Run antenatal future T2DM prediction", type="primary", use_container_width=True):
        st.session_state.ant_prob = predict_antenatal_t2dm_after_gdm()

    if st.session_state.ant_prob is not None:
        ant_prob = float(st.session_state.ant_prob)
        ant_band = published_model_band(ant_prob, 0.096)
        ant_label = classification_text(ant_prob, 0.096)
        render_result_cards(
            ant_prob,
            ant_band,
            "Future T2DM risk after delivery - antenatal model",
            0.096,
            ant_label,
            "Published antenatal logistic model for women with GDM.",
        )
        st.markdown("#### Suggested next action")
        render_actions(next_best_action("antenatal_after_gdm", ant_prob, 0.096))
        st.info("Paper-reported action threshold used in this demo: 0.096.")

with tab_postnatal:
    st.markdown(
        """
        <div class="panel">
            <div class="section-kicker">Stage 3</div>
            <h3>Postnatal follow-up: updated future T2DM risk</h3>
            <p class="muted">Use postpartum glucose values and BMI to update long-term future T2DM risk after a pregnancy affected by GDM.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "post_view_antenatal_2h_ogtt" not in st.session_state:
        st.session_state.post_view_antenatal_2h_ogtt = float(st.session_state.antenatal_2h_ogtt)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.number_input(
            "Pregnancy 2h-OGTT (mirrored)",
            min_value=0.0,
            max_value=40.0,
            step=0.1,
            key="post_view_antenatal_2h_ogtt",
            on_change=sync_post_view_to_antenatal,
            help="This mirrors the antenatal 2h-OGTT value used in the postnatal model.",
        )
        st.number_input("Postnatal FPG", min_value=0.0, max_value=30.0, step=0.1, key="postnatal_fpg")
    with p2:
        st.number_input("Postnatal 2h-OGTT", min_value=0.0, max_value=40.0, step=0.1, key="postnatal_2h_ogtt")
        st.number_input("Postnatal BMI", min_value=10.0, max_value=80.0, step=0.1, key="postnatal_bmi")
    with p3:
        st.markdown(
            '<div class="callout"><strong>Practical note:</strong> this module is strongest when postpartum testing has actually been completed.</div>',
            unsafe_allow_html=True,
        )
        st.metric("Pregnancy 2h-OGTT linked value", f"{float(st.session_state.post_view_antenatal_2h_ogtt):.1f}")

    if st.button("Run postnatal future T2DM prediction", type="primary", use_container_width=True):
        st.session_state.post_prob = predict_postnatal_t2dm_after_gdm()

    if st.session_state.post_prob is not None:
        post_prob = float(st.session_state.post_prob)
        post_band = published_model_band(post_prob, 0.086)
        post_label = classification_text(post_prob, 0.086)
        render_result_cards(
            post_prob,
            post_band,
            "Future T2DM risk after delivery - postnatal model",
            0.086,
            post_label,
            "Published postnatal logistic model using postpartum glucose values and BMI.",
        )
        st.markdown("#### Suggested next action")
        render_actions(next_best_action("postnatal_after_gdm", post_prob, 0.086))
        st.info("Paper-reported action threshold used in this demo: 0.086.")

with tab_report:
    st.markdown(
        """
        <div class="panel">
            <div class="section-kicker">Report output</div>
            <h3>PDF-style patient report page</h3>
            <p class="muted">This page is styled like a one-page report for screenshots, slide decks, posters, and prototype demonstrations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    summary_df = summary_dataframe()
    html_report = report_html(summary_df)
    st.markdown(html_report, unsafe_allow_html=True)

    csv_bytes = summary_df.to_csv(index=False).encode("utf-8") if not summary_df.empty else b"Module,Probability\n"
    st.download_button(
        "Download summary CSV",
        data=csv_bytes,
        file_name="maternal_diabetes_risk_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download HTML report",
        data=html_report,
        file_name="maternal_diabetes_report.html",
        mime="text/html",
    )

with tab_about:
    top_left, top_right = st.columns([1.1, 0.9])
    with top_left:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">About this demo</div>
                <h3>What this final clean version includes</h3>
                <p class="muted">This version is designed for a public research demo: custom logo area, institution footer, QR-enabled sharing, cleaner result cards, a publication block, and Streamlit Cloud deployment files.</p>
                <div class="callout"><strong>Public message:</strong> this platform supports prevention across the maternal metabolic pathway, rather than focusing on one isolated prediction point.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="deploy-grid">
                <div class="panel">
                    <div class="section-kicker">GitHub checklist</div>
                    <h4>Repo contents</h4>
                    <ul>
                        <li><code>app.py</code></li>
                        <li><code>requirements.txt</code></li>
                        <li><code>.streamlit/config.toml</code></li>
                        <li><code>README.md</code></li>
                        <li>Saved model and scaler</li>
                        <li><code>assets/custom_logo.png</code> (optional)</li>
                    </ul>
                </div>
                <div class="panel">
                    <div class="section-kicker">Deployment</div>
                    <h4>Streamlit Community Cloud</h4>
                    <ol>
                        <li>Push the project to GitHub.</li>
                        <li>Choose the repo and <code>app.py</code> in Streamlit Community Cloud.</li>
                        <li>Add model files to the repo if the app is for demo use.</li>
                        <li>Deploy and add the public app URL back into the sidebar to generate the QR code.</li>
                    </ol>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with top_right:
        st.markdown(
            """
            <div class="panel">
                <div class="section-kicker">Publication block</div>
                <h4>Suggested public-sharing text</h4>
                <p class="muted">Add your final paper title, DOI, collaborators, ethics note, and contact details here before public deployment.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(
            f"Institution: {st.session_state.institution_name}\n"
            f"Unit: {st.session_state.institution_unit}\n"
            f"Contact: {st.session_state.contact_email}\n"
            f"App URL: {st.session_state.public_app_url}\n"
            f"GitHub: {st.session_state.github_url}\n"
            f"Publication: {st.session_state.publication_url}\n",
            language="text",
        )
        qr_bytes = qr_image(st.session_state.public_app_url)
        if qr_bytes is not None:
            st.image(qr_bytes, width=180, caption="QR for public app")

    st.markdown("#### Suggested footer text")
    st.code(
        "Research demonstration only. This app is intended for model presentation and decision-support prototyping. "
        "It does not replace local screening policy, clinician judgment, or governance requirements.",
        language="text",
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    f"""
    <div class="footer-wrap">
        <div class="footer-top">
            <div>
                <span class="footer-pill">{escape(st.session_state.institution_name)}</span>
                <span class="footer-pill">{escape(st.session_state.institution_unit)}</span>
                <span class="footer-pill">{escape(st.session_state.model_version)}</span>
            </div>
            <div>{safe_link(st.session_state.github_url, 'GitHub')} · {safe_link(st.session_state.publication_url, 'Publication')} · {safe_link(st.session_state.public_app_url, 'Public app')}</div>
        </div>
        <div><strong>M-Guide | Maternal Diabetes Prevention Platform</strong> — Monash-style prototype for research demonstration and public sharing. Replace placeholder branding with approved institutional assets before external release.</div>
        <div>Contact: {escape(st.session_state.contact_email)} | {escape(st.session_state.report_note)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)
