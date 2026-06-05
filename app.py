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


# =========================================================
# STATE
# =========================================================
def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


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
    logo_html = ""
    if logo_path is not None:
        logo_html = "<div></div>"
    st.markdown(
        f"""
        <div class="app-header">
            <div>
                <div class="brand-kicker">{escape(st.session_state.model_version)} | Research demo</div>
                <h1 class="brand-title">M-Guide Maternal Diabetes Prevention Platform</h1>
                <div class="brand-subtitle">{escape(st.session_state.app_tagline)} across booking, pregnancy after GDM, and postnatal review.</div>
            </div>
            {logo_html}
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
            <span class="chip">Active: {escape(str(st.session_state.active_module))}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="notice"><strong>Research demo only.</strong> This platform is not a stand-alone diagnostic or clinical deployment tool without governance, local validation, security review, and approval.</div>',
        unsafe_allow_html=True,
    )


def render_module_intro(title: str, text: str) -> None:
    st.markdown(
        f"""
        <div class="panel">
            <div class="panel-title">{escape(title)}</div>
            <div class="panel-text">{escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result(prob: float, band: str, threshold: float, title: str, subtitle: str, payload: dict) -> None:
    pct = max(0.0, min(100.0, prob * 100.0))
    actions = "".join(f"<li>{escape(item)}</li>" for item in payload["actions"])
    reasons = "".join(f"<li>{escape(item)}</li>" for item in payload["reasons"])
    note = ""
    if payload.get("tailored_note"):
        note = f'<p class="small-muted"><strong>Tailored support:</strong> {escape(payload["tailored_note"])}</p>'

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-kicker">{escape(title)}</div>
            <div class="result-value">{pct:.1f}%</div>
            <div>{risk_pill(band)} <span class="small-muted"> {escape(action_label(prob, threshold))}</span></div>
            <div class="meter"><div class="meter-fill" style="width:{pct:.1f}%"></div></div>
            <div class="small-muted">{escape(subtitle)} Threshold: {threshold:.3f}.</div>
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


def render_empty_result() -> None:
    st.markdown(
        """
        <div class="result-card">
            <div class="result-kicker">Result</div>
            <div class="result-value" style="font-size:1.45rem;">Ready when you are</div>
            <div class="small-muted">Enter the predictors and calculate risk. Results stay on screen while you move between modules.</div>
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
        )
    else:
        render_empty_result()


# =========================================================
# MODULES
# =========================================================
def render_booking_module() -> None:
    left, right = st.columns([1.15, 0.85])
    with left:
        render_module_intro(
            "Booking visit: GDM risk",
            "Estimate the probability of developing GDM using routine booking predictors and the saved CatBoost model.",
        )
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
        render_module_intro(
            "Pregnancy after GDM: future T2DM risk",
            "Use the published seven-predictor antenatal equation to support postpartum prevention planning.",
        )
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
        render_module_intro(
            "Postnatal review: updated future T2DM risk",
            "Update long-term risk using linked antenatal OGTT plus postpartum glucose values and BMI.",
        )
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

selected_module = st.radio(
    "Choose pathway stage",
    MODULES,
    horizontal=True,
    key="active_module",
    label_visibility="collapsed",
)

if selected_module == "Booking visit":
    render_booking_module()
elif selected_module == "Pregnancy after GDM":
    render_antenatal_module()
else:
    render_postnatal_module()

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
