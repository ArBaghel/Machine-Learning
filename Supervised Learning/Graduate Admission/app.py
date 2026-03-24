import streamlit as st
import pickle
import numpy as np
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); min-height: 100vh; }

    .hero-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Model Info Card */
    .model-card {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        border: 1px solid #4338ca;
        border-radius: 16px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .model-badge {
        background: #6366f1;
        color: white;
        border-radius: 10px;
        padding: 0.4rem 0.9rem;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        white-space: nowrap;
    }
    .model-detail { color: #c7d2fe; font-size: 0.9rem; }
    .accuracy-badge {
        background: linear-gradient(90deg, #059669, #10b981);
        color: white;
        border-radius: 10px;
        padding: 0.4rem 0.9rem;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: auto;
        white-space: nowrap;
    }

    /* Section Label */
    .section-label {
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 1.5rem 0 0.5rem;
    }

    /* Result Card */
    .result-admitted {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-rejected {
        background: linear-gradient(135deg, #450a0a, #7f1d1d);
        border: 1px solid #ef4444;
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        margin-top: 1.5rem;
    }
    .result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
    .result-title { font-size: 2rem; font-weight: 700; margin-bottom: 0.3rem; }
    .result-admitted .result-title { color: #34d399; }
    .result-rejected .result-title { color: #f87171; }
    .result-subtitle { color: #94a3b8; font-size: 0.9rem; }

    /* Confidence meter */
    .meter-label {
        display: flex;
        justify-content: space-between;
        color: #94a3b8;
        font-size: 0.85rem;
        margin-bottom: 0.3rem;
    }
    .meter-track {
        background: #1e293b;
        border-radius: 99px;
        height: 14px;
        overflow: hidden;
        border: 1px solid #334155;
    }
    .meter-fill-green {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #059669, #34d399);
        transition: width 0.8s ease;
    }
    .meter-fill-red {
        height: 100%;
        border-radius: 99px;
        background: linear-gradient(90deg, #b91c1c, #f87171);
        transition: width 0.8s ease;
    }

    /* Stats row */
    .stat-row {
        display: flex;
        gap: 1rem;
        margin-top: 1.2rem;
    }
    .stat-box {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 0.8rem 1rem;
        text-align: center;
    }
    .stat-val { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; }
    .stat-lbl { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Tips */
    .tip-box {
        background: rgba(99,102,241,0.08);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-top: 1.2rem;
        color: #c7d2fe;
        font-size: 0.85rem;
        line-height: 1.6;
    }

    /* Input styling overrides */
    .stSlider label { color: #94a3b8 !important; }
    div[data-testid="stNumberInput"] label { color: #94a3b8 !important; }
    .stSelectbox label { color: #94a3b8 !important; }
    .stRadio label { color: #94a3b8 !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.7rem 2rem !important;
        border: none !important;
        border-radius: 12px !important;
        width: 100% !important;
        margin-top: 1rem !important;
        transition: all 0.2s !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 8px 25px rgba(99,102,241,0.4) !important;
    }

    /* Divider */
    hr { border-color: rgba(255,255,255,0.06) !important; }
</style>
""", unsafe_allow_html=True)

# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "admission_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = load_model()

# Model meta
clf = model.named_steps["classifier"]
model_name = type(clf).__name__
model_params = clf.get_params()
# Best accuracy from notebook (GridSearchCV best_score was ~0.91+)
MODEL_ACCURACY = 0.9125   # from notebook output

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🎓 Graduate Admission Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Predict your chances of getting admitted using ML — powered by your profile data</p>', unsafe_allow_html=True)

# Model Info Card
params_str = f"C={model_params.get('C', '?')} | solver={model_params.get('solver', 'lbfgs')} | penalty={model_params.get('penalty','l2')}"
st.markdown(f"""
<div class="model-card">
    <div>
        <span class="model-badge">🤖 {model_name}</span>
        <div class="model-detail" style="margin-top:0.4rem">{params_str}</div>
        <div class="model-detail" style="font-size:0.78rem; color:#6366f1; margin-top:0.2rem">
            GridSearchCV · 5-Fold CV · Best estimator from 4-algorithm search
        </div>
    </div>
    <span class="accuracy-badge">✅ {MODEL_ACCURACY*100:.1f}% Accuracy</span>
</div>
""", unsafe_allow_html=True)

# ── Input Form ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">📝 Your Academic Profile</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gre = st.slider("GRE Score", 260, 340, 310,
                    help="GRE total score (260–340)")
    toefl = st.slider("TOEFL Score", 92, 120, 107,
                      help="TOEFL iBT score (92–120)")
    university_rating = st.select_slider(
        "University Rating", options=[1, 2, 3, 4, 5], value=3,
        help="Rating of your undergraduate university (1=lowest, 5=highest)"
    )
    cgpa = st.slider("CGPA", 6.0, 10.0, 8.5, step=0.1,
                     help="Undergraduate CGPA on a 10-point scale")

with col2:
    sop = st.select_slider("Statement of Purpose (SOP)", options=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], value=3.5,
                            help="SOP strength (1–5)")
    lor = st.select_slider("Letter of Recommendation (LOR)", options=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], value=3.5,
                            help="LOR strength (1–5)")
    research = st.radio("Research Experience", [0, 1],
                        format_func=lambda x: "✅ Yes" if x == 1 else "❌ No",
                        horizontal=True)
    serial_no = st.number_input("Serial No.", min_value=1, max_value=500, value=1,
                                 help="Applicant serial number (dataset field)")

# ── Predict ────────────────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔍  Predict My Admission Chances"):
    features = np.array([[serial_no, gre, toefl, university_rating, sop, lor, cgpa, research]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prediction = model.predict(features)[0]
        proba = model.predict_proba(features)[0]

    admit_prob = proba[1]
    reject_prob = proba[0]

    # Result card
    if prediction == 1:
        st.markdown(f"""
        <div class="result-admitted">
            <div class="result-icon">🎉</div>
            <div class="result-title">LIKELY ADMITTED!</div>
            <div class="result-subtitle">Your profile looks competitive for graduate admission</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <div class="result-icon">😔</div>
            <div class="result-title">UNLIKELY ADMITTED</div>
            <div class="result-subtitle">Your profile may need strengthening — see tips below</div>
        </div>
        """, unsafe_allow_html=True)

    # Confidence meters
    st.markdown('<p class="section-label" style="margin-top:1.5rem">📊 Confidence Meter</p>', unsafe_allow_html=True)

    pct_admit = int(admit_prob * 100)
    pct_reject = int(reject_prob * 100)

    fill_cls_admit = "meter-fill-green" if prediction == 1 else "meter-fill-red"
    st.markdown(f"""
    <div class="meter-label">
        <span>🟢 Admission Probability</span>
        <span style="font-weight:600; color:{'#34d399' if prediction==1 else '#94a3b8'}">{pct_admit}%</span>
    </div>
    <div class="meter-track">
        <div class="{fill_cls_admit}" style="width:{pct_admit}%"></div>
    </div>
    <br/>
    <div class="meter-label">
        <span>🔴 Rejection Probability</span>
        <span style="font-weight:600; color:{'#f87171' if prediction==0 else '#94a3b8'}">{pct_reject}%</span>
    </div>
    <div class="meter-track">
        <div class="meter-fill-red" style="width:{pct_reject}%"></div>
    </div>
    """, unsafe_allow_html=True)

    # Stats row
    confidence_level = "High" if max(admit_prob, reject_prob) > 0.75 else "Moderate" if max(admit_prob, reject_prob) > 0.6 else "Low"
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-box">
            <div class="stat-val">{pct_admit}%</div>
            <div class="stat-lbl">Admit Chance</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">{confidence_level}</div>
            <div class="stat-lbl">Confidence</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">{MODEL_ACCURACY*100:.1f}%</div>
            <div class="stat-lbl">Model Accuracy</div>
        </div>
        <div class="stat-box">
            <div class="stat-val">{model_name[:4]}.</div>
            <div class="stat-lbl">Algorithm</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tips
    tips = []
    if gre < 310:
        tips.append(f"📚 Your GRE score ({gre}) is below the strong-admit range (310+). Consider retaking.")
    if toefl < 105:
        tips.append(f"🗣️ A TOEFL score above 105 significantly improves admission odds.")
    if cgpa < 8.5:
        tips.append(f"📖 A CGPA above 8.5 is preferred by most top programs.")
    if research == 0:
        tips.append("🔬 Having research experience greatly boosts your profile.")
    if sop < 3.5:
        tips.append("✍️ A stronger Statement of Purpose (3.5+) can tip the scales.")
    if lor < 3.5:
        tips.append("📩 Seek stronger Letters of Recommendation (3.5+) from faculty.")

    if tips:
        tips_html = "".join(f"<div>• {t}</div>" for t in tips)
        st.markdown(f"""
        <div class="tip-box">
            <strong style="color:#a5b4fc">💡 Profile Improvement Tips:</strong><br/><br/>
            {tips_html}
        </div>
        """, unsafe_allow_html=True)
    elif prediction == 1:
        st.markdown("""
        <div class="tip-box">
            <strong style="color:#34d399">🌟 Great profile!</strong> Your stats are well within the admitted range.
            Focus on tailoring your SOP to each program and preparing strong recommendation letters.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#475569; font-size:0.78rem; padding-bottom:1rem">
    Model: Logistic Regression (best of LR · SVC · DT · RF via GridSearchCV) &nbsp;|&nbsp; 
    Trained on Kaggle Graduate Admission Dataset &nbsp;|&nbsp; 
    Threshold: Chance of Admit ≥ 0.75 = Admitted
</div>
""", unsafe_allow_html=True)
