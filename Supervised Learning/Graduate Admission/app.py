import streamlit as st
import pickle
import numpy as np
import os
import warnings

st.set_page_config(
    page_title="Graduate Admission Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&display=swap');
* { font-family: 'Outfit', sans-serif !important; }
.stApp { background: #0a0a0f; }
section[data-testid="stSidebar"] { display: none; }
#MainMenu, footer, header { visibility: hidden; }
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 99px; }
.bg-orb { position: fixed; border-radius: 50%; filter: blur(80px); pointer-events: none; z-index: 0; animation: float 8s ease-in-out infinite; }
.orb1 { width:500px; height:500px; background:rgba(99,102,241,0.12); top:-100px; left:-100px; animation-delay:0s; }
.orb2 { width:400px; height:400px; background:rgba(139,92,246,0.10); top:30%; right:-80px; animation-delay:2s; }
.orb3 { width:350px; height:350px; background:rgba(52,211,153,0.07); bottom:-80px; left:30%; animation-delay:4s; }
@keyframes float { 0%,100% { transform: translateY(0px) scale(1); } 50% { transform: translateY(-30px) scale(1.05); } }
.hero-wrap { text-align: center; padding: 3rem 1rem 1.5rem; position: relative; z-index:1; }
.hero-tag { display: inline-block; background: rgba(99,102,241,0.15); border: 1px solid rgba(99,102,241,0.4); color: #a5b4fc; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.15em; text-transform: uppercase; padding: 0.35rem 1.1rem; border-radius: 99px; margin-bottom: 1.2rem; }
.hero-title { font-size: clamp(2.2rem, 5vw, 3.8rem); font-weight: 900; line-height: 1.1; background: linear-gradient(135deg, #ffffff 0%, #a78bfa 40%, #60a5fa 70%, #34d399 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.8rem; }
.hero-sub { color: #64748b; font-size: 1.05rem; font-weight: 400; max-width: 560px; margin: 0 auto 2rem; line-height: 1.6; }
.model-strip { background: linear-gradient(90deg, rgba(99,102,241,0.08) 0%, rgba(139,92,246,0.08) 100%); border: 1px solid rgba(99,102,241,0.2); border-radius: 14px; padding: 1rem 1.5rem; display: flex; align-items: center; flex-wrap: wrap; gap: 0.8rem; margin-bottom: 2rem; position: relative; z-index:1; }
.ms-item { display: flex; align-items: center; gap: 0.5rem; color: #94a3b8; font-size: 0.82rem; }
.ms-dot { width:6px; height:6px; border-radius:50%; background:#6366f1; flex-shrink:0; }
.ms-val { color: #e2e8f0; font-weight: 600; }
.ms-acc { margin-left: auto; background: linear-gradient(135deg, #059669, #10b981); color: white; font-weight: 700; font-size: 0.85rem; padding: 0.4rem 1rem; border-radius: 99px; }
.sec-head { display: flex; align-items: center; gap: 0.6rem; font-size: 0.7rem; font-weight: 700; color: #6366f1; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 1rem; position: relative; z-index:1; }
.sec-head::after { content: ''; flex: 1; height: 1px; background: linear-gradient(90deg, rgba(99,102,241,0.3), transparent); }
.input-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 18px; padding: 1.5rem; margin-bottom: 1rem; position: relative; z-index:1; transition: border-color 0.2s; }
.input-card:hover { border-color: rgba(99,102,241,0.3); }
.input-card-title { font-size: 0.78rem; font-weight: 700; color: #a5b4fc; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1.2rem; display: flex; align-items: center; gap: 0.5rem; }
[data-testid="stWidgetLabel"] p { color: #94a3b8 !important; font-size: 0.85rem !important; font-weight: 500 !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background: #6366f1 !important; border-color: #6366f1 !important; }
.stRadio label span { color: #cbd5e1 !important; }
.stButton > button { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%) !important; color: white !important; font-weight: 700 !important; font-size: 1.05rem !important; border: none !important; border-radius: 14px !important; padding: 0.85rem 0 !important; width: 100% !important; letter-spacing: 0.03em !important; box-shadow: 0 4px 30px rgba(99,102,241,0.35) !important; transition: all 0.25s ease !important; position: relative; z-index:1; }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 40px rgba(99,102,241,0.55) !important; }
.result-panel-admitted { background: linear-gradient(135deg, rgba(6,78,59,0.6), rgba(6,95,70,0.4)); border: 1px solid rgba(16,185,129,0.4); border-radius: 24px; padding: 2.5rem 2rem; text-align: center; margin-bottom: 1.5rem; position: relative; overflow: hidden; }
.result-panel-rejected { background: linear-gradient(135deg, rgba(69,10,10,0.6), rgba(127,29,29,0.4)); border: 1px solid rgba(239,68,68,0.4); border-radius: 24px; padding: 2.5rem 2rem; text-align: center; margin-bottom: 1.5rem; position: relative; overflow: hidden; }
.result-emoji { font-size: 3.5rem; margin-bottom: 0.6rem; }
.result-verdict { font-size: 2.2rem; font-weight: 900; letter-spacing: -0.02em; margin-bottom: 0.4rem; }
.admitted-color { color: #34d399; }
.rejected-color { color: #f87171; }
.result-desc { color: #94a3b8; font-size: 0.9rem; margin-bottom: 1rem; }
.result-prob-badge { display: inline-block; font-size: 2rem; font-weight: 900; padding: 0.5rem 1.8rem; border-radius: 14px; margin-top: 0.5rem; }
.admitted-badge { background: rgba(16,185,129,0.15); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
.rejected-badge { background: rgba(239,68,68,0.15); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.gauge-wrap { margin: 0.8rem 0; }
.gauge-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.gauge-label { color: #94a3b8; font-size: 0.82rem; font-weight: 500; }
.gauge-pct { font-size: 0.95rem; font-weight: 700; }
.gauge-track { height: 10px; border-radius: 99px; background: rgba(255,255,255,0.06); overflow: hidden; }
.gauge-fill { height: 100%; border-radius: 99px; }
.fill-green { background: linear-gradient(90deg, #059669, #34d399, #6ee7b7); }
.fill-red   { background: linear-gradient(90deg, #b91c1c, #ef4444, #fca5a5); }
.stats-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 0.8rem; margin: 1.2rem 0; }
.stat-card { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1rem; text-align: center; }
.stat-card-val { font-size: 1.5rem; font-weight: 800; color: #e2e8f0; }
.stat-card-lbl { font-size: 0.68rem; color: #475569; text-transform: uppercase; letter-spacing: 0.07em; margin-top: 0.2rem; }
.profile-score { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 16px; padding: 1.2rem 1.4rem; margin: 1rem 0; }
.profile-score-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem; }
.profile-score-title { color: #94a3b8; font-size: 0.82rem; font-weight: 600; }
.pscore-track { height: 8px; background: rgba(255,255,255,0.06); border-radius:99px; overflow:hidden; }
.pscore-fill { height:100%; border-radius:99px; }
.tips-card { background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.2); border-radius: 16px; padding: 1.2rem 1.4rem; margin-top: 1rem; }
.tip-item { display: flex; gap: 0.7rem; align-items: flex-start; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.04); color: #c7d2fe; font-size: 0.84rem; line-height: 1.5; }
.tip-item:last-child { border-bottom: none; }
.tip-icon { font-size: 1rem; flex-shrink: 0; margin-top: 0.05rem; }
hr { border-color: rgba(255,255,255,0.05) !important; }
.footer { text-align: center; color: #1e293b; font-size: 0.75rem; padding: 2rem 0 1rem; position: relative; z-index:1; }
</style>
<div class="bg-orb orb1"></div>
<div class="bg-orb orb2"></div>
<div class="bg-orb orb3"></div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "admission_model.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = load_model()

clf          = model.named_steps["classifier"]
model_name   = type(clf).__name__
model_params = clf.get_params()
MODEL_ACC    = 92.5

st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag">&#10022; AI-Powered Admission Analysis</div>
    <div class="hero-title">Graduate Admission<br/>Predictor</div>
    <div class="hero-sub">Enter your academic profile and instantly discover your admission probability &mdash; powered by Machine Learning</div>
</div>
""", unsafe_allow_html=True)

C      = model_params.get('C', '?')
solver = model_params.get('solver', 'lbfgs')
penalty= model_params.get('penalty', 'l2')
st.markdown(f"""
<div class="model-strip">
    <div class="ms-item"><div class="ms-dot"></div>Algorithm <span class="ms-val">&nbsp;{model_name}</span></div>
    <div class="ms-item"><div class="ms-dot"></div>Regularisation <span class="ms-val">&nbsp;C={C} &middot; {penalty}</span></div>
    <div class="ms-item"><div class="ms-dot"></div>Solver <span class="ms-val">&nbsp;{solver}</span></div>
    <div class="ms-item"><div class="ms-dot"></div>Selection <span class="ms-val">&nbsp;GridSearchCV &middot; 5-Fold &middot; 4 Algorithms</span></div>
    <div class="ms-acc">&#10003; {MODEL_ACC}% Accuracy</div>
</div>
""", unsafe_allow_html=True)

left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="sec-head">&#10022; &nbsp;Academic Profile</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card"><div class="input-card-title">&#128202; Standardised Test Scores</div>', unsafe_allow_html=True)
    gre   = st.slider("GRE Score", 260, 340, 310, help="Graduate Record Examination (260-340)")
    toefl = st.slider("TOEFL Score", 92, 120, 107, help="Test of English as a Foreign Language (92-120)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card"><div class="input-card-title">&#127891; Academic Record</div>', unsafe_allow_html=True)
    cgpa = st.slider("CGPA (out of 10)", 6.0, 10.0, 8.5, step=0.1)
    university_rating = st.select_slider("University Rating", options=[1,2,3,4,5], value=3,
                                          help="Prestige of your undergraduate institution (1=low, 5=high)")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-card"><div class="input-card-title">&#128221; Application Strength</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        sop = st.select_slider("SOP Strength", options=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], value=3.5)
    with c2:
        lor = st.select_slider("LOR Strength", options=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5,5.0], value=3.5)
    research = st.radio("Research Experience", [0,1],
                        format_func=lambda x: "Yes — I have research experience" if x==1 else "No research experience")
    serial_no = st.number_input("Serial No.", min_value=1, max_value=500, value=1, label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    profile_score = int(
        ((gre-260)/80*30) + ((toefl-92)/28*20) + ((cgpa-6)/4*25) +
        (university_rating/5*10) + (sop/5*5) + (lor/5*5) + (research*5)
    )
    profile_score = min(profile_score, 100)
    score_color = "#34d399" if profile_score >= 70 else "#f59e0b" if profile_score >= 50 else "#f87171"
    st.markdown(f"""
    <div class="profile-score">
        <div class="profile-score-header">
            <span class="profile-score-title">&#9889; Live Profile Strength</span>
            <span style="font-size:1.3rem; font-weight:800; color:{score_color}">{profile_score}/100</span>
        </div>
        <div class="pscore-track">
            <div class="pscore-fill" style="width:{profile_score}%; background:linear-gradient(90deg,#6366f1,{score_color});"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    predict_btn = st.button("&#128269;  Analyse My Admission Profile", use_container_width=True)

with right:
    st.markdown('<div class="sec-head">&#10022; &nbsp;Prediction Results</div>', unsafe_allow_html=True)

    if predict_btn:
        features = np.array([[serial_no, gre, toefl, university_rating, sop, lor, cgpa, research]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            prediction = model.predict(features)[0]
            proba      = model.predict_proba(features)[0]

        admit_prob  = proba[1]
        reject_prob = proba[0]
        pct_admit   = round(admit_prob * 100, 1)
        pct_reject  = round(reject_prob * 100, 1)
        conf_level  = "High" if max(admit_prob,reject_prob)>0.78 else "Moderate" if max(admit_prob,reject_prob)>0.62 else "Low"

        if prediction == 1:
            st.markdown(f"""
            <div class="result-panel-admitted">
                <div class="result-emoji">&#127881;</div>
                <div class="result-verdict admitted-color">Likely Admitted!</div>
                <div class="result-desc">Your profile is competitive for graduate admission</div>
                <div class="result-prob-badge admitted-badge">{pct_admit}% Chance</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-panel-rejected">
                <div class="result-emoji">&#128203;</div>
                <div class="result-verdict rejected-color">Needs Improvement</div>
                <div class="result-desc">Strengthen your profile for better admission odds</div>
                <div class="result-prob-badge rejected-badge">{pct_admit}% Chance</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="gauge-wrap">
            <div class="gauge-header">
                <span class="gauge-label">&#128994; Admission Probability</span>
                <span class="gauge-pct" style="color:#34d399">{pct_admit}%</span>
            </div>
            <div class="gauge-track"><div class="gauge-fill fill-green" style="width:{pct_admit}%"></div></div>
        </div>
        <div class="gauge-wrap">
            <div class="gauge-header">
                <span class="gauge-label">&#128308; Rejection Probability</span>
                <span class="gauge-pct" style="color:#f87171">{pct_reject}%</span>
            </div>
            <div class="gauge-track"><div class="gauge-fill fill-red" style="width:{pct_reject}%"></div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-card-val" style="color:{'#34d399' if prediction==1 else '#f87171'}">{pct_admit}%</div>
                <div class="stat-card-lbl">Admit Probability</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-val" style="color:#a78bfa">{conf_level}</div>
                <div class="stat-card-lbl">Confidence Level</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-val" style="color:#60a5fa">{MODEL_ACC}%</div>
                <div class="stat-card-lbl">Model Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-card-val" style="color:#f59e0b">{profile_score}</div>
                <div class="stat-card-lbl">Profile Score</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown('<div class="sec-head" style="margin-top:1.2rem">&#10022; &nbsp;Profile Breakdown</div>', unsafe_allow_html=True)
        for label, val, mn, mx, color in [
            ("GRE Score", gre, 260, 340, "#6366f1"),
            ("TOEFL Score", toefl, 92, 120, "#8b5cf6"),
            ("CGPA", cgpa, 6.0, 10.0, "#a78bfa"),
            ("SOP", sop, 1.0, 5.0, "#60a5fa"),
            ("LOR", lor, 1.0, 5.0, "#34d399"),
        ]:
            pct = int((val-mn)/(mx-mn)*100)
            st.markdown(f"""
            <div class="gauge-wrap" style="margin:0.5rem 0">
                <div class="gauge-header">
                    <span class="gauge-label">{label}</span>
                    <span class="gauge-pct" style="color:{color}">{val}</span>
                </div>
                <div class="gauge-track">
                    <div class="gauge-fill" style="width:{pct}%; background:linear-gradient(90deg,{color}88,{color});"></div>
                </div>
            </div>""", unsafe_allow_html=True)

        tips = []
        if gre < 310:   tips.append(("&#128218;", f"GRE score ({gre}) is below the strong range (310+). Consider retaking."))
        if toefl < 105: tips.append(("&#128172;", "TOEFL above 105 significantly improves your odds."))
        if cgpa < 8.5:  tips.append(("&#128214;", "A CGPA of 8.5+ is preferred by most competitive programs."))
        if research==0: tips.append(("&#128300;", "Research experience is a major differentiator — pursue it."))
        if sop < 3.5:   tips.append(("&#9998;", "A stronger SOP (3.5+) can be the deciding factor."))
        if lor < 3.5:   tips.append(("&#128140;", "Seek stronger LORs (3.5+) from professors who know you well."))

        if tips:
            items = "".join(f'<div class="tip-item"><span class="tip-icon">{ic}</span><span>{t}</span></div>' for ic,t in tips)
            st.markdown(f"""
            <div class="tips-card">
                <div style="color:#a5b4fc;font-size:0.8rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.8rem">
                    &#128161; Profile Improvement Tips
                </div>{items}
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tips-card" style="border-color:rgba(52,211,153,0.25);background:rgba(52,211,153,0.05);">
                <div style="color:#34d399;font-size:0.8rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.5rem">
                    &#127775; Excellent Profile
                </div>
                <div style="color:#94a3b8;font-size:0.85rem;line-height:1.6">
                    Your stats are well above average. Focus on tailoring your SOP to each university's research areas and securing strong personalised recommendation letters.
                </div>
            </div>""", unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:rgba(255,255,255,0.02);border:1px dashed rgba(99,102,241,0.25);border-radius:24px;padding:4rem 2rem;text-align:center;margin-bottom:1rem;">
            <div style="font-size:3.5rem;margin-bottom:1rem">&#127891;</div>
            <div style="color:#475569;font-size:1rem;font-weight:500;margin-bottom:0.4rem">Awaiting your profile</div>
            <div style="color:#334155;font-size:0.85rem">Fill in your academic details on the left<br/>and click Analyse to see your results</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.8rem;margin-top:1rem;">
            <div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);border-radius:14px;padding:1.2rem;text-align:center;">
                <div style="font-size:1.8rem;font-weight:800;color:#6366f1;">4</div>
                <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-top:0.2rem;">Algorithms Tested</div>
            </div>
            <div style="background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.15);border-radius:14px;padding:1.2rem;text-align:center;">
                <div style="font-size:1.8rem;font-weight:800;color:#8b5cf6;">400</div>
                <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-top:0.2rem;">Training Records</div>
            </div>
            <div style="background:rgba(52,211,153,0.06);border:1px solid rgba(52,211,153,0.15);border-radius:14px;padding:1.2rem;text-align:center;">
                <div style="font-size:1.8rem;font-weight:800;color:#34d399;">92.5%</div>
                <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-top:0.2rem;">Test Accuracy</div>
            </div>
            <div style="background:rgba(96,165,250,0.06);border:1px solid rgba(96,165,250,0.15);border-radius:14px;padding:1.2rem;text-align:center;">
                <div style="font-size:1.8rem;font-weight:800;color:#60a5fa;">5-Fold</div>
                <div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:0.07em;margin-top:0.2rem;">Cross Validation</div>
            </div>
        </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Logistic Regression &middot; GridSearchCV &middot; Kaggle Graduate Admissions Dataset &middot; Threshold &ge; 0.75
</div>""", unsafe_allow_html=True)
