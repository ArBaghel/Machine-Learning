import os
import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #ede7f6 0%, #fce4ec 60%, #e8f5e9 100%);
    }

    .stSlider label, .stSlider label p,
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"],
    .stMarkdown p, .stMarkdown,
    label, p, span {
        color: #1a1a2e !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #555 !important;
        font-size: 0.78rem !important;
    }
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #7b2ff7 !important;
        border-color: #7b2ff7 !important;
    }
    .main-title {
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6a0dad, #c2185b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        padding-top: 10px;
    }
    .sub-title {
        text-align: center;
        color: #5a4a6a !important;
        font-size: 1rem;
        font-weight: 500 !important;
        margin-bottom: 2rem;
        opacity: 1 !important;
    }
    .card {
        background: #ffffff;
        border-radius: 18px;
        padding: 22px 26px;
        box-shadow: 0 4px 24px rgba(100,0,200,0.10);
        margin-bottom: 18px;
    }
    .card-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #2d0060 !important;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 2px solid #f3e8ff;
        padding-bottom: 10px;
    }
    .result-box {
        background: linear-gradient(135deg, #6a0dad, #c2185b);
        border-radius: 20px;
        padding: 32px 20px;
        text-align: center;
        color: #ffffff !important;
        box-shadow: 0 8px 32px rgba(106,13,173,0.30);
        margin-bottom: 18px;
    }
    .result-emoji { font-size: 3.5rem; margin-bottom: 8px; }
    .result-species { font-size: 2rem; font-weight: 800; color: #ffffff !important; letter-spacing: 1px; }
    .result-desc { font-size: 0.9rem; color: rgba(255,255,255,0.88) !important; margin-top: 6px; font-weight: 400 !important; }
    .result-conf {
        margin-top: 16px; font-size: 1.4rem; font-weight: 800;
        color: #ffffff !important; background: rgba(255,255,255,0.18);
        border-radius: 12px; padding: 10px 20px; display: inline-block;
    }
    .conf-row { margin: 10px 0; }
    .conf-header { display: flex; justify-content: space-between; margin-bottom: 5px; }
    .conf-name { font-size: 0.9rem; font-weight: 700; color: #2d0060 !important; }
    .conf-pct  { font-size: 0.9rem; font-weight: 700; }
    .bar-bg { background: #ede7f6; border-radius: 99px; height: 13px; overflow: hidden; }
    .bar-fill { height: 13px; border-radius: 99px; }
    .metric-card {
        background: #f9f4ff; border-radius: 14px;
        padding: 16px 12px; text-align: center; border: 1.5px solid #e9d5ff;
    }
    .metric-val { font-size: 1.7rem; font-weight: 800; color: #6a0dad !important; }
    .metric-lbl {
        font-size: 0.72rem; color: #7a6a9a !important;
        text-transform: uppercase; letter-spacing: 1px;
        margin-top: 4px; font-weight: 600 !important;
    }
    .badge {
        display: inline-block; background: #f3e8ff; color: #5b21b6 !important;
        border-radius: 8px; padding: 5px 14px; margin: 4px 3px;
        font-size: 0.82rem; font-weight: 700 !important; border: 1.5px solid #c4b5fd;
    }
    div.stButton > button {
        background: linear-gradient(90deg, #6a0dad, #c2185b) !important;
        color: #ffffff !important; border: none !important;
        border-radius: 14px !important; padding: 14px 0 !important;
        font-size: 1.05rem !important; font-weight: 700 !important;
        width: 100% !important; box-shadow: 0 4px 18px rgba(106,13,173,0.35) !important;
        letter-spacing: 0.5px !important; margin-top: 6px !important;
    }
    div.stButton > button:hover { opacity: 0.88 !important; }
    .icon-label {
        display: flex; align-items: center; gap: 10px;
        font-size: 0.95rem; font-weight: 700;
        color: #2d0060 !important; margin-bottom: 6px;
    }
    .icon-label svg { flex-shrink: 0; }
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── SVG Icons ──────────────────────────────────────────────────────────────────
def icon(name):
    icons = {
        "ruler": '<svg width="18" height="18" fill="none" stroke="#6a0dad" stroke-width="2" viewBox="0 0 24 24"><path d="M3 12l18-9-9 18-2-7-7-2z"/></svg>',
        "cpu":   '<svg width="18" height="18" fill="none" stroke="#6a0dad" stroke-width="2" viewBox="0 0 24 24"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "bar":   '<svg width="18" height="18" fill="none" stroke="#6a0dad" stroke-width="2" viewBox="0 0 24 24"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/><line x1="2" y1="20" x2="22" y2="20"/></svg>',
        "list":  '<svg width="18" height="18" fill="none" stroke="#6a0dad" stroke-width="2" viewBox="0 0 24 24"><line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><circle cx="3" cy="6" r="1" fill="#6a0dad"/><circle cx="3" cy="12" r="1" fill="#6a0dad"/><circle cx="3" cy="18" r="1" fill="#6a0dad"/></svg>',
        "sepal": '<svg width="18" height="18" fill="none" stroke="#388e3c" stroke-width="2" viewBox="0 0 24 24"><path d="M12 2C6 2 2 8 2 14c0 4 2 7 5 8l5-10 5 10c3-1 5-4 5-8 0-6-4-12-10-12z"/></svg>',
        "petal": '<svg width="18" height="18" fill="none" stroke="#c2185b" stroke-width="2" viewBox="0 0 24 24"><ellipse cx="12" cy="12" rx="4" ry="9" transform="rotate(-30 12 12)"/><ellipse cx="12" cy="12" rx="4" ry="9" transform="rotate(30 12 12)"/><circle cx="12" cy="12" r="2" fill="#c2185b"/></svg>',
    }
    return icons.get(name, "")


# ── Load Model ─────────────────────────────────────────────────────────────────
# os.path.dirname(__file__) ensures the pickle is found relative to app.py
# on both local machine AND Streamlit Cloud
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'iris_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# REMOVE this entire function
@st.cache_data
def get_accuracy():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    iris = load_iris()
    X, y = iris.data, iris.target
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.33)
    mdl = load_model()
    return round(accuracy_score(y_test, mdl.predict(X_test)) * 100, 2)

accuracy = get_accuracy()

model     = load_model()
clf       = model.named_steps['classifier']
algo_name = type(clf).__name__
params    = clf.get_params()
accuracy  = get_accuracy()

CLASS_NAMES  = ["Setosa",  "Versicolor", "Virginica"]
CLASS_ICONS  = ["&#9734;", "&#9670;",    "&#9654;"]
CLASS_COLORS = ["#6a0dad", "#c2185b",    "#0288d1"]
CLASS_DESC   = [
    "Small, compact — narrow petals & short sepals",
    "Medium-sized — purple-veined, oval petals",
    "Largest species — broad, overlapping petals"
]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">&#127800; Iris Flower Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Machine Learning classification using Support Vector Machine (SVM)</div>', unsafe_allow_html=True)

left, right = st.columns([1, 1], gap="large")

# ╔══ LEFT ══════════════════════════════════════════════════════════════════════╗
with left:
    st.markdown(f'<div class="card"><div class="card-title">{icon("ruler")} Enter Flower Measurements</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="icon-label">{icon("sepal")} Sepal Length (cm)</div>', unsafe_allow_html=True)
    sepal_length = st.slider("sepal_length", 4.0, 8.0, 5.8, 0.1, label_visibility="collapsed")

    st.markdown(f'<div class="icon-label">{icon("sepal")} Sepal Width (cm)</div>', unsafe_allow_html=True)
    sepal_width = st.slider("sepal_width", 2.0, 4.5, 3.0, 0.1, label_visibility="collapsed")

    st.markdown(f'<div class="icon-label">{icon("petal")} Petal Length (cm)</div>', unsafe_allow_html=True)
    petal_length = st.slider("petal_length", 1.0, 7.0, 4.0, 0.1, label_visibility="collapsed")

    st.markdown(f'<div class="icon-label">{icon("petal")} Petal Width (cm)</div>', unsafe_allow_html=True)
    petal_width = st.slider("petal_width", 0.1, 2.5, 1.2, 0.1, label_visibility="collapsed")

    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("&#128269;  Predict Species")

    st.markdown(f'<div class="card" style="margin-top:18px"><div class="card-title">{icon("cpu")} Model Information</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-val">{accuracy}%</div><div class="metric-lbl">Test Accuracy</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.2rem;">{algo_name}</div><div class="metric-lbl">Algorithm</div></div>', unsafe_allow_html=True)

    st.markdown('<p style="color:#2d0060 !important; font-weight:700 !important; margin:14px 0 6px 0;">&#9881; Best Parameters</p>', unsafe_allow_html=True)
    show_keys = ('C', 'kernel', 'gamma', 'degree', 'max_iter')
    badges = "".join(f'<span class="badge">{k} = {v}</span>' for k, v in params.items() if k in show_keys)
    st.markdown(badges, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ╔══ RIGHT ═════════════════════════════════════════════════════════════════════╗
with right:
    inp      = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    pred     = model.predict(inp)[0]
    dec      = clf.decision_function(inp)[0]
    exp_d    = np.exp(dec - np.max(dec))
    confs    = exp_d / exp_d.sum()
    conf_pct = round(confs[pred] * 100, 1)

    if predict_clicked:
        st.markdown(f'''
        <div class="result-box">
            <div class="result-emoji">&#127800;</div>
            <div class="result-species">Iris {CLASS_NAMES[pred]}</div>
            <div class="result-desc">{CLASS_DESC[pred]}</div>
            <div class="result-conf">&#9989; Confidence: {conf_pct}%</div>
        </div>''', unsafe_allow_html=True)
    else:
        st.markdown('''
        <div class="result-box" style="background:linear-gradient(135deg,#b0bec5,#78909c);box-shadow:none;">
            <div class="result-emoji">&#128269;</div>
            <div class="result-species" style="font-size:1.5rem;">Awaiting Prediction</div>
            <div class="result-desc">Adjust the sliders and click Predict Species</div>
        </div>''', unsafe_allow_html=True)

    st.markdown(f'<div class="card"><div class="card-title">{icon("bar")} Confidence Breakdown</div>', unsafe_allow_html=True)
    for i, (name, conf, color) in enumerate(zip(CLASS_NAMES, confs, CLASS_COLORS)):
        pct     = round(conf * 100, 1)
        is_pred = (i == pred)
        tick    = " &#10003;" if is_pred else ""
        bold    = "font-weight:800;" if is_pred else "font-weight:600;"
        st.markdown(f'''
        <div class="conf-row">
            <div class="conf-header">
                <span class="conf-name" style="color:{color} !important; {bold}">{CLASS_ICONS[i]} Iris {name}{tick}</span>
                <span class="conf-pct" style="color:{color} !important;">{pct}%</span>
            </div>
            <div class="bar-bg">
                <div class="bar-fill" style="width:{pct}%; background:{color};"></div>
            </div>
        </div>''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="card"><div class="card-title">{icon("list")} Input Summary</div>', unsafe_allow_html=True)
    labels = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"]
    values = [sepal_length, sepal_width, petal_length, petal_width]
    cols   = st.columns(4)
    for col, lbl, val in zip(cols, labels, values):
        col.markdown(f'<div class="metric-card"><div class="metric-val" style="font-size:1.4rem;">{val}</div><div class="metric-lbl">{lbl} (cm)</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#7a6a9a; font-size:0.82rem; padding:10px 0 20px 0; font-weight:500;'>
    &#127800; Iris Flower Predictor &nbsp;|&nbsp; SVC · Linear Kernel &nbsp;|&nbsp; UCI Iris Dataset · 150 samples · 3 classes
</div>
""", unsafe_allow_html=True)
