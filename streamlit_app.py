"""
NephroAI — CKD Prediction Streamlit App
Run: streamlit run app.py
"""
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json, warnings
warnings.filterwarnings('ignore')

# ─── Page config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="NephroAI — CKD Prediction",
    page_icon="🫘",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background: #0a0f1e; color: #e8edf8; }
  .metric-card {
    background: #111827; border: 1px solid #1e2d4a;
    border-radius: 12px; padding: 20px; text-align: center;
  }
  .risk-high { color: #f43f5e; font-size: 2.5rem; font-weight: bold; }
  .risk-low  { color: #10b981; font-size: 2.5rem; font-weight: bold; }
  .feature-bar { height: 8px; border-radius: 4px; margin: 2px 0; }
  .sidebar .stSelectbox, .sidebar .stNumberInput { background: #111827; }
  h1, h2, h3 { color: #e8edf8 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Artifacts ───────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    try:
        with open('ckd_artifacts.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model artifacts not found. Please run `python ckd_pipeline.py` first.")
        return None

artifacts = load_artifacts()

model = artifacts["model"]
imputer = artifacts["imputer"]
scaler = artifacts["scaler"]
encoders = artifacts["encoders"]
rfe = artifacts["rfe"]
selected_feats = artifacts["selected_features"]
all_feats = artifacts["all_features"]

# ⭐ Defensive check
if not hasattr(imputer, "transform"):
    st.error("❌ Imputer not loaded correctly. Please rerun ckd_pipeline.py")
    st.stop()

# ─── Header ───────────────────────────────────────────────────────
st.markdown("# 🫘 NephroAI — Chronic Kidney Disease Prediction")
st.markdown("**Explainable AI for Early CKD Detection** | UCI CKD Dataset | SHAP-Style Explanations")
st.divider()

if artifacts is None:
    st.stop()

model          = artifacts['model']
imputer        = artifacts['imputer']
scaler         = artifacts['scaler']
encoders       = artifacts['encoders']
rfe            = artifacts['rfe']
selected_feats = artifacts['selected_features']
all_feats      = artifacts['all_features']
results_dict   = artifacts['model_results']
global_imps    = artifacts['global_importances']
feat_stats     = artifacts['feature_stats']
best_name      = artifacts['best_model_name']

# ─── Sidebar: Patient Input ────────────────────────────────────────
st.sidebar.markdown("## 🩺 Patient Clinical Parameters")
st.sidebar.markdown("*Leave blank to auto-impute from population mean*")

def si(label, min_v, max_v, step, default, help_txt, key):
    return st.sidebar.number_input(label, min_value=float(min_v), max_value=float(max_v),
                                    value=float(default), step=float(step), help=help_txt, key=key)

st.sidebar.markdown("#### 📊 Renal & Metabolic")
sc   = si("Serum Creatinine (mg/dL)", 0.4, 20.0, 0.1, 3.1, "Normal: 0.7–1.3", "sc")
bu   = si("Blood Urea (mg/dL)", 5, 400, 1, 58, "Normal: 7–20", "bu")
sod  = si("Sodium (mEq/L)", 100, 165, 0.5, 136.0, "Normal: 136–145", "sod")
bgr  = si("Blood Glucose (mg/dL)", 50, 500, 1, 132, "Fasting normal: 70–100", "bgr")

st.sidebar.markdown("#### 🩸 Hematological")
hemo = si("Hemoglobin (g/dL)", 2.0, 20.0, 0.1, 11.7, "Normal M>13, F>12", "hemo")
pcv  = si("Packed Cell Volume (%)", 9, 60, 0.5, 36.0, "Normal: 37–52", "pcv")
rbcc = si("RBC Count (M/μL)", 1.0, 8.0, 0.1, 4.1, "Normal: 4.2–5.4", "rbcc")
wbcc = si("WBC Count (cells/μL)", 2000, 30000, 100, 8400, "Normal: 4500–11000", "wbcc")

st.sidebar.markdown("#### 🧪 Urinalysis")
sg_opts = {"1.005": 1.005, "1.010": 1.010, "1.015": 1.015, "1.020": 1.020, "1.025": 1.025}
sg_sel  = st.sidebar.selectbox("Specific Gravity", list(sg_opts.keys()), index=2)
sg      = sg_opts[sg_sel]
al      = st.sidebar.slider("Albumin (0–5 scale)", 0, 5, 1)
pcc_sel = st.sidebar.selectbox("Pus Cell Clumps", ["Not Present", "Present"])
pcc     = 1 if pcc_sel == "Present" else 0

st.sidebar.markdown("#### 👤 Demographics & History")
age     = si("Age (years)", 0, 100, 1, 52, "Patient age", "age")
bp      = si("Blood Pressure (mm Hg)", 40, 200, 1, 76, "Diastolic BP", "bp")
htn_sel = st.sidebar.selectbox("Hypertension", ["No", "Yes"])
htn     = 1 if htn_sel == "Yes" else 0
app_sel = st.sidebar.selectbox("Appetite", ["Good", "Poor"])
appet   = 1 if app_sel == "Poor" else 0

# ─── Predict ──────────────────────────────────────────────────────
predict_btn = st.sidebar.button("🔬 Predict CKD Risk", type="primary", use_container_width=True)

# Build input row
input_data = {
    'age': age,
    'bp': bp,
    'sg': sg,
    'al': al,
    'su': 0,

    # ⭐ categorical as strings (IMPORTANT)
    'rbc': 'normal',
    'pc': 'normal',
    'pcc': 'yes' if pcc else 'no',
    'ba': 'no',

    'bgr': bgr,
    'bu': bu,
    'sc': sc,
    'sod': sod,
    'pot': 4.5,

    'hemo': hemo,
    'pcv': pcv,
    'wbcc': wbcc,
    'rbcc': rbcc,

    'htn': 'yes' if htn else 'no',
    'dm': 'no',
    'cad': 'no',
    'appet': 'poor' if appet else 'good',
    'pe': 'no',
    'ane': 'no'
}

def run_prediction(input_data):

    # Create dataframe
    df = pd.DataFrame([input_data])

    # ⭐ Ensure ALL training columns exist
    for col in all_feats:
        if col not in df.columns:
            df[col] = np.nan

    # ⭐ Force correct column order
    df = df[all_feats]

    df = df.apply(pd.to_numeric, errors='ignore')

    # ⭐ Encode categorical safely
    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str)
            try:
                df[col] = df[col].apply(
    lambda x: enc.transform([x])[0] if x in enc.classes_ else np.nan
)
            except:
                df[col] = np.nan

    # ⭐ Impute
    X_imp = pd.DataFrame(imputer.transform(df), columns=all_feats)

    # ⭐ Scale
    X_sc = pd.DataFrame(scaler.transform(X_imp), columns=all_feats)

    # ⭐ Feature selection
    X_sel = X_sc.values[:, rfe.support_]

    # ⭐ Predict
    prob = model.predict_proba(X_sel)[0, 1]

    return prob, X_sel[0], X_sc

def local_explain(model, sample, features):
    base = model.predict_proba(sample.reshape(1,-1))[0,1]
    contribs = {}
    for i, f in enumerate(features):
        p = sample.copy(); p[i] = 0.0
        c = model.predict_proba(p.reshape(1,-1))[0,1]
        contribs[f] = float(base - c)
    return contribs

# ─── Main content ──────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Running XAI pipeline..."):
        prob, sel_arr, X_sc = run_prediction(input_data)
        contribs = local_explain(model, sel_arr, selected_feats)

    isCKD = prob >= 0.5
    pct   = prob * 100

    # ─── Result header
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='{"risk-high" if isCKD else "risk-low"}'>{'CKD Detected' if isCKD else 'No CKD'}</div>
            <div style='color:#8899bb;margin-top:8px'>Prediction Result</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div style='font-size:2.5rem;font-weight:bold;color:{"#f43f5e" if prob>0.7 else "#f59e0b" if prob>0.4 else "#10b981"}'>{pct:.1f}%</div>
            <div style='color:#8899bb;margin-top:8px'>CKD Probability</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        risk_lv = "HIGH" if prob >= 0.7 else "MODERATE" if prob >= 0.4 else "LOW"
        colors  = {"HIGH":"#f43f5e","MODERATE":"#f59e0b","LOW":"#10b981"}
        st.markdown(f"""<div class='metric-card'>
            <div style='font-size:2.5rem;font-weight:bold;color:{colors[risk_lv]}'>{risk_lv}</div>
            <div style='color:#8899bb;margin-top:8px'>Risk Level</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ─── Two columns: Local + Global
    left, right = st.columns(2)

    with left:
        st.markdown("### 🔍 Local Feature Attribution (LIME-style)")
        st.caption("How each feature influenced THIS prediction")

        sorted_c = sorted(contribs.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
        names  = [f[0] for f in sorted_c]
        vals   = [f[1] for f in sorted_c]
        colors_bar = ['#f43f5e' if v > 0 else '#10b981' for v in vals]

        fig, ax = plt.subplots(figsize=(7,5))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')
        bars = ax.barh(names[::-1], vals[::-1], color=colors_bar[::-1], height=0.6)
        ax.axvline(0, color='#4a5f82', linewidth=1)
        ax.set_xlabel('Feature Contribution to CKD Probability', color='#8899bb')
        ax.tick_params(colors='#8899bb')
        for spine in ax.spines.values(): spine.set_edgecolor('#1e2d4a')
        ax.set_title('Local LIME-style Explanation', color='#e8edf8', pad=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with right:
        st.markdown("### 📊 Global Feature Importance (SHAP-style)")
        st.caption("Average impact across all predictions — from model permutation analysis")

        gi_sorted = sorted(global_imps.items(), key=lambda x: x[1], reverse=True)[:10]
        gnames = [x[0] for x in gi_sorted]
        gvals  = [x[1]*100 for x in gi_sorted]
        gcols  = ['#f43f5e' if global_imps.get(n,0) > 0.08 else '#3b82f6' if global_imps.get(n,0) > 0.04 else '#10b981' for n in gnames]

        fig2, ax2 = plt.subplots(figsize=(7,5))
        fig2.patch.set_facecolor('#111827')
        ax2.set_facecolor('#111827')
        ax2.barh(gnames[::-1], gvals[::-1], color=gcols[::-1], height=0.65)
        ax2.set_xlabel('Normalized Feature Importance (%)', color='#8899bb')
        ax2.tick_params(colors='#8899bb')
        for spine in ax2.spines.values(): spine.set_edgecolor('#1e2d4a')
        ax2.set_title('Global SHAP-Equivalent Importance', color='#e8edf8', pad=10)
        red_p   = mpatches.Patch(color='#f43f5e', label='High impact (risk)')
        blue_p  = mpatches.Patch(color='#3b82f6', label='Moderate impact')
        green_p = mpatches.Patch(color='#10b981', label='Lower impact')
        ax2.legend(handles=[red_p, blue_p, green_p], facecolor='#0a0f1e', labelcolor='#8899bb', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # ─── Clinical Narrative
    st.markdown("### 📋 Clinical Narrative")
    top_risk = [f for f,v in sorted(contribs.items(), key=lambda x:-x[1]) if v > 0.05][:3]
    top_prot = [f for f,v in sorted(contribs.items(), key=lambda x: x[1]) if v < -0.05][:2]

    feat_labels = {
        'hemo':'Hemoglobin', 'rbcc':'RBC Count', 'pcv':'PCV', 'sc':'Serum Creatinine',
        'bu':'Blood Urea', 'al':'Albumin', 'appet':'Appetite', 'htn':'Hypertension',
        'sod':'Sodium', 'bp':'Blood Pressure', 'wbcc':'WBC Count', 'pcc':'Pus Cell Clumps',
        'bgr':'Blood Glucose', 'sg':'Specific Gravity', 'age':'Age'
    }

    narrative = f"""
**Assessment:** The NephroAI ensemble model predicts **{'CKD PRESENT' if isCKD else 'NO CKD'}** with a probability of **{pct:.1f}%** ({'HIGH' if prob>=0.7 else 'MODERATE' if prob>=0.4 else 'LOW'} risk).

**Key Risk Drivers:** {', '.join(feat_labels.get(f,f) for f in top_risk) if top_risk else 'None identified above threshold'}.

**Protective Factors:** {', '.join(feat_labels.get(f,f) for f in top_prot) if top_prot else 'None identified'}.

**Hematological Status:** Hemoglobin {hemo:.1f} g/dL — {'⚠ Consistent with renal anemia' if hemo < 10 else '✓ Within acceptable range'}. PCV {pcv:.0f}% — {'⚠ Compromised' if pcv < 30 else '✓ Adequate'}.

**Renal Function:** Serum creatinine {sc:.1f} mg/dL — {'⚠ Elevated, suggesting reduced GFR' if sc > 1.5 else '✓ Normal range'}. Blood urea {bu:.0f} mg/dL — {'⚠ Elevated uremia risk' if bu > 40 else '✓ Normal'}.

**Clinical Comorbidities:** Hypertension: {'⚠ Present — major CKD risk and progression driver' if htn else '✓ Absent'}. Appetite: {'⚠ Poor — uremic symptom' if appet else '✓ Good'}.

**Recommendation:** {'🔴 Nephrology referral advised. Further workup: GFR estimation (CKD-EPI), 24h urine protein, renal ultrasound, electrolyte panel. Optimize blood pressure with ACE-I/ARB if confirmed.' if isCKD else '🟢 Continue routine monitoring. Annual renal function tests, urine dipstick, BP measurement. Lifestyle optimization: hydration, low-sodium diet.'}

> ⚠ *AI-generated summary — must be reviewed by a qualified clinician.*
    """
    st.info(narrative)

else:
    # ─── Landing state
    st.markdown("### 👈 Enter patient values in the sidebar and click **Predict CKD Risk**")

    # Model comparison
    st.markdown("## 🏆 Model Performance Leaderboard")
    st.markdown("5-fold stratified cross-validation on SMOTE-balanced UCI CKD data")

    df_results = pd.DataFrame([
        {'Model': k, 'AUC-ROC': v['mean_auc'], 'Std Dev': v['std_auc']}
        for k,v in results_dict.items()
    ]).sort_values('AUC-ROC', ascending=False)
    df_results['AUC-ROC'] = df_results['AUC-ROC'].map('{:.4f}'.format)
    df_results['Std Dev']  = df_results['Std Dev'].map('±{:.4f}'.format)
    df_results['Best'] = df_results['Model'].apply(lambda x: '⭐' if x == best_name else '')
    st.dataframe(df_results, use_container_width=True, hide_index=True)

    # Global importance chart
    st.markdown("## 📊 Global Feature Importance (SHAP-Equivalent)")
    gi_sorted = sorted(global_imps.items(), key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')
    names = [x[0] for x in gi_sorted]
    vals  = [x[1]*100 for x in gi_sorted]
    cols  = ['#f43f5e' if v > 8 else '#3b82f6' if v > 4 else '#10b981' for v in vals]
    ax.bar(names, vals, color=cols)
    ax.set_ylabel('Importance (%)', color='#8899bb')
    ax.tick_params(colors='#8899bb', axis='both')
    ax.set_title('SHAP-Equivalent Feature Importance — All Selected Features', color='#e8edf8', pad=12)
    for spine in ax.spines.values(): spine.set_edgecolor('#1e2d4a')
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.caption("Red = strong risk driver · Blue = moderate · Green = lower impact")

# ─── Footer ───────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center;color:#4a5f82;font-size:12px;'>
<strong>NephroAI</strong> — Explainable AI for CKD Early Detection<br>
Built with: UCI CKD Dataset · Scikit-learn · SMOTE · RFE · Ensemble Learning · SHAP/LIME XAI
<br><br>
⚠ <em>For research and educational purposes only. Not a substitute for clinical diagnosis.</em>
</div>
""", unsafe_allow_html=True)
