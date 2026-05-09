"""
Heart Disease Prediction – Streamlit Web Application
======================================================
Run:  streamlit run app.py
"""

import os, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             roc_curve, roc_auc_score)

# ── Optional TF ────────────────────────────────────────────────────────────────
try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #1e3a5f, #e63946);
        color: white; padding: 2rem 2.5rem; border-radius: 16px;
        margin-bottom: 1.5rem; text-align: center;
        box-shadow: 0 8px 32px rgba(30,58,95,0.35);
    }
    .main-header h1 { font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .main-header p  { font-size: 1.05rem; opacity: 0.9; margin-top: 0.5rem; }

    .metric-card {
        background: white; border-radius: 14px; padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08); text-align: center;
        border-top: 4px solid #e63946; transition: transform 0.2s;
    }
    .metric-card:hover { transform: translateY(-3px); }
    .metric-value  { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .metric-label  { font-size: 0.85rem; color: #666; font-weight: 500; margin-top: 4px; }

    .predict-card {
        background: linear-gradient(135deg, #f8fafd, #eef2f7);
        border-radius: 16px; padding: 2rem; border: 1px solid #dde4ee;
    }
    .result-positive {
        background: linear-gradient(135deg, #fff0f0, #ffe0e0);
        border: 2px solid #e63946; border-radius: 14px; padding: 1.5rem;
        text-align: center; color: #c0152a;
    }
    .result-negative {
        background: linear-gradient(135deg, #f0fff4, #dcffe8);
        border: 2px solid #2eb872; border-radius: 14px; padding: 1.5rem;
        text-align: center; color: #1a7a42;
    }
    .section-title {
        font-size: 1.3rem; font-weight: 700; color: #1e3a5f;
        border-left: 4px solid #e63946; padding-left: 12px;
        margin-bottom: 1rem;
    }
    .stSelectbox, .stSlider { margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_csv("heart.csv").dropna().reset_index(drop=True)

@st.cache_resource
def load_models():
    models = {}
    base = "outputs/models"
    if not os.path.exists(base):
        return models
    for name, fname in [
        ("Logistic Regression", "logistic_regression.pkl"),
        ("Decision Tree",        "decision_tree.pkl"),
        ("Random Forest",        "random_forest.pkl"),
    ]:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            models[name] = joblib.load(p)
    if HAS_TF:
        p = os.path.join(base, "neural_network.keras")
        if os.path.exists(p):
            models["Neural Network"] = tf.keras.models.load_model(p)
    return models

@st.cache_resource
def load_scaler():
    p = "outputs/models/scaler.pkl"
    return joblib.load(p) if os.path.exists(p) else None

def load_results():
    p = "outputs/results/model_comparison.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

FEATURE_COLS = ['age','sex','cp','trestbps','chol','fbs','restecg',
                'thalach','exang','oldpeak','slope','ca','thal']

# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
  <h1>🫀 Heart Disease Prediction</h1>
  <p>AI-powered classification using Machine Learning & Neural Networks</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/color/96/heart-health.png", width=80)
    st.markdown("## 🔬 Navigation")
    page = st.radio("", ["🏠 Overview", "📊 EDA", "⚖️ Model Comparison",
                          "🔮 Predict", "📈 ROC Curves"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("**Dataset Info**")
    df = load_data()
    st.info(f"Rows: **{len(df)}** | Features: **{len(FEATURE_COLS)}**")
    st.markdown("**Target**")
    counts = df['target'].value_counts()
    st.metric("Heart Disease", f"{counts.get(1,0)} ({counts.get(1,0)/len(df)*100:.1f}%)")
    st.metric("No Disease",    f"{counts.get(0,0)} ({counts.get(0,0)/len(df)*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# Page: Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<div class="section-title">📋 Dataset Overview</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in [
        (c1, "Total Patients", len(df)),
        (c2, "With Disease",   int(df['target'].sum())),
        (c3, "Without Disease",int((df['target']==0).sum())),
        (c4, "Features",       len(FEATURE_COLS))
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📄 Sample Data</div>', unsafe_allow_html=True)
    st.dataframe(df.head(10).style.background_gradient(cmap='Blues', subset=['age','chol','thalach']),
                 use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">📐 Statistical Summary</div>', unsafe_allow_html=True)
    st.dataframe(df.describe().round(2).style.background_gradient(cmap='Blues'),
                 use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-title">🧬 Feature Descriptions</div>', unsafe_allow_html=True)
    desc = {
        "age":      ("Age", "Patient age in years"),
        "sex":      ("Sex", "1 = Male, 0 = Female"),
        "cp":       ("Chest Pain Type", "0=Typical Angina, 1=Atypical, 2=Non-anginal, 3=Asymptomatic"),
        "trestbps": ("Resting BP", "Resting blood pressure (mm Hg)"),
        "chol":     ("Cholesterol", "Serum cholesterol in mg/dl"),
        "fbs":      ("Fasting Blood Sugar", "1 if > 120 mg/dl, else 0"),
        "restecg":  ("Resting ECG", "0=Normal, 1=ST-T abnormality, 2=LV hypertrophy"),
        "thalach":  ("Max Heart Rate", "Maximum heart rate achieved"),
        "exang":    ("Exercise Angina", "1=Yes, 0=No"),
        "oldpeak":  ("ST Depression", "Induced by exercise relative to rest"),
        "slope":    ("ST Slope", "0=Downsloping, 1=Flat, 2=Upsloping"),
        "ca":       ("Major Vessels", "# of vessels coloured by fluoroscopy (0-4)"),
        "thal":     ("Thalassemia", "0=Normal, 1=Fixed defect, 2=Reversible defect"),
        "target":   ("Target", "1=Heart Disease, 0=No Heart Disease"),
    }
    desc_df = pd.DataFrame([(k,v[0],v[1]) for k,v in desc.items()],
                            columns=["Column","Name","Description"])
    st.dataframe(desc_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page: EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.markdown('<div class="section-title">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlations", "Feature Analysis", "Box Plots"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            counts = df['target'].value_counts()
            ax.pie(counts.values, labels=['No Disease','Heart Disease'],
                   colors=['#457b9d','#e63946'], autopct='%1.1f%%',
                   startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
            ax.set_title('Target Distribution', fontweight='bold')
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c2:
            fig, ax = plt.subplots()
            for label, color, name in [(1,'#e63946','Heart Disease'),(0,'#457b9d','No Disease')]:
                ax.hist(df[df['target']==label]['age'], bins=20, alpha=0.7,
                        color=color, label=name, edgecolor='white')
            ax.set_title('Age Distribution by Target', fontweight='bold')
            ax.set_xlabel('Age'); ax.set_ylabel('Count'); ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(11, 8))
        corr = df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5, annot_kws={"size": 8})
        ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=13)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab3:
        feature = st.selectbox("Select Feature", FEATURE_COLS)
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots()
            for label, color, name in [(1,'#e63946','Heart Disease'),(0,'#457b9d','No Disease')]:
                ax.hist(df[df['target']==label][feature], bins=20, alpha=0.7,
                        color=color, label=name, edgecolor='white')
            ax.set_title(f'{feature} Distribution by Target', fontweight='bold')
            ax.legend()
            st.pyplot(fig, use_container_width=True)
            plt.close()
        with c2:
            fig, ax = plt.subplots()
            df.boxplot(column=feature, by='target', ax=ax,
                       boxprops=dict(color='#1e3a5f'),
                       medianprops=dict(color='#e63946', linewidth=2))
            ax.set_title(f'{feature} vs Target', fontweight='bold')
            ax.set_xlabel('Target (0=No Disease, 1=Heart Disease)')
            plt.suptitle('')
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with tab4:
        num_cols = ['age','trestbps','chol','thalach','oldpeak']
        fig, axes = plt.subplots(1, 5, figsize=(18, 5))
        for i, col in enumerate(num_cols):
            axes[i].boxplot([df[df['target']==0][col].dropna(),
                             df[df['target']==1][col].dropna()],
                            labels=['No Disease','Heart Disease'],
                            boxprops=dict(color='#1e3a5f'),
                            medianprops=dict(color='#e63946', linewidth=2))
            axes[i].set_title(col, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=20)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Page: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚖️ Model Comparison":
    st.markdown('<div class="section-title">⚖️ Model Performance Comparison</div>', unsafe_allow_html=True)
    results_df = load_results()

    if results_df is None:
        st.warning("⚠️ No trained models found. Please run `python train_models.py` first.")
    else:
        # Metric cards
        best_model = results_df.loc[results_df['Accuracy'].idxmax()]
        st.success(f"🏆 **Best Model:** {best_model['Model']} with **{best_model['Accuracy']:.2f}%** accuracy")

        st.markdown("### 📊 Full Metrics Table")
        styled_df = results_df.set_index('Model').style\
            .background_gradient(cmap='RdYlGn', subset=['Accuracy','F1-Score','ROC-AUC'])\
            .format("{:.2f}%")
        st.dataframe(styled_df, use_container_width=True)

        st.markdown("### 📈 Metric Charts")
        metrics_list = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        colors = ['#457b9d', '#2a9d8f', '#e9c46a', '#e63946']
        for i, metric in enumerate(metrics_list):
            bars = axes[i].bar(results_df['Model'], results_df[metric],
                               color=colors[:len(results_df)], edgecolor='white')
            axes[i].set_title(metric, fontweight='bold')
            axes[i].set_ylim(0, 115)
            axes[i].set_ylabel('%')
            axes[i].tick_params(axis='x', rotation=30)
            for bar, val in zip(bars, results_df[metric]):
                axes[i].text(bar.get_x() + bar.get_width()/2,
                             bar.get_height() + 1, f'{val:.1f}',
                             ha='center', fontsize=9, fontweight='bold')
        fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Saved confusion matrix plots
        st.markdown("### 🔲 Confusion Matrices")
        cm_paths = {
            "Logistic Regression": "outputs/plots/cm_Logistic_Regression.png",
            "Decision Tree":        "outputs/plots/cm_Decision_Tree.png",
            "Random Forest":        "outputs/plots/cm_Random_Forest.png",
            "Neural Network":       "outputs/plots/cm_Neural_Network.png",
        }
        cols = st.columns(len([p for p in cm_paths.values() if os.path.exists(p)]))
        col_idx = 0
        for name, path in cm_paths.items():
            if os.path.exists(path):
                cols[col_idx].image(path, caption=name, use_container_width=True)
                col_idx += 1

        # Feature importance
        fi_rf = "outputs/plots/08_feature_importance_rf.png"
        if os.path.exists(fi_rf):
            st.markdown("### 🌲 Random Forest Feature Importance")
            st.image(fi_rf, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# Page: Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict":
    st.markdown('<div class="section-title">🔮 Predict Heart Disease Risk</div>', unsafe_allow_html=True)
    models  = load_models()
    scaler  = load_scaler()

    if not models:
        st.warning("⚠️ No trained models found. Please run `python train_models.py` first.")
    else:
        st.markdown('<div class="predict-card">', unsafe_allow_html=True)
        st.markdown("### 📝 Enter Patient Information")

        c1, c2, c3 = st.columns(3)
        with c1:
            age      = st.slider("Age",             20, 80, 50, key="age")
            sex      = st.selectbox("Sex",          ["Male (1)", "Female (0)"], key="sex")
            cp       = st.selectbox("Chest Pain Type",
                                    ["Asymptomatic (0)", "Typical Angina (1)",
                                     "Atypical Angina (2)", "Non-anginal (3)"], key="cp")
            trestbps = st.slider("Resting BP (mmHg)", 80, 200, 120, key="trestbps")
            chol     = st.slider("Cholesterol (mg/dl)", 100, 600, 200, key="chol")
        with c2:
            fbs      = st.selectbox("Fasting Blood Sugar > 120 mg/dl",
                                    ["No (0)", "Yes (1)"], key="fbs")
            restecg  = st.selectbox("Resting ECG",
                                    ["Normal (0)", "ST-T Abnormality (1)", "LV Hypertrophy (2)"], key="restecg")
            thalach  = st.slider("Max Heart Rate", 60, 220, 150, key="thalach")
            exang    = st.selectbox("Exercise Induced Angina",
                                    ["No (0)", "Yes (1)"], key="exang")
        with c3:
            oldpeak  = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0, step=0.1, key="oldpeak")
            slope    = st.selectbox("Slope of ST Segment",
                                    ["Downsloping (0)", "Flat (1)", "Upsloping (2)"], key="slope")
            ca       = st.slider("Major Vessels (0-4)", 0, 4, 0, key="ca")
            thal     = st.selectbox("Thalassemia",
                                    ["Normal (0)", "Fixed Defect (1)", "Reversible Defect (2)"], key="thal")

        model_choice = st.selectbox("🤖 Select Model", list(models.keys()))
        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("🫀 Predict", type="primary", use_container_width=True):
            sex_val     = int(sex.split("(")[1][0])
            cp_val      = int(cp.split("(")[1][0])
            fbs_val     = int(fbs.split("(")[1][0])
            restecg_val = int(restecg.split("(")[1][0])
            exang_val   = int(exang.split("(")[1][0])
            slope_val   = int(slope.split("(")[1][0])
            thal_val    = int(thal.split("(")[1][0])

            X_input = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val,
                                  restecg_val, thalach, exang_val, oldpeak,
                                  slope_val, ca, thal_val]])
            if scaler:
                X_input = scaler.transform(X_input)

            selected_model = models[model_choice]
            if model_choice == "Neural Network":
                prob = float(selected_model.predict(X_input)[0][0])
                pred = int(prob >= 0.5)
            else:
                prob = float(selected_model.predict_proba(X_input)[0][1])
                pred = int(selected_model.predict(X_input)[0])

            st.markdown("---")
            if pred == 1:
                st.markdown(f"""
                <div class="result-positive">
                    <h2>⚠️ Heart Disease Detected</h2>
                    <h3>Confidence: {prob*100:.1f}%</h3>
                    <p>The model predicts a high risk of heart disease. Please consult a cardiologist immediately.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-negative">
                    <h2>✅ No Heart Disease Detected</h2>
                    <h3>Confidence: {(1-prob)*100:.1f}%</h3>
                    <p>The model predicts a low risk of heart disease. Maintain a healthy lifestyle!</p>
                </div>""", unsafe_allow_html=True)

            # Probability gauge
            fig, ax = plt.subplots(figsize=(6, 1.5))
            ax.barh(0, prob,    color='#e63946', height=0.5, label='Disease Risk')
            ax.barh(0, 1-prob, left=prob, color='#2eb872', height=0.5, label='No Disease')
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel('Probability')
            ax.set_title('Risk Probability Bar', fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Page: ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 ROC Curves":
    st.markdown('<div class="section-title">📈 ROC Curves & AUC Analysis</div>', unsafe_allow_html=True)
    roc_path = "outputs/plots/07_roc_curves.png"
    if os.path.exists(roc_path):
        st.image(roc_path, use_container_width=True)
    else:
        st.warning("⚠️ ROC plot not found. Please run `python train_models.py` first.")

    results_df = load_results()
    if results_df is not None:
        st.markdown("### AUC Scores")
        auc_df = results_df[['Model','ROC-AUC']].sort_values('ROC-AUC', ascending=False)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#e63946' if v == auc_df['ROC-AUC'].max() else '#457b9d'
                  for v in auc_df['ROC-AUC']]
        bars = ax.barh(auc_df['Model'], auc_df['ROC-AUC'], color=colors, edgecolor='white')
        ax.set_xlabel('ROC-AUC (%)')
        ax.set_title('ROC-AUC by Model', fontweight='bold')
        for bar, val in zip(bars, auc_df['ROC-AUC']):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                    f'{val:.2f}%', va='center', fontweight='bold')
        ax.set_xlim(0, 115)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#888;font-size:0.85rem;'>"
    "🫀 Heart Disease Prediction System | Built with Scikit-learn & TensorFlow | Major Project</p>",
    unsafe_allow_html=True
)
