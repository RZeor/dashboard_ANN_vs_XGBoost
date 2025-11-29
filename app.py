import streamlit as st
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

# Set page config
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI/UX
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Header Styles */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        animation: fadeInDown 1s ease-in-out;
    }
    
    .subtitle {
        text-align: center;
        color: #e0e7ff;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        font-weight: 300;
        animation: fadeIn 1.5s ease-in-out;
    }
    
    /* Card Styles */
    .stMetric {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 48px rgba(0,0,0,0.3);
    }
    
    .stMetric label {
        color: #e0e7ff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric [data-testid="stMetricDelta"] {
        color: #a5f3fc !important;
    }
    
    /* Sidebar Styles */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 2px solid #334155;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0px);
    }
    
    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(15, 23, 42, 0.6);
        border-radius: 12px;
        padding: 0.5rem;
        backdrop-filter: blur(10px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #e2e8f0;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(102, 126, 234, 0.2);
        color: white;
    }
    
    /* Input Styles */
    .stSelectbox, .stNumberInput, .stTextInput {
        border-radius: 8px;
    }
    
    .stSelectbox > div > div,
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background-color: rgba(30, 41, 59, 0.95) !important;
        color: #e2e8f0 !important;
        border: 2px solid #475569;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus,
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.3);
        background-color: rgba(30, 41, 59, 1) !important;
    }
    
    /* Select box dropdown */
    .stSelectbox [data-baseweb="select"] > div {
        background-color: rgba(30, 41, 59, 0.95) !important;
        color: #e2e8f0 !important;
    }
    
    .stSelectbox [data-baseweb="select"] span {
        color: #e2e8f0 !important;
    }
    
    /* Dropdown menu */
    [data-baseweb="popover"] {
        background-color: rgba(15, 23, 42, 0.98) !important;
    }
    
    [role="listbox"] {
        background-color: rgba(15, 23, 42, 0.98) !important;
    }
    
    [role="option"] {
        background-color: transparent !important;
        color: #e2e8f0 !important;
    }
    
    [role="option"]:hover {
        background-color: rgba(102, 126, 234, 0.3) !important;
        color: #ffffff !important;
    }
    
    [aria-selected="true"] {
        background-color: rgba(102, 126, 234, 0.5) !important;
        color: #ffffff !important;
    }
    
    /* Card Container */
    .element-container {
        animation: fadeIn 0.5s ease-in-out;
    }
    
    /* Info/Success/Error Boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Dataframe Styles */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Expander Styles */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(118, 75, 162, 0.2) 100%);
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* File Uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.05);
        border: 2px dashed #667eea;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: #764ba2;
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e293b;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Plotly Chart Container */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Title with Animation
st.markdown('''
<h1 class="main-header">🔍 Fraud Detection Dashboard</h1>
<p class="subtitle">AI-Powered Transaction Analysis with ANN & XGBoost</p>
''', unsafe_allow_html=True)

# Add a dynamic status banner
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(34, 197, 94, 0.1); border-radius: 10px; border-left: 4px solid #22c55e;'>
        <h3 style='color: #22c55e; margin: 0;'>🟢</h3>
        <p style='color: #e2e8f0; margin: 0; font-size: 0.9rem;'>System Online</p>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(59, 130, 246, 0.1); border-radius: 10px; border-left: 4px solid #3b82f6;'>
        <h3 style='color: #3b82f6; margin: 0;'>🧠</h3>
        <p style='color: #e2e8f0; margin: 0; font-size: 0.9rem;'>ANN Ready</p>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(168, 85, 247, 0.1); border-radius: 10px; border-left: 4px solid #a855f7;'>
        <h3 style='color: #a855f7; margin: 0;'>🌳</h3>
        <p style='color: #e2e8f0; margin: 0; font-size: 0.9rem;'>XGBoost Ready</p>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: rgba(234, 179, 8, 0.1); border-radius: 10px; border-left: 4px solid #eab308;'>
        <h3 style='color: #eab308; margin: 0;'>⚡</h3>
        <p style='color: #e2e8f0; margin: 0; font-size: 0.9rem;'>Real-time Analysis</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Enhanced Sidebar
st.sidebar.markdown("""
<div style='text-align: center; padding: 1.5rem 0;'>
    <h1 style='color: #667eea; font-size: 2rem; margin: 0;'>🎛️</h1>
    <h2 style='color: #ffffff; margin: 0.5rem 0 0 0; font-size: 1.5rem;'>Control Panel</h2>
    <p style='color: #94a3b8; font-size: 0.85rem; margin: 0.5rem 0 0 0;'>Configure your analysis</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🤖 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose your model",
    ["ANN", "XGBoost", "Compare Both"],
    index=2,
    help="Select which model(s) to use for predictions"
)

# Load models
st.cache_resource
def load_models():
    try:
        # Gunakan path relatif dari lokasi app.py
        ann_model = load_model('model/ANN/ann_model.keras')
        st.sidebar.success("✅ ANN model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading ANN model: {e}")
        ann_model = None
    
    try:
        with open('model/XGBoost/xgboost_model.pkl', 'rb') as f:
            xgb_model = pickle.load(f)
        st.sidebar.success("✅ XGBoost model loaded")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading XGBoost model: {e}")
        xgb_model = None
    
    return ann_model, xgb_model

# Load scaler
@st.cache_resource
def load_scaler():
    try:
        with open('model/robust/robust_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        st.sidebar.success("✅ Scaler loaded")
        return scaler
    except Exception as e:
        st.sidebar.warning(f"⚠️ Scaler not found: {e}")
        return None

# Load data for inference and analysis
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data/data.csv')
        st.sidebar.success("✅ Data loaded")
        return df
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}")
        return None

# Load evaluation data
@st.cache_data
def load_eval_data():
    try:
        df_eval = pd.read_csv('data/data_eval.csv')
        st.sidebar.success("✅ Evaluation data loaded")
        return df_eval
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error loading evaluation data: {e}")
        return None

# Cache preprocessing operations for visualization
@st.cache_data
def preprocess_data_for_visualization(_df, _scaler):
    """Cache the expensive preprocessing operations"""
    df_viz = decode_one_hot(_df.copy())
    df_viz_denorm = inverse_transform_data(df_viz.copy(), _scaler)
    return df_viz_denorm

# Inverse transform function
def inverse_transform_data(df_normalized, scaler):
    """Kembalikan data yang sudah dinormalisasi ke nilai asli"""
    df_original = df_normalized.copy()
    
    # Kolom numerik yang dinormalisasi
    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    
    # Cek kolom yang ada
    available_cols = [col for col in numerical_cols if col in df_normalized.columns]
    
    if scaler is not None and available_cols:
        try:
            # Inverse transform
            df_original[available_cols] = scaler.inverse_transform(df_normalized[available_cols])
        except Exception as e:
            st.warning(f"Could not inverse transform: {e}")
    
    return df_original

# Decode one-hot encoding
def decode_one_hot(df):
    """Kembalikan one-hot encoding ke kategori asli"""
    df_decoded = df.copy()
    
    # Cek kolom type
    type_cols = [col for col in df.columns if col.startswith('type_')]
    
    if type_cols:
        # Buat kolom 'type' dari one-hot encoding
        type_mapping = {col: col.replace('type_', '') for col in type_cols}
        
        # Cari nilai maksimum untuk setiap baris
        type_values = df[type_cols].idxmax(axis=1)
        df_decoded['type'] = type_values.map(type_mapping)
        
        # Jika semua nilai 0 (drop_first=True), maka kategori pertama
        all_zero_mask = df[type_cols].sum(axis=1) == 0
        if all_zero_mask.any():
            df_decoded.loc[all_zero_mask, 'type'] = 'CASH_OUT'  # Kategori yang di-drop
        
        # Drop kolom one-hot
        df_decoded.drop(columns=type_cols, inplace=True)
    
    return df_decoded

# Format currency
def format_currency(value):
    """Format nilai sebagai currency"""
    return f"${value:,.2f}"

# Main app
ann_model, xgb_model = load_models()
scaler = load_scaler()
df = load_data()
eval_df = load_eval_data()

if df is not None and (ann_model is not None or xgb_model is not None):
    
    # Sidebar - Data Info with Enhanced Design
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Dataset Overview")
    
    st.sidebar.markdown(f"""
    <div style='background: rgba(102, 126, 234, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <p style='color: #94a3b8; font-size: 0.85rem; margin: 0;'>Total Records</p>
        <h3 style='color: #ffffff; margin: 0.25rem 0 0 0;'>{len(df):,}</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if 'isFraud' in df.columns:
        fraud_count = df['isFraud'].sum()
        non_fraud_count = len(df) - fraud_count
        fraud_percentage = (fraud_count / len(df)) * 100
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.markdown(f"""
            <div style='background: rgba(239, 68, 68, 0.1); padding: 0.75rem; border-radius: 8px; text-align: center;'>
                <p style='color: #fca5a5; font-size: 0.75rem; margin: 0;'>Fraud</p>
                <h4 style='color: #ef4444; margin: 0.25rem 0 0 0;'>{fraud_count:,}</h4>
                <p style='color: #fca5a5; font-size: 0.7rem; margin: 0.25rem 0 0 0;'>{fraud_percentage:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div style='background: rgba(34, 197, 94, 0.1); padding: 0.75rem; border-radius: 8px; text-align: center;'>
                <p style='color: #86efac; font-size: 0.75rem; margin: 0;'>Legit</p>
                <h4 style='color: #22c55e; margin: 0.25rem 0 0 0;'>{non_fraud_count:,}</h4>
                <p style='color: #86efac; font-size: 0.7rem; margin: 0.25rem 0 0 0;'>{100-fraud_percentage:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Sidebar - Sample Size with Enhanced Design
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🎯 Inference Settings")
    
    st.sidebar.markdown("""
    <div style='background: rgba(59, 130, 246, 0.1); padding: 0.5rem 1rem; border-radius: 8px; margin-bottom: 1rem;'>
        <p style='color: #93c5fd; font-size: 0.8rem; margin: 0;'>💡 Adjust parameters for optimal results</p>
    </div>
    """, unsafe_allow_html=True)
    
    sample_size = st.sidebar.slider(
        "📏 Sample Size",
        min_value=100,
        max_value=min(10000, len(df)),
        value=min(1000, len(df)),
        step=100,
        help="Number of records to analyze"
    )
    
    confidence_threshold = st.sidebar.slider(
        "🎚️ Confidence Threshold (ANN)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Threshold for classifying transactions as fraud"
    )
    
    # Add visual indicator for threshold
    st.sidebar.markdown(f"""
    <div style='background: rgba(168, 85, 247, 0.1); padding: 0.75rem; border-radius: 8px; margin-top: 0.5rem;'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <span style='color: #c4b5fd; font-size: 0.85rem;'>Current Threshold:</span>
            <span style='color: #a855f7; font-weight: bold; font-size: 1.1rem;'>{confidence_threshold:.2f}</span>
        </div>
        <div style='background: rgba(255,255,255,0.1); height: 6px; border-radius: 3px; margin-top: 0.5rem; overflow: hidden;'>
            <div style='background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); height: 100%; width: {confidence_threshold*100}%; transition: width 0.3s ease;'></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Model Evaluation", "🔮 Inference", "🎯 Manual Prediction", "📤 Upload CSV", "📈 Data Analysis", "ℹ️ About"])
    
    with tab1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                    padding: 2rem; border-radius: 15px; margin-bottom: 2rem; border-left: 5px solid #667eea;'>
            <h2 style='color: #e2e8f0; margin: 0 0 0.5rem 0;'>📊 Model Evaluation Metrics</h2>
            <p style='color: #94a3b8; margin: 0; font-size: 0.95rem;'>
                Comprehensive performance analysis using validation dataset • Real-time metrics • Confusion matrices
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if eval_df is None:
            st.error("❌ Evaluation data (data_eval.csv) not found! Please ensure data_eval.csv is in the same directory as app.py")
        elif 'isFraud' not in eval_df.columns:
            st.warning("⚠️ Column 'isFraud' not found in evaluation data. Cannot evaluate models.")
        else:
            X = eval_df.drop('isFraud', axis=1)
            y = eval_df['isFraud']
            
            # Display eval data info
            eval_info_col1, eval_info_col2, eval_info_col3 = st.columns(3)
            with eval_info_col1:
                st.metric("📊 Total Eval Records", f"{len(eval_df):,}")
            with eval_info_col2:
                fraud_eval = eval_df['isFraud'].sum()
                st.metric("🚨 Fraud Cases", f"{fraud_eval:,}")
            with eval_info_col3:
                non_fraud_eval = len(eval_df) - fraud_eval
                st.metric("✅ Non-Fraud Cases", f"{non_fraud_eval:,}")
            
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            
            # ANN Evaluation
            if model_choice in ["ANN", "Compare Both"] and ann_model is not None:
                with col1:
                    st.subheader("🧠 ANN Model")
                    
                    with st.spinner("🧠 Neural network is analyzing patterns..."):
                        y_pred_ann_prob = ann_model.predict(X, verbose=0)
                        y_pred_ann = (y_pred_ann_prob > confidence_threshold).astype(int).flatten()
                    
                    # Metrics
                    acc = accuracy_score(y, y_pred_ann)
                    prec = precision_score(y, y_pred_ann, zero_division=0)
                    rec = recall_score(y, y_pred_ann, zero_division=0)
                    f1 = f1_score(y, y_pred_ann, zero_division=0)
                    
                    # Display metrics in cards
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("📈 Accuracy", f"{acc:.4f}", delta=f"{acc*100:.2f}%")
                        st.metric("🎯 Precision", f"{prec:.4f}", delta=f"{prec*100:.2f}%")
                    with metrics_col2:
                        st.metric("🔍 Recall", f"{rec:.4f}", delta=f"{rec*100:.2f}%")
                        st.metric("⚖️ F1 Score", f"{f1:.4f}", delta=f"{f1*100:.2f}%")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y, y_pred_ann)
                    fig = px.imshow(cm, 
                                    labels=dict(x="Predicted", y="Actual", color="Count"),
                                    x=['Non-Fraud', 'Fraud'],
                                    y=['Non-Fraud', 'Fraud'],
                                    text_auto=True,
                                    color_continuous_scale='Greens',
                                    aspect="auto")
                    fig.update_layout(
                        title="Confusion Matrix - ANN",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification Report
                    with st.expander("📋 Detailed Classification Report"):
                        report = classification_report(y, y_pred_ann, target_names=['Non-Fraud', 'Fraud'], output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
            
            # XGBoost Evaluation
            if model_choice in ["XGBoost", "Compare Both"] and xgb_model is not None:
                with col2:
                    st.subheader("🌳 XGBoost Model")
                    
                    with st.spinner("🌳 XGBoost is processing decision trees..."):
                        y_pred_xgb = xgb_model.predict(X)
                    
                    # Metrics
                    acc = accuracy_score(y, y_pred_xgb)
                    prec = precision_score(y, y_pred_xgb, zero_division=0)
                    rec = recall_score(y, y_pred_xgb, zero_division=0)
                    f1 = f1_score(y, y_pred_xgb, zero_division=0)
                    
                    # Display metrics in cards
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("📈 Accuracy", f"{acc:.4f}", delta=f"{acc*100:.2f}%")
                        st.metric("🎯 Precision", f"{prec:.4f}", delta=f"{prec*100:.2f}%")
                    with metrics_col2:
                        st.metric("🔍 Recall", f"{rec:.4f}", delta=f"{rec*100:.2f}%")
                        st.metric("⚖️ F1 Score", f"{f1:.4f}", delta=f"{f1*100:.2f}%")
                    
                    # Confusion Matrix
                    cm = confusion_matrix(y, y_pred_xgb)
                    fig = px.imshow(cm,
                                    labels=dict(x="Predicted", y="Actual", color="Count"),
                                    x=['Non-Fraud', 'Fraud'],
                                    y=['Non-Fraud', 'Fraud'],
                                    text_auto=True,
                                    color_continuous_scale='Blues',
                                    aspect="auto")
                    fig.update_layout(
                        title="Confusion Matrix - XGBoost",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Classification Report
                    with st.expander("📋 Detailed Classification Report"):
                        report = classification_report(y, y_pred_xgb, target_names=['Non-Fraud', 'Fraud'], output_dict=True)
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df.style.highlight_max(axis=0), use_container_width=True)
            
            # Model Comparison
            if model_choice == "Compare Both" and ann_model is not None and xgb_model is not None:
                st.markdown("---")
                st.subheader("📊 Model Comparison")
                
                # Comparison metrics
                comparison_data = {
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    'ANN': [
                        accuracy_score(y, (ann_model.predict(X, verbose=0) > confidence_threshold).astype(int)),
                        precision_score(y, (ann_model.predict(X, verbose=0) > confidence_threshold).astype(int), zero_division=0),
                        recall_score(y, (ann_model.predict(X, verbose=0) > confidence_threshold).astype(int), zero_division=0),
                        f1_score(y, (ann_model.predict(X, verbose=0) > confidence_threshold).astype(int), zero_division=0)
                    ],
                    'XGBoost': [
                        accuracy_score(y, xgb_model.predict(X)),
                        precision_score(y, xgb_model.predict(X), zero_division=0),
                        recall_score(y, xgb_model.predict(X), zero_division=0),
                        f1_score(y, xgb_model.predict(X), zero_division=0)
                    ]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(name='ANN', x=comparison_df['Metric'], y=comparison_df['ANN'], marker_color='lightgreen'))
                fig.add_trace(go.Bar(name='XGBoost', x=comparison_df['Metric'], y=comparison_df['XGBoost'], marker_color='lightblue'))
                
                fig.update_layout(
                    title='Model Performance Comparison',
                    xaxis_title='Metric',
                    yaxis_title='Score',
                    barmode='group',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("⚠️ Column 'isFraud' not found in evaluation data. Cannot evaluate models.")
    
    with tab2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%); 
                    padding: 2rem; border-radius: 15px; margin-bottom: 2rem; border-left: 5px solid #22c55e;'>
            <h2 style='color: #e2e8f0; margin: 0 0 0.5rem 0;'>🔮 Batch Inference Engine</h2>
            <p style='color: #94a3b8; margin: 0; font-size: 0.95rem;'>
                Process thousands of transactions instantly • Advanced filtering • Real-time denormalization
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample data for inference
        X_inference = df.drop('isFraud', axis=1) if 'isFraud' in df.columns else df
        
        # Random sampling with seed
        random_seed = st.number_input("Random Seed (for reproducibility)", min_value=0, max_value=1000, value=42)
        X_inference_sample = X_inference.sample(n=min(sample_size, len(X_inference)), random_state=random_seed)
        
        # Store original indices
        original_indices = X_inference_sample.index
        
        # Perform inference
        results_normalized = X_inference_sample.copy()
        
        if model_choice in ["ANN", "Compare Both"] and ann_model is not None:
            with st.spinner("🧠 ANN is analyzing transaction patterns..."):
                ann_pred_prob = ann_model.predict(X_inference_sample, verbose=0)
                results_normalized['ANN_Probability'] = ann_pred_prob.flatten()
                results_normalized['ANN_Prediction'] = (ann_pred_prob > confidence_threshold).astype(int).flatten()
        
        if model_choice in ["XGBoost", "Compare Both"] and xgb_model is not None:
            with st.spinner("🌳 XGBoost is processing ensemble predictions..."):
                xgb_pred = xgb_model.predict(X_inference_sample)
                results_normalized['XGBoost_Prediction'] = xgb_pred
        
        # Inverse transform and decode
        results_denormalized = inverse_transform_data(results_normalized.copy(), scaler)
        results_display = decode_one_hot(results_denormalized)
        
        # Add true labels if available
        if 'isFraud' in df.columns:
            results_display['True_Label'] = df.loc[original_indices, 'isFraud'].values
        
        # Display summary statistics
        st.subheader(f"📋 Inference Results Summary (Sample: {len(results_display):,} records)")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'ANN_Prediction' in results_display.columns:
                ann_fraud = results_display['ANN_Prediction'].sum()
                ann_fraud_pct = (ann_fraud / len(results_display)) * 100
                st.metric("🧠 ANN: Predicted Fraud", f"{ann_fraud:,}", delta=f"{ann_fraud_pct:.1f}%")
        
        with col2:
            if 'XGBoost_Prediction' in results_display.columns:
                xgb_fraud = results_display['XGBoost_Prediction'].sum()
                xgb_fraud_pct = (xgb_fraud / len(results_display)) * 100
                st.metric("🌳 XGBoost: Predicted Fraud", f"{xgb_fraud:,}", delta=f"{xgb_fraud_pct:.1f}%")
        
        with col3:
            if 'ANN_Prediction' in results_display.columns and 'XGBoost_Prediction' in results_display.columns:
                disagreement = (results_display['ANN_Prediction'] != results_display['XGBoost_Prediction']).sum()
                disagreement_pct = (disagreement / len(results_display)) * 100
                st.metric("⚠️ Model Disagreement", f"{disagreement:,}", delta=f"{disagreement_pct:.1f}%")
        
        with col4:
            if 'True_Label' in results_display.columns:
                actual_fraud = results_display['True_Label'].sum()
                actual_fraud_pct = (actual_fraud / len(results_display)) * 100
                st.metric("✅ Actual Fraud", f"{actual_fraud:,}", delta=f"{actual_fraud_pct:.1f}%")
        
        # Filter options
        st.markdown("---")
        col_filter, col_sort = st.columns([2, 1])
        
        with col_filter:
            filter_option = st.selectbox(
                "🔍 Filter Results",
                ["All Records", "Predicted Fraud Only (ANN)", "Predicted Fraud Only (XGBoost)", 
                 "Model Disagreement", "True Positives (ANN)", "False Positives (ANN)"]
            )
        
        with col_sort:
            if 'ANN_Probability' in results_display.columns:
                sort_option = st.selectbox(
                    "📊 Sort By",
                    ["Index", "ANN Probability (High to Low)", "ANN Probability (Low to High)"]
                )
        
        # Apply filters
        filtered_results = results_display.copy()
        
        if filter_option == "Predicted Fraud Only (ANN)" and 'ANN_Prediction' in filtered_results.columns:
            filtered_results = filtered_results[filtered_results['ANN_Prediction'] == 1]
        elif filter_option == "Predicted Fraud Only (XGBoost)" and 'XGBoost_Prediction' in filtered_results.columns:
            filtered_results = filtered_results[filtered_results['XGBoost_Prediction'] == 1]
        elif filter_option == "Model Disagreement":
            if 'ANN_Prediction' in filtered_results.columns and 'XGBoost_Prediction' in filtered_results.columns:
                filtered_results = filtered_results[filtered_results['ANN_Prediction'] != filtered_results['XGBoost_Prediction']]
        elif filter_option == "True Positives (ANN)":
            if 'ANN_Prediction' in filtered_results.columns and 'True_Label' in filtered_results.columns:
                filtered_results = filtered_results[(filtered_results['ANN_Prediction'] == 1) & (filtered_results['True_Label'] == 1)]
        elif filter_option == "False Positives (ANN)":
            if 'ANN_Prediction' in filtered_results.columns and 'True_Label' in filtered_results.columns:
                filtered_results = filtered_results[(filtered_results['ANN_Prediction'] == 1) & (filtered_results['True_Label'] == 0)]
        
        # Apply sorting
        if 'ANN_Probability' in filtered_results.columns:
            if sort_option == "ANN Probability (High to Low)":
                filtered_results = filtered_results.sort_values('ANN_Probability', ascending=False)
            elif sort_option == "ANN Probability (Low to High)":
                filtered_results = filtered_results.sort_values('ANN_Probability', ascending=True)
        
        # Display table
        st.subheader(f"📄 Filtered Results ({len(filtered_results):,} records)")
        
        # Format numeric columns for display
        display_df = filtered_results.copy()
        numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "")
        
        if 'ANN_Probability' in display_df.columns:
            display_df['ANN_Probability'] = display_df['ANN_Probability'].apply(lambda x: f"{x:.4f}")
        
        st.dataframe(display_df, use_container_width=True, height=400)
        
        # Visualization
        if len(filtered_results) > 0:
            st.markdown("---")
            st.subheader("📊 Inference Visualization")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                if 'ANN_Probability' in filtered_results.columns:
                    fig = px.histogram(filtered_results, x='ANN_Probability', 
                                     nbins=50,
                                     title='ANN Prediction Probability Distribution',
                                     color_discrete_sequence=['#2ecc71'])
                    fig.add_vline(x=confidence_threshold, line_dash="dash", line_color="red", 
                                annotation_text=f"Threshold: {confidence_threshold}")
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_col2:
                if 'type' in filtered_results.columns:
                    type_counts = filtered_results['type'].value_counts()
                    fig = px.pie(values=type_counts.values, names=type_counts.index,
                               title='Transaction Type Distribution in Sample',
                               hole=0.3)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Download button
        st.markdown("---")
        csv = filtered_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"inference_results_{filter_option.replace(' ', '_').lower()}.csv",
            mime="text/csv"
        )
    
    with tab3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(234, 179, 8, 0.1) 0%, rgba(249, 115, 22, 0.1) 100%); 
                    padding: 2rem; border-radius: 15px; margin-bottom: 2rem; border-left: 5px solid #eab308;'>
            <h2 style='color: #e2e8f0; margin: 0 0 0.5rem 0;'>🎯 Single Transaction Analyzer</h2>
            <p style='color: #94a3b8; margin: 0; font-size: 0.95rem;'>
                Input custom transaction details • Get instant AI predictions • Compare model confidence levels
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Get sample data to understand the structure
        sample_data = X_inference.iloc[0] if len(X_inference) > 0 else None
        
        # Create two columns for input form
        st.subheader("📝 Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Transaction Type")
            # Transaction type selection
            transaction_types = ['CASH_OUT', 'TRANSFER']
            selected_type = st.selectbox("Select Transaction Type", transaction_types)
            
            st.markdown("#### Transaction Amount")
            amount_input = st.number_input(
                "Amount ($)",
                min_value=0.0,
                max_value=10000000.0,
                value=1000.0,
                step=100.0,
                format="%.2f"
            )
            
            st.markdown("#### Origin Account Balance")
            old_balance_orig = st.number_input(
                "Old Balance Origin ($)",
                min_value=0.0,
                max_value=100000000.0,
                value=10000.0,
                step=100.0,
                format="%.2f"
            )
            
            new_balance_orig = st.number_input(
                "New Balance Origin ($)",
                min_value=0.0,
                max_value=100000000.0,
                value=old_balance_orig - amount_input if old_balance_orig >= amount_input else 0.0,
                step=100.0,
                format="%.2f"
            )
        
        with col2:
            st.markdown("#### Destination Account Balance")
            old_balance_dest = st.number_input(
                "Old Balance Destination ($)",
                min_value=0.0,
                max_value=100000000.0,
                value=5000.0,
                step=100.0,
                format="%.2f"
            )
            
            new_balance_dest = st.number_input(
                "New Balance Destination ($)",
                min_value=0.0,
                max_value=100000000.0,
                value=old_balance_dest + amount_input,
                step=100.0,
                format="%.2f"
            )
            
            st.markdown("#### Additional Options")
            use_random_sample = st.checkbox("📋 Use Random Sample from Data", value=False)
            
            if use_random_sample:
                if st.button("🎲 Load Random Transaction"):
                    random_idx = np.random.randint(0, len(df))
                    st.session_state['random_transaction'] = random_idx
                    st.rerun()
        
        # Load random transaction if selected
        if use_random_sample and 'random_transaction' in st.session_state:
            random_idx = st.session_state['random_transaction']
            random_row = df.iloc[random_idx]
            
            # Denormalize and decode
            random_display = inverse_transform_data(pd.DataFrame([random_row]), scaler).iloc[0]
            random_display_decoded = decode_one_hot(pd.DataFrame([random_display])).iloc[0]
            
            st.info(f"📋 Loaded transaction from row {random_idx}")
            
            # Update values
            if 'type' in random_display_decoded.index:
                selected_type = random_display_decoded['type']
            if 'amount' in random_display_decoded.index:
                amount_input = random_display_decoded['amount']
            if 'oldbalanceOrg' in random_display_decoded.index:
                old_balance_orig = random_display_decoded['oldbalanceOrg']
            if 'newbalanceOrig' in random_display_decoded.index:
                new_balance_orig = random_display_decoded['newbalanceOrig']
            if 'oldbalanceDest' in random_display_decoded.index:
                old_balance_dest = random_display_decoded['oldbalanceDest']
            if 'newbalanceDest' in random_display_decoded.index:
                new_balance_dest = random_display_decoded['newbalanceDest']
        
        st.markdown("---")
        
        # Enhanced Predict button with animation
        st.markdown("<br>", unsafe_allow_html=True)
        predict_button = st.button("🔮 Analyze Transaction", type="primary", use_container_width=True)
        
        if predict_button:
            # Add a progress animation
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("⚙️ Preprocessing transaction data...")
            progress_bar.progress(25)
            
            with st.spinner("🤖 AI models are analyzing..."):
                # Create input dataframe
                input_data = {
                    'amount': amount_input,
                    'oldbalanceOrg': old_balance_orig,
                    'newbalanceOrig': new_balance_orig,
                    'oldbalanceDest': old_balance_dest,
                    'newbalanceDest': new_balance_dest
                }
                
                # One-hot encode transaction type
                for t_type in transaction_types:
                    col_name = f'type_{t_type}'
                    if col_name in X_inference.columns:
                        input_data[col_name] = 1 if t_type == selected_type else 0
                
                # Create DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Normalize using scaler if available
                if scaler is not None:
                    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                    available_cols = [col for col in numerical_cols if col in input_df.columns]
                    if available_cols:
                        input_df[available_cols] = scaler.transform(input_df[available_cols])
                
                # Ensure all required columns are present
                for col in X_inference.columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                # Reorder columns to match training data
                input_df = input_df[X_inference.columns]
                
                status_text.text("🔮 Running AI predictions...")
                progress_bar.progress(50)
                
                # Make predictions
                progress_bar.progress(75)
                status_text.text("📊 Generating results...")
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.markdown("---")
                st.markdown("""
                <div style='text-align: center; padding: 1.5rem; background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%); 
                            border-radius: 12px; margin-bottom: 1.5rem;'>
                    <h2 style='color: #e2e8f0; margin: 0;'>🎯 Prediction Results</h2>
                </div>
                """, unsafe_allow_html=True)
                
                result_col1, result_col2 = st.columns(2)
                
                # ANN Prediction with Enhanced Design
                if model_choice in ["ANN", "Compare Both"] and ann_model is not None:
                    with result_col1:
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(99, 102, 241, 0.1) 100%); 
                                    padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(59, 130, 246, 0.3);'>
                            <h3 style='color: #60a5fa; margin: 0 0 1rem 0; text-align: center;'>🧠 Neural Network</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        ann_prob = float(ann_model.predict(input_df, verbose=0)[0][0])
                        ann_pred = 1 if ann_prob > confidence_threshold else 0
                        
                        # Display probability with animation
                        st.metric("🎯 Fraud Probability", f"{ann_prob:.4f}", delta=f"{ann_prob*100:.2f}%")
                        
                        # Display prediction with enhanced design
                        if ann_pred == 1:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                                        padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;
                                        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.4);'>
                                <h2 style='color: white; margin: 0; font-size: 1.5rem;'>🚨 FRAUD DETECTED</h2>
                                <p style='color: #fecaca; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>High risk transaction identified</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(min(ann_prob, 1.0))
                        else:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); 
                                        padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;
                                        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.4);'>
                                <h2 style='color: white; margin: 0; font-size: 1.5rem;'>✅ LEGITIMATE</h2>
                                <p style='color: #bbf7d0; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Transaction appears safe</p>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(min(ann_prob, 1.0))
                        
                        st.caption(f"⚖️ Decision Threshold: {confidence_threshold}")
                
                # XGBoost Prediction with Enhanced Design
                if model_choice in ["XGBoost", "Compare Both"] and xgb_model is not None:
                    with result_col2:
                        st.markdown("""
                        <div style='background: linear-gradient(135deg, rgba(168, 85, 247, 0.1) 0%, rgba(147, 51, 234, 0.1) 100%); 
                                    padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(168, 85, 247, 0.3);'>
                            <h3 style='color: #c084fc; margin: 0 0 1rem 0; text-align: center;'>🌳 Gradient Boosting</h3>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        xgb_pred = int(xgb_model.predict(input_df)[0])
                        
                        # Get probability if available
                        try:
                            xgb_prob = float(xgb_model.predict_proba(input_df)[0][1])
                            st.metric("🎯 Fraud Probability", f"{xgb_prob:.4f}", delta=f"{xgb_prob*100:.2f}%")
                            st.progress(min(xgb_prob, 1.0))
                        except:
                            st.metric("🎯 Prediction", "Fraud" if xgb_pred == 1 else "Non-Fraud")
                        
                        # Display prediction with enhanced design
                        if xgb_pred == 1:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                                        padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;
                                        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.4);'>
                                <h2 style='color: white; margin: 0; font-size: 1.5rem;'>🚨 FRAUD DETECTED</h2>
                                <p style='color: #fecaca; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>High risk transaction identified</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div style='background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%); 
                                        padding: 1.5rem; border-radius: 12px; text-align: center; margin: 1rem 0;
                                        box-shadow: 0 8px 32px rgba(34, 197, 94, 0.4);'>
                                <h2 style='color: white; margin: 0; font-size: 1.5rem;'>✅ LEGITIMATE</h2>
                                <p style='color: #bbf7d0; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>Transaction appears safe</p>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Summary
                st.markdown("---")
                st.subheader("📊 Transaction Summary")
                
                summary_data = {
                    "Field": ["Transaction Type", "Amount", "Old Balance (Origin)", "New Balance (Origin)", 
                             "Old Balance (Dest)", "New Balance (Dest)"],
                    "Value": [
                        selected_type,
                        f"${amount_input:,.2f}",
                        f"${old_balance_orig:,.2f}",
                        f"${new_balance_orig:,.2f}",
                        f"${old_balance_dest:,.2f}",
                        f"${new_balance_dest:,.2f}"
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
                
                # Enhanced Model agreement
                if model_choice == "Compare Both" and ann_model is not None and xgb_model is not None:
                    st.markdown("---")
                    if ann_pred == xgb_pred:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(16, 185, 129, 0.2) 100%); 
                                    padding: 1.5rem; border-radius: 12px; border-left: 5px solid #22c55e; text-align: center;
                                    box-shadow: 0 4px 20px rgba(34, 197, 94, 0.2);'>
                            <h3 style='color: #22c55e; margin: 0 0 0.5rem 0;'>✅ Models in Agreement</h3>
                            <p style='color: #86efac; margin: 0; font-size: 1.1rem;'>
                                Both models predict: <strong>{'FRAUD 🚨' if ann_pred == 1 else 'LEGITIMATE ✓'}</strong>
                            </p>
                            <p style='color: #bbf7d0; margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
                                High confidence in prediction
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, rgba(234, 179, 8, 0.2) 0%, rgba(245, 158, 11, 0.2) 100%); 
                                    padding: 1.5rem; border-radius: 12px; border-left: 5px solid #eab308; text-align: center;
                                    box-shadow: 0 4px 20px rgba(234, 179, 8, 0.2);'>
                            <h3 style='color: #eab308; margin: 0 0 0.5rem 0;'>⚠️ Models Disagree</h3>
                            <p style='color: #fde047; margin: 0; font-size: 0.95rem;'>
                                🧠 ANN predicts: <strong>{'FRAUD' if ann_pred == 1 else 'LEGITIMATE'}</strong><br>
                                🌳 XGBoost predicts: <strong>{'FRAUD' if xgb_pred == 1 else 'LEGITIMATE'}</strong>
                            </p>
                            <p style='color: #fef08a; margin: 0.5rem 0 0 0; font-size: 0.85rem;'>
                                Consider additional verification
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    
    with tab4:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(168, 85, 247, 0.1) 0%, rgba(236, 72, 153, 0.1) 100%); 
                    padding: 2rem; border-radius: 15px; margin-bottom: 2rem; border-left: 5px solid #a855f7;'>
            <h2 style='color: #e2e8f0; margin: 0 0 0.5rem 0;'>📤 CSV Upload & Batch Processing</h2>
            <p style='color: #94a3b8; margin: 0; font-size: 0.95rem;'>
                Upload your dataset • Automatic preprocessing • Get comprehensive fraud analysis report
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### 📋 CSV Format Requirements
        Your CSV file should contain the following columns (can be normalized or not):
        - `amount`: Transaction amount
        - `oldbalanceOrg`: Origin account balance before transaction
        - `newbalanceOrig`: Origin account balance after transaction
        - `oldbalanceDest`: Destination account balance before transaction
        - `newbalanceDest`: Destination account balance after transaction
        - `type` or `type_*`: Transaction type (CASH_OUT or TRANSFER only)
        
        **Note**: If your data is not normalized, it will be automatically normalized using the scaler.
        """)
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key='csv_uploader')
        
        if uploaded_file is not None:
            try:
                # Read uploaded CSV
                uploaded_df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! ({len(uploaded_df):,} records)")
                
                # Display preview
                with st.expander("👁️ Preview Data (First 10 rows)", expanded=True):
                    st.dataframe(uploaded_df.head(10), use_container_width=True)
                
                st.markdown("---")
                
                # Data validation
                st.subheader("🔍 Data Validation")
                
                col_check1, col_check2 = st.columns(2)
                
                with col_check1:
                    st.write("**Detected Columns:**")
                    st.write(uploaded_df.columns.tolist())
                
                with col_check2:
                    st.write("**Data Info:**")
                    st.write(f"- Total Rows: {len(uploaded_df):,}")
                    st.write(f"- Total Columns: {len(uploaded_df.columns)}")
                    st.write(f"- Missing Values: {uploaded_df.isnull().sum().sum()}")
                
                # Check if data needs preprocessing
                needs_preprocessing = st.checkbox("📊 Data is in original scale (needs normalization and encoding)", value=True)
                
                st.markdown("---")
                
                # Process button
                if st.button("🔮 Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner("Processing predictions..."):
                        try:
                            # Store original data for display
                            uploaded_df_original = uploaded_df.copy()
                            
                            # Preprocessing if needed
                            if needs_preprocessing:
                                st.info("⚙️ Preprocessing data (normalization and encoding)...")
                                
                                # Check if 'type' column exists (categorical)
                                if 'type' in uploaded_df.columns:
                                    # One-hot encode type
                                    type_dummies = pd.get_dummies(uploaded_df['type'], prefix='type')
                                    uploaded_df = pd.concat([uploaded_df.drop('type', axis=1), type_dummies], axis=1)
                                
                                # Normalize numerical columns
                                if scaler is not None:
                                    numerical_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
                                    available_cols = [col for col in numerical_cols if col in uploaded_df.columns]
                                    
                                    if available_cols:
                                        uploaded_df[available_cols] = scaler.transform(uploaded_df[available_cols])
                                        st.success(f"✅ Normalized {len(available_cols)} numerical columns")
                            
                            # Remove 'isFraud' if exists (ground truth)
                            has_ground_truth = 'isFraud' in uploaded_df.columns
                            if has_ground_truth:
                                y_true = uploaded_df['isFraud'].copy()
                                uploaded_df = uploaded_df.drop('isFraud', axis=1)
                            
                            # Ensure all required columns are present
                            for col in X_inference.columns:
                                if col not in uploaded_df.columns:
                                    uploaded_df[col] = 0
                            
                            # Reorder columns to match training data
                            uploaded_df = uploaded_df[X_inference.columns]
                            
                            st.success("✅ Data preprocessing completed!")
                            
                            # Make predictions
                            results_df = uploaded_df_original.copy()
                            
                            if model_choice in ["ANN", "Compare Both"] and ann_model is not None:
                                st.info("🧠 Running ANN predictions...")
                                ann_predictions_prob = ann_model.predict(uploaded_df, verbose=0)
                                results_df['ANN_Probability'] = ann_predictions_prob.flatten()
                                results_df['ANN_Prediction'] = (ann_predictions_prob > confidence_threshold).astype(int).flatten()
                                st.success(f"✅ ANN predictions completed!")
                            
                            if model_choice in ["XGBoost", "Compare Both"] and xgb_model is not None:
                                st.info("🌳 Running XGBoost predictions...")
                                xgb_predictions = xgb_model.predict(uploaded_df)
                                results_df['XGBoost_Prediction'] = xgb_predictions
                                
                                # Try to get probabilities
                                try:
                                    xgb_predictions_prob = xgb_model.predict_proba(uploaded_df)[:, 1]
                                    results_df['XGBoost_Probability'] = xgb_predictions_prob
                                except:
                                    pass
                                
                                st.success(f"✅ XGBoost predictions completed!")
                            
                            # Add ground truth if available
                            if has_ground_truth:
                                results_df['True_Label'] = y_true.values
                            
                            st.markdown("---")
                            st.subheader("📊 Prediction Results Summary")
                            
                            # Summary metrics
                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                            
                            with summary_col1:
                                st.metric("📄 Total Records", f"{len(results_df):,}")
                            
                            with summary_col2:
                                if 'ANN_Prediction' in results_df.columns:
                                    ann_fraud_count = results_df['ANN_Prediction'].sum()
                                    ann_fraud_pct = (ann_fraud_count / len(results_df)) * 100
                                    st.metric("🧠 ANN Fraud Detected", f"{ann_fraud_count:,}", delta=f"{ann_fraud_pct:.1f}%")
                            
                            with summary_col3:
                                if 'XGBoost_Prediction' in results_df.columns:
                                    xgb_fraud_count = results_df['XGBoost_Prediction'].sum()
                                    xgb_fraud_pct = (xgb_fraud_count / len(results_df)) * 100
                                    st.metric("🌳 XGBoost Fraud Detected", f"{xgb_fraud_count:,}", delta=f"{xgb_fraud_pct:.1f}%")
                            
                            with summary_col4:
                                if has_ground_truth:
                                    true_fraud_count = results_df['True_Label'].sum()
                                    true_fraud_pct = (true_fraud_count / len(results_df)) * 100
                                    st.metric("✅ Actual Fraud", f"{true_fraud_count:,}", delta=f"{true_fraud_pct:.1f}%")
                            
                            # Model agreement analysis
                            if 'ANN_Prediction' in results_df.columns and 'XGBoost_Prediction' in results_df.columns:
                                st.markdown("---")
                                st.subheader("🤝 Model Agreement Analysis")
                                
                                agreement_col1, agreement_col2 = st.columns(2)
                                
                                with agreement_col1:
                                    agree_count = (results_df['ANN_Prediction'] == results_df['XGBoost_Prediction']).sum()
                                    agree_pct = (agree_count / len(results_df)) * 100
                                    st.metric("✅ Agreement", f"{agree_count:,}", delta=f"{agree_pct:.1f}%")
                                
                                with agreement_col2:
                                    disagree_count = (results_df['ANN_Prediction'] != results_df['XGBoost_Prediction']).sum()
                                    disagree_pct = (disagree_count / len(results_df)) * 100
                                    st.metric("⚠️ Disagreement", f"{disagree_count:,}", delta=f"{disagree_pct:.1f}%")
                            
                            # Performance metrics if ground truth available
                            if has_ground_truth:
                                st.markdown("---")
                                st.subheader("📈 Model Performance on Uploaded Data")
                                
                                perf_col1, perf_col2 = st.columns(2)
                                
                                if 'ANN_Prediction' in results_df.columns:
                                    with perf_col1:
                                        st.markdown("#### 🧠 ANN Metrics")
                                        ann_acc = accuracy_score(results_df['True_Label'], results_df['ANN_Prediction'])
                                        ann_prec = precision_score(results_df['True_Label'], results_df['ANN_Prediction'], zero_division=0)
                                        ann_rec = recall_score(results_df['True_Label'], results_df['ANN_Prediction'], zero_division=0)
                                        ann_f1 = f1_score(results_df['True_Label'], results_df['ANN_Prediction'], zero_division=0)
                                        
                                        metric_col1, metric_col2 = st.columns(2)
                                        with metric_col1:
                                            st.metric("Accuracy", f"{ann_acc:.4f}")
                                            st.metric("Precision", f"{ann_prec:.4f}")
                                        with metric_col2:
                                            st.metric("Recall", f"{ann_rec:.4f}")
                                            st.metric("F1 Score", f"{ann_f1:.4f}")
                                
                                if 'XGBoost_Prediction' in results_df.columns:
                                    with perf_col2:
                                        st.markdown("#### 🌳 XGBoost Metrics")
                                        xgb_acc = accuracy_score(results_df['True_Label'], results_df['XGBoost_Prediction'])
                                        xgb_prec = precision_score(results_df['True_Label'], results_df['XGBoost_Prediction'], zero_division=0)
                                        xgb_rec = recall_score(results_df['True_Label'], results_df['XGBoost_Prediction'], zero_division=0)
                                        xgb_f1 = f1_score(results_df['True_Label'], results_df['XGBoost_Prediction'], zero_division=0)
                                        
                                        metric_col1, metric_col2 = st.columns(2)
                                        with metric_col1:
                                            st.metric("Accuracy", f"{xgb_acc:.4f}")
                                            st.metric("Precision", f"{xgb_prec:.4f}")
                                        with metric_col2:
                                            st.metric("Recall", f"{xgb_rec:.4f}")
                                            st.metric("F1 Score", f"{xgb_f1:.4f}")
                            
                            # Display results table
                            st.markdown("---")
                            st.subheader("📋 Detailed Prediction Results")
                            
                            # Filter options
                            filter_col1, filter_col2 = st.columns(2)
                            
                            with filter_col1:
                                result_filter = st.selectbox(
                                    "Filter Results",
                                    ["All Records", "Predicted Fraud Only (ANN)", "Predicted Fraud Only (XGBoost)", 
                                     "Model Disagreement", "High Risk (Both Models)", "Non-Fraud (Both Models)"],
                                    key='upload_filter'
                                )
                            
                            with filter_col2:
                                show_limit = st.number_input("Show rows", min_value=10, max_value=len(results_df), value=min(100, len(results_df)), step=10)
                            
                            # Apply filter
                            filtered_results = results_df.copy()
                            
                            if result_filter == "Predicted Fraud Only (ANN)" and 'ANN_Prediction' in filtered_results.columns:
                                filtered_results = filtered_results[filtered_results['ANN_Prediction'] == 1]
                            elif result_filter == "Predicted Fraud Only (XGBoost)" and 'XGBoost_Prediction' in filtered_results.columns:
                                filtered_results = filtered_results[filtered_results['XGBoost_Prediction'] == 1]
                            elif result_filter == "Model Disagreement":
                                if 'ANN_Prediction' in filtered_results.columns and 'XGBoost_Prediction' in filtered_results.columns:
                                    filtered_results = filtered_results[filtered_results['ANN_Prediction'] != filtered_results['XGBoost_Prediction']]
                            elif result_filter == "High Risk (Both Models)":
                                if 'ANN_Prediction' in filtered_results.columns and 'XGBoost_Prediction' in filtered_results.columns:
                                    filtered_results = filtered_results[(filtered_results['ANN_Prediction'] == 1) & (filtered_results['XGBoost_Prediction'] == 1)]
                            elif result_filter == "Non-Fraud (Both Models)":
                                if 'ANN_Prediction' in filtered_results.columns and 'XGBoost_Prediction' in filtered_results.columns:
                                    filtered_results = filtered_results[(filtered_results['ANN_Prediction'] == 0) & (filtered_results['XGBoost_Prediction'] == 0)]
                            
                            st.write(f"**Showing {min(show_limit, len(filtered_results)):,} of {len(filtered_results):,} filtered records**")
                            st.dataframe(filtered_results.head(show_limit), use_container_width=True, height=400)
                            
                            # Visualizations
                            st.markdown("---")
                            st.subheader("📊 Prediction Visualizations")
                            
                            viz_col1, viz_col2 = st.columns(2)
                            
                            with viz_col1:
                                if 'ANN_Probability' in results_df.columns:
                                    fig = px.histogram(results_df, x='ANN_Probability', 
                                                     nbins=50,
                                                     title='ANN Prediction Probability Distribution',
                                                     color_discrete_sequence=['#2ecc71'])
                                    fig.add_vline(x=confidence_threshold, line_dash="dash", line_color="red", 
                                                annotation_text=f"Threshold: {confidence_threshold}")
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            with viz_col2:
                                if 'XGBoost_Probability' in results_df.columns:
                                    fig = px.histogram(results_df, x='XGBoost_Probability', 
                                                     nbins=50,
                                                     title='XGBoost Prediction Probability Distribution',
                                                     color_discrete_sequence=['#3498db'])
                                    st.plotly_chart(fig, use_container_width=True)
                            
                            # Download results
                            st.markdown("---")
                            st.subheader("💾 Download Results")
                            
                            download_col1, download_col2 = st.columns(2)
                            
                            with download_col1:
                                csv_all = results_df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download All Results (CSV)",
                                    data=csv_all,
                                    file_name="batch_prediction_results.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                            
                            with download_col2:
                                csv_filtered = filtered_results.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="📥 Download Filtered Results (CSV)",
                                    data=csv_filtered,
                                    file_name=f"batch_prediction_{result_filter.replace(' ', '_').lower()}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        except Exception as e:
                            st.error(f"❌ Error during prediction: {str(e)}")
                            st.exception(e)
            
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {str(e)}")
                st.exception(e)
        else:
            st.info("👆 Please upload a CSV file to begin batch prediction")
            
            # Show example
            st.markdown("---")
            st.subheader("📝 Example CSV Format")
            
            example_data = {
                'type': ['CASH_OUT', 'TRANSFER', 'CASH_OUT'],
                'amount': [1000.50, 5000.00, 2500.75],
                'oldbalanceOrg': [10000.00, 50000.00, 25000.00],
                'newbalanceOrig': [9000.50, 45000.00, 22500.25],
                'oldbalanceDest': [5000.00, 10000.00, 15000.00],
                'newbalanceDest': [6000.50, 15000.00, 17500.75]
            }
            
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df, use_container_width=True)
            
            # Download example
            csv_example = example_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Example CSV Template",
                data=csv_example,
                file_name="example_template.csv",
                mime="text/csv"
            )
    
    with tab5:
        st.markdown("""
        <div style='background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(14, 165, 233, 0.1) 100%); 
                    padding: 2rem; border-radius: 15px; margin-bottom: 2rem; border-left: 5px solid #06b6d4;'>
            <h2 style='color: #e2e8f0; margin: 0 0 0.5rem 0;'>📈 Advanced Data Analytics</h2>
            <p style='color: #94a3b8; margin: 0; font-size: 0.95rem;'>
                Interactive visualizations • Statistical insights • Pattern discovery • Correlation analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Use cached preprocessing
        with st.spinner("🔄 Preprocessing data for visualization..."):
            df_viz_denorm = preprocess_data_for_visualization(df, scaler)
        
        # Add sections with expandable visualizations
        st.markdown("### 📊 Select Analysis Sections")
        st.info("💡 Expand sections below to view specific visualizations and reduce loading time")
        
        # Section 1: Basic Distribution (Always visible - lightweight)
        with st.expander("📊 Basic Distributions", expanded=True):
            viz_row1_col1, viz_row1_col2 = st.columns(2)
            
            with viz_row1_col1:
                # Transaction Type Distribution
                if 'type' in df_viz_denorm.columns:
                    type_counts = df_viz_denorm['type'].value_counts()
                    fig = px.bar(x=type_counts.index, y=type_counts.values,
                               title='Transaction Type Distribution',
                               labels={'x': 'Transaction Type', 'y': 'Count'},
                               color=type_counts.values,
                               color_continuous_scale='Viridis')
                    fig.update_layout(showlegend=False, height=350)
                    st.plotly_chart(fig, use_container_width=True)
            
            with viz_row1_col2:
                # Fraud Distribution
                if 'isFraud' in df_viz_denorm.columns:
                    fraud_dist = df_viz_denorm['isFraud'].value_counts()
                    fig = go.Figure(data=[go.Pie(
                        labels=['Non-Fraud', 'Fraud'],
                        values=fraud_dist.values,
                        hole=.4,
                        marker_colors=['#2ecc71', '#e74c3c']
                    )])
                    fig.update_layout(title='Fraud vs Non-Fraud Distribution', height=350)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Section 2: Fraud by Transaction Type
        with st.expander("💰 Fraud Analysis by Transaction Type", expanded=False):
            if 'type' in df_viz_denorm.columns and 'isFraud' in df_viz_denorm.columns:
                fraud_by_type = df_viz_denorm.groupby('type')['isFraud'].agg(['sum', 'count', 'mean'])
                fraud_by_type.columns = ['Fraud Count', 'Total Transactions', 'Fraud Rate']
                fraud_by_type['Fraud Rate'] = fraud_by_type['Fraud Rate'] * 100
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('Fraud Count by Type', 'Fraud Rate by Type'),
                    specs=[[{"type": "bar"}, {"type": "bar"}]]
                )
                
                fig.add_trace(
                    go.Bar(x=fraud_by_type.index, y=fraud_by_type['Fraud Count'], 
                          name='Fraud Count', marker_color='#e74c3c'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=fraud_by_type.index, y=fraud_by_type['Fraud Rate'],
                          name='Fraud Rate (%)', marker_color='#3498db'),
                    row=1, col=2
                )
                
                fig.update_xaxes(title_text="Transaction Type", row=1, col=1)
                fig.update_xaxes(title_text="Transaction Type", row=1, col=2)
                fig.update_yaxes(title_text="Count", row=1, col=1)
                fig.update_yaxes(title_text="Percentage (%)", row=1, col=2)
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        # Section 3: Amount Analysis
        with st.expander("💵 Transaction Amount Analysis", expanded=False):
            if 'amount' in df_viz_denorm.columns:
                amount_col1, amount_col2 = st.columns(2)
                
                with amount_col1:
                    # Histogram dengan data denormalized
                    fig = px.histogram(df_viz_denorm, x='amount', 
                                     title='Distribution of Transaction Amounts (Original Scale)',
                                     nbins=50,
                                     color_discrete_sequence=['#9b59b6'])
                    fig.update_xaxes(title='Amount ($)')
                    fig.update_yaxes(title='Frequency')
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                with amount_col2:
                    if 'isFraud' in df_viz_denorm.columns:
                        # Box plot dengan data denormalized
                        fig = px.box(df_viz_denorm, x='isFraud', y='amount',
                                   title='Transaction Amount by Fraud Status (Original Scale)',
                                   labels={'isFraud': 'Is Fraud', 'amount': 'Amount ($)'},
                                   color='isFraud',
                                   color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
        
        # Section 4: Balance Analysis
        with st.expander("💰 Balance Analysis", expanded=False):
            balance_col1, balance_col2 = st.columns(2)
            
            with balance_col1:
                if 'oldbalanceOrg' in df_viz_denorm.columns and 'isFraud' in df_viz_denorm.columns:
                    fig = px.box(df_viz_denorm, x='isFraud', y='oldbalanceOrg',
                               title='Origin Old Balance by Fraud Status',
                               labels={'isFraud': 'Is Fraud', 'oldbalanceOrg': 'Old Balance Origin ($)'},
                               color='isFraud',
                               color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            
            with balance_col2:
                if 'oldbalanceDest' in df_viz_denorm.columns and 'isFraud' in df_viz_denorm.columns:
                    fig = px.box(df_viz_denorm, x='isFraud', y='oldbalanceDest',
                               title='Destination Old Balance by Fraud Status',
                               labels={'isFraud': 'Is Fraud', 'oldbalanceDest': 'Old Balance Dest ($)'},
                               color='isFraud',
                               color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Section 5: Statistical Summary (Lightweight - always expanded)
        with st.expander("📊 Statistical Summary", expanded=True):
            numeric_cols_denorm = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            available_cols = [col for col in numeric_cols_denorm if col in df_viz_denorm.columns]
            
            if available_cols:
                summary_stats = df_viz_denorm[available_cols].describe()
                
                # Format as currency
                for col in summary_stats.columns:
                    summary_stats[col] = summary_stats[col].apply(lambda x: f"${x:,.2f}")
                
                st.dataframe(summary_stats, use_container_width=True)
        
        # Section 6: Feature Correlations (Heavy - collapsed by default)
        with st.expander("🔗 Feature Correlations (Click to load)", expanded=False):
            st.info("⚠️ This visualization may take a moment to load for large datasets")
            
            numeric_cols_for_corr = df_viz_denorm.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols_for_corr) > 1:
                # Select features for correlation
                selected_features = st.multiselect(
                    "Select features to analyze",
                    numeric_cols_for_corr,
                    default=numeric_cols_for_corr[:min(6, len(numeric_cols_for_corr))],
                    help="Select fewer features for faster rendering"
                )
                
                if selected_features and len(selected_features) > 1:
                    with st.spinner("🔄 Calculating correlations..."):
                        corr_matrix = df_viz_denorm[selected_features].corr()
                        fig = px.imshow(corr_matrix,
                                      labels=dict(color="Correlation"),
                                      x=corr_matrix.columns,
                                      y=corr_matrix.columns,
                                      color_continuous_scale='RdBu_r',
                                      aspect="auto",
                                      text_auto='.2f')
                        fig.update_layout(title="Feature Correlation Heatmap (Original Scale)", height=600)
                        st.plotly_chart(fig, use_container_width=True)
                elif selected_features and len(selected_features) == 1:
                    st.warning("Please select at least 2 features to show correlations")
        
        # Section 7: Fraud vs Amount by Type
        with st.expander("💸 Fraud Amount Analysis by Transaction Type", expanded=False):
            if 'type' in df_viz_denorm.columns and 'amount' in df_viz_denorm.columns and 'isFraud' in df_viz_denorm.columns:
                # Average amount by type and fraud status
                avg_amount = df_viz_denorm.groupby(['type', 'isFraud'])['amount'].mean().reset_index()
                avg_amount['isFraud'] = avg_amount['isFraud'].map({0: 'Non-Fraud', 1: 'Fraud'})
                
                fig = px.bar(avg_amount, x='type', y='amount', color='isFraud',
                            title='Average Transaction Amount by Type and Fraud Status',
                            labels={'amount': 'Average Amount ($)', 'type': 'Transaction Type'},
                            color_discrete_map={'Non-Fraud': '#2ecc71', 'Fraud': '#e74c3c'},
                            barmode='group')
                fig.update_yaxes(tickprefix="$", tickformat=",.0f")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Performance tip
        st.markdown("---")
        st.markdown("""
        <div style='background: rgba(59, 130, 246, 0.1); padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;'>
            <h4 style='color: #60a5fa; margin: 0 0 0.5rem 0;'>💡 Performance Tips</h4>
            <ul style='color: #94a3b8; margin: 0; font-size: 0.9rem;'>
                <li>Visualizations are loaded on-demand when you expand sections</li>
                <li>Data preprocessing is cached for faster subsequent loads</li>
                <li>Select fewer features in correlation analysis for faster rendering</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with tab6:
        st.header("ℹ️ About This Dashboard")
        
        st.markdown("""
        ### 🎯 Purpose
        This dashboard provides comprehensive evaluation and inference capabilities for fraud detection models.
        
        ### 🔧 Features
        - **Model Evaluation**: Compare ANN and XGBoost model performance with detailed metrics
        - **Inference**: Run predictions on new data with configurable parameters
        - **Manual Prediction**: Input custom transaction details for real-time fraud prediction
        - **Upload CSV**: Batch prediction on uploaded CSV files with automatic preprocessing
        - **Data Analysis**: Visualize data distributions and patterns
        - **Data Denormalization**: View inference results in original scale
        
        ### 📊 Models
        - **ANN (Artificial Neural Network)**: Deep learning model with customizable confidence threshold
        - **XGBoost**: Gradient boosting model for high-performance classification
        
        ### 📁 Required Files
        - `data.csv`: Preprocessed data (normalized with RobustScaler, one-hot encoded)
        - `ann_model.keras`: Trained ANN model
        - `xgboost_model.pkl`: Trained XGBoost model
        - `robust_scaler.pkl`: Fitted scaler for denormalization (optional)
        
        ### 🚀 Usage Tips
        1. Select a model from the sidebar
        2. Adjust inference parameters (sample size, threshold)
        3. Navigate through tabs to explore different features
        4. Download results for further analysis
        
        ### 📈 Metrics Explained
        - **Accuracy**: Overall correctness of predictions
        - **Precision**: Proportion of correct fraud predictions among all fraud predictions
        - **Recall**: Proportion of actual frauds correctly identified
        - **F1 Score**: Harmonic mean of precision and recall
        
        ### 👨‍💻 Developer
        Built with ❤️ using Streamlit, TensorFlow, XGBoost, and Plotly
        """)
        
        st.markdown("---")
        st.info("💡 **Tip**: Use the sidebar to configure model selection and inference parameters!")

else:
    st.error("❌ Required files not found!")
    st.markdown("""
    ### Please ensure the following files exist in the application directory:
    - `data.csv` - Preprocessed data for inference
    - `ann_model.keras` - Trained ANN model
    - `xgboost_model.pkl` - Trained XGBoost model
    - `robust_scaler.pkl` - Scaler for denormalization (optional)
    
    ### Current Directory:
    """)
    st.code(os.getcwd())
    
    st.markdown("### Files in Current Directory:")
    try:
        files = os.listdir('.')
        for file in files:
            st.write(f"- {file}")
    except Exception as e:
        st.error(f"Error listing files: {e}")

# Enhanced Footer
st.markdown("---")
st.markdown("""
<div style='background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(30, 41, 59, 0.9) 100%); 
            padding: 2rem; border-radius: 15px; margin-top: 3rem; text-align: center;'>
    <h3 style='color: #94a3b8; margin: 0 0 1rem 0; font-size: 1.2rem;'>
        Built with Rzeror
    </h3>
    <div style='display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap; margin-bottom: 1rem;'>
        <div style='color: #64748b;'>
            <span style='color: #3b82f6; font-weight: bold;'>⚡</span> Powered by TensorFlow & XGBoost
        </div>
        <div style='color: #64748b;'>
            <span style='color: #8b5cf6; font-weight: bold;'>📊</span> Visualized with Plotly
        </div>
        <div style='color: #64748b;'>
            <span style='color: #ef4444; font-weight: bold;'>🎨</span> UI/UX with Streamlit
        </div>
    </div>
    <div style='color: #475569; font-size: 0.85rem; margin-top: 1rem;'>
        © 2025 Fraud Detection Dashboard • <a href='https://github.com/RZeor' style='color: #667eea; text-decoration: none;'>@RZeor</a>
    </div>
    <div style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(100, 116, 139, 0.2);'>
        <p style='color: #64748b; font-size: 0.8rem; margin: 0;'>
            🔒 Secure • 🚀 Fast • 🎯 Accurate
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
