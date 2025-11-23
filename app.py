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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔍 Fraud Detection Model Evaluation Dashboard</h1>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["ANN", "XGBoost", "Compare Both"],
    index=2
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
        df = pd.read_csv('data.csv')
        st.sidebar.success("✅ Data loaded")
        return df
    except Exception as e:
        st.sidebar.error(f"❌ Error loading data: {e}")
        return None

# Load evaluation data
@st.cache_data
def load_eval_data():
    try:
        df_eval = pd.read_csv('data_eval.csv')
        st.sidebar.success("✅ Evaluation data loaded")
        return df_eval
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error loading evaluation data: {e}")
        return None

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
    
    # Sidebar - Data Info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Data Information")
    st.sidebar.write(f"**Total Records:** {len(df):,}")
    
    if 'isFraud' in df.columns:
        fraud_count = df['isFraud'].sum()
        non_fraud_count = len(df) - fraud_count
        fraud_percentage = (fraud_count / len(df)) * 100
        
        st.sidebar.write(f"**Fraud Cases:** {fraud_count:,} ({fraud_percentage:.2f}%)")
        st.sidebar.write(f"**Non-Fraud Cases:** {non_fraud_count:,} ({100-fraud_percentage:.2f}%)")
    
    # Sidebar - Sample Size
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎯 Inference Settings")
    sample_size = st.sidebar.slider(
        "Sample Size",
        min_value=100,
        max_value=min(10000, len(df)),
        value=min(1000, len(df)),
        step=100
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold (ANN)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05
    )
    
    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Model Evaluation", "🔮 Inference", "🎯 Manual Prediction", "📈 Data Analysis", "ℹ️ About"])
    
    with tab1:
        st.header("Model Evaluation Metrics")
        st.info("📊 Using data_eval.csv for model evaluation")
        
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
                    
                    with st.spinner("Evaluating ANN model on data_eval.csv..."):
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
                    
                    with st.spinner("Evaluating XGBoost model on data_eval.csv..."):
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
        st.header("Model Inference on New Data")
        st.info("🔮 Using data.csv for inference (will be denormalized and decoded for display)")
        
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
            with st.spinner("Running ANN inference..."):
                ann_pred_prob = ann_model.predict(X_inference_sample, verbose=0)
                results_normalized['ANN_Probability'] = ann_pred_prob.flatten()
                results_normalized['ANN_Prediction'] = (ann_pred_prob > confidence_threshold).astype(int).flatten()
        
        if model_choice in ["XGBoost", "Compare Both"] and xgb_model is not None:
            with st.spinner("Running XGBoost inference..."):
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
        st.header("🎯 Manual Prediction")
        st.info("💡 Input transaction details to get fraud prediction from both models")
        
        # Get sample data to understand the structure
        sample_data = X_inference.iloc[0] if len(X_inference) > 0 else None
        
        # Create two columns for input form
        st.subheader("📝 Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Transaction Type")
            # Transaction type selection
            transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
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
        
        # Predict button
        if st.button("🔮 Predict Fraud", type="primary", use_container_width=True):
            with st.spinner("Processing prediction..."):
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
                
                # Make predictions
                st.markdown("---")
                st.subheader("🎯 Prediction Results")
                
                result_col1, result_col2 = st.columns(2)
                
                # ANN Prediction
                if model_choice in ["ANN", "Compare Both"] and ann_model is not None:
                    with result_col1:
                        st.markdown("### 🧠 ANN Model")
                        ann_prob = float(ann_model.predict(input_df, verbose=0)[0][0])
                        ann_pred = 1 if ann_prob > confidence_threshold else 0
                        
                        # Display probability
                        st.metric("Fraud Probability", f"{ann_prob:.4f}", delta=f"{ann_prob*100:.2f}%")
                        
                        # Display prediction with color
                        if ann_pred == 1:
                            st.error("🚨 **FRAUD DETECTED**")
                            st.progress(min(ann_prob, 1.0))
                        else:
                            st.success("✅ **LEGITIMATE TRANSACTION**")
                            st.progress(min(ann_prob, 1.0))
                        
                        st.caption(f"Threshold: {confidence_threshold}")
                
                # XGBoost Prediction
                if model_choice in ["XGBoost", "Compare Both"] and xgb_model is not None:
                    with result_col2:
                        st.markdown("### 🌳 XGBoost Model")
                        xgb_pred = int(xgb_model.predict(input_df)[0])
                        
                        # Get probability if available
                        try:
                            xgb_prob = float(xgb_model.predict_proba(input_df)[0][1])
                            st.metric("Fraud Probability", f"{xgb_prob:.4f}", delta=f"{xgb_prob*100:.2f}%")
                            st.progress(min(xgb_prob, 1.0))
                        except:
                            st.metric("Prediction", "Fraud" if xgb_pred == 1 else "Non-Fraud")
                        
                        # Display prediction with color
                        if xgb_pred == 1:
                            st.error("🚨 **FRAUD DETECTED**")
                        else:
                            st.success("✅ **LEGITIMATE TRANSACTION**")
                
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
                
                # Model agreement
                if model_choice == "Compare Both" and ann_model is not None and xgb_model is not None:
                    st.markdown("---")
                    if ann_pred == xgb_pred:
                        st.success(f"✅ **Models Agree**: Both predict {'FRAUD' if ann_pred == 1 else 'NON-FRAUD'}")
                    else:
                        st.warning(f"⚠️ **Models Disagree**: ANN predicts {'FRAUD' if ann_pred == 1 else 'NON-FRAUD'}, XGBoost predicts {'FRAUD' if xgb_pred == 1 else 'NON-FRAUD'}")
    
    with tab4:
        st.header("Data Analysis & Visualization")
        st.info("📈 Using data.csv with denormalization and decoding for analysis")
        
        # Decode data for visualization - gunakan data yang sudah di-decode dan di-denormalized
        df_viz = decode_one_hot(df.copy())
        df_viz_denorm = inverse_transform_data(df_viz.copy(), scaler)
        
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
                fig.update_layout(showlegend=False)
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
                fig.update_layout(title='Fraud vs Non-Fraud Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        # Fraud by Transaction Type
        if 'type' in df_viz_denorm.columns and 'isFraud' in df_viz_denorm.columns:
            st.markdown("---")
            st.subheader("💰 Fraud Analysis by Transaction Type")
            
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
        
        # Amount distribution - GUNAKAN DATA DENORMALIZED
        st.markdown("---")
        st.subheader("💵 Transaction Amount Analysis (Original Scale)")
        
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
                st.plotly_chart(fig, use_container_width=True)
            
            with amount_col2:
                if 'isFraud' in df_viz_denorm.columns:
                    # Box plot dengan data denormalized
                    fig = px.box(df_viz_denorm, x='isFraud', y='amount',
                               title='Transaction Amount by Fraud Status (Original Scale)',
                               labels={'isFraud': 'Is Fraud', 'amount': 'Amount ($)'},
                               color='isFraud',
                               color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                    st.plotly_chart(fig, use_container_width=True)
        
        # Balance Analysis - GUNAKAN DATA DENORMALIZED
        st.markdown("---")
        st.subheader("💰 Balance Analysis (Original Scale)")
        
        balance_col1, balance_col2 = st.columns(2)
        
        with balance_col1:
            if 'oldbalanceOrg' in df_viz_denorm.columns and 'isFraud' in df_viz_denorm.columns:
                fig = px.box(df_viz_denorm, x='isFraud', y='oldbalanceOrg',
                           title='Origin Old Balance by Fraud Status',
                           labels={'isFraud': 'Is Fraud', 'oldbalanceOrg': 'Old Balance Origin ($)'},
                           color='isFraud',
                           color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
        
        with balance_col2:
            if 'oldbalanceDest' in df_viz_denorm.columns and 'isFraud' in df_viz_denorm.columns:
                fig = px.box(df_viz_denorm, x='isFraud', y='oldbalanceDest',
                           title='Destination Old Balance by Fraud Status',
                           labels={'isFraud': 'Is Fraud', 'oldbalanceDest': 'Old Balance Dest ($)'},
                           color='isFraud',
                           color_discrete_map={0: '#2ecc71', 1: '#e74c3c'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Statistics Summary - GUNAKAN DATA DENORMALIZED
        st.markdown("---")
        st.subheader("📊 Statistical Summary (Original Scale)")
        
        numeric_cols_denorm = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
        available_cols = [col for col in numeric_cols_denorm if col in df_viz_denorm.columns]
        
        if available_cols:
            summary_stats = df_viz_denorm[available_cols].describe()
            
            # Format as currency
            for col in summary_stats.columns:
                summary_stats[col] = summary_stats[col].apply(lambda x: f"${x:,.2f}")
            
            st.dataframe(summary_stats, use_container_width=True)
        
        # Feature correlations - GUNAKAN DATA DENORMALIZED
        st.markdown("---")
        st.subheader("🔗 Feature Correlations (Original Scale)")
        
        numeric_cols_for_corr = df_viz_denorm.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols_for_corr) > 1:
            # Select features for correlation
            selected_features = st.multiselect(
                "Select features to analyze",
                numeric_cols_for_corr,
                default=numeric_cols_for_corr[:min(8, len(numeric_cols_for_corr))]
            )
            
            if selected_features:
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
        
        # Fraud vs Amount Analysis by Type - GUNAKAN DATA DENORMALIZED
        if 'type' in df_viz_denorm.columns and 'amount' in df_viz_denorm.columns and 'isFraud' in df_viz_denorm.columns:
            st.markdown("---")
            st.subheader("💸 Fraud Amount Analysis by Transaction Type")
            
            # Average amount by type and fraud status
            avg_amount = df_viz_denorm.groupby(['type', 'isFraud'])['amount'].mean().reset_index()
            avg_amount['isFraud'] = avg_amount['isFraud'].map({0: 'Non-Fraud', 1: 'Fraud'})
            
            fig = px.bar(avg_amount, x='type', y='amount', color='isFraud',
                        title='Average Transaction Amount by Type and Fraud Status',
                        labels={'amount': 'Average Amount ($)', 'type': 'Transaction Type'},
                        color_discrete_map={'Non-Fraud': '#2ecc71', 'Fraud': '#e74c3c'},
                        barmode='group')
            fig.update_yaxes(tickprefix="$", tickformat=",.0f")
            st.plotly_chart(fig, use_container_width=True)
        
    with tab5:
        st.header("ℹ️ About This Dashboard")
        
        st.markdown("""
        ### 🎯 Purpose
        This dashboard provides comprehensive evaluation and inference capabilities for fraud detection models.
        
        ### 🔧 Features
        - **Model Evaluation**: Compare ANN and XGBoost model performance with detailed metrics
        - **Inference**: Run predictions on new data with configurable parameters
        - **Manual Prediction**: Input custom transaction details for real-time fraud prediction
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
