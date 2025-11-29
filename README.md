# 🔍 Fraud Detection Dashboard: ANN vs XGBoost

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

A comprehensive interactive dashboard for fraud detection that compares **Artificial Neural Network (ANN)** and **XGBoost** models. Built with Streamlit, this application provides real-time fraud prediction, batch processing, and detailed model evaluation with stunning visualizations.

## 🌐 Live Demo

**🚀 [Try the Live Dashboard Here](https://dashboardannvsxgboost.streamlit.app/)**

Experience the full functionality of the fraud detection system without any installation!

## 🌟 Features

### 📊 **Model Evaluation**
- Side-by-side comparison of ANN and XGBoost performance
- Detailed metrics: Accuracy, Precision, Recall, F1 Score
- Interactive confusion matrices
- Comprehensive classification reports
- Visual performance comparison charts

### 🔮 **Batch Inference**
- Run predictions on sample data from `data.csv`
- Configurable sample size (100-10,000 records)
- Adjustable confidence threshold for ANN
- Automatic denormalization to original scale
- Multiple filtering options:
  - Predicted fraud only (by model)
  - Model disagreement analysis
  - True/False positives
- Download results as CSV

### 🎯 **Manual Prediction**
- Input individual transaction details through interactive form
- Real-time fraud prediction from both models
- Probability scores with visual progress bars
- Model agreement/disagreement indicator
- Load random sample from dataset option
- Transaction summary display

### 📤 **Upload CSV for Batch Prediction**
- Upload custom CSV files for bulk predictions
- Automatic preprocessing:
  - Data normalization using RobustScaler
  - One-hot encoding for categorical variables
  - Column matching with training data
- Comprehensive results summary:
  - Fraud detection statistics
  - Model agreement analysis
  - Performance metrics (if ground truth available)
- Multiple download options:
  - All results
  - Filtered results
- Visual probability distributions
- Example CSV template provided

### 📈 **Data Analysis & Visualization**
- Transaction type distribution with interactive charts
- Fraud vs non-fraud analysis with donut charts
- Amount and balance analysis in original scale
- Statistical summaries with currency formatting
- Correlation heatmaps with customizable features
- Fraud analysis by transaction type
- **Performance optimized** with lazy loading and expandable sections
- **Smart caching** for instant reloads
- Interactive Plotly visualizations with smooth animations

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/RZeor/dashboard_ANN_vs_XGBoost.git
cd dashboard_ANN_vs_XGBoost
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv .env
```

3. **Activate the virtual environment**
   - Windows:
   ```bash
   .env\Scripts\activate
   ```
   - macOS/Linux:
   ```bash
   source .env/bin/activate
   ```

4. **Install required packages**
```bash
pip install -r requirements.txt
```

### Required Files

Ensure the following files are in the project directory:

```
dashboard_ANN_vs_XGBoost/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── data.csv                        # Preprocessed data for inference
├── data_eval.csv                   # Data for model evaluation
├── README.md                       # This file
└── model/
    ├── ANN/
    │   └── ann_model.keras         # Trained ANN model
    ├── XGBoost/
    │   └── xgboost_model.pkl       # Trained XGBoost model
    └── robust/
        └── robust_scaler.pkl       # Fitted RobustScaler
```

### Running the Application

**Option 1: Use the Live Demo (Recommended)**

Visit the deployed application: **[https://dashboardannvsxgboost.streamlit.app/](https://dashboardannvsxgboost.streamlit.app/)**

**Option 2: Run Locally**

```bash
streamlit run app.py
```

The dashboard will open automatically in your default web browser at `http://localhost:8501`

## 📊 Data Format

### Input Data Requirements

Your CSV files should contain the following columns:

| Column | Description | Type |
|--------|-------------|------|
| `amount` | Transaction amount | Float |
| `oldbalanceOrg` | Origin account balance before transaction | Float |
| `newbalanceOrig` | Origin account balance after transaction | Float |
| `oldbalanceDest` | Destination account balance before transaction | Float |
| `newbalanceDest` | Destination account balance after transaction | Float |
| `type` | Transaction type (CASH_OUT or TRANSFER) | String |
| `isFraud` | Fraud label (optional, for evaluation) | Binary (0/1) |

### Transaction Types

The models are trained to detect fraud in these transaction types:
- **CASH_OUT**: Cash withdrawal transactions
- **TRANSFER**: Money transfer transactions

> **Note**: Only CASH_OUT and TRANSFER transactions are analyzed for fraud, as these are the primary vectors for fraudulent activity in financial systems.

### Data Preprocessing

If your data is in original scale (not normalized):
1. ✅ The dashboard will automatically normalize numerical features using RobustScaler
2. ✅ Categorical variables will be one-hot encoded
3. ✅ Missing columns will be filled with default values

## 🎯 Usage Guide

### 1. Model Evaluation Tab
- Select model(s) to evaluate from sidebar
- View comprehensive metrics on evaluation dataset
- Compare model performance side-by-side
- Analyze confusion matrices and classification reports

### 2. Inference Tab
- Adjust sample size and confidence threshold in sidebar
- Choose random seed for reproducibility
- Filter results by prediction type
- Sort by ANN probability
- Download filtered results

### 3. Manual Prediction Tab
- Select transaction type (CASH_OUT or TRANSFER)
- Input transaction amount and balance details
- Click "Predict Fraud" to get real-time predictions
- View probability scores from both models
- Check model agreement status

### 4. Upload CSV Tab
- Upload your custom CSV file
- Preview data and check validation
- Toggle preprocessing option if needed
- Click "Run Batch Prediction"
- Analyze results with interactive filters
- Download predictions as CSV

### 5. Data Analysis Tab
- **Performance optimized**: Expand only the sections you need
- Explore data distributions (lightweight, always visible)
- Analyze fraud patterns by transaction type
- View correlation heatmaps (lazy loaded on-demand)
- Examine statistical summaries with currency formatting
- Interactive visualizations with Plotly
- Cached preprocessing for instant reloads

## 🛠️ Technology Stack

- **Frontend**: Streamlit with custom CSS styling
- **UI/UX Design**: 
  - Custom gradient themes (purple/indigo palette)
  - Glass morphism effects
  - Smooth CSS animations
  - Inter font family
- **Machine Learning**: 
  - TensorFlow/Keras (ANN)
  - XGBoost (Gradient Boosting)
- **Data Processing**: Pandas, NumPy
- **Preprocessing**: Scikit-learn (RobustScaler)
- **Visualization**: Plotly Express, Plotly Graph Objects
- **Performance**: Streamlit caching (@st.cache_data, @st.cache_resource)
- **Metrics**: Scikit-learn metrics

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
tensorflow>=2.13.0
xgboost>=2.0.0
scikit-learn>=1.3.0
plotly>=5.17.0
```

## 🎨 Key Highlights

- **Modern UI/UX**: Beautiful gradient design with smooth animations and hover effects
- **Interactive Dashboard**: User-friendly interface with intuitive navigation
- **Real-time Predictions**: Instant fraud detection on manual inputs with progress indicators
- **Batch Processing**: Handle large datasets efficiently with smart caching
- **Model Comparison**: Direct comparison between deep learning and ensemble methods
- **Automatic Preprocessing**: No manual data preparation needed
- **Beautiful Visualizations**: Interactive charts and plots with Plotly
- **Performance Optimized**: Lazy loading and cached preprocessing for faster experience
- **Export Capabilities**: Download predictions and filtered results
- **Responsive Design**: Dark theme with purple gradient and glass morphism effects

## 📈 Model Performance Metrics

The dashboard tracks and displays:
- **Accuracy**: Overall prediction correctness
- **Precision**: Ratio of correct fraud predictions
- **Recall**: Ability to identify all fraud cases
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual breakdown of predictions with interactive heatmaps
- **Model Agreement**: Shows when both models agree/disagree on predictions

## ⚡ Performance Features

- **Smart Caching**: Data preprocessing and model loading are cached for instant access
- **Lazy Loading**: Visualizations load on-demand to reduce initial loading time
- **Expandable Sections**: Control which analyses to view in the Data Analysis tab
- **Progress Indicators**: Real-time feedback during model predictions
- **Optimized Rendering**: Reduced chart heights and efficient data structures

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**RZeor**
- GitHub: [@RZeor](https://github.com/RZeor)

## 🎨 UI/UX Features

- **Modern Design**: Purple gradient theme with dark mode aesthetic
- **Smooth Animations**: FadeIn, slideIn, and hover effects throughout
- **Glass Morphism**: Translucent cards with backdrop blur effects
- **Interactive Elements**: 
  - Hover effects on metrics and buttons
  - Animated tabs with gradient backgrounds
  - Custom styled input fields with focus states
  - Progress bars with gradient fills
- **Status Indicators**: Real-time system status badges
- **Enhanced Sidebar**: Organized control panel with visual indicators
- **Custom Scrollbar**: Themed scrollbar matching the dashboard design
- **Responsive Layout**: Optimized for different screen sizes

## 🙏 Acknowledgments

- Built with ❤️ using Streamlit
- Powered by TensorFlow and XGBoost
- Visualizations by Plotly
- UI/UX inspired by modern design principles

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Issues](https://github.com/RZeor/dashboard_ANN_vs_XGBoost/issues) page
2. Create a new issue with detailed description
3. Include error messages and screenshots if applicable

---

**⭐ Star this repository if you find it helpful!**
