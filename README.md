# 🚗 Car Price Prediction ML Project

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![Flask](https://img.shields.io/badge/Flask-2.0+-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen)
![Model Comparison](https://img.shields.io/badge/Model%20Comparison-Advanced-purple)

**An end-to-end Machine Learning pipeline for predicting car prices with hyperparameter optimization, advanced model comparison with statistical tests, comprehensive visualizations, and Docker containerization.**

[Getting Started](#-quick-start) • [Features](#-features) • [Model Comparison](#-advanced-model-comparison) • [Architecture](#-architecture) • [Results](#-results--metrics)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Advanced Model Comparison](#-advanced-model-comparison)
- [Tech Stack](#-tech-stack)
- [Architecture](#-architecture)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)
- [Results & Metrics](#-results--metrics)
- [Key Decisions](#-key-decisions)
- [Production Deployment](#-production-deployment)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 Overview

A complete machine learning application that predicts car prices based on vehicle attributes (company, engine type, transmission, color, body style, dealer region). The project demonstrates best practices in ML engineering, including:

- ✅ **Automated ML Pipeline**: End-to-end data processing and model training
- ✅ **Hyperparameter Optimization**: Optuna-based parameter tuning with patience logic
- ✅ **Advanced Model Comparison**: 4 statistical tests, 10+ visualizations, HTML reports
- ✅ **Production-Ready**: Dockerized, scalable, and reproducible
- ✅ **Modern UI**: Interactive web interface with real-time predictions
- ✅ **Full Dataset Retraining**: Uses best parameters to train on complete dataset

**Problem Statement**: Given car features, predict the selling price.
**Solution**: Multiple models compared with statistical tests. CatBoost achieves ~14.65% MAPE and 0.86 R² score.

---

## ✨ Features

### 🤖 ML Pipeline
- **Data Ingestion**: Automatic CSV loading with train/test split (70/30)
- **Data Transformation**: OneHot encoding, missing value imputation, standardization
- **Model Training**: 4 models (RandomForest, GradientBoosting, XGBoost, CatBoost)
- **Hyperparameter Optimization**: Optuna framework with early stopping
- **Full Dataset Retraining**: Combines train+test for production model
- **Logging & Monitoring**: Detailed logs at each stage

### 🌐 Web Interface
- **Modern Design**: Gradient background, smooth animations
- **Form Validation**: Dropdown selections for all features
- **Real-time Predictions**: Instant price estimates
- **Error Handling**: User-friendly error messages
- **Responsive**: Works on desktop and mobile

### 🐳 Deployment
- **Docker Ready**: One-command deployment
- **docker-compose**: Orchestration with environment config
- **Auto-pipeline**: Runs training on container start
- **Reproducible**: Same environment everywhere

### 📊 **Advanced Model Comparison** ⭐ NEW
- **4 Statistical Tests**: Paired t-tests, ANOVA, Kruskal-Wallis, Confidence Intervals
- **10+ Visualizations**: Box plots, violin plots, residuals, Q-Q plots, feature importance
- **Professional HTML Report**: Interactive, embedded visualizations, exportable
- **Ensemble Methods**: Voting and Stacking regression
- **Residual Analysis**: Normality, heteroscedasticity, and pattern detection
- **Complete Diagnostics**: Shapiro-Wilk, Levene's tests for assumptions

---

## 📊 Advanced Model Comparison

### Statistical Tests Performed
```
✅ Paired t-tests (pairwise model comparisons)
✅ One-way ANOVA (test if all models differ)
✅ Kruskal-Wallis (non-parametric ANOVA)
✅ Shapiro-Wilk (residual normality)
✅ Levene's test (variance homogeneity)
✅ 95% Confidence Intervals (all metrics)
✅ Effect Sizes (Cohen's d, η²)
```

### Visualizations Generated (10+ plots)
```
📊 Cross-Validation Analysis
├─ Box plots (score distributions)
├─ Violin plots (distribution shapes)
└─ Metrics comparison (MAPE, MAE, R²)

📈 Model Performance
├─ Test set comparison
├─ Ensemble vs individual models
└─ Actual vs predicted scatter

🔍 Residual Diagnostics
├─ Residuals vs fitted values (heteroscedasticity)
├─ Residuals distribution (normality check)
├─ Q-Q plots (normal probability)
└─ Feature importance (top features)
```

### Output Reports
```
📄 model_comparison_report.html
├─ Interactive sections
├─ Embedded visualizations
├─ Statistical test results
├─ Model recommendations
└─ Professional styling

📊 model_comparison_report.json
├─ All metrics (CV, test set)
├─ Statistical test results
├─ Ensemble performance
└─ Best model info
```

---

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Framework** | scikit-learn 1.3.2 | Model training & preprocessing |
| **Optimization** | Optuna | Hyperparameter tuning |
| **Advanced Tests** | SciPy | Statistical analysis |
| **Visualization** | Matplotlib, Seaborn | Plots and diagnostics |
| **Boosting** | XGBoost, CatBoost | Advanced models |
| **Backend** | Flask | REST API & web server |
| **Frontend** | HTML/CSS/JavaScript | Interactive UI |
| **Deployment** | Docker + docker-compose | Containerization |
| **Data Processing** | pandas, numpy | Data manipulation |
| **Language** | Python 3.10 | Core implementation |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA INGESTION PIPELINE                  │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  notebook/car_data.csv                                      │
│           ↓                                                  │
│   [DataIngestion]  → Load raw data                          │
│           ↓                                                  │
│   Train/Test Split (70/30)                                 │
│           ↓                                                  │
│  [DataTransformation] → Preprocessing Pipeline             │
│   ├─ OneHotEncoder (categorical features)                 │
│   ├─ SimpleImputer (missing values)                        │
│   └─ StandardScaler (normalization)                        │
│           ↓                                                  │
│  [ModelComparison] → Compare 4 Models                      │
│   ├─ 5-fold cross-validation                              │
│   ├─ Statistical significance tests                        │
│   ├─ 10+ visualizations                                    │
│   ├─ Ensemble methods (Voting + Stacking)                │
│   └─ HTML & JSON reports                                   │
│           ↓                                                  │
│  [Optional] Hyperparameter Optimization                    │
│   ├─ Optuna framework                                      │
│   ├─ Minimize MAPE metric                                  │
│   └─ Patience logic (5 trials, min 8 total)               │
│           ↓                                                  │
│  [Optional] Retrain on Full Dataset                        │
│   └─ Use best parameters + combine train+test             │
│           ↓                                                  │
│  ✅ Save: model.pkl + preprocessor.pkl                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended) ⭐

**One command to run everything:**

```bash
docker-compose up --build
```

Then open: **http://localhost:9090**

---

### Option 2: Local Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run pipeline with model comparison
python src/components/data_ingestion.py
# When prompted: "Compare multiple models?" → yes

# Start app
python app.py
```

Then open: **http://localhost:9090**

---

## 📊 Results & Metrics

| Metric | Value | Test Set |
|--------|-------|----------|
| **MAPE (Best Model)** | 14.65% ± 1.78% | 14.68% |
| **R² Score** | 0.8593 | 0.8593 |
| **MAE** | ~$2,025 | - |
| **Training Samples** | 1,234 | - |
| **Test Samples** | 529 | - |
| **CV Folds** | 5 | - |

### Statistical Significance
- **ANOVA p-value**: < 0.05 ✅ (Models significantly differ)
- **Best vs Worst**: p = 0.018 ✅ (CatBoost significantly better)
- **Effect Size (η²)**: 0.34 (Large effect)

---

## 💡 Key Decisions

### Why Compare 4 Models?
- **RandomForest**: Baseline, fast, interpretable
- **GradientBoosting**: Strong sequential learner
- **XGBoost**: Industry standard, optimized
- **CatBoost**: Categorical features handling

### Why Statistical Tests?
- Validate differences are real, not random
- Provide confidence intervals for metrics
- Assess normality and variance assumptions
- Enable informed model selection

### Why Visualizations?
- Detect overfitting/underfitting patterns
- Check residual assumptions
- Compare distributions visually
- Identify unstable folds

### Why Ensemble?
- Combine strengths of multiple models
- Reduce overfitting through averaging
- Improve generalization (sometimes)
- Stacking: Meta-learner finds optimal weights

### Why HTML Report?
- Professional presentation
- Shareable across teams
- Embedded visualizations
- Interactive navigation

---

## 🔧 Troubleshooting

### Port 9090 Already in Use
Edit `docker-compose.yml` and change port to `9091:9090`

### Model Not Found
Run: `python src/components/data_ingestion.py`

### Docker Build Fails
```bash
docker-compose down
docker system prune -a
docker-compose up --build
```

### Missing Visualization Libraries
```bash
pip install matplotlib seaborn
```

### Statistical Test Errors
Ensure scipy >= 1.10 is installed for all tests

---

## 📚 Documentation

- **DOCKER_GUIDE.md** - Docker commands reference
- **DOCKER_LEARNING_GUIDE.md** - Learn Docker concepts
- **DOCKER_FIRST_TIME_SETUP.md** - Step-by-step walkthrough
- **MODEL_COMPARISON_GUIDE.md** - Original model comparison guide
- **MODEL_COMPARISON_ENHANCED.md** - ⭐ NEW - Complete enhanced features guide

---

## 🎓 Interview Talking Points

✅ **"End-to-end ML pipeline with data → model → deployment"**
✅ **"Compared 4 models using 5-fold cross-validation"**
✅ **"Validated with 4 statistical tests (t-test, ANOVA, Kruskal-Wallis, CIs)"**
✅ **"Generated 10+ diagnostic visualizations for model assessment"**
✅ **"Created interactive HTML report with embedded analysis"**
✅ **"Ensemble methods (Voting/Stacking) improved predictions"**
✅ **"Hyperparameter optimization with smart early stopping"**
✅ **"Full dataset retraining for production model"**
✅ **"Docker containerization for reproducibility"**
✅ **"Modern web UI with real-time predictions"**

---

## 📝 License

MIT License

---

<div align="center">

### ⭐ If this project helped you, please give it a star!

**Now with advanced statistical testing and comprehensive visualizations!**

</div>