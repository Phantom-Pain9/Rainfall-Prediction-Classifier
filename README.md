# 🌧️ Australian Rainfall Prediction: End-to-End Machine Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning classification project predicting whether it will rain tomorrow in Australia based on historical daily meteorological observations from the Australian Bureau of Meteorology.

---

## 📌 Project Overview

Predicting next-day precipitation is critical for water management, agricultural operations, and disaster preparedness. This project implements a complete, leak-free machine learning workflow:
- **Feature Engineering & Cleaning:** Extracting seasonal cyclic signals from dates and filtering geographical locations.
- **Preprocessing Pipeline:** Integrated `ColumnTransformer` handling numerical imputation/scaling and categorical one-hot encoding without data leakage.
- **Hyperparameter Optimization:** Stratified 5-Fold Cross-Validation via `GridSearchCV`.
- **Model Comparison:** Evaluating and comparing **Random Forest Classifier** against **Logistic Regression**.
- **Model Interpretability:** Extracting feature importances back through the encoding pipeline to identify top weather predictors.

---

## 📊 Dataset Description

The dataset contains daily weather metrics from 2008 to 2017 across Australian weather stations (sourced from Kaggle / Australian Bureau of Meteorology):

| Feature Category | Features Included |
| :--- | :--- |
| **Atmospheric Pressure** | `Pressure9am`, `Pressure3pm` (hPa) |
| **Temperature** | `MinTemp`, `MaxTemp`, `Temp9am`, `Temp3pm` (°C) |
| **Humidity & Moisture** | `Humidity9am`, `Humidity3pm` (%), `Rainfall` (mm), `Evaporation` (mm) |
| **Solar & Sky Cover** | `Sunshine` (hours), `Cloud9am`, `Cloud3pm` (oktas) |
| **Wind Dynamics** | `WindGustDir`, `WindGustSpeed`, `WindDir9am`, `WindDir3pm`, `WindSpeed9am`, `WindSpeed3pm` |
| **Target Variable** | `RainTomorrow` (Binary: `Yes` / `No`) |

---

## ⚙️ Methodology & Pipeline Architecture

```
Raw Data ──> Location Filtering & Date Feature Extraction
             │
             ├──> Numerical Pipeline: StandardScaler
             │
             └──> Categorical Pipeline: OneHotEncoder (handle_unknown='ignore')
                  │
                  └──> ColumnTransformer ──> Estimator (Random Forest / Logistic Regression)
                                              │
                                              └──> GridSearchCV (Stratified 5-Fold CV)
```

1. **Stratified Splitting:** 80/20 train-test split stratified on `RainTomorrow` to preserve target class proportions (~76.3% No, ~23.7% Yes).
2. **ColumnTransformer:** Prevents data leakage by computing statistics strictly within training folds.
3. **Cross-Validation:** 5-fold Stratified K-Fold CV ensuring robust hyperparameter tuning.

---

## 📈 Model Performance & Comparison

Both models were evaluated on the unseen test set ($N = 1,512$):

| Metric | Random Forest Classifier | Logistic Regression |
| :--- | :---: | :---: |
| **Accuracy** | **83.60%** | **83.47%** |
| **Precision (Rain: Yes)** | **0.71** | **0.68** |
| **Recall / TPR (Rain: Yes)** | **0.51** | **0.51** |
| **F1-Score (Rain: Yes)** | **0.59** | **0.58** |
| **ROC / Discrimination** | Higher precision on true rainy days | Balanced linear boundary |

### Key Findings
- **Predictive Performance:** Both models achieved comparable overall accuracy (~83.5%) and identical True Positive Rate (Recall $\approx$ 51%) on rainy days.
- **Precision Advantage:** Random Forest produced fewer false alarms (precision of 0.71 vs 0.68), making it the superior operational model for rain forecasting.
- **Key Determinants:** Feature importance analysis revealed that **`Humidity3pm`**, **`Sunshine`**, **`Pressure3pm`**, and **`Cloud3pm`** are the strongest predictors of next-day rainfall.

---

## 💻 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/Phantom-Pain9/Rainfall-Prediction-Classifier.git
cd Rainfall-Prediction-Classifier
```

### 2. Set up virtual environment & install dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the Jupyter Notebook
```bash
jupyter notebook Rainfall_Prediction_Classifier.ipynb
```

---

## 📜 License & Credits

- Dataset provided by the Australian Bureau of Meteorology via Kaggle.
- Developed as part of the **IBM Machine Learning with Python** course within the **IBM Data Science Professional Certificate**.
