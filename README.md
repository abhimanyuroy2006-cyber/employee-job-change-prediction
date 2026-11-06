# Employee Job Change Prediction (Capstone Project)

An end-to-end Machine Learning pipeline to predict whether a data science professional will seek a job change after completing a training program.

## Problem Statement
A Big Data & Data Science company wants to identify candidates who genuinely intend to join after training — to reduce onboarding costs and improve hiring efficiency.

## Project Highlights
- **19,158 candidate records** with demographic, education, and experience features
- **Class Imbalance handled with SMOTE** (applied strictly after train/test split — no leakage)
- **4 Models Benchmarked**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **GridSearchCV Hyperparameter Tuning** for all models
- **Full EDA**: Univariate & Bivariate analysis with visualizations

## Tech Stack
`Python` `Scikit-Learn` `XGBoost` `LightGBM` `Imbalanced-Learn (SMOTE)` `Pandas` `Seaborn` `Matplotlib`

## Key Features Analyzed
- City Development Index
- Education Level
- Years of Experience
- Company Size & Type
- Training Hours

## Project Structure
```
capstone project/
├── aug_train.csv          # Training dataset
├── aug_test.csv           # Test dataset
├── job_change_prediction.ipynb   # Main Jupyter Notebook
└── chart/                 # EDA visualizations
```

## Run the Notebook
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm imbalanced-learn
jupyter notebook job_change_prediction.ipynb
```
