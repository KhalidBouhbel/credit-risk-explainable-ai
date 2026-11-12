# Credit Risk Prediction & Explainable AI

An end-to-end machine learning project for predicting credit default risk, combining a Random Forest classifier, a FastAPI backend, an interactive Streamlit dashboard, and SHAP explainability.

## Project Overview

This project explores credit risk assessment using machine learning, with particular attention to model interpretability and responsible decision support.

The application provides:
- Individual credit default risk predictions
- Interactive exploration of client profiles
- Local explanations using SHAP values
- Global feature importance analysis
- Exploratory decision threshold adjustment
- Data drift analysis using Evidently

## Technologies

- Python, Pandas, NumPy, Scikit-learn
- Random Forest Classifier
- FastAPI
- Streamlit
- SHAP
- Evidently
- Matplotlib

## Model Explainability

SHAP decomposes individual predictions into a baseline prediction and feature contributions.

For a tested client, the Random Forest predicted a default risk score of 0.76. The SHAP contributions reconstructed this prediction with a numerical discrepancy below 1e-10.

SHAP explanations describe model behavior, not causal relationships.

## Dataset

Source: Home Credit Default Risk (Kaggle)

https://www.kaggle.com/competitions/home-credit-default-risk

The dataset is not redistributed in this repository. Local execution requires preparing the corresponding data files.

## Repository Structure

- `Projet_7_analyse_1.ipynb`: exploratory data analysis
- `Projet_7_modelisation-1-Officiel.ipynb`: model development
- `API_dashboard/app_fastapi_heroku/`: prediction API
- `API_dashboard/app_streamlit_heroku/`: interactive dashboard
- `evidently_datadrift/`: data drift report
- `Projet7_presentation.pdf`: project presentation
- `Projet7_note_technique.odt`: technical documentation

## Limitations

The model is presented as an exploratory decision-support tool.

Probability calibration, fairness, decision thresholds, and deployment robustness require further validation before any real-world credit decision-making.

## Author

Khalid Bouhbel

Data Scientist | Mathematics Educator | AI & Education Research Interests
