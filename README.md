 ## Loan Default Prediction Web App

This project is a machine learning-based web app that predicts whether a loan applicant is likely to default. The application is built using *Streamlit* and trained using a *Random Forest* model. The app takes user input for financial attributes, performs preprocessing, and predicts the loan default risk. It also uses *SHAP* to provide model explainability.

---

## Project Overview

Loan default prediction is essential for financial institutions to reduce risk and make data-driven decisions. In this project, a classification model is trained on historical loan data to determine whether a person is likely to default on a loan.

We built a user-friendly Streamlit web app where users can enter input features and see real-time predictions, including interpretability with SHAP plots.


##  Features

-  Predict loan default risk with a trained ML model
-  Visualize SHAP values for transparency and decision-making
-  Scaled input data using StandardScaler
-  Deployable to Streamlit Cloud
-  Organized and modular code structure


##  Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- SHAP
- Joblib


## Project Structure
*Jupyter files 
1.loan_default.ipynb
2.EDA.ipynb
*Python file(streamlit)
1. app.py
 *DataSet
1. cs-training from kaagle
 *Models and scaler
1.scaler.pkl
2. loan_default_model.pkl
 *Requirement.txt
 * READmd file
   *Outputs.txt
   *Screenshots of EDA outputs
   ##  Input Features

The model uses the following input features:

- Revolving Utilization of Unsecured Lines
- Debt Ratio
- Monthly Income
- Number of Open Credit Lines and Loans
- Number of Times 30-59 Days Past Due
- Number of Times 90 Days Late
- Number of Dependents
- Age
- Number of Real Estate Loans or Lines

---

## Model Information

- Algorithm: RandomForestClassifier
- Preprocessing: StandardScaler
- Performance: ~85% Accuracy
- Interpretability: SHAP (SHapley Additive exPlanations)

---

##  How to Run the App Locally

1. *Clone the repository*:
   ```bash
   git clone https://github.com/Hepziba29/loan-default-predictor-with-Streamlit&SHAPgit
   cd loan-default-predictor
   
   
