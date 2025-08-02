 ## Loan Default Prediction Web App

This project is a Python and machine learning-based web app that predicts whether a loan applicant is likely to default. The application is built using *Streamlit* and trained using a *Random Forest* model. The app takes user input for financial attributes, performs preprocessing, and predicts the loan default risk. It also uses *SHAP* to provide model explainability.

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
1.Jupyter files 
loan_default.ipynb
EDA.ipynb
2.Python file(streamlit)
  app.py
  train_model.py
3.DataSet
cs-training from kaagle
4.Models and scaler
  scaler.pkl
 loan_default_model.pkl
 5.Requirement.txt
 6. READmd file
 7. Result.png 
   
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
   git clone https://github.com/Hepziba29/loan-default-predictor-with-Streamlit-SHAPgit
   cd loan-default-predictor
   Install dependencies
   pip install -rrequirement.txt
   Run the Streamlit app:
   streamlit run app.py
 ## Live app
   Click here to try the app live [Loan Default Prediction App]
  (https://loan-default-predictor-with-app-shap-8nej5yx88npgz7mcm3bxgp.streamlit.app)
   

 #  Exploratory Data Analysis: Loan Default Prediction

This notebook explores and analyzes the *UCI Credit Risk Dataset* to understand patterns behind customer loan defaults. The purpose of this EDA is to identify data distributions, detect missing values, understand correlations, and prepare the data for model building.

---

##  Dataset Overview

The dataset contains historical loan applicant data with the target column:

- SeriousDlqin2yrs → Indicates whether a person defaulted within 2 years.

### Key Features:

- RevolvingUtilizationOfUnsecuredLines
- age
- NumberOfTime30-59DaysPastDueNotWorse
- DebtRatio
- MonthlyIncome
- NumberOfOpenCreditLinesAndLoans
- NumberOfTimes90DaysLate
- NumberRealEstateLoansOrLines
- NumberOfTime60-89DaysPastDueNotWorse
- NumberOfDependents

---

##  Key Objectives of EDA

- Understand the distribution of numerical features.
- Check for missing values and handle them.
- Detect outliers and imbalances in the target variable.
- Explore relationships between input features and loan default risk.

---

##  What This EDA Covers

-  *Data cleaning:* Handled null values in MonthlyIncome and NumberOfDependents.
-  *Distribution plots:* Age, Income, Credit Lines, etc.
-  *Target imbalance check:* Checked for class imbalance in defaults.
-  *Correlation heatmap:* Explored feature relationships.
-  *Boxplots & histograms* for visualizing feature distributions and outliers.
-  *Insight extraction* from feature trends affecting default risk.

---

## Outputs
Histogram of age and income
Target Variable Distruibution
Boxplot of Credit Lines vs Default
Correlation Heatmap
   
   
