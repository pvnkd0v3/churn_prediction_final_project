# Churn Prediction Model Training and Evaluation: TripleTen Data Science Final Project 
This project was completed as the final capstone for the TripleTen Data Science Bootcamp. The goal was to develop a machine learning model that predicts customer churn based on historical user behavior and demographics.

##  Project Overview
- **Goal:** Predict whether a customer will churn (i.e., stop using the service).
- **Type:** Supervised binary classification
- **Tools Used:** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter Notebook

## Skills Demonstrated
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Logistic Regression, Random Forest, and Gradient Boosting (XGBoost, LightGBM)
- Cross-validation and hyperparameter tuning
- Data visualization and presentation

## Project Structure
- contract.csv: Contract data by unique customer ID of 0ver 7000 Interconnect Telecom customers
- personal.csv: Demographics of Interconnect customers by customer ID
- internet.csv: What Telecom internet services are included in contract by customer ID of Interconnect internet users
- phone.csv: How many lines are included in contract by customer ID of Interconnect phone users
- requirements.txt: Requirements to run project
- work_plan.ipynb: Project introduction, initial view of data, and written plan for project
- EDA_and_model_training.ipynb: Main project notebook containing exploratory data analysis, model training, and model evaluation
- project_report.ipynb: A brief summary of project and its results

## How to Run Project
- Clone repository
 ```bash
git clone https://github.com/pvnkd0v3/churn_prediction_final_project
   cd churn-prediction
```
- Install dependencies
 ```bash
pip install -r requirements.txt
```
- Launch jupyter Notebook
```bash
  jupyter notebook
```
- Open and run: EDA_and_churn_prediction.ipynb

## Results Summary:
- Model: LightGBM
- ROC-AUC score: 0.8902
