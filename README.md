# Customer Churn Prediction

## **Overview**

This project aims to predict customer churn for gambling operators by analyzing player behavior using historical data. The goal is to develop a machine learning model that identifies players likely to churn (i.e., stop engaging with the service) based on their activity. Early prediction of churn can help businesses take proactive measures to retain customers, thereby improving customer loyalty and reducing churn.

The project involves preprocessing and analyzing a dataset consisting of 7,000 players, where daily aggregation data is provided for various player activities. The dataset includes features such as the number of bets placed, turnover, approved deposits/withdrawals, net gaming revenue (NGR), and session counts. The churn is defined as a player who has been inactive for 30 days, and the task is to predict whether a player will churn.

## **Key Steps**

1. **Data Preprocessing**
    - Filling missing values
    - One-hot encoding categorical variables
    - Creating the churn target variable based on a 30-day inactivity rule

2. **Exploratory Data Analysis (EDA)**
    - Visualizing churn distribution
    - Analyzing feature correlations
    - Plotting feature distributions and box plots for churn vs. features

3. **Model Building**
    - Four different models are used for prediction:
        - Logistic Regression
        - Random Forest
        - XGBoost
        - LightGBM

4. **Model Evaluation**
    - Evaluation of models using:
        - Confusion Matrix
        - ROC Curve and AUC
        - Precision-Recall Curve
    - Comparison of model performance using key metrics such as precision, recall, F1-score, and accuracy.

## **Getting Started**

To run the code, you will need to have the following libraries installed:

- pandas
- scikit-learn
- XGBoost
- LightGBM
- matplotlib
- seaborn

You can install the necessary libraries using pip


### **Prerequisites**

1. **Dataset**: The dataset should be in the format of an Excel file named `data.xlsx`, containing player activity data.
2. **Python 3.x**: This code is compatible with Python 3.x versions.

### **How to Run**

1. Clone the repository to your local machine.
   
   ```bash
   git clone https://github.com/abdghad2025/Customer-Churn-Prediction.git


