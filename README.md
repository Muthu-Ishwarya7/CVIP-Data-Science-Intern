# CVIP-Data-Science-Intern
**COVID 19 ANALYSIS (Normal Task)**

***Covid19 Detection using Machine Learning Models***

About Dataset
Data Description: Data collected is from March 2020 - November 2021

Symptoms: Cough, Fever, Sore Throat, Shortness of Breath & Headache.

Other Features: Gender, Age 60 and above, Test indication & Test date.

Target Feature: Corona Result.

Implementations
1. Data Extraction - Collected the data from Kaggle.
2. Data Preprocessing - Null Values and checking features datatypes Label.
3. Feature Encoding - Label Encoding & One Hot Encoding.
4. Exploratory Data Analysis - Used Plotly.
5. Data abundent - Undersampling Majority Class & removed no information columns.
6. Data split - 70:30 split as Train - Test Split.
7. Feature Scaling - Normalizing the entire data.
8. Feature Selection - Applied Information Gain test & classified important features.
9. Data Modeling - Applied GridSearchCv for Logistic Regression, Random Forest & XGBoost models.
10. Model Evaluation - Calculated all the metrics using Confusion Matrix.
11. Model Comparison - Compared the accuracy of all models.

**DIABETICS PREDICTION (Golden Task)**

***Metaheuristic optimization and MLP based Diabetes Prediction***

Pima Indians Diabetes Dataset

This dataset describes the medical records for Pima Indians and whether or not each patient will have an onset of diabetes with the consideration of several medical predictor (independent) variables and one target (dependent) variable, Outcome.

Fields description:

Pregnancies = Number of times pregnant

Glucose = Plasma glucose concentration a 2 hours in an oral glucose tolerance test

BloodPressure = Diastolic blood pressure (mm Hg)

SkinThickness = Triceps skin fold thickness (mm)

Insulin = 2-Hour serum insulin (mu U/ml)

BMI = Body mass index (weight in kg/(height in m)^2)

DiabetesPedigreeFunction = Diabetes pedigree function

Age = in (years)

Outcome = (1:tested positive for diabetes, 0: tested negative for diabetes)

Implementations
1. Data Extraction - Collected the data from Kaggle.
2. Data Preprocessing - EDA using plotly
3. Data abundent - Random Undersampling
4. Statistical Measures - Skewness, Normality Test & Pearson Correlation.
5. Data split - 70:30 split as Train - Test Split.
6. Data Metaheuristic Optimization - Grey Wolf Optimizer (GWO)
7. Defining the Model Architecture
8. Data Modeling - Multi-Layer Perceptron (MLP)
9. Training the Model
10. Model Evaluation - Confusion Matrix.
