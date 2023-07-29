# CVIP-Data-Science-Intern
## *****Phase 1*****
# **COVID 19 ANALYSIS (Normal Task)**

# ***Covid19 Detection using Machine Learning Models***

# About Dataset
### Data Description: 
Data collected is from March 2020 - November 2021

### Symptoms: 
Cough, Fever, Sore Throat, Shortness of Breath & Headache.

### Other Features: 
Gender, Age 60 and above, Test indication & Test date.

### Target Feature: 
Corona Result.

# Implementations
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

________________________________________________________________________________________________________________________________________

# **DIABETICS PREDICTION (Golden Task)**

# ***Metaheuristic optimization and MLP based Diabetes Prediction***

# Pima Indians Diabetes Dataset

This dataset describes the medical records for Pima Indians and whether or not each patient will have an onset of diabetes with the consideration of several medical predictor (independent) variables and one target (dependent) variable, Outcome.

# Fields description:

Pregnancies = Number of times pregnant

Glucose = Plasma glucose concentration a 2 hours in an oral glucose tolerance test

BloodPressure = Diastolic blood pressure (mm Hg)

SkinThickness = Triceps skin fold thickness (mm)

Insulin = 2-Hour serum insulin (mu U/ml)

BMI = Body mass index (weight in kg/(height in m)^2)

DiabetesPedigreeFunction = Diabetes pedigree function

Age = in (years)

Outcome = (1:tested positive for diabetes, 0: tested negative for diabetes)

# Implementations
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

________________________________________________________________________________________________________________________________________

## *****Phase 2*****
# **MOBILE PRICE CLASSIFICATION (Normal Task)**

# ***Ensemble Learning for Mobile Price Classification: Leveraging LDA and Variance Threshold for Enhanced Accuracy***

# Data Description :

battery_power - Total energy a battery can store in one time measured in mAh

blue - Has bluetooth or not

clock_speed - speed at which microprocessor executes instructions

dual_sim - Has dual sim support or not

fc - Front Camera mega pixels

four_g - Has 4G or not

int_memory - Internal Memory in Gigabytes

m_dep - Mobile Depth in cm

mobile_wt - Weight of mobile phone

n_cores - Number of cores of processor

pc - Primary Camera mega pixels

px_height - Pixel Resolution Height

px_width - Pixel Resolution Width

ram - Random Access Memory in Mega Bytes

sc_h - Screen Height of mobile in cm

sc_w - Screen Width of mobile in cm

talk_time - longest time that a single battery charge will last when you are

three_g - Has 3G or not

touch_screen - Has touch screen or not

wifi - Has wifi or not

price_range - This is the target variable with value of 0(low cost), 1(medium cost), 2(high cost) and 3(very high cost).

# Implementations

1. Data Extraction - Collected the data from Kaggle.
2. Data Preprocessing - EDA using Pandas Profiling
3. Feature Scaling - Standardization the entire data.
4. Dimentionality Reduction Technique - LDA
5. Feature Selection - Variance Threshold
6. Data split - 80:20 split as Train-Test Split.
7. Data Modeling - Ensemble Learning(Boosting algorithms - AdaBoost, Gradient Boosting, XGBoost, Light GBM)
8. Model Evaluation - Calculated all the metrics using Confusion Matrix.
9. Model Comparison - Compared the accuracy of all models.
10. Model validation techniques - Cross Validation of Each Model
11. Validation Comparison - Compared CrossValidation Score of all models.
12. Predictions on Test Set

________________________________________________________________________________________________________________________________________

# **Image Caption Generator (Golden Task)**

# ***Image Captioning with ResNet50 and Greedy Search: A Deep Learning Approach to Generating Image Descriptions***

# Image Caption Generator
Input image ----> Image Caption DL Model ----> Output Caption
- ResNet50 (Residual Network 50)is a deep convolutional neural network architecture that addresses the vanishing gradient problem with skip connections and residual blocks.
- Greedy Search is a decoding algorithm where the model predicts the word with the highest probability at each step, producing a sequence iteratively.
- In image captioning, ResNet50 is used to encode images, while Greedy/Beam Search is applied to generate captions based on the encoded features and language model predictions.
# Flickr 8k Image Dataset
Data is properly labelled, each image contain 5 different captions.

After extracting zip files we will find below folders,

## Flickr8k_Dataset:
Contains a total of 8092 images in JPEG format with different shapes and sizes.
- Predefined training dataset of 6000 images
- Validation dataset of 1000 images
- Testing dataset of 1000 images

## Flickr8k_text :
Contains text files describing train_set ,test_set.
- Flickr8k.token.txt contains 5 captions for each image i.e. total 40460 captions.

# Implementations
1.  Data Download and Unzipping
2.  Library Imports
3.  Data Visualization and Preprocessing
4.  ResNet50 Model for Image Encoding
5.  Setting Hyperparameters for Vocabulary Size and Maximum Caption Length
6.  Creating Dictionaries Containing Mapping of Words to Indices and Indices to Words
7.  Transforming Data into a Dictionary Mapping of Image ID to Encoded Captions
8.  Data Generator for Modeling
9.  Model Architecture - Designed a caption generation model using LSTM, Dense, and Embedding layers
10. Model Compilation and Training - Categorical cross-entropy loss function & Adam optimizer
11. Caption Generation
12. Evaluation - BLEU score for  Greedy Search
________________________________________________________________________________________________________________________________________
