# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 14:08:20 2026

@author: amcol
"""

#### Import Libraries

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import preprocessing




#### Load in Data

# Declare filename
filename_kaggle = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset.csv"
filename_kaggle_w_labels = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset_for_DT.csv"

# Read in data and drop all non-numeric columns.
kaggle_DF = pd.read_csv(filename_kaggle)
kaggle_DF = kaggle_DF.drop(columns=["Unnamed: 0","Name","Platform","Genre","Publisher","Developer","Rating"])
kaggle_DF_w_labels = pd.read_csv(filename_kaggle_w_labels)
kaggle_DF_w_labels = kaggle_DF_w_labels.drop(columns=["Unnamed: 0","Labels"])




#### Perform Multiple Linear Regression (10-D)

# Choose which variable to predict.
Y = kaggle_DF[["NA_Sales"]]
print(Y)
print("")

# Set rest of variables as "X" dataset (dependent variables), and display it.
X = kaggle_DF.drop("NA_Sales", axis= 1) 
print(X) # Here X has 9 variables;
print(type(X))
print(X.shape)
print("")

# Split into training and testing data.
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size =0.33, random_state=42)

# Instantiate instance of LinearRegression().
my_MLR = LinearRegression()

# Train the model.
my_MLR.fit(x_train, y_train)
print(my_MLR.coef_) # These are the coefficients of variables used to train (used to create linear equation)
print(my_MLR.intercept_) # This is the y-intercept
print("Our MODEL is:\n")
print("y=%3.3fx1 + %3.2fx2 + %3.2fx3 + %3.2fx4 + %3.2fx5 + %3.2fx6 + %3.2fx7 + %3.2fx8 + %3.2fx9 + %3.2f"%(my_MLR.coef_[0,0] , my_MLR.coef_[0,1], my_MLR.coef_[0,2], my_MLR.coef_[0,3], my_MLR.coef_[0,4], my_MLR.coef_[0,5], my_MLR.coef_[0,6], my_MLR.coef_[0,7], my_MLR.coef_[0,8], my_MLR.intercept_[0]))
print("")

# Create predictions based on the trained model.
my_prediction = my_MLR.predict(x_test)

# Determine how accurate the model is by printing the score.
print("Score of multiple-linear regression model prediciting North-American sales:")
print(my_MLR.score(x_train,y_train))
print("")

# Model evaluation 
print('mean_squared_error : ', mean_squared_error(y_test, my_prediction)) 
print('mean_absolute_error : ', mean_absolute_error(y_test, my_prediction)) 
print("")




#### Create Visualizations for Multiple Linear Regression

# Flatten y_test and predictions into 1-D arrays for plotting.
y_test_flat = y_test.values.flatten()
my_prediction_flat = my_prediction.flatten()

# Compute residuals for residual plots.
residuals = y_test_flat - my_prediction_flat

# Create scatter plot of actual vs predicted values.
plt.figure(figsize=(8,6))
plt.scatter(y_test_flat, my_prediction_flat, alpha=0.6)
plt.plot([y_test_flat.min(), y_test_flat.max()],
         [y_test_flat.min(), y_test_flat.max()],
         'r--')
plt.xlabel("Actual NA_Sales")
plt.ylabel("Predicted NA_Sales")
plt.title("Actual vs Predicted NA_Sales (Multiple Linear Regression)")
plt.grid(True)
plt.show()

# Create residual plot to assess model fit.
plt.figure(figsize=(8,6))
plt.scatter(my_prediction_flat, residuals, alpha=0.6)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted NA_Sales")
plt.ylabel("Residuals")
plt.title("Residual Plot (Multiple Linear Regression)")
plt.grid(True)
plt.show()

# Create histogram of residuals to see error distribution.
plt.figure(figsize=(8,6))
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.title("Histogram of Residuals (Multiple Linear Regression)")
plt.grid(True)
plt.show()

# Create bar chart of regression coefficients to show variable importance.
plt.figure(figsize=(10,6))
plt.bar(X.columns, my_MLR.coef_[0])
plt.xticks(rotation=45, ha='right')
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.title("Feature Coefficients (Multiple Linear Regression)")
plt.grid(axis='y')
plt.tight_layout()
plt.show()




#### Perform Logistic Regression

# Create training and testing datasets.
training_dataset, testing_dataset = train_test_split(kaggle_DF_w_labels, test_size = 0.3, random_state = 42)

# Extract training and testing labels, and then remove them from the dataset.
training_labels = training_dataset["Numeric_Labels"]
testing_labels = testing_dataset["Numeric_Labels"]
training_dataset_no_label = training_dataset.drop(columns=["Numeric_Labels"])
testing_dataset_no_label = testing_dataset.drop(columns=["Numeric_Labels"])

# Scale the data as according to https://scikit-learn.org/stable/modules/preprocessing.html
scaler = preprocessing.StandardScaler().fit(training_dataset_no_label)
training_dataset_no_label = scaler.transform(training_dataset_no_label)
testing_dataset_no_label = scaler.transform(testing_dataset_no_label)

# Instantiate instance of LogisticRegression()
my_LogR = LogisticRegression(max_iter=1000)

# Train model with training data.
my_LogR_model = my_LogR.fit(training_dataset_no_label, training_labels)

# Create predictions using trained model on testing data.
predictions = my_LogR_model.predict(testing_dataset_no_label)

# Create confusion matrix to evaluate performance of model.
cm=confusion_matrix(testing_labels, predictions)
print("\nThe confusion matrix is for the Logistic Regression Model is:")
print(cm)
print("")

# Compute accuracy by dividing trace of confusion matrix by the sum of all 
# entries in the confusion matrix.
number_correct = np.trace(cm)
total_number = np.sum(cm)
accuracy = number_correct / total_number
print("Accuracy of Logistic Regression Model:")
print(accuracy)
print("")




#### Create Visualizations for Logistic Regression

# Use Seaborn to create a pretty confusion matrix visualization.
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Logistic Regression)")
plt.show()

# Create normalized confusion matrix visualization.
plt.figure(figsize=(8,6))
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Normalized Confusion Matrix (Logistic Regression)")
plt.show()

# Create bar chart showing how many predictions were made for each class.
unique_classes, class_counts = np.unique(predictions, return_counts=True)

plt.figure(figsize=(8,6))
plt.bar(unique_classes.astype(str), class_counts)
plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Predicted Class Distribution (Logistic Regression)")
plt.grid(axis='y')
plt.show()

# Create bar chart of logistic regression coefficients.
plt.figure(figsize=(10,6))
plt.bar(training_dataset_no_label.shape[1] * [""], np.zeros(training_dataset_no_label.shape[1]))  # Placeholder
plt.clf()

feature_names = training_dataset.drop(columns=["Numeric_Labels"]).columns

# If binary classification, coefficients will be one row.
if my_LogR.coef_.shape[0] == 1:
    plt.figure(figsize=(10,6))
    plt.bar(feature_names, my_LogR.coef_[0])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Coefficient Value")
    plt.title("Feature Coefficients (Logistic Regression)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
else:
    # If multiclass, plot coefficient magnitudes averaged across classes.
    avg_abs_coef = np.mean(np.abs(my_LogR.coef_), axis=0)
    
    plt.figure(figsize=(10,6))
    plt.bar(feature_names, avg_abs_coef)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Features")
    plt.ylabel("Average Absolute Coefficient Value")
    plt.title("Average Feature Importance (Logistic Regression)")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()