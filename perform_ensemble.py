# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 16:29:57 2026

@author: Austin Coleman
"""

#### Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns




#### Load in Data

filename_kaggle_w_labels = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset_for_DT.csv"

kaggle_DF = pd.read_csv(filename_kaggle_w_labels)
kaggle_DF = kaggle_DF.drop(columns=["Unnamed: 0", "Labels"])




#### Prepare Data

# Extract training and testing datasets.
training_dataset, testing_dataset = train_test_split(kaggle_DF, test_size=0.3, random_state=42)

# Extract the labels.
training_labels = training_dataset["Numeric_Labels"]
testing_labels = testing_dataset["Numeric_Labels"]

# Remove the labels from the training and testing data.
training_data = training_dataset.drop(columns=["Numeric_Labels"])
testing_data = testing_dataset.drop(columns=["Numeric_Labels"])

# Scale the data
scaler = preprocessing.StandardScaler().fit(training_data)
training_data = scaler.transform(training_data)
testing_data = scaler.transform(testing_data)




#### Train Random Forest Model (Ensemble Learning)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(training_data, training_labels)
pred_rf = rf_model.predict(testing_data)




#### Confusion Matrix

# Compute confusion matrix
cm_rf = confusion_matrix(testing_labels, pred_rf)

# Print results
print("")
print("Random Forest Confusion Matrix:")
print(cm_rf)
print("")




#### Accuracy Calculation

# Function to compute accuracy
def compute_accuracy(cm):
    return np.trace(cm) / np.sum(cm)

print("")
print("Random Forest Accuracy:", compute_accuracy(cm_rf))
print("")




#### Visualization

plt.figure(figsize=(6,5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




#### Feature Importance Visualization

importances = rf_model.feature_importances_
feature_names = kaggle_DF.drop(columns=["Numeric_Labels"]).columns

plt.figure(figsize=(8,5))
plt.barh(feature_names, importances)
plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()