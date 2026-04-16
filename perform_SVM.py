# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:35:40 2026

@author: Austin Coleman
"""

#### Import Libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn import preprocessing
from sklearn.svm import SVC




#### Load in Data

filename_kaggle_w_labels = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset_for_DT.csv"

kaggle_DF = pd.read_csv(filename_kaggle_w_labels)
kaggle_DF = kaggle_DF.drop(columns=["Unnamed: 0","Labels"])




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




#### Train SVM Models (3 Kernels)

# Linear Kernel
svm_linear = SVC(kernel='linear', C=1)
svm_linear.fit(training_data, training_labels)
pred_linear = svm_linear.predict(testing_data)

# Polynomial Kernel
svm_poly = SVC(kernel='poly', degree=3, C=1)
svm_poly.fit(training_data, training_labels)
pred_poly = svm_poly.predict(testing_data)

# RBF Kernel
svm_rbf = SVC(kernel='rbf', C=1, gamma='scale')
svm_rbf.fit(training_data, training_labels)
pred_rbf = svm_rbf.predict(testing_data)




#### Confusion Matrices

# Compute the confusion matrices for all 3 models.
cm_linear = confusion_matrix(testing_labels, pred_linear)
cm_poly = confusion_matrix(testing_labels, pred_poly)
cm_rbf = confusion_matrix(testing_labels, pred_rbf)

# Print the results.
print("")
print("Linear Kernel Confusion Matrix:")
print(cm_linear)
print("")

print("Polynomial Kernel Confusion Matrix:")
print(cm_poly)
print("")

print("RBF Kernel Confusion Matrix:")
print(cm_rbf)
print("")




#### Accuracy Calculation

# Create tiny function to compute accuracy for each model.
def compute_accuracy(cm):
    return np.trace(cm) / np.sum(cm)

# Print the results.
print("")
print("Linear Accuracy:", compute_accuracy(cm_linear))
print("Polynomial Accuracy:", compute_accuracy(cm_poly))
print("RBF Accuracy:", compute_accuracy(cm_rbf))
print("")




#### Visualization

# Linear Kernel CM
plt.figure(figsize=(6,5))
sns.heatmap(cm_linear, annot=True, fmt='d', cmap='Blues')
plt.title("Linear Kernel Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Polynomial Kernel CM
plt.figure(figsize=(6,5))
sns.heatmap(cm_poly, annot=True, fmt='d', cmap='Greens')
plt.title("Polynomial Kernel Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# RBF Kernel CM
plt.figure(figsize=(6,5))
sns.heatmap(cm_rbf, annot=True, fmt='d', cmap='Reds')
plt.title("RBF Kernel Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()