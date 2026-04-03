# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:02:55 2026

@author: Austin Coleman
"""

#### Import Libraries

# Import all libraries used.
import seaborn as sns
import matplotlib.pyplot as plt     
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB

# Add path for graphviz
os.environ["PATH"] = r"C:\Program Files (x86)\Graphviz\bin" + os.pathsep + os.environ["PATH"]




#### Load in Data

# Declare filename of data to load in.
gaussian_dataset_filename = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset_for_DT.csv"
bernoulli_dataset_filename = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/videogame_bernoulli_dataset.csv"
multinomial_dataset_filename = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/videogame_multinomial_dataset.csv"
label_filename = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/labels_for_DT.csv"

# Read in data as dataframe using pandas.
gaussian_DF = pd.read_csv(gaussian_dataset_filename)
bernoulli_DF = pd.read_csv(bernoulli_dataset_filename)
multinomial_DF = pd.read_csv(multinomial_dataset_filename)

print(gaussian_DF.dtypes)

# Remove column Unnamed: 0 if it exists.
if "Unnamed: 0" in gaussian_DF.columns:
    gaussian_DF = gaussian_DF.drop(columns=["Unnamed: 0"])

if "Unnamed: 0" in bernoulli_DF.columns:
    bernoulli_DF = bernoulli_DF.drop(columns=["Unnamed: 0"])

if "Unnamed: 0" in multinomial_DF.columns:
    multinomial_DF = multinomial_DF.drop(columns=["Unnamed: 0"])

# Read in labels as array using numpy.
labels_for_DT = np.loadtxt(label_filename, dtype="str")




#### Show Data Preparation Information

# Print first few rows of each dataset.
print(" ")
print("MULTINOMIAL DATASET HEAD:")
print(multinomial_DF.head())
print(" ")
print("MULTINOMIAL DATASET SHAPE:")
print(multinomial_DF.shape)
print(" ")

print("BERNOULLI DATASET HEAD:")
print(bernoulli_DF.head())
print(" ")
print("BERNOULLI DATASET SHAPE:")
print(bernoulli_DF.shape)
print(" ")

print("GAUSSIAN DATASET HEAD:")
print(gaussian_DF.head())
print(" ")
print("GAUSSIAN DATASET SHAPE:")
print(gaussian_DF.shape)
print(" ")

# Print column names for each dataset.
print("MULTINOMIAL DATASET COLUMNS:")
print(multinomial_DF.columns)
print(" ")

print("BERNOULLI DATASET COLUMNS:")
print(bernoulli_DF.columns)
print(" ")

print("GAUSSIAN DATASET COLUMNS:")
print(gaussian_DF.columns)
print(" ")




##### Perform Multinomial Naive Bayes

# Use train_test_split to create our training dataset and our testing dataset.
training_dataset, testing_dataset = train_test_split(multinomial_DF, test_size=0.3, random_state=42)
print(training_dataset)
print(testing_dataset)

# Print training and testing set sizes.
print("Training Set Shape (MNB):")
print(training_dataset.shape)
print("Testing Set Shape (MNB):")
print(testing_dataset.shape)
print(" ")

# Print first few rows of training and testing sets.
print("Training Set Head (MNB):")
print(training_dataset.head())
print(" ")
print("Testing Set Head (MNB):")
print(testing_dataset.head())
print(" ")

# Check if training and testing sets overlap.
overlap_rows = pd.merge(training_dataset, testing_dataset, how="inner")
print("Number of overlapping rows between training and testing sets (MNB):")
print(len(overlap_rows))
print(" ")

# Extract training and testing labels, and then remove them from the dataset.
training_labels = training_dataset["Popular"]
testing_labels = testing_dataset["Popular"]
training_dataset_no_label = training_dataset.drop(columns=["Popular"])
testing_dataset_no_label = testing_dataset.drop(columns=["Popular"])

# Create the model.
myMNB = MultinomialNB()

# Fit the model to the training data.
myMNB.fit(training_dataset_no_label, training_labels)

# Run model on test data.
prediction = myMNB.predict(testing_dataset_no_label)

# Compute and print confusion matrix on predicted results.
cm = confusion_matrix(testing_labels, prediction)
print("\nThe confusion matrix for Multinomial NB is:")
print(cm)

# Compute accuracy by dividing trace of confusion matrix by the sum of all 
# entries in the confusion matrix.
number_correct = cm[0,0] + cm[1,1]
total_number = number_correct + cm[0,1] + cm[1,0]
accuracy = number_correct / total_number
print("Accuracy of Multinomial NB:")
print(accuracy)
print("")

# Print summary for easier copy/paste into report.
print("===== MULTINOMIAL NB SUMMARY =====")
print("Confusion Matrix:")
print(cm)
print("Accuracy:")
print(accuracy)
print("Classes:")
print(myMNB.classes_)
print(" ")

# Display the confusion matrix for visualization.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["L", "H"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Multinomial Naive Bayes")
plt.show()

# Get predicted probabilities.
probs = myMNB.predict_proba(testing_dataset_no_label)

# Convert to DataFrame for easy plotting.
prob_df = pd.DataFrame(probs, columns=myMNB.classes_)

# Plot probabilities for first 20 test examples.
prob_df.iloc[:20].plot(kind="bar", figsize=(12,6))
plt.title("Predicted Class Probabilities for First 20 Test Samples (MNB)")
plt.xlabel("Test Sample Index")
plt.ylabel("Predicted Probability")
plt.legend(title="Class")
plt.tight_layout()
plt.show()

# Create DataFrame of feature log probabilities.
feature_log_probs = pd.DataFrame(
    myMNB.feature_log_prob_,
    index=myMNB.classes_,
    columns=training_dataset_no_label.columns
)

# Plot heatmap.
plt.figure(figsize=(14,6))
sns.heatmap(feature_log_probs, cmap="coolwarm", annot=False)
plt.title("Feature Log Probabilities by Class (Multinomial Naive Bayes)")
plt.xlabel("Features")
plt.ylabel("Class")
plt.tight_layout()
plt.show()

# Create data frame containing the actual and predicted results.
results_df = pd.DataFrame({
    "Actual": testing_labels.values,
    "Predicted": prediction
})

# Plot counts.
plt.figure(figsize=(8,5))
sns.countplot(data=results_df, x="Actual", hue="Predicted")
plt.title("Actual vs Predicted Class Counts (MNB)")
plt.xlabel("Actual Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()




#### Perform Bernoulli Naive Bayes

# Use train_test_split to create our training dataset and our testing dataset.
training_dataset, testing_dataset = train_test_split(bernoulli_DF, test_size=0.3, random_state=42)
print(training_dataset)
print(testing_dataset)

# Print training and testing set sizes.
print("Training Set Shape (Bernoulli):")
print(training_dataset.shape)
print("Testing Set Shape (Bernoulli):")
print(testing_dataset.shape)
print(" ")

# Print first few rows of training and testing sets.
print("Training Set Head (Bernoulli):")
print(training_dataset.head())
print(" ")
print("Testing Set Head (Bernoulli):")
print(testing_dataset.head())
print(" ")

# Check if training and testing sets overlap.
overlap_rows = pd.merge(training_dataset, testing_dataset, how="inner")
print("Number of overlapping rows between training and testing sets (Bernoulli):")
print(len(overlap_rows))
print(" ")

# Extract training and testing labels, and then remove them from the dataset.
training_labels = training_dataset["Popular"]
testing_labels = testing_dataset["Popular"]
training_dataset_no_label = training_dataset.drop(columns=["Popular"])
testing_dataset_no_label = testing_dataset.drop(columns=["Popular"])

# Instantiate instance of BernoulliNB and train the model.
BernModel = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
BernModel.fit(training_dataset_no_label, training_labels)

# Run model on test data.
prediction = BernModel.predict(testing_dataset_no_label)

# Compute and print confusion matrix on predicted results.
cm = confusion_matrix(testing_labels, prediction)
print("\nThe confusion matrix for Bernoulli NB is:")
print(cm)

# Compute accuracy by dividing trace of confusion matrix by the sum of all 
# entries in the confusion matrix.
number_correct = cm[0,0] + cm[1,1]
total_number = number_correct + cm[0,1] + cm[1,0]
accuracy = number_correct / total_number
print("Accuracy of Bernoulli NB:")
print(accuracy)
print("")

# Print summary for easier copy/paste into report.
print("===== BERNOULLI NB SUMMARY =====")
print("Confusion Matrix:")
print(cm)
print("Accuracy:")
print(accuracy)
print("Classes:")
print(BernModel.classes_)
print(" ")

# Display the confusion matrix for visualization.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["L", "H"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Bernoulli Naive Bayes")
plt.show()

# Get predicted probabilities.
probs = BernModel.predict_proba(testing_dataset_no_label)

# Convert to DataFrame for easy plotting.
prob_df = pd.DataFrame(probs, columns=BernModel.classes_)

# Plot probabilities for first 20 test examples.
prob_df.iloc[:20].plot(kind="bar", figsize=(12,6))
plt.title("Predicted Class Probabilities for First 20 Test Samples (Bernoulli)")
plt.xlabel("Test Sample Index")
plt.ylabel("Predicted Probability")
plt.legend(title="Class")
plt.tight_layout()
plt.show()

# Create DataFrame of feature log probabilities.
feature_log_probs = pd.DataFrame(
    BernModel.feature_log_prob_,
    index=BernModel.classes_,
    columns=training_dataset_no_label.columns
)

# Plot heatmap.
plt.figure(figsize=(14,6))
sns.heatmap(feature_log_probs, cmap="coolwarm", annot=False)
plt.title("Feature Log Probabilities by Class (Bernoulli Naive Bayes)")
plt.xlabel("Features")
plt.ylabel("Class")
plt.tight_layout()
plt.show()

# Create data frame containing the actual and predicted results.
results_df = pd.DataFrame({
    "Actual": testing_labels.values,
    "Predicted": prediction
})

# Plot counts.
plt.figure(figsize=(8,5))
sns.countplot(data=results_df, x="Actual", hue="Predicted")
plt.title("Actual vs Predicted Class Counts (Bernoulli)")
plt.xlabel("Actual Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()




#### Perform Gaussian Naive Bayes

# Use train_test_split to create our training dataset and our testing dataset.
training_dataset, testing_dataset = train_test_split(gaussian_DF, test_size=0.3, random_state=42)
print(training_dataset)
print(testing_dataset)

# Print training and testing set sizes.
print("Training Set Shape (Gaussian):")
print(training_dataset.shape)
print("Testing Set Shape (Gaussian):")
print(testing_dataset.shape)
print(" ")

# Print first few rows of training and testing sets.
print("Training Set Head (Gaussian):")
print(training_dataset.head())
print(" ")
print("Testing Set Head (Gaussian):")
print(testing_dataset.head())
print(" ")

# Check if training and testing sets overlap.
overlap_rows = pd.merge(training_dataset, testing_dataset, how="inner")
print("Number of overlapping rows between training and testing sets (Gaussian):")
print(len(overlap_rows))
print(" ")

# Extract training and testing labels, and then remove them from the dataset.
training_labels = training_dataset["Numeric_Labels"]
testing_labels = testing_dataset["Numeric_Labels"]
training_dataset_no_label = training_dataset.drop(columns=["Labels", "Numeric_Labels"])
testing_dataset_no_label = testing_dataset.drop(columns=["Labels", "Numeric_Labels"])

# Instantiate instance of GaussianNB and train the model.
GaussModel = GaussianNB()
GaussModel.fit(training_dataset_no_label, training_labels)

# Run model on test data.
prediction = GaussModel.predict(testing_dataset_no_label)

# Compute and print confusion matrix on predicted results.
cm = confusion_matrix(testing_labels, prediction)
print("\nThe confusion matrix for Gaussian NB is:")
print(cm)

# Compute accuracy by dividing trace of confusion matrix by the sum of all 
# entries in the confusion matrix.
number_correct = cm[0,0] + cm[1,1]
total_number = number_correct + cm[0,1] + cm[1,0]
accuracy = number_correct / total_number
print("Accuracy of Gaussian NB:")
print(accuracy)
print("")

# Print summary for easier copy/paste into report.
print("===== GAUSSIAN NB SUMMARY =====")
print("Confusion Matrix:")
print(cm)
print("Accuracy:")
print(accuracy)
print("Classes:")
print(GaussModel.classes_)
print(" ")

# Display the confusion matrix for visualization.
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["L", "H"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix for Gaussian Naive Bayes")
plt.show()

# Get predicted probabilities.
probs = GaussModel.predict_proba(testing_dataset_no_label)

# Convert to DataFrame for easy plotting.
prob_df = pd.DataFrame(probs, columns=GaussModel.classes_)

# Plot probabilities for first 20 test examples.
prob_df.iloc[:20].plot(kind="bar", figsize=(12,6))
plt.title("Predicted Class Probabilities for First 20 Test Samples (Gaussian)")
plt.xlabel("Test Sample Index")
plt.ylabel("Predicted Probability")
plt.legend(title="Class")
plt.tight_layout()
plt.show()

# Create data frame containing the actual and predicted results.
results_df = pd.DataFrame({
    "Actual": testing_labels.values,
    "Predicted": prediction
})

# Plot counts.
plt.figure(figsize=(8,5))
sns.countplot(data=results_df, x="Actual", hue="Predicted")
plt.title("Actual vs Predicted Class Counts (Gaussian)")
plt.xlabel("Actual Class")
plt.ylabel("Count")
plt.tight_layout()
plt.show()