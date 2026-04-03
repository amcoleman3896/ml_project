# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 17:43:16 2026

@author: Austin Coleman
"""

#### Import Libraries

# Import all libraries used.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz 
import pydotplus
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Add path for graphviz
os.environ["PATH"] = r"C:\Program Files (x86)\Graphviz\bin" + os.pathsep + os.environ["PATH"]




#### Load in Data

# Declare filename of data to load in.
dataset_filename="C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset_for_DT.csv"
label_filename = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/labels_for_DT.csv"

# Read in data as dataframe using pandas.
videogameDF = pd.read_csv(dataset_filename)
print(videogameDF.dtypes)

# Remove columns Unnamed: 0
videogameDF = videogameDF.drop(columns=["Unnamed: 0"])

# Read in labels as array using numpy.
labels_for_DT = np.loadtxt(label_filename,dtype="str")




##### Perform Decisicion Tree Classification No Max Depth

# Use train_test_split to create our training dataset and our testing dataset.
training_dataset, testing_dataset = train_test_split(videogameDF, test_size = 0.3, random_state = 42)
print(training_dataset)
print(testing_dataset)

# Extract training and testing labels, and then remove them from the dataset.
training_labels = training_dataset["Numeric_Labels"]
testing_labels = testing_dataset["Numeric_Labels"]
training_dataset_no_label = training_dataset.drop(columns=["Labels","Numeric_Labels"])
testing_dataset_no_label = testing_dataset.drop(columns=["Labels","Numeric_Labels"])


# Instantiate an instance of DecisionTreeClassifier
MyDT=DecisionTreeClassifier(random_state=42)

# Train the decision tree.
MyDT = MyDT.fit(training_dataset_no_label, training_labels) 

# Create visualization object.
TREE_Vis = tree.export_graphviz(MyDT, 
                    out_file=None, 
                    feature_names=training_dataset_no_label.columns,  
                    class_names=["L","H"],    
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(TREE_Vis)  
graph 

# Export to pdf
DT_graph = pydotplus.graph_from_dot_data(TREE_Vis)
DT_graph.write_png("My_DT_tree_no_max_depth.png")

# Print the accuracy of the training and testing data. Tree was huge, likely
# overfit. Thus, testing accuracy will likely be less.
print("Training Accuracy:", MyDT.score(training_dataset_no_label, training_labels))
print("Testing Accuracy:", MyDT.score(testing_dataset_no_label, testing_labels))
print(" ")




#### Determine Best Max Depth

# Create empty lists to store max depth values and accuracies.
depth_values = []
training_accuracies = []
testing_accuracies = []

# Loop through different max_depth values.
for depth in range(1, 21):
    
    # Instantiate an instance of DecisionTreeClassifier with current max_depth.
    MyDT = DecisionTreeClassifier(max_depth = depth, random_state = 42)
    
    # Fit the Decision Tree model using the training dataset.
    MyDT = MyDT.fit(training_dataset_no_label, training_labels)
    
    # Calculate training accuracy and testing accuracy.
    train_accuracy = MyDT.score(training_dataset_no_label, training_labels)
    test_accuracy = MyDT.score(testing_dataset_no_label, testing_labels)
    
    # Append current depth and accuracies to lists.
    depth_values.append(depth)
    training_accuracies.append(train_accuracy)
    testing_accuracies.append(test_accuracy)

# Print results for each max_depth value.
for i in range(len(depth_values)):
    print("Max Depth:", depth_values[i], 
          " Training Accuracy:", training_accuracies[i], 
          " Testing Accuracy:", testing_accuracies[i])
print(" ")




#### Perform DT With Best Max Depth (3) w GINI Choosing Root Node

# Instantiate instance of DecisionTreeClassifier with set random seed generator
# and best max depth (2 or 3, going with 3 to have more leaves).
MyDT=DecisionTreeClassifier(max_depth = 3, random_state = 42)

# Train the decision tree.
MyDT = MyDT.fit(training_dataset_no_label, training_labels) 

# Create visualization object.
TREE_Vis = tree.export_graphviz(MyDT, 
                    out_file=None, 
                    feature_names=training_dataset_no_label.columns,  
                    class_names=["L","H"],   
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(TREE_Vis)  
graph 

# Export to pdf
DT_graph = pydotplus.graph_from_dot_data(TREE_Vis)
DT_graph.write_png("My_DT_tree_best_max_depth_3_root_1.png")

# Print the accuracy of the training and testing data for best max depth.
print("Training Accuracy Max Depth 3 Root 1 (User Count):", MyDT.score(training_dataset_no_label, training_labels))
print("Testing Accuracy Max Depth 3 Root 1 (User Count):", MyDT.score(testing_dataset_no_label, testing_labels))

# Compute confusion matrix and print it.
test_pred = MyDT.predict(testing_dataset_no_label)
# Define numeric labels
labels = [0, 1]  # 0=L, 1=H
label_names = ["L", "H"]

# Compute confusion matrix, forcing order of labels
cm = confusion_matrix(testing_labels, test_pred, labels=labels)
print("Confusion Matrix Max Depth 3 Root 1 (User Count):")
print(cm)
print("")


plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names,  # Predicted labels
            yticklabels=label_names)  # True labels
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (DT) User Count Root")
plt.show()


#### Perform DT With Best Max Depth With User_Count Removed

# Remove the column User_Count from the data (was the root node above).
videogameDF = videogameDF.drop(columns=["User_Count"])

# Create new training and testing datasets.
training_dataset, testing_dataset = train_test_split(videogameDF, test_size = 0.3, random_state = 42)

# Extract training and testing labels, and then remove them from the dataset.
training_labels = training_dataset["Numeric_Labels"]
testing_labels = testing_dataset["Numeric_Labels"]
training_dataset_no_label = training_dataset.drop(columns=["Labels","Numeric_Labels"])
testing_dataset_no_label = testing_dataset.drop(columns=["Labels","Numeric_Labels"])

# Instantiate instance of DecisionTreeClassifier with set random seed generator
# and best max depth (3).
MyDT=DecisionTreeClassifier(max_depth = 3, random_state = 42)

# Train the decision tree.
MyDT = MyDT.fit(training_dataset_no_label, training_labels) 

# Create visualization object.
TREE_Vis = tree.export_graphviz(MyDT, 
                    out_file=None, 
                    feature_names=training_dataset_no_label.columns,  
                    class_names=["L","H"],     
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(TREE_Vis)  
graph 

# Export to pdf
DT_graph = pydotplus.graph_from_dot_data(TREE_Vis)
DT_graph.write_png("My_DT_tree_best_max_depth_3_root_2.png")

# Print the accuracy of the training and testing data for 2nd best max depth.
print("Training Accuracy Max Depth 3 Root 2 (Critic Count):", MyDT.score(training_dataset_no_label, training_labels))
print("Testing Accuracy Max Depth 3 Root 2 (Critic Count):", MyDT.score(testing_dataset_no_label, testing_labels))

# Compute confusion matrix and print it.
test_pred = MyDT.predict(testing_dataset_no_label)
# Define numeric labels
labels = [0, 1]  # 0=L, 1=H
label_names = ["L", "H"]

# Compute confusion matrix, forcing order of labels
cm = confusion_matrix(testing_labels, test_pred, labels=labels)
print("Confusion Matrix Max Depth 3 Root 2 (Critic Count):")
print(cm)
print("")


plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names,  # Predicted labels
            yticklabels=label_names)  # True labels
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (DT) Critic Count Root")
plt.show()




#### Perform DT With Best Max Depth With User_Count Removed

# Remove the column User_Count from the data (was the root node above).
videogameDF = videogameDF.drop(columns=["Critic_Count"])

# Create new training and testing datasets.
training_dataset, testing_dataset = train_test_split(videogameDF, test_size = 0.3, random_state = 42)

# Extract training and testing labels, and then remove them from the dataset.
training_labels = training_dataset["Numeric_Labels"]
testing_labels = testing_dataset["Numeric_Labels"]
training_dataset_no_label = training_dataset.drop(columns=["Labels","Numeric_Labels"])
testing_dataset_no_label = testing_dataset.drop(columns=["Labels","Numeric_Labels"])

# Instantiate instance of DecisionTreeClassifier with set random seed generator
# and best max depth (3).
MyDT=DecisionTreeClassifier(max_depth = 3, random_state = 42)

# Train the decision tree.
MyDT = MyDT.fit(training_dataset_no_label, training_labels) 

# Create visualization object.
TREE_Vis = tree.export_graphviz(MyDT, 
                    out_file=None, 
                    feature_names=training_dataset_no_label.columns,  
                    class_names=["L","H"], 
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(TREE_Vis)  
graph 

# Export to pdf
DT_graph = pydotplus.graph_from_dot_data(TREE_Vis)
DT_graph.write_png("My_DT_tree_best_max_depth_3_root_3.png")

# Print the accuracy of the training and testing data for 2nd best max depth.
print("Training Accuracy Max Depth 3 Root 3 (Global Sales):", MyDT.score(training_dataset_no_label, training_labels))
print("Testing Accuracy Max Depth 3 Root 3 (Global Sales):", MyDT.score(testing_dataset_no_label, testing_labels))

# Compute confusion matrix and print it.
test_pred = MyDT.predict(testing_dataset_no_label)
cm = confusion_matrix(testing_labels, test_pred)
print("Confusion Matrix Max Depth 3 Root 3 (Global Sales):")
print(cm)
print("")

# Map numeric indices to label strings
label_names = ["L", "H"]  # 0=L, 1=H

plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_names,  # Predicted labels
            yticklabels=label_names)  # True labels
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (DT) Global Sales Root")
plt.show()





