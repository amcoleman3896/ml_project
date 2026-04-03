# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:04:38 2026

@author: Austin Coleman
"""
#### Import Libraries

# Import all libraries used.
import pandas as pd
import numpy as np




#### Load in Data

# Declare filename of data to load in.
filename="C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset.csv"

# Read in data as dataframe using pandas.
videogameDF = pd.read_csv(filename)
print(videogameDF.dtypes)




#### Create Label Dataset

# Use a for-loop to go over each row in the dataset to create label for each
# observation. Will take average of critic score and user score - if it is 8
# or above, the label column titled "High_Score" will be labeled either as 
# "Y" or "N" (yes or no).
high_score_labels = np.zeros([len(videogameDF["User_Score"]),1],dtype="str")
high_score_labels_int = np.zeros([len(videogameDF["User_Score"]),1],dtype='int')
high_score_count = 0
num_games = 0
for i in range(len(videogameDF["User_Score"])):
    
    # Compute the average score for the current videogame. Must divide critic
    # score by 10 because user score is rated out of 10, not 100.
    current_average_score = (videogameDF.loc[i,"User_Score"] + (videogameDF.loc[i,"Critic_Score"]/10)) / 2
    
    # If the score is 8 or greater, assign "Y" as the label. Otherwise, label
    # as "N".
    if current_average_score >= 8:
        high_score_labels[i] = "H"
        high_score_labels_int[i] = 1
        high_score_count = high_score_count + 1
    else:
        high_score_labels[i] = "L"
        high_score_labels_int[i] = 0
    
    num_games = num_games + 1
        
# Add that as a column to the DF.
videogameDF['Labels'] = high_score_labels
videogameDF['Numeric_Labels'] = high_score_labels_int

# Now need to remove the rows that were used to create the label.
videogameDF = videogameDF.drop(columns=["User_Score", "Critic_Score"])    

# Next need to drop all columns that aren't quantitative.
videogameDF = videogameDF.drop(columns=["Rating", "Developer","Publisher","Genre","Platform","Name","Unnamed: 0"])

# Save cleaned dataframe as separate dataset and save labels as well.
videogameDF.to_csv("cleaned_kaggle_dataset_for_DT.csv")
np.savetxt('labels_for_DT.csv', high_score_labels, fmt='%s')