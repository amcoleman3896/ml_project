# -*- coding: utf-8 -*-
"""
Created on Wed Apr 1 2026

@author: Austin Coleman
"""

#### Import Libraries

# Import libraries used.
import pandas as pd
import numpy as np




#### Load in Data

# Declare filename of data to load in.
dataset_filename = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset.csv"

# Read in data as dataframe using pandas.
videogameDF = pd.read_csv(dataset_filename)

print(videogameDF.head())
print(videogameDF.columns)
print(videogameDF.shape)

# Remove column Unnamed: 0 if it exists.
if "Unnamed: 0" in videogameDF.columns:
    videogameDF = videogameDF.drop(columns=["Unnamed: 0"])




#### Create Binary Target Variable

# Create binary target variable where:
# 1 = above median global sales
# 0 = below or equal to median global sales
sales_threshold = videogameDF["Global_Sales"].median()
videogameDF["Popular"] = (videogameDF["Global_Sales"] > sales_threshold).astype(int)

print("\nMedian Global Sales Threshold:")
print(sales_threshold)

print("\nPopular Class Counts:")
print(videogameDF["Popular"].value_counts())




#### Remove Columns That Would Cause Data Leakage

# Remove columns that directly contain sales information.
videogameDF = videogameDF.drop(columns=[
    "Name",
    "NA_Sales",
    "EU_Sales",
    "JP_Sales",
    "Other_Sales",
    "Global_Sales"
], errors="ignore")




#### Fill Missing Values

# Fill missing categorical values with "Unknown".
categorical_columns = ["Platform", "Genre", "Publisher", "Developer", "Rating"]

for col in categorical_columns:
    if col in videogameDF.columns:
        videogameDF[col] = videogameDF[col].fillna("Unknown").astype(str)

# Fill missing numeric/count values with the median of the column.
numeric_columns = ["Critic_Count", "User_Count", "Year_of_Release", "Critic_Score", "User_Score"]

for col in numeric_columns:
    if col in videogameDF.columns:
        videogameDF[col] = pd.to_numeric(videogameDF[col], errors="coerce")
        videogameDF[col] = videogameDF[col].fillna(videogameDF[col].median())




############################################################
#### Create Dataset for Bernoulli Naive Bayes
############################################################

# Create empty dataframe to hold Bernoulli NB variables.
bernoulli_dataset = pd.DataFrame()




#### Platform Binary Features

# Create binary platform indicator variables.
if "Platform" in videogameDF.columns:
    videogameDF["Platform"] = videogameDF["Platform"].astype(str)

    bernoulli_dataset["Is_PlayStation"] = videogameDF["Platform"].str.contains("PS", case=False, na=False).astype(int)
    bernoulli_dataset["Is_Xbox"] = videogameDF["Platform"].str.contains("X", case=False, na=False).astype(int)
    bernoulli_dataset["Is_Nintendo"] = videogameDF["Platform"].isin(
        ["Wii", "DS", "3DS", "WiiU", "GB", "GBA", "GC", "N64", "NES", "SNES", "Switch"]
    ).astype(int)
    bernoulli_dataset["Is_PC"] = videogameDF["Platform"].str.contains("PC", case=False, na=False).astype(int)
    bernoulli_dataset["Is_Handheld"] = videogameDF["Platform"].isin(
        ["DS", "3DS", "GB", "GBA", "PSP", "PSV"]
    ).astype(int)




#### Genre Binary Features

# Create binary indicator variables for each genre.
if "Genre" in videogameDF.columns:
    genres = videogameDF["Genre"].dropna().unique()

    for genre in genres:
        col_name = "Genre_" + str(genre).replace(" ", "_")
        bernoulli_dataset[col_name] = (videogameDF["Genre"] == genre).astype(int)




#### Rating Binary Features

# Create binary ESRB rating indicator variables.
if "Rating" in videogameDF.columns:
    bernoulli_dataset["Is_E_Rated"] = (videogameDF["Rating"] == "E").astype(int)
    bernoulli_dataset["Is_T_Rated"] = (videogameDF["Rating"] == "T").astype(int)
    bernoulli_dataset["Is_M_Rated"] = (videogameDF["Rating"] == "M").astype(int)




#### Score and Review Binary Features

# Create binary indicator variables based on score/review thresholds.
if "Critic_Score" in videogameDF.columns:
    bernoulli_dataset["High_Critic_Score"] = (videogameDF["Critic_Score"] >= 75).astype(int)

if "User_Score" in videogameDF.columns:
    bernoulli_dataset["High_User_Score"] = (videogameDF["User_Score"] >= 8).astype(int)

if "Critic_Count" in videogameDF.columns:
    critic_count_median = videogameDF["Critic_Count"].median()
    bernoulli_dataset["High_Critic_Count"] = (videogameDF["Critic_Count"] > critic_count_median).astype(int)

if "User_Count" in videogameDF.columns:
    user_count_median = videogameDF["User_Count"].median()
    bernoulli_dataset["High_User_Count"] = (videogameDF["User_Count"] > user_count_median).astype(int)




#### Year Binary Features

# Create binary indicator variables based on release year.
if "Year_of_Release" in videogameDF.columns:
    bernoulli_dataset["Released_After_2010"] = (videogameDF["Year_of_Release"] >= 2010).astype(int)
    bernoulli_dataset["Released_Before_2000"] = (videogameDF["Year_of_Release"] < 2000).astype(int)




#### Top Publisher and Developer Binary Features

# Create binary indicator variables for top publishers.
if "Publisher" in videogameDF.columns:
    top_publishers = videogameDF["Publisher"].value_counts().head(10).index

    for pub in top_publishers:
        col_name = "Publisher_" + str(pub).replace(" ", "_").replace("/", "_")
        bernoulli_dataset[col_name] = (videogameDF["Publisher"] == pub).astype(int)

# Create binary indicator variables for top developers.
if "Developer" in videogameDF.columns:
    top_developers = videogameDF["Developer"].value_counts().head(10).index

    for dev in top_developers:
        col_name = "Developer_" + str(dev).replace(" ", "_").replace("/", "_")
        bernoulli_dataset[col_name] = (videogameDF["Developer"] == dev).astype(int)




#### Add Target Variable Back In

# Add target variable to Bernoulli dataset.
bernoulli_dataset["Popular"] = videogameDF["Popular"]

print("\nBernoulli Dataset Shape:")
print(bernoulli_dataset.shape)

print("\nBernoulli Dataset Preview:")
print(bernoulli_dataset.head())

# Save Bernoulli dataset to csv.
bernoulli_dataset.to_csv("videogame_bernoulli_dataset.csv", index=False)

print("\nBernoulli dataset saved as: videogame_bernoulli_dataset.csv")




############################################################
#### Create Dataset for Multinomial Naive Bayes
############################################################

# Create empty dataframe to hold Multinomial NB variables.
multinomial_dataset = pd.DataFrame()




#### Keep Count-Based Variables

# Keep count variables already present in the dataset.
if "Critic_Count" in videogameDF.columns:
    multinomial_dataset["Critic_Count"] = videogameDF["Critic_Count"].astype(int)

if "User_Count" in videogameDF.columns:
    multinomial_dataset["User_Count"] = videogameDF["User_Count"].astype(int)




#### Frequency Encode Categorical Variables

# Convert categorical variables into frequency-based count variables.
for col in ["Platform", "Genre", "Publisher", "Developer", "Rating"]:
    if col in videogameDF.columns:
        freq_map = videogameDF[col].value_counts().to_dict()
        multinomial_dataset[col + "_Freq"] = videogameDF[col].map(freq_map).astype(int)




#### Add Other Integer-Valued Variables

# Add integer versions of other useful numeric variables.
if "Year_of_Release" in videogameDF.columns:
    multinomial_dataset["Year_of_Release"] = videogameDF["Year_of_Release"].round().astype(int)

if "Critic_Score" in videogameDF.columns:
    multinomial_dataset["Critic_Score"] = videogameDF["Critic_Score"].round().astype(int)

if "User_Score" in videogameDF.columns:
    multinomial_dataset["User_Score_x10"] = (videogameDF["User_Score"] * 10).round().astype(int)




#### Ensure All Values Are Nonnegative

# Clip all values at 0 to make sure nothing is negative.
multinomial_dataset = multinomial_dataset.clip(lower=0)




#### Add Target Variable Back In

# Add target variable to Multinomial dataset.
multinomial_dataset["Popular"] = videogameDF["Popular"]

print("\nMultinomial Dataset Shape:")
print(multinomial_dataset.shape)

print("\nMultinomial Dataset Preview:")
print(multinomial_dataset.head())

# Save Multinomial dataset to csv.
multinomial_dataset.to_csv("videogame_multinomial_dataset.csv", index=False)

print("\nMultinomial dataset saved as: videogame_multinomial_dataset.csv")