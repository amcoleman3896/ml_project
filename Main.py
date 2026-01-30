# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 11:46:41 2026

@author: Austin Coleman
"""

# Import necessary libraries.
import pandas as pd
import numpy as np
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt
import random


#### Load in Data

## Declare Dataset ##
## https://drive.google.com/file/d/1J4hQPngIfP5rez1ciFQL4mXvCrGKXk2t/view?usp=sharing

# Load in the data in using pandas.
filename="C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/Video_Game_Main_Data_Sheet.csv"

# Check out the data just to see how it is read in.
videogameDF = pd.read_csv(filename)
print(videogameDF.dtypes)



#### Dirty Up the Data

# Create list of random crap we can add to spots of the data to dirty it up.
garbage_values = ['N/A', 1782, -999, None, 2030, 'unknown']

# Decide how many rows to dirty up.
num_rows = len(videogameDF)
percent_dirty = 0.02   # 2% of rows
num_dirty = int(percent_dirty * num_rows)

# Go over each column to randomly add dirty data to.
for each_column in videogameDF.columns:
    
    # Specify the percent of data we wish to dirty based on the column.
    if each_column == "Name":
        percent_dirty = .005
    elif each_column == "Platform":
        percent_dirty = .005
    elif each_column == "Genre":
        percent_dirty = .005
    elif each_column == "Publisher":
        percent_dirty = .005
    elif each_column == "Developer":
        percent_dirty = .005
    elif each_column == "Rating":
        percent_dirty = .005
    else:
        percent_dirty = .05
    
    # Establish how many rows we will dirty up.
    num_dirty = int(percent_dirty * num_rows)
    
    # Pick random row indices to mess up for the current column.
    dirty_indices = random.sample(range(num_rows), num_dirty)
    
    # Go over each random index for the current column to reassign garbage 
    # value to.
    for idx in dirty_indices:
       
        # For the random indices chosen for this column, randomly assign 
        # garbage values based on the list defined above.
        videogameDF.loc[idx, each_column] = random.choice(garbage_values)
    
    
# See how dirtying the data changed the data types.
print(videogameDF.dtypes)





#### Create Numeric Columns and Remove Negative Indices

# Before we even start to visualize the dirty data, we know that the following 
# columns should only be numeric: year of release, NA sales, EU sales, JP 
# sales, other sales, global sales, critic and user scores, critic and user 
# counts. Change columns to numeric to automatically turn other values to ''.
# Also, find all negative values and set to '' as well.
numeric_columns = ["Year_of_Release","NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales",
                   "Critic_Score","Critic_Count","User_Score","User_Count"]
for current_column in numeric_columns:
    videogameDF[current_column] = pd.to_numeric(videogameDF[current_column], errors='coerce')
    negative_indices = videogameDF[current_column] <= 0
    videogameDF.loc[negative_indices,current_column] = None




#### Clean Year of Release


# Plot the Year_of_Release to visualize what is included after keeping only 
# numeric values and removing all negative values.
year_of_release_counts = videogameDF["Year_of_Release"].value_counts().sort_index()
year_of_release_counts.columns = ["Year_of_Release","Count"]
year_of_release_counts.plot(kind='line', figsize=(4,4), x="Year_of_Release", y="Count")
plt.xlabel('Year of Release')
plt.ylabel('Count')
plt.title('Number of Video Games Released Per Year (Pre-Cleaning)')
plt.show()

# Determine the minimum and maximum year all the games released.
print('Max Year of Release:')
print(videogameDF["Year_of_Release"].max())
print('Min Year of Release:')
print(videogameDF["Year_of_Release"].min())
print('')

# Remove all instances of video games that were labeled to be before the year 
# 1980 and all instances of video games released after 2025.
indices_above_max_year_of_release = videogameDF["Year_of_Release"] >= 2025
indices_below_min_year_of_release = videogameDF["Year_of_Release"] <= 1980
videogameDF.loc[indices_above_max_year_of_release,"Year_of_Release"] = None
videogameDF.loc[indices_below_min_year_of_release,"Year_of_Release"] = None

# Delete rows where we don't have year of release
videogameDF = videogameDF.dropna(subset=["Year_of_Release"])

# Replot the data.
year_of_release_counts = videogameDF["Year_of_Release"].value_counts().sort_index()
year_of_release_counts.columns = ["Year_of_Release","Count"]
year_of_release_counts.plot(kind='line', figsize=(4,4), x="Year_of_Release", y="Count")
plt.xlabel('Year of Release')
plt.ylabel('Count')
plt.title('Number of Video Games Released Per Year (Post-Cleaning)')
plt.show()






#### Clean Sales Columns

# Be sure to remove outliers in sales prior to approximating empty results. 

## Box Plot of NA_Sales per Year Prior to Cleaning ##
sales_per_year_plot = videogameDF.dropna(subset=["NA_Sales", "Year_of_Release"])
ax = sales_per_year_plot.boxplot(
    column="NA_Sales",
    by="Year_of_Release",
    figsize=(4 ,4),
    grid=False,
    showfliers=True  # optional: hides extreme outliers for cleaner view
)
plt.xlabel("Year of Release")
plt.ylabel("NA Sales (millions)")
plt.title("NA Sales per Year (Pre-Cleaning)")
plt.suptitle("")  # remove default 'Boxplot grouped by ...' title
# show every 5th year (change 5 to whatever you want)
ticks = ax.get_xticks()
labels = ax.get_xticklabels()

ax.set_xticks(ticks[::5])
ax.set_xticklabels(labels[::5], rotation=90)
plt.show()

# Remove outliers - either 2030 or 1782 million sales are not realistic. Remove
# these values from all sales columns.
sales_columns = ["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]
for current_sale_column in sales_columns:
    indices_remove_sales = videogameDF[current_sale_column] >= 1782
    videogameDF.loc[indices_remove_sales,current_sale_column] = None
    


## NA_Sales ##

# Determine the indices where there is a NaN value for NA_Sales
blank_sales_indices = videogameDF["NA_Sales"].isna()

# Use a for-loop to go over each blank NA_Sales cell and populate it with the 
# average sales for that year in that region.
for each_index in videogameDF.index[blank_sales_indices]:
    
    # Extract the current year.
    current_year = videogameDF.loc[each_index,"Year_of_Release"]
    
    # Determine all indices that correspond to games released this year.
    current_year_indices = videogameDF["Year_of_Release"] == current_year
    current_year_indices = videogameDF.index[current_year_indices]
    
    # Extract the sales in NA for this year.
    current_year_sales = videogameDF.loc[current_year_indices,"NA_Sales"]
    
        
    if current_year_sales.count() == 0:
        fill_value = videogameDF["NA_Sales"].median()
    else:
        fill_value = current_year_sales.median()
    
    # Determine the average of sales in NA for this year and assign to the 
    # blank cell.
    videogameDF.loc[each_index,"NA_Sales"] = fill_value
    
    
    
## EU_Sales ##

# Determine the indices where there is a NaN value for EU_Sales
blank_sales_indices = videogameDF["EU_Sales"].isna()

# Use a for-loop to go over each blank EU_Sales cell and populate it with the 
# average sales for that year in that region.
for each_index in videogameDF.index[blank_sales_indices]:
    
    # Extract the current year.
    current_year = videogameDF.loc[each_index,"Year_of_Release"]
    
    # Determine all indices that correspond to games released this year.
    current_year_indices = videogameDF["Year_of_Release"] == current_year
    current_year_indices = videogameDF.index[current_year_indices]
    
    # Extract the sales in EU for this year.
    current_year_sales = videogameDF.loc[current_year_indices,"EU_Sales"]
    
        
    if current_year_sales.count() == 0:
        fill_value = videogameDF["EU_Sales"].median()
    else:
        fill_value = current_year_sales.median()
    
    # Determine the average of sales in NA for this year and assign to the 
    # blank cell.
    videogameDF.loc[each_index,"EU_Sales"] = fill_value
       
    
    
## JP_Sales ##

# Determine the indices where there is a NaN value for JP_Sales
blank_sales_indices = videogameDF["JP_Sales"].isna()

# Use a for-loop to go over each blank JP_Sales cell and populate it with the 
# average sales for that year in that region.
for each_index in videogameDF.index[blank_sales_indices]:
    
    # Extract the current year.
    current_year = videogameDF.loc[each_index,"Year_of_Release"]
    
    # Determine all indices that correspond to games released this year.
    current_year_indices = videogameDF["Year_of_Release"] == current_year
    current_year_indices = videogameDF.index[current_year_indices]
    
    # Extract the sales in JP for this year.
    current_year_sales = videogameDF.loc[current_year_indices,"JP_Sales"]
    
    if current_year_sales.count() == 0:
        fill_value = videogameDF["JP_Sales"].median()
    else:
        fill_value = current_year_sales.median()
    
    # Determine the average of sales in NA for this year and assign to the 
    # blank cell.
    videogameDF.loc[each_index,"JP_Sales"] = fill_value
    
   
    
## Other_Sales ##

# Determine the indices where there is a NaN value for JP_Sales
blank_sales_indices = videogameDF["Other_Sales"].isna()

# Use a for-loop to go over each blank Other_Sales cell and populate it with
# the average sales for that year in that region.
for each_index in videogameDF.index[blank_sales_indices]:
    
    # Extract the current year.
    current_year = videogameDF.loc[each_index,"Year_of_Release"]
    
    # Determine all indices that correspond to games released this year.
    current_year_indices = videogameDF["Year_of_Release"] == current_year
    current_year_indices = videogameDF.index[current_year_indices]
    
    # Extract the sales in Other for this year.
    current_year_sales = videogameDF.loc[current_year_indices,"Other_Sales"]
    
    if current_year_sales.count() == 0:
        fill_value = videogameDF["Other_Sales"].median()
    else:
        fill_value = current_year_sales.median()
    
    # Determine the average of sales in NA for this year and assign to the 
    # blank cell.
    videogameDF.loc[each_index,"Other_Sales"] = fill_value
    
   
    
## Global_Sales ##

# Determine the indices where there is a NaN value for JP_Sales
blank_sales_indices = videogameDF["Global_Sales"].isna()

# Use a for-loop to go over each blank Global_Sales cell and populate it with
# the sum of the sales from everywhere else.
for each_index in videogameDF.index[blank_sales_indices]:
    
    # Extract the sales from all regions and add together.
    total_sales = videogameDF.loc[each_index,"NA_Sales"] + videogameDF.loc[each_index,"EU_Sales"] + videogameDF.loc[each_index,"JP_Sales"] + videogameDF.loc[each_index,"Other_Sales"]
    
    # Save to global sales.
    videogameDF.loc[each_index,"Global_Sales"] = total_sales
    
    
    
# Replot results.
sales_per_year_plot = videogameDF.dropna(subset=["NA_Sales", "Year_of_Release"])
ax = sales_per_year_plot.boxplot(
    column="NA_Sales",
    by="Year_of_Release",
    figsize=(4,4),
    grid=False,
    showfliers=True  # optional: hides extreme outliers for cleaner view
)
plt.xlabel("Year of Release")
plt.ylabel("NA Sales (millions)")
plt.title("NA Sales per Year (Post-Cleaning)")
plt.suptitle("")  # remove default 'Boxplot grouped by ...' title
# show every 5th year (change 5 to whatever you want)
ticks = ax.get_xticks()
labels = ax.get_xticklabels()

ax.set_xticks(ticks[::5])
ax.set_xticklabels(labels[::5], rotation=90)
plt.show()






#### Clean up Genre

# Gather collection of all genres.
genres = videogameDF["Genre"].value_counts()
print("All genres and their count:")
print(genres)

# Remove all rows that have incorrect genres. Cannot make any guess to what 
# the genre would have been unless manually going over every game and deciding.
remove_these_indices = videogameDF["Genre"] == 2030
videogameDF = videogameDF.loc[~remove_these_indices]

remove_these_indices = videogameDF["Genre"] == -999
videogameDF = videogameDF.loc[~remove_these_indices]

remove_these_indices = videogameDF["Genre"] == 1782
videogameDF = videogameDF.loc[~remove_these_indices]

remove_these_indices = videogameDF["Genre"] == "unknown"
videogameDF = videogameDF.loc[~remove_these_indices]

remove_these_indices = videogameDF["Genre"] == "N/A"
videogameDF = videogameDF.loc[~remove_these_indices]

videogameDF = videogameDF.dropna(subset=["Genre"])


genres = videogameDF["Genre"].value_counts()
print("All genres and their count after cleaning:")
print(genres)


#### Clean up Critic/User Score/Count Columns

# Look at max and min of these columns.
print("")
print("Max Critic Score = ")
print(videogameDF["Critic_Score"].max())
print("Min Critic Score = ")
print(videogameDF["Critic_Score"].min())
print("Max Critic Count = ")
print(videogameDF["Critic_Count"].max())
print("Min Critic Count = ")
print(videogameDF["Critic_Count"].min())
print("")

print("")
print("Max User Score = ")
print(videogameDF["User_Score"].max())
print("Min User Score = ")
print(videogameDF["User_Score"].min())
print("Max User Count = ")
print(videogameDF["User_Count"].max())
print("Min User Count = ")
print(videogameDF["User_Count"].min())
print("")


## Critic Score/User Score Cleaning ##

# Set values greater than 10 to None in user score (range is 1-10)
indices = videogameDF["User_Score"] > 10
videogameDF.loc[indices,"User_Score"] = None

# Set critic score values greater than 100 to None. (range is 1-100)
indices = videogameDF["Critic_Score"] > 100
videogameDF.loc[indices,"Critic_Score"] = None

# Look at histogram of user score.
plt.figure(figsize=(4, 4))
sns.histplot(videogameDF["User_Score"], kde=True, bins=100)
plt.title("User Score Distribution (Pre-Cleaning)")
plt.xlabel("User Score")
plt.ylabel("Count")
plt.show()

# Look at histogram of critic score.
plt.figure(figsize=(4, 4))
sns.histplot(videogameDF["Critic_Score"], kde=True, bins=100)
plt.title("Critic Score Distribution (Pre-Cleaning)")
plt.xlabel("Critic Score")
plt.ylabel("Count")
plt.show()


# Now set NaN values based on the median from that genre.
blank_score_indices = videogameDF["Critic_Score"].isna()
for each_index in videogameDF.index[blank_score_indices]:
    
    # Get the current genre of the game.
    current_genre = videogameDF.loc[each_index, "Genre"]
    
    # Get indices that correspond to all games that are this current genre.
    current_genre_indices = videogameDF["Genre"] == current_genre
    current_genre_indices = videogameDF.index[current_genre_indices]
    
    # Get all critic scores from games of this genre.
    all_scores_current_genre = videogameDF.loc[current_genre_indices,"Critic_Score"]
    
    # If there are no valid critic scores for this genre (there should be), 
    # just set the score as the median of all critic scores. 
    if all_scores_current_genre.count() == 0:
        
        fill_value = videogameDF["Critic_Score"].median()
    
    else:
        fill_value = all_scores_current_genre.median()
        
    # Add value to replace the NaN
    videogameDF.loc[each_index,"Critic_Score"] = fill_value
       
        
blank_score_indices = videogameDF["User_Score"].isna()
for each_index in videogameDF.index[blank_score_indices]:
    
    # Get the current genre of the game.
    current_genre = videogameDF.loc[each_index, "Genre"]
    
    # Get indices that correspond to all games that are this current genre.
    current_genre_indices = videogameDF["Genre"] == current_genre
    current_genre_indices = videogameDF.index[current_genre_indices]
    
    # Get all critic scores from games of this genre.
    all_scores_current_genre = videogameDF.loc[current_genre_indices,"User_Score"]
    
    # If there are no valid critic scores for this genre (there should be), 
    # just set the score as the median of all critic scores. 
    if all_scores_current_genre.count() == 0:
        
        fill_value = videogameDF["User_Score"].median()
    
    else:
        fill_value = all_scores_current_genre.median()
        
    # Add value to replace the NaN
    videogameDF.loc[each_index,"User_Score"] = fill_value


# Now visualize Critic_Score and User_Score after cleaning
plt.figure(figsize=(4, 4))
sns.histplot(videogameDF["Critic_Score"], kde=True, bins=100)
plt.title("Critic Score Distribution (Post-Cleaning)")
plt.xlabel("Critic Score")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(4, 4))
sns.histplot(videogameDF["User_Score"], kde=True, bins=100)
plt.title("User Score Distribution (Post-Cleaning)")
plt.xlabel("User Score")
plt.ylabel("Count")
plt.show()



## Visualize Critic Count and User Count ##

# Look at histogram of critic count.
plt.figure(figsize=(4, 4))
sns.histplot(videogameDF["Critic_Count"], kde=True, bins=100)
plt.title("Critic Count Distribution (Pre-Cleaning)")
plt.xlabel("Critic Count")
plt.ylabel("Count")
plt.show()

# Look at histogram of user count.
plt.figure(figsize=(4, 4))
sns.histplot(videogameDF["User_Count"], kde=True, bins=100)
plt.title("User Count Distribution (Pre-Cleaning)")
plt.xlabel("User Count")
plt.ylabel("Count")
plt.show()

# Manually remove values 2030 and 1782.
remove_these_indices = videogameDF["Critic_Count"] == 2030
videogameDF = videogameDF.loc[~remove_these_indices]
remove_these_indices = videogameDF["Critic_Count"] == 1782
videogameDF = videogameDF.loc[~remove_these_indices]
remove_these_indices = videogameDF["User_Count"] == 2030
videogameDF = videogameDF.loc[~remove_these_indices]
remove_these_indices = videogameDF["User_Count"] == 1782
videogameDF = videogameDF.loc[~remove_these_indices]



# Now set NaN values based on the median from that genre.
blank_count_indices = videogameDF["Critic_Count"].isna()
for each_index in videogameDF.index[blank_count_indices]:
    
    # Get the current genre of the game.
    current_genre = videogameDF.loc[each_index, "Genre"]
    
    # Get indices that correspond to all games that are this current genre.
    current_genre_indices = videogameDF["Genre"] == current_genre
    current_genre_indices = videogameDF.index[current_genre_indices]
    
    # Get all critic scores from games of this genre.
    all_counts_current_genre = videogameDF.loc[current_genre_indices,"Critic_Count"]
    
    # If there are no valid critic scores for this genre (there should be), 
    # just set the score as the median of all critic scores. 
    if all_counts_current_genre.count() == 0:
        
        fill_value = videogameDF["Critic_Count"].median()
    
    else:
        fill_value = all_counts_current_genre.median()
        
    # Add value to replace the NaN
    videogameDF.loc[each_index,"Critic_Count"] = fill_value
       
        
blank_count_indices = videogameDF["User_Count"].isna()
for each_index in videogameDF.index[blank_count_indices]:
    
    # Get the current genre of the game.
    current_genre = videogameDF.loc[each_index, "Genre"]
    
    # Get indices that correspond to all games that are this current genre.
    current_genre_indices = videogameDF["Genre"] == current_genre
    current_genre_indices = videogameDF.index[current_genre_indices]
    
    # Get all user counts from this current genre.
    all_counts_current_genre = videogameDF.loc[current_genre_indices,"User_Count"]
    
    # If there are no valid user counts for this genre, replace with median 
    # across all genres. Otherwise just median from this genre.
    if all_counts_current_genre.count() == 0:
        
        fill_value = videogameDF["User_Count"].median()
    
    else:
        fill_value = all_counts_current_genre.median()

    # Add value to replace the NaN
    videogameDF.loc[each_index,"User_Count"] = fill_value


# Look at histogram of critic count.
plt.figure(figsize=(4, 4))
sns.histplot(videogameDF["Critic_Count"], kde=True, bins=100)
plt.title("Critic Count Distribution (Post-Cleaning)")
plt.xlabel("Critic Count")
plt.ylabel("Count")
plt.show()

# Look at histogram of user count.
plt.figure(figsize=(4, 4))
sns.histplot(videogameDF["User_Count"], kde=True, bins=100)
plt.title("User Count Distribution (Post-Cleaning)")
plt.xlabel("User Count")
plt.ylabel("Count")
plt.show()


#### Print Cleaning Results

print("# NAs in Year_of_Release = ")
print(videogameDF["Year_of_Release"].isna().sum())
print("")

print("# NAs in NA_Sales = ")
print(videogameDF["NA_Sales"].isna().sum())
print("")

print("# NAs in EU_Sales = ")
print(videogameDF["EU_Sales"].isna().sum())
print("")

print("# NAs in JP_Sales = ")
print(videogameDF["JP_Sales"].isna().sum())
print("")

print("# NAs in Other_Sales = ")
print(videogameDF["Other_Sales"].isna().sum())
print("")

print("# NAs in Global_Sales = ")
print(videogameDF["Global_Sales"].isna().sum())
print("")

print("# NAs in Critic_Score = ")
print(videogameDF["Critic_Score"].isna().sum())
print("")

print("# NAs in Critic_Count = ")
print(videogameDF["Critic_Count"].isna().sum())
print("")

print("# NAs in User_Score = ")
print(videogameDF["User_Score"].isna().sum())
print("")

print("# NAs in User_Count = ")
print(videogameDF["User_Count"].isna().sum())
print("")






#### Clean Rest of Data

# The rest of the data that has NA values cannot be approximated. Must remove 
# them.
videogameDF = videogameDF.dropna()

# Print results.
print("# NAs in Name:")
print(videogameDF["Name"].isna().sum())
print("")

print("# NAs in Platform:")
print(videogameDF["Platform"].isna().sum())
print("")

print("# NAs in Publisher:")
print(videogameDF["Publisher"].isna().sum())
print("")

print("# NAs in Developer:")
print(videogameDF["Developer"].isna().sum())
print("")

print("# NAs in Rating:")
print(videogameDF["Rating"].isna().sum())
print("")

# See what values are in rating and publisher that don't belong.
print(videogameDF["Platform"].value_counts())
print(videogameDF["Genre"].value_counts())
print(videogameDF["Publisher"].value_counts())
print(videogameDF["Developer"].value_counts())
print(videogameDF["Rating"].value_counts())

# Kind of cheating because I had to dirty the data myself, but "by inspection"
# above I can see that the following values are the above columns that don't
# belong. Delete these rows.
garbage_values = ['N/A', 1782, -999, 2030, 'unknown']
columns = ["Name","Platform", "Genre", "Publisher", "Developer", "Rating"]
for i in range(len(columns)):
    for j in range(len(garbage_values)):
        indices_to_remove = videogameDF[columns[i]] == garbage_values[j]
        videogameDF = videogameDF.loc[~indices_to_remove]






#### Make Plots

# Count how many games per platform
platform_counts = videogameDF['Platform'].value_counts()

# Plot as bar chart
platform_counts.plot(kind='pie', figsize=(4,4))
plt.xlabel('Platform')
plt.ylabel('Number of Games')
plt.title('Number of Games per Platform (Post-Cleaning)')
plt.show()


# Count how many games per publisher
rating_counts = videogameDF['Rating'].value_counts()

# Plot as pie chart with improvements
rating_counts.plot(kind='bar', figsize=(4,4))            # square figure for pie)

plt.ylabel('')                # remove the y-label (default shows 'Rating')
plt.title('Distribution of Games by Rating (Post-Cleaning)')
plt.tight_layout()            # adjust layout so labels don't get cut off
plt.show()


## User Score per Genre ##    
user_score_per_genre = videogameDF.dropna(subset = ["Genre","User_Score"])
user_score_per_genre.boxplot(
    column="User_Score",
    by="Genre",
    figsize=(4,4),
    grid=False,
    showfliers=True  # optional: hides extreme outliers for cleaner view
)
plt.xlabel("Genre")
plt.ylabel("User Score")
plt.title("User Score per Genre (Post-Cleaning)")
plt.suptitle("")  # remove default 'Boxplot grouped by ...' title
plt.xticks(rotation=90)  # rotate x-axis labels if many years
plt.show()


# Make plot that shows average critic score by year.
average_critic_score = videogameDF.groupby("Year_of_Release")["Critic_Score"].mean()
average_critic_score = average_critic_score.sort_index()
plt.figure(figsize=(4,4))
average_critic_score.plot(kind="line")
plt.xlabel("Year")
plt.ylabel("Average Critic Score")
plt.title("Average Critic Score by Year (Post-Cleaning)")
plt.tight_layout()
plt.show()
