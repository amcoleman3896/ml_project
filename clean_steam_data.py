# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 13:50:13 2026

@author: Austin Coleman
"""

#### Import Libraries 

# Import libraries being used to clean API data.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats




#### Load in Data

# Declare filename used
filename = 'C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/steam_user_game_data_bigger_data_sheet.csv'

# Load in file and print to see what we're working with.
steam_DF = pd.read_csv(filename)
print(steam_DF.dtypes)




#### Clean Genre and Release Date Columns

# Compute the number of blank/NaN values in the genres and release_date prior
# to cleaning.
number_blank_genres_before_cleaning = sum((steam_DF['genres'] == "[]").to_numpy(dtype="bool"))
number_blank_release_dates_before_cleaning = sum(steam_DF['release_date'].isna())

# Compute total number of rows prior to cleaning these variables.
number_rows_before_cleaning = len(steam_DF['genres'])

# Determine the value counts of all games in the DF to determine if a game 
# appears more than once. Will allow us to possibly populate empty values for 
# genre or release date if that game appeared more than once.
game_value_counts = steam_DF.value_counts("name")

# Determine the indices where all the "[]" genre values occur, and where all 
# the NaN release date values occur.
genre_na_indices = steam_DF["genres"] == "[]"
genre_na_indices = steam_DF.index[genre_na_indices]
release_date_na_indices = steam_DF["release_date"].isna()
release_date_na_indices = steam_DF.index[release_date_na_indices]



# Use a for-loop to go over all genre indices and see if we can possibly 
# replace the empty genre value with the actual genre of the game (if it
# appeared more than once and was populated with genre on its repeat).
for index in genre_na_indices:
    
    # Grab the current game and the current player count of that game.
    current_game = steam_DF.loc[index,"name"]
    current_game_player_count = game_value_counts[current_game]
    
    # If the player count is not equal to 1, then we know we have a chance to
    # replace the "[]" genre value.
    if current_game_player_count != 1:
        
        # Grab the indices corresponding to the multiple occurrences of the 
        # current game.
        current_game_indices = steam_DF["name"] == current_game
        current_game_indices = steam_DF.index[current_game_indices]
        
        # Grab the genre entries for the multiple listings of the game.
        current_game_genre_query = steam_DF.loc[current_game_indices,"genres"]
        
        # Determine out of the list extracted above which entries are "[]" 
        # entries. Use this to then determine which entries are actually 
        # populated.
        empty_genre_indices = current_game_genre_query == "[]"
        empty_genre_indices = empty_genre_indices.to_numpy(dtype="bool")
        populated_genre_indices = ~empty_genre_indices
        empty_genre_indices = current_game_genre_query.index[empty_genre_indices]
        
        # If the sum of populated_genre_indices is not 0, then that means there
        # was at least one entry of the current game that listed the genre. 
        # Use this to replace the emtpy entries.
        if sum(populated_genre_indices) != 0:
            
            # Get the indices of the populated entries, and extract just the
            # first entry if there are multiple entries populated with the
            # genre.
            replacement_index = current_game_genre_query.index[populated_genre_indices][0]
            replacement_genre = str(current_game_genre_query.loc[replacement_index])
            
            # Replace the empty genres accordingly.
            steam_DF.loc[empty_genre_indices,"genres"] = replacement_genre
            


# Now do pretty much same thing to see if we can replace release dates that are
# NaN values.
for index in release_date_na_indices:
    
    # Grab the current game and the current player count of that game.
    current_game = steam_DF.loc[index,"name"]
    current_game_player_count = game_value_counts[current_game]
    
    # If the player count is not equal to 1, then we know we have a chance to
    # replace the NaN release date.
    if current_game_player_count != 1:
        
        # Grab the indices corresponding to the multiple occurrences of the 
        # current game.
        current_game_indices = steam_DF["name"] == current_game
        current_game_indices = steam_DF.index[current_game_indices]
        
        # Grab the release date entries for all the occurrences of this game.
        current_game_release_date_query = steam_DF.loc[current_game_indices,"release_date"]
        
        # Determine out of the list extracted above which entries are NaN 
        # entries. Use this to then determine which entries are actually 
        # populated.
        empty_release_date_indices = current_game_release_date_query.isna()
        empty_release_date_indices = empty_release_date_indices.to_numpy(dtype="bool")
        populated_release_date_indices = ~empty_release_date_indices
        empty_release_date_indices = current_game_release_date_query.index[empty_release_date_indices]
        
        # If the sum of populated_release_date_indices is not 0, then that
        # means there was at least one entry of the current game that listed 
        # the release date. Use this to replace the emtpy entries.
        if sum(populated_release_date_indices) != 0:
            
            # Get the indices of the populated entries, and extract just the
            # first entry if there are multiple entries populated with the
            # release date.
            replacement_index = current_game_release_date_query.index[populated_release_date_indices][0]
            replacement_release_date = str(current_game_release_date_query.loc[replacement_index])
            
            # Replace the empty entries accordingly.
            steam_DF.loc[empty_release_date_indices,"release_date"] = replacement_release_date

            

# Count the number of instances of genre and release date that were cleaned.
number_genres_cleaned = number_blank_genres_before_cleaning - sum((steam_DF['genres'] == "[]").to_numpy(dtype="bool")) 
number_release_dates_cleaned = number_blank_release_dates_before_cleaning - sum(steam_DF['release_date'].isna())
print("Number of empty genre values populated = ")
print(number_genres_cleaned)
print("Number of empty release date values populated = ")
print(number_release_dates_cleaned)



# Now we have no choice but to remove observations that still have blank 
# genre entries and blank release date entries. Determine the indices to 
# delete.
empty_genre_indices = (steam_DF['genres'] == "[]").to_numpy(dtype="bool")
empty_genre_indices = steam_DF.index[empty_genre_indices]
empty_release_date_indices = steam_DF['release_date'].isna()
empty_release_date_indices = steam_DF.index[empty_release_date_indices]
indices_to_delete = empty_genre_indices.append(empty_release_date_indices)
indices_to_delete = indices_to_delete.sort_values()
indices_to_delete = indices_to_delete.drop_duplicates()

# Remove these indices.
steam_DF = steam_DF.drop(indices_to_delete)

# Compute total number of rows after cleaning these variables.
number_rows_after_cleaning = len(steam_DF['genres'])

# Display the total number of rows removed in the process, and the percentage
# of data removed.
number_rows_removed = number_rows_before_cleaning - number_rows_after_cleaning
percent_rows_removed = (number_rows_removed / number_rows_before_cleaning) * 100
print("Total number of rows removed:")
print(number_rows_removed)
print("Percent of rows removed:")
print(percent_rows_removed)



# Reorder the indices since rows have been deleted.
new_indices = np.arange(number_rows_after_cleaning)
new_indices = pd.Index(new_indices)            
steam_DF.index = new_indices




#### Add Release Year Column

# Create the blank storage list that will be used to store just the year of the
# release for each game. Also create blank list that will be used to store 
# which indices contain non-valid release dates (e.g., "to be announced")
all_indices = steam_DF.index
release_year = []
indices_to_drop = []

# Use a for-loop to go over all release dates to extract just the year from
# the release_date string.
save_index = 0
for each_index in all_indices:
    
    # Extract the entire release date string.
    current_release_date_string = str(steam_DF.loc[each_index,'release_date'])

    # Try and turn the last 4 indices in the string into an integer (last 4
    # should be the year). But, there are some "to be announced" entries.
    try:
        current_year = int(current_release_date_string[-4:])
        release_year.append(current_year)
    
    # If it's a "to be announced" entry, save the release year as NaN.
    except:
        release_year.append(None)

# Create new column to the data frame that contains the release year for each 
# game.
steam_DF["release_year"] = release_year

# Drop the rows that had NaNs.
steam_DF = steam_DF.dropna(subset=["release_year"])

# Determine if there are any standalone years.
value_counts_release_year = steam_DF["release_year"].value_counts()
print(value_counts_release_year)




#### Clean Price Column   
 
# Start by dividing all the price column values by 100 to get them in dollars.
steam_DF['price'] = steam_DF['price'] / 100

# Plot to determine outliers.
plt.figure(figsize=(4, 4))
plt.plot(steam_DF.loc[:,'price'],'k.')
plt.show()

# Can see there are multiple outliers. Typically, a deluxe edition version of a 
# game would cost ~$90. Make all values greater than this NA values.
price_outlier_indices = (steam_DF['price'] > 90).to_numpy(dtype='bool')
price_outlier_indices = steam_DF.index[price_outlier_indices]
steam_DF.loc[price_outlier_indices,'price'] = None

# Replot to show the outliers removed and to better visualize prices for rest 
# of games.
plt.figure(figsize=(4, 4))
plt.plot(steam_DF.loc[:,'price'],'k.')
plt.show()    
   

    
# Now need to try and approximate the NaN values. Start by determing the
# indices of the NaN values (values that were outliers or were NaNs already).
nan_indices = steam_DF['price'].isna()
nan_indices = steam_DF.index[nan_indices]

# Use a for-loop to loop over these indices and approximate them with values
# so as not to have to delete the observations.
for nan_index in nan_indices:
    
    # Extract the indices of all games that came out that same year.
    current_year = steam_DF.loc[nan_index,'release_year']
    same_year_indices = steam_DF['release_year'] == current_year
    same_year_indices = steam_DF.index[same_year_indices]

    # Now determine the prices of the games that came out that same year, to 
    # then determine that there are prices available for games that came out 
    # that year.
    same_year_game_prices = steam_DF.loc[same_year_indices,'price']
    
    # If the length of same_year_game_prices is greater than or equal to 1, 
    # then we can see if there are games that came out that year that are of
    # the same genre. If there isn't then we will either (1) use the median
    # of the price of games that came out that year or (2) leave it NaN if 
    # there are no populated price entries for games that came out that year.
    if len(same_year_game_prices) >= 1:
        
        # Determine the genre of the current game.
        current_genre = steam_DF.loc[nan_index,'genres']
        
        # Extract list of the genres from the DF that are from the same year 
        # that also have the same genre.
        same_genre_same_year = steam_DF.loc[same_year_indices,'genres'] == current_genre
        same_genre_same_year_indices = same_genre_same_year.index
        
        # If the sum of same_genre_indices is greater than 1, then that means
        # there was at least one other game that had the same genre.
        if len(same_genre_same_year_indices) != 0:
            
            # Determine the median of the price of all these games.
            replacement_price = (steam_DF.loc[same_genre_same_year_indices,'price']).median()
            
        # Otherwise, the replacement price will be determined by the median of 
        # games that came out that year.
        else:
            
            # Extract the replacement price.
            replacement_price = same_year_game_prices.median()
            
        # Now replace the price accordingly.
        steam_DF.loc[nan_index,'price'] = replacement_price
                
        
        
# After all is said and done, if there weren't price values that could be 
# replaced, then need to remove them.
steam_DF = steam_DF.dropna(subset=["price"])
                   

      

#### Clean Metacritic Score Column


# Repeat process for metacritic score values. Start by determining where the 
# NaN values are.
nan_indices = steam_DF['metacritic'].isna()
nan_indices = steam_DF.index[nan_indices]

# Use a for-loop to loop over these indices and approximate them with values
# so as not to have to delete the observations.
for nan_index in nan_indices:
    
    # Extract the indices of all games that came out that same year.
    current_year = steam_DF.loc[nan_index,'release_year']
    same_year_indices = steam_DF['release_year'] == current_year
    same_year_indices = steam_DF.index[same_year_indices]

    # Now determine the metacritic score of the games that came out that same 
    # year, to then determine that there are scores available for games that 
    # came out that year.
    same_year_game_scores = steam_DF.loc[same_year_indices,'metacritic']
    
    # If the length of same_year_game_scores is greater than or equal to 1, 
    # then we can see if there are games that came out that year that are of
    # the same genre. If there isn't then we will either (1) use the median
    # of the metacritic score of games that came out that year or (2) leave it
    # NaN if there are no populated metacritic score entries for games that 
    # came out that year.
    if len(same_year_game_scores) >= 1:
        
        # Determine the genre of the current game.
        current_genre = steam_DF.loc[nan_index,'genres']
        
        # Extract list of the genres from the DF that are from the same year 
        # that also have the same genre.
        same_genre_same_year = steam_DF.loc[same_year_indices,'genres'] == current_genre
        same_genre_same_year_indices = same_genre_same_year.index
        
        # If the sum of same_genre_indices is greater than 1, then that means
        # there was at least one other game that had the same genre.
        if len(same_genre_same_year_indices) != 0:
            
            # Determine the median of the metacritic score of all these games.
            replacement_score = (steam_DF.loc[same_genre_same_year_indices,'metacritic']).median()
            
        # Otherwise, the replacement metacritc score will be determined by the
        # median games that came out that year.
        else:
            
            # Extract the replacement metacritic score.
            replacement_score = same_year_game_scores.median()
            
        # Now replace the metacritic score accordingly.
        steam_DF.loc[nan_index,'metacritic'] = replacement_score
                
        
        
# After all is said and done, if there weren't metacritic score values that
# could be replaced, then need to remove them.
steam_DF = steam_DF.dropna(subset=["metacritic"])
        



#### Plot Distributions of Price and Metacritic Score

# Price Distribution
plt.figure(figsize=(4,4))
plt.hist(steam_DF.loc[:,'price'],bins=15)
plt.show()

# Metacritic Score Distribution
plt.figure(figsize=(4,4))
plt.hist(steam_DF.loc[:,'metacritic'],bins=15)
plt.show()




#### Make Sure Types are Correct

# Print dtypes of cleaned dataframe.
print(steam_DF.dtypes)




#### Save DF as .csv

# Save results.
steam_DF.to_csv("steam_data_cleaned.csv")
