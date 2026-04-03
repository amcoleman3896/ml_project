# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 15:03:36 2026

@author: Austin Coleman
"""

#### Import Necessary Libraries
import requests  #to query the API 
import re  #regular expressions
import pandas as pd   # for dataframes
import time
from sklearn.feature_extraction.text import CountVectorizer   




#### Define Function to Get Friends
def get_friends(steamid):
    url = (
        "http://api.steampowered.com/"
        "ISteamUser/"
        "GetFriendList/"
        "v0001/"
        f"?key={MY_API_KEY}"
        f"&steamid={steamid}"
        "&relationship=friend"
    )
    response = requests.get(url)
    jsontext = response.json()
    
    # Safe check
    if "friendslist" not in jsontext:
        return []
    
    return [f["steamid"] for f in jsontext["friendslist"]["friends"]]
    

#### Define Function to Get Public Profiles on Accounts (Friends)
def get_public_profiles(steamids):
    
    # If steam ID is empty, return empty list.
    if not steamids:
        return []
    
    # Create URL to make API request with.
    ids = ",".join(steamids)
    url = (
        "http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v0002/"
        f"?key={MY_API_KEY}&steamids={ids}"
    )
    
    # Make API request.
    response = requests.get(url)
    players = response.json()["response"]["players"]
    
    # Return values accordingly.
    return [p["steamid"] for p in players if p.get("communityvisibilitystate") == 3] # return steam ID if the profile is public.


#### Define Function to Get Owned Games
def get_owned_games(steamid):
    url = (
        "http://api.steampowered.com/"
        "IPlayerService/"
        "GetOwnedGames/"
        "v0001/"
        f"?key={MY_API_KEY}"
        f"&steamid={steamid}"
        "&format=json"
        "&include_appinfo=true"
        "&include_played_free_games=true"
    )
    reponse = requests.get(url)
    return reponse.json()["response"]


#### Define Function to Get Game Metadata
def get_game_metadata(appid):
    
    # Declare the URL to be used.
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    
    # Try to make an API request
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
    # If it doesn't work, print message saying so and return blank values to 
    # avoid code breaking.
    except Exception as e:
        print(f"Error fetching appid {appid}: {e}")
        return {"genres": [], "release_date": None, "price": None, "metacritic": None}

    # Check if data is None or empty
    if not data or str(appid) not in data or not data[str(appid)]:
        return {"genres": [], "release_date": None, "price": None, "metacritic": None}

    # Check success flag. If it works, pull all the info for this game.
    if data[str(appid)].get("success"):
        info = data[str(appid)].get("data", {})
        genres = [g["description"] for g in info.get("genres", [])]
        release_date = info.get("release_date", {}).get("date", None)
        price = info.get("price_overview", {}).get("final", None)
        metacritic = info.get("metacritic", {}).get("score", None)
        return {
            "genres": genres,
            "release_date": release_date,
            "price": price,
            "metacritic": metacritic
        }
    
    # If not, once again return blank values.
    else:
        return {"genres": [], "release_date": None, "price": None, "metacritic": None}




#### Declare Own Steam Info
MY_API_KEY = "D84F2F9F3CA0061A76F9B4AD7F652E6C"  # replace with your own key
MY_STEAMID = "76561198820105042"



#### Determine Steam IDs for Friends and Friends-of-Friends

# Call get_friends() to get all friends' Steam IDs.
friend_ids = get_friends(MY_STEAMID)

# Set parameters that determine how far we go into "friends-of-friends+..."
MAX_DEPTH = 3             # how many layers of friends to go (1 = just my friends)
MAX_FRIENDS_PER_USER = 15 # how many friends to take per user to limit explosion

# Initialize BFS queue (Code obtained from chatGPT to enable going into 
# "friends-of-friends+...)
from collections import deque
queue = deque([(MY_STEAMID, 0)])  # (steamid, depth)
visited = set()
all_user_ids = set()

# Use while-loop to go into multiple layers of friend Steam ID extraction.
while queue:
    steamid, depth = queue.popleft()

    # Stop if already visited or beyond max depth
    if steamid in visited or depth > MAX_DEPTH:
        continue

    visited.add(steamid)
    all_user_ids.add(steamid)

    # Get friends
    friends = get_friends(steamid)[:MAX_FRIENDS_PER_USER]
    public_friends = get_public_profiles(friends)

    # Add friends to queue with increased depth
    for f in public_friends:
        if f not in visited:
            queue.append((f, depth + 1))



#### Get All Owned Games for Each Steam ID

# Preallocate dictionary that will store data for all the games owned by all 
# users in friend_ids.
all_users_games = {}

# Loop over every ID in all_user_ids to populate all_users_games by calling 
# get_owned_games() for each id.
for steam_id in all_user_ids:
    try:
        games = get_owned_games(steam_id)
        if "games" in games:
            all_users_games[steam_id] = games
        time.sleep(1)
    except Exception as e:
        print(f"Failed for {steam_id}: {e}")
print(f"Total users with games: {len(all_users_games)}")



#### Save Data in Data Frame Format

# Preallocate empty list that will be used to store each row of the data frame 
# that will store all the info for all users' games.
rows = []

# Loop over every Steam ID (that may be repeated due to someone owning more 
# than one game) and populate what will be each row of the data frame.
for steamid, data in all_users_games.items():
    for g in data.get("games", []):
        metadata = get_game_metadata(g["appid"])
        
        rows.append({
            "steamid": steamid,
            "appid": g["appid"],
            "name": g.get("name", ""),
            "playtime_forever": g.get("playtime_forever", 0),
            "genres": metadata["genres"],
            "release_date": metadata["release_date"],
            "price": metadata["price"],
            "metacritic": metadata["metacritic"]
        })
        
        time.sleep(0.2)  # be polite to Steam API

# Use the results to create the data frame, display to screen.
all_users_games_df = pd.DataFrame(rows)
print(all_users_games_df.head())
print(f"Total rows in DataFrame: {len(all_users_games_df)}")



#### Save Data to .CSV
all_users_games_df.to_csv("steam_user_game_data_bigger_data_sheet.csv", index=False)
