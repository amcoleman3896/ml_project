# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:33:01 2026

@author: Austin Coleman
"""

# -------------------------------------------------------------
# Association Rule Mining (ARM) using Apriori
# -------------------------------------------------------------
# This code:
# 1. Runs Apriori on transaction data
# 2. Generates association rules
# 3. Shows top 15 rules by support, confidence, and lift
# 4. Visualizes rules as a network graph
# -------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import ast

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

filename_transactional = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/Retail_Transactions_Dataset.csv"

# Load in cleaned datasets.
df = pd.read_csv(filename_transactional)

# Convert string list -> actual Python list
df['items'] = df['items'].apply(ast.literal_eval)

# Convert each transaction list into dummy variables (one-hot encoding)
transaction_matrix = df['items'].str.join('|').str.get_dummies()

# View result
print(transaction_matrix.head())


# -------------------------------------------------------------
# LOAD OR DEFINE TRANSACTION DATA
# -------------------------------------------------------------
# transaction_data should be a dataframe where:
# rows = transactions
# columns = items
# values = 1/0 or True/False

transaction_data = transaction_dataframe   # <-- replace with your dataset


# -------------------------------------------------------------
# RUN APRIORI ALGORITHM
# -------------------------------------------------------------
# min_support determines how frequent itemsets must be
# Adjust threshold depending on dataset size

frequent_itemsets = apriori(
    transaction_data,
    min_support=0.02,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets.head())


# -------------------------------------------------------------
# GENERATE ASSOCIATION RULES
# -------------------------------------------------------------
# metric determines which rule strength measure is used
# Here we generate rules based on lift

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
)

print("\nAll Rules Generated:", len(rules))


# -------------------------------------------------------------
# TOP 15 RULES BY SUPPORT
# -------------------------------------------------------------

top_support = rules.sort_values(by="support", ascending=False).head(15)

print("\nTop 15 Rules by Support")
print(top_support[['antecedents','consequents','support','confidence','lift']])


# -------------------------------------------------------------
# TOP 15 RULES BY CONFIDENCE
# -------------------------------------------------------------

top_confidence = rules.sort_values(by="confidence", ascending=False).head(15)

print("\nTop 15 Rules by Confidence")
print(top_confidence[['antecedents','consequents','support','confidence','lift']])


# -------------------------------------------------------------
# TOP 15 RULES BY LIFT
# -------------------------------------------------------------

top_lift = rules.sort_values(by="lift", ascending=False).head(15)

print("\nTop 15 Rules by Lift")
print(top_lift[['antecedents','consequents','support','confidence','lift']])


# -------------------------------------------------------------
# NETWORK VISUALIZATION OF ASSOCIATION RULES
# -------------------------------------------------------------
# This shows relationships between items

G = nx.DiGraph()

# Use top lift rules for visualization
rules_to_plot = top_lift

for _, row in rules_to_plot.iterrows():

    antecedent = list(row['antecedents'])[0]
    consequent = list(row['consequents'])[0]

    G.add_edge(antecedent, consequent, weight=row['lift'])

plt.figure(figsize=(10,8))

pos = nx.spring_layout(G)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    node_color="lightblue",
    font_size=10,
    arrows=True
)

edge_labels = nx.get_edge_attributes(G,'weight')

nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels=edge_labels
)

plt.title("Association Rule Network (Based on Lift)")
plt.show()


# -------------------------------------------------------------
# THRESHOLDS USED
# -------------------------------------------------------------
print("\nThresholds Used:")
print("Minimum Support:", 0.02)
print("Minimum Lift:", 1)