# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 21:16:06 2026

@author: Austin Coleman
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import ast

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules



# Load your transaction CSV file
filename = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/Retail_Transactions_Dataset.csv"
transactions_DF = pd.read_csv(filename)

# Display first few rows
print("Sample of Raw Data:")
print(transactions_DF.head())




#### CONVERT TO LIST OF TRANSACTIONS

# Convert 'Product' strings to actual lists
transaction_list = []

for i in range(int(len(transactions_DF)/10)):
    
    # Safely evaluate string as Python literal
    items = ast.literal_eval(transactions_DF.loc[i, "Product"])
    
    # Strip whitespace just in case
    cleaned_items = [item.strip() for item in items]
    
    transaction_list.append(cleaned_items)




#### ONE-HOT ENCODING (REQUIRED FOR APRIORI)


transaction_encoder_object = TransactionEncoder()

transaction_encoded_array = transaction_encoder_object.fit(transaction_list).transform(transaction_list)

transaction_encoded_DF = pd.DataFrame(
    transaction_encoded_array,
    columns=transaction_encoder_object.columns_
)

print("\nOne-Hot Encoded Data:")
print(transaction_encoded_DF.head())




#### APPLY APRIORI

minimum_support_threshold = 0.001  # Adjust as needed

frequent_itemsets_DF = apriori(
    transaction_encoded_DF,
    min_support=minimum_support_threshold,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets_DF.head())




#### GENERATE ASSOCIATION RULES

minimum_confidence_threshold = 0.002  # Adjust as needed

association_rules_DF = association_rules(
    frequent_itemsets_DF,
    metric="confidence",
    min_threshold=minimum_confidence_threshold
)

print("\nAssociation Rules:")
print(association_rules_DF.head())




#### TOP 15 RULES BY SUPPORT

top_15_support = association_rules_DF.sort_values(
    by="support",
    ascending=False
).head(15)

print("\nTop 15 Rules by Support:")
print(top_15_support[["antecedents", "consequents", "support"]])




#### TOP 15 RULES BY CONFIDENCE

top_15_confidence = association_rules_DF.sort_values(
    by="confidence",
    ascending=False
).head(15)

print("\nTop 15 Rules by Confidence:")
print(top_15_confidence[["antecedents", "consequents", "confidence"]])




#### TOP 15 RULES BY LIFT

top_15_lift = association_rules_DF.sort_values(
    by="lift",
    ascending=False
).head(15)

print("\nTop 15 Rules by Lift:")
print(top_15_lift[["antecedents", "consequents", "lift"]])




#### VISUALIZATION 

# Sort rules by lift
top_rules = association_rules_DF.sort_values('lift', ascending=False).head(10)

# Create labels like "A → B"
labels = [f"{', '.join(list(a))} → {', '.join(list(b))}" 
          for a, b in zip(top_rules['antecedents'], top_rules['consequents'])]

plt.figure(figsize=(10,6))
plt.barh(labels, top_rules['lift'], color='skyblue')
plt.xlabel('Lift')
plt.title('Top 10 Association Rules by Lift')
plt.gca().invert_yaxis()  # largest lift on top
plt.show()



