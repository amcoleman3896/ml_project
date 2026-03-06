# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 16:11:42 2026

@author: Austin Coleman
"""

#### Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler





#### Load in Data

# Declare filename
filename_kaggle = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset.csv"
filename_steam = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/steam_data_cleaned.csv"

# Load in cleaned datasets.
kaggle_DF = pd.read_csv(filename_kaggle)
steam_DF = pd.read_csv(filename_steam)
print(kaggle_DF.dtypes)
print(steam_DF.dtypes)
print("")




#### Extract Quantitative Data

# Cannot use qualitative variables in PCA. Extract only quantitative columns.
quantitative_kaggle_DF = kaggle_DF.loc[:,["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales","Critic_Score","Critic_Count","User_Score","User_Score","User_Count"]]
quantitative_steam_DF = steam_DF.loc[:,["playtime_forever","price","metacritic","release_year"]]




#### Standardize Both Datasets

# Now we need to loop over each column, and subtract the mean and divide by
# the standard deviation for the kaggle dataset.
# =============================================================================
# all_columns = quantitative_kaggle_DF.columns
# for current_column in all_columns:
#     
#     # Extract the mean and std for clarity.
#     current_mean = (quantitative_kaggle_DF.loc[:,current_column]).mean()
#     current_std = (quantitative_kaggle_DF.loc[:,current_column]).std()
#     
#     # Standardize the current column.
#     quantitative_kaggle_DF.loc[:,current_column] = (quantitative_kaggle_DF.loc[:,current_column] - current_mean) / current_std
# =============================================================================

# Do the same for the steam dataset.
# =============================================================================
# all_columns = quantitative_steam_DF.columns
# for current_column in all_columns:
#     
#     # Extract the mean and std for clarity.
#     current_mean = (quantitative_steam_DF.loc[:,current_column]).mean()
#     current_std = (quantitative_steam_DF.loc[:,current_column]).std()
#     
#     # Standardize the current column.
#     quantitative_steam_DF.loc[:,current_column] = (quantitative_steam_DF.loc[:,current_column] - current_mean) / current_std 
# =============================================================================

# Standardize using StandardScalar() (specified in module assignment)
scaler = StandardScaler()
standardized_array_kaggle = scaler.fit_transform(quantitative_kaggle_DF) 
standardized_array_steam = scaler.fit_transform(quantitative_steam_DF) 

# Save data to add to github page for submission requirements (need link).
np.savetxt('kaggle_data_cleaned_for_pca.csv', standardized_array_kaggle)
np.savetxt('steam_data_cleaned_for_pca.csv', standardized_array_steam)




#### Compute Covariance Matrix

# Compute covariance matrix of the standardized data.
covariance_matrix_kaggle = np.cov(standardized_array_kaggle, rowvar=False)
covariance_matrix_steam = np.cov(standardized_array_steam, rowvar=False)




#### Perform Eigendecomposition On Covariance Matrix

# Extract eigenvalues and eigenvectors.
eigenvalues_kaggle, eigenvectors_kaggle = np.linalg.eigh(covariance_matrix_kaggle)
eigenvalues_steam, eigenvectors_steam = np.linalg.eigh(covariance_matrix_steam)

# Now need to sort the eigenvalues and eigenvectors.
idx = np.argsort(eigenvalues_kaggle)[::-1]
eigenvalues_kaggle = eigenvalues_kaggle[idx]
eigenvectors_kaggle = eigenvectors_kaggle[:, idx]
idx = np.argsort(eigenvalues_steam)[::-1]
eigenvalues_steam = eigenvalues_steam[idx]
eigenvectors_steam = eigenvectors_steam[:, idx]

# Print the results to check them out.
print("Kaggle Dataset Eigenvalues = ")
print(eigenvalues_kaggle)
print("Kaggle Dataset Eigenvectors = ")
print(eigenvectors_kaggle)
print("")
print("Steam Dataset Eigenvalues = ")
print(eigenvalues_steam)
print("Steam Dataset Eigenvectors = ")
print(eigenvectors_steam)
print("")




#### Investigate Principle Components

# Determine percentage that each eigenvalue contributes to the overall sum
# of all eigenvalues.
eigenvalue_percentage_kaggle = (eigenvalues_kaggle / sum(eigenvalues_kaggle)) * 100
eigenvalue_percentage_steam = (eigenvalues_steam / sum(eigenvalues_steam)) * 100
print("Percentage contribution of Kaggle eigenvalues = ")
print(eigenvalue_percentage_kaggle)
print("Percentage contribution of Steam eigenvalues = ")
print(eigenvalue_percentage_steam)
print("")

# Sum the first 3 eigenvalues to determine what percentage of information we 
# capture.
print("% of info from first 3 principle components Kaggle dataset = ")
print(sum(eigenvalue_percentage_kaggle[:3]))
print("% of info from first 3 principle components Steam dataset = ")
print(sum(eigenvalue_percentage_steam[:3]))
print("")

# Sum the first 2 eigenvalues to determine what percentage of information we 
# capture.
print("% of info from first 2 principle components Kaggle dataset = ")
print(sum(eigenvalue_percentage_kaggle[:2]))
print("% of info from first 2 principle components Steam dataset = ")
print(sum(eigenvalue_percentage_steam[:2]))
print("")

# Extract the first 3 eigenvalues/vectors and store them for clarity.
eigenvalues_for_transformation_kaggle = eigenvalues_kaggle[:3]
eigenvectors_for_3D_transformation_kaggle = eigenvectors_kaggle[:3]
eigenvectors_for_2D_transformation_kaggle = eigenvectors_kaggle[:2]
eigenvalues_for_transformation_steam = eigenvalues_steam[:3]
eigenvectors_for_transformation_steam = eigenvectors_steam[:3]

# Determine how many eigenvalues we need to add together to capture at least
# 95% of the data.
percent_captured = 0
index = 0
num_eigenvalues_95_percent_kaggle = 0
while percent_captured < 95:
    percent_captured = percent_captured + eigenvalue_percentage_kaggle[index]
    index = index + 1
    num_eigenvalues_95_percent_kaggle = num_eigenvalues_95_percent_kaggle + 1
    
percent_captured = 0
index = 0
num_eigenvalues_95_percent_steam = 0
while percent_captured < 95:
    percent_captured = percent_captured + eigenvalue_percentage_steam[index]
    index = index + 1
    num_eigenvalues_95_percent_steam = num_eigenvalues_95_percent_steam + 1
    
# Print results.
print("Number of principle components necessary to capture >95% of the Kaggle data: ")
print(num_eigenvalues_95_percent_kaggle)
print("Number of principle components necessary to capture >95% of the Steam data: ")
print(num_eigenvalues_95_percent_steam)
print("")


#### Perform PCA Using PCA Function

# Run PCA on the standardized dataset
pca = PCA()
pca.fit(standardized_array_kaggle)

# Variance explained by each component
variance_explained = pca.explained_variance_ratio_

# Cumulative variance
cumulative_variance = np.cumsum(variance_explained)

print("Variance explained by each component:")
print(variance_explained)

print("\nCumulative variance explained:")
print(cumulative_variance)

# Information retained
info_2D = cumulative_variance[1] * 100
info_3D = cumulative_variance[2] * 100

print("\nInformation retained in 2D dataset:", info_2D, "%")
print("Information retained in 3D dataset:", info_3D, "%")


#### Transform Data

# Use eigenvectors to transform data into 2D and 3D
kaggle_data_2D = standardized_array_kaggle @ eigenvectors_for_2D_transformation_kaggle.T
kaggle_data_3D = standardized_array_kaggle @ eigenvectors_for_3D_transformation_kaggle.T

# Plot results.
fig = plt.figure(figsize=(12,5))

# 2D Plot
ax1 = fig.add_subplot(1,2,1)
ax1.scatter(kaggle_data_2D[:,0], kaggle_data_2D[:,1])
ax1.set_xlabel("Principal Component 1")
ax1.set_ylabel("Principal Component 2")
ax1.set_title("2D PCA Projection")

# 3D Plot
ax2 = fig.add_subplot(1,2,2, projection='3d')
ax2.scatter(
    kaggle_data_3D[:,0],
    kaggle_data_3D[:,1],
    kaggle_data_3D[:,2]
)

ax2.set_xlabel("Principal Component 1")
ax2.set_ylabel("Principal Component 2")
ax2.set_zlabel("Principal Component 3")
ax2.set_title("3D PCA Projection")

plt.tight_layout()
plt.show()

# X values = number of components
components = np.arange(1, len(cumulative_variance) + 1)

# Plot
plt.figure()

plt.plot(components, cumulative_variance, marker='o')

plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Explained (%)")
plt.title("Cumulative Variance Captured by PCA")

plt.grid(True)

plt.show()
