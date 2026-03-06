# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 11:38:24 2026

@author: Austin Coleman
"""

#### Import Libraries

import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from matplotlib import cm
from sklearn.cluster import DBSCAN



from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score



#### Load in Data

# Declare filenames used.
filename = "C:/Users/Austin Coleman/Documents/Spring 2026/Machine Learning/Project/cleaned_kaggle_dataset.csv"

# Load in data.
kaggle_DF = pd.read_csv(filename)
print(kaggle_DF.dtypes)




#### Remove the Labels

quantitative_kaggle_DF = kaggle_DF.loc[:,["NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales","Critic_Score","Critic_Count","User_Score","User_Score","User_Count"]]




#### Normalize Data

# Instantiate scalar.
scalar = StandardScaler()

# Normalize data
standardized_data = scalar.fit_transform(quantitative_kaggle_DF)


X, y = make_blobs(
    n_samples=500,
    n_features=2,
    centers=4,
    cluster_std=1,
    center_box=(-10.0, 10.0),
    shuffle=True,
    random_state=1,
)  # For reproducibility



range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(standardized_data) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(standardized_data)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(standardized_data, cluster_labels)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(standardized_data, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2 = fig.add_subplot(1,2,2, projection='3d')
    
    ax2.scatter(
        standardized_data[:,0],
        standardized_data[:,1],
        standardized_data[:,2],
        c=colors,
        s=20,
        alpha=0.7
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    
    # Draw white circles at cluster centers
    ax2.scatter(
       centers[:,0],
       centers[:,1],
       centers[:,2],
       marker="o",
       c="white",
       s=200,
       edgecolor="k"
   )

    for i, c in enumerate(centers):
       ax2.scatter(c[0], c[1], c[2], marker="$%d$" % i, s=50, edgecolor="k")

    ax2.set_title("3D visualization of clustered data")
    ax2.set_xlabel("Feature 1")
    ax2.set_ylabel("Feature 2")
    ax2.set_zlabel("Feature 3")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % n_clusters,
        fontsize=14,
        fontweight="bold",
    )

plt.show()





kmeans_object_Count = sklearn.cluster.KMeans(n_clusters=2)
#print(kmeans_object)
kmeans_object_Count.fit(standardized_data)
# Get cluster assignment labels
labels = kmeans_object_Count.labels_
prediction_kmeans = kmeans_object_Count.predict(standardized_data)
#print(labels)
print(prediction_kmeans)
# Format results as a DataFrame





#### Hierarchical



Z = linkage(standardized_data, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(
    Z,
    truncate_mode='level',  # only show the top levels
    p=5,                    # show last 5 merges
    leaf_rotation=90.,
    leaf_font_size=12.,
)
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('Cluster merge level')
plt.ylabel('Distance')
plt.show()





#### DBSCAN 

dbscan = DBSCAN(eps=2, min_samples=5)  
dbscan_labels = dbscan.fit_predict(standardized_data)


# Count clusters 
n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)

colors = cm.nipy_spectral((dbscan_labels.astype(float) + 1) / (n_clusters_db + 1))  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    standardized_data[:, 0],
    standardized_data[:, 1],
    standardized_data[:, 2],
    c=colors,
    marker='.',
    s=30,
    edgecolor='k',
    alpha=0.7
)
ax.view_init(elev=30, azim=20)
plt.title(f'DBSCAN Clustering (n_clusters = {n_clusters_db})')
plt.show()



