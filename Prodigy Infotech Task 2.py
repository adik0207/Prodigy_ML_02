# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

customer_data = pd.read_csv('D:/Engineering/Prodigy Infotech/Progidy Infotech Task 2/Mall_Customers.csv')

X = customer_data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determining the number of clusters (k) using the Elbow method
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow method results to determine the optimal k value
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method for Optimal k')
plt.show()

# Performing K-means clustering with the selected k value
k = 5
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Adding cluster labels to the original dataset
customer_data['Cluster'] = kmeans.labels_

# Exploring the clusters
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)
cluster_sizes = customer_data['Cluster'].value_counts().sort_index()

# Visualizing the clusters (scatter plot)
plt.figure(figsize=(12, 6))

for i in range(k):
    cluster_data = customer_data[customer_data['Cluster'] == i]
    plt.scatter(cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {i+1}')

plt.scatter(cluster_centers['Annual Income (k$)'], cluster_centers['Spending Score (1-100)'], c='black', marker='X', s=100, label='Centroids')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Clusters based on Annual Income and Spending Score')
plt.legend()
plt.show()
