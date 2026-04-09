import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score

# Load the breast cancer dataset
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reduce dimensionality for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Perform agglomerative clustering
n_clusters = 2  # We know there are 2 classes in the breast cancer dataset
agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='average')
cluster_labels = agg_clustering.fit_predict(X_scaled)

# Create a figure with two subplots
plt.figure(figsize=(16, 8))

# Plot 1: Hierarchical Clustering Dendrogram
plt.subplot(1, 2, 1)
Z = linkage(X_scaled, method='ward')
dendrogram(Z, truncate_mode='lastp', p=30, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram', fontsize=14)
plt.xlabel('Sample index or cluster size', fontsize=12)
plt.ylabel('Distance', fontsize=12)


# Evaluate the clustering performance
print("Clustering Performance Metrics:")
print(f"Silhouette Score: {silhouette_score(X_scaled, cluster_labels):.4f}")


plt.tight_layout()
plt.show()
