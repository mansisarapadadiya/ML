import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Load dataset
df = pd.read_csv("titanic.csv")

# Select useful features
data = df[['Pclass','Sex','Age','SibSp','Parch','Fare']].copy()

# Convert categorical column to numeric
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].mean())
data['Fare'] = data['Fare'].fillna(data['Fare'].mean())

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(data)

# Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
clusters = model.fit_predict(X)

# Add cluster labels to dataset
df['Cluster'] = clusters

# Print cluster results
print("\nCluster Assignment (First 20 Rows):")
print(df[['PassengerId','Cluster']].head(20))

# -----------------------------
# Dendrogram Visualization
# -----------------------------
linked = linkage(X, method='ward')

plt.figure(figsize=(10,6))
dendrogram(linked)
plt.title("Dendrogram for Titanic Dataset")
plt.xlabel("Passengers")
plt.ylabel("Euclidean Distance")
plt.show()
