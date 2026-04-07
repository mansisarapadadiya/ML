import sys
print(sys.executable)
#import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

from sklearn_extra.cluster import KMedoids
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import silhouette_score

#load the iris dataset
iris=datasets.load_iris()
x=iris.data
y=iris.target

#perform k-means,kmedoids clustering
k=3 #number of clusters
kmeans=KMeans(n_clusters=k,random_state=0).fit(x)
kmedoids = KMedoids(n_clusters=k, random_state=0).fit(x)

#get cluster labels
kmeans_labels=kmeans.labels_
kmedoids_labels=kmedoids.labels_

#compute silhouette scores
kmeans_score=silhouette_score(x,kmeans_labels)
kmedoids_score=silhouette_score(x,kmedoids_labels)

print("kmeans silhoutte score:",kmeans_score)
print("kmedoids silhoutte score:",kmedoids_score)

#visualize the clusters (using first two features)
#kmeans plot
plt.scatter(x[:,0],x[:,1],c=kmeans_labels,s=50,cmap='viridis')
centers=kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75)
plt.title('KMeans Clustering of Iris Dataset')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()

#kmedoids plot
plt.scatter(x[:,0],x[:,1],c=kmedoids_labels,s=50,cmap='viridis')
plt.title('kmeans clustering of iris dataset')
plt.xlabel('sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()
