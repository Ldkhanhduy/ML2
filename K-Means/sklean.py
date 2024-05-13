import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

data = pd.read_csv('D:/data/k_means.csv', index_col=None)
data = data[['Average salary per month (USD)', 'Average working hours']]

# plt.scatter(data['data1'], data['data2'])
# plt.show()
#
model = KMeans(n_clusters=4, random_state=42)
model.fit(data)
#
# centroid = model.cluster_centers_
label = model.labels_
# print(np.zeros((20,3)))
cluster_data1 = data.values[[model.labels_[i] == 0 for i in range(data.shape[0])]]
plt.scatter(cluster_data1[:,0], cluster_data1[:,1],
            marker='^', label='Cluster 1')
cluster_data1 = data.values[[model.labels_[i] == 1 for i in range(data.shape[0])]]
plt.scatter(cluster_data1[:,0], cluster_data1[:,1],
            marker='+', label='Cluster 2')
cluster_data1 = data.values[[model.labels_[i] == 2 for i in range(data.shape[0])]]
plt.scatter(cluster_data1[:,0], cluster_data1[:,1],
            marker='o', label='Cluster 3')
cluster_data1 = data.values[[model.labels_[i] == 3 for i in range(data.shape[0])]]
plt.scatter(cluster_data1[:,0], cluster_data1[:,1],
            marker='s', label='Cluster 4')
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], c='red', marker='x')
plt.xlabel('Average salary per month (USD)', fontweight = 'bold')
plt.ylabel('Average working hours per day', fontweight = 'bold')
plt.title('Scatter plot about working hours and salary each country', size = 20, fontweight='bold')
plt.legend()
plt.show()

print(f"Silhouette score: {silhouette_score(data, model.labels_)}")
print(f"Davies Bouldin score: {davies_bouldin_score(data, model.labels_)}")