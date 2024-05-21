import pandas as pd
from handmade_kmeans import Kmeans
import matplotlib.pyplot as plt
import sklearn.cluster as sc
from sklearn.metrics import silhouette_score, davies_bouldin_score, mean_squared_error


data = pd.read_csv("D:/data/HUET.csv", index_col=None)
data = data[['Height', 'Weight', 'Porsion']]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

model1 = Kmeans(k=3)
model1.fit(data)
print(f'Sillhouette score: {silhouette_score(data, model1.label_mark)}')
print(f'Davies boudlin score: {davies_bouldin_score(data, model1.label_mark)}')
print(f'SSE: {model1.SSE}')

model2 = sc.KMeans(n_clusters=3, random_state=42)
model2.fit(data)
print(f'Sillhouette score: {silhouette_score(data, model2.labels_)}')
print(f'Davies boudlin score: {davies_bouldin_score(data, model2.labels_)}')
print(f'SSE: {model2.inertia_}')

# cluster_data1 = data.values[[model2.labels_[i] == 0 for i in range(data.shape[0])]]
# ax.scatter(cluster_data1[:,0], cluster_data1[:,1], cluster_data1[:,2],
#             marker='^', label='Cluster 1', alpha=1, s=40)
# cluster_data1 = data.values[[model2.labels_[i] == 1 for i in range(data.shape[0])]]
# ax.scatter(cluster_data1[:,0], cluster_data1[:,1], cluster_data1[:,2],
#             marker='+', label='Cluster 2', alpha=1, s=40)
# cluster_data1 = data.values[[model2.labels_[i] == 2 for i in range(data.shape[0])]]
# ax.scatter(cluster_data1[:,0], cluster_data1[:,1], cluster_data1[:,2],
#             marker='o', label='Cluster 3', alpha=1, s=40)
# ax.scatter(model2.cluster_centers_[:,0], model2.cluster_centers_[:,1], model2.cluster_centers_[:,2], marker='x', alpha=1, c='red', s=50)
# plt.show()
# cluster_data1 = data.values[[model1.label_mark[i] == 0 for i in range(data.shape[0])]]
# ax.scatter(cluster_data1[:,0], cluster_data1[:,1], cluster_data1[:,2],
#             marker='^', label='Cluster 1', alpha=1, s=40)
# cluster_data1 = data.values[[model1.label_mark[i] == 1 for i in range(data.shape[0])]]
# ax.scatter(cluster_data1[:,0], cluster_data1[:,1], cluster_data1[:,2],
#             marker='+', label='Cluster 2', alpha=1, s=40)
# cluster_data1 = data.values[[model1.label_mark[i] == 2 for i in range(data.shape[0])]]
# ax.scatter(cluster_data1[:,0], cluster_data1[:,1], cluster_data1[:,2],
#             marker='o', label='Cluster 3', alpha=1, s=40)
# ax.scatter(model1.centroid[:,0], model1.centroid[:,1],model1.centroid[:,2], marker='x', alpha=1, c='red', s=50)
ax.scatter(data['Height'], data['Weight'], data['Porsion'], alpha=1)
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Porsion')
plt.title("K-Means model clustering HUETers's data")
plt.legend()
plt.show()