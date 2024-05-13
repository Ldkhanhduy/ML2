import pandas as pd
from handmade_kmeans import Kmeans
import matplotlib.pyplot as plt
import sklearn.cluster as sc
from sklearn.metrics import silhouette_score, davies_bouldin_score


data = pd.read_csv("D:/data/HUET.csv", index_col=None)
data = data[['Height', 'Weight', 'Porsion']]

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

model1 = Kmeans(k=4)
model1.fit(data)
print(f'Sillhouette score: {silhouette_score(data, model1.label_mark)}')
print(f'Davies boudlin score: {davies_bouldin_score(data, model1.label_mark)}')

model2 = sc.KMeans(n_clusters=4, init='k-means++', random_state=42, n_init=10, max_iter=200)
model2.fit(data)
print(f'Sillhouette score: {silhouette_score(data, model2.labels_)}')
print(f'Davies boudlin score: {davies_bouldin_score(data, model2.labels_)}')

# ax.scatter(data['Height'], data['Weight'], data['Porsion'], c=model1.label_mark, alpha=1)
# ax.scatter(model1.centroid[:,0], model1.centroid[:,1], marker='x', alpha=1, c='red', s=50)
# plt.show()
ax.scatter(data['Height'], data['Weight'], data['Porsion'], c=model2.labels_, alpha=1)
ax.scatter(model2.cluster_centers_[:,0], model2.cluster_centers_[:,1], marker='x', alpha=1, c='red', s=50)
ax.set_xlabel('Height')
ax.set_ylabel('Weight')
ax.set_zlabel('Porsion')
plt.title("K-Means model clustering HUETers's data")
plt.show()