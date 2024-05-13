from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("D:/data/HUET.csv", index_col=None)
data = data[['Height', 'Weight', 'Porsion']]

# Dùng phương pháp elbow để tìm số cụm tốt nhất
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Vẽ biểu đồ elbow
plt.plot(range(1, 11), wcss, marker='*', c='red', label='Elbow line')
plt.grid(True)
plt.title('Phương pháp Elbow', size=20, fontweight='bold')
plt.xlabel('Số cụm')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.legend()
plt.show()