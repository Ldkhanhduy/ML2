import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

class Kmeans:
    def __init__(self, k, random_state=2024):
        self.k = k
        self.random_state = random_state
        self.centroid = []
        self.SSE = 0
        self.label_mark = []

    # khởi tạo centroid ban đầu
    def find_centroid(self, x):
        a = np.random.choice(x.shape[0], self.k, replace=False)
        self.centroid = x.values[a]
        # self.centroid = [[1000,170],[1000, 145],[5100, 163],[4100, 140]]
        return self.centroid

    # có centroid, tìm nhãn ban đầu cho các điểm dữ liệu và cập nhật lại tâm
    def find_label(self, x):
        label = np.zeros((x.shape[0], self.k))
        self.label_mark = []
        #tìm nhãn
        for i in range(x.shape[0]):
            norm = [np.linalg.norm(x.values[i] - self.centroid[j]) for j in range(self.k)]
            label[i][np.argmin(norm)] = 1
            self.label_mark.append(np.argmin(norm))
        #cập nhật tâm
        for i in range(self.k):
            above = np.array([0 for i in range(x.shape[1])])
            below = 0
            for j in range(x.shape[0]):
                a = x.values[j] * label[j][i]
                b = label[j][i]
                above = np.add(above, a)
                below += b
            mi = above / below
            self.centroid[i] = mi
        return label

    # Tính hàm mục tiêu
    def sse(self, x, label):
        for i in range(x.shape[0]):
            sse_one = 0
            for j in range(self.k):
                each_data = label[i][j] * np.linalg.norm(x.values[i] - self.centroid[j])
                sse_one += each_data
            self.SSE += sse_one
        return self.SSE

    #tối ưu hàm mục tiêu
    def fit(self, x):
        self.centroid = Kmeans.find_centroid(self, x)
        label1 = Kmeans.find_label(self, x)
        label2 = Kmeans.find_label(self, x)
        self.SSE = Kmeans.sse(self,x, label1)
        SSE_update = Kmeans.sse(self,x, label2)
        while self.SSE > SSE_update:
            label = Kmeans.find_label(self, x)
            #cập nhật hàm mục tiêu
            self.SSE = SSE_update
            SSE_update = Kmeans.sse(self, x, label)
        return self.SSE, np.array(self.centroid)

if __name__ == '__main__':
    data = pd.read_csv('D:/data/k_means.csv', index_col=None)
    data = data[['Average salary per month (USD)', 'Average working hours']]
    model = Kmeans(k=4)
    model.fit(data)
    print(model.label_mark)
    centroid = np.array(model.centroid)
    # plt.scatter(data['Average salary per month (USD)'], data['Average working hours'],
    #             c=model.label_mark)
    print(f"Silhouette score: {silhouette_score(data, model.label_mark)}")
    print(f"Davies Bouldin score: {davies_bouldin_score(data, model.label_mark)}")

    cluster_data1 = data.values[[model.label_mark[i] == 0 for i in range(data.shape[0])]]
    plt.scatter(cluster_data1[:,0], cluster_data1[:,1],
                marker='^', label='Cluster 1')
    cluster_data1 = data.values[[model.label_mark[i] == 1 for i in range(data.shape[0])]]
    plt.scatter(cluster_data1[:,0], cluster_data1[:,1],
                marker='+', label='Cluster 2')
    cluster_data1 = data.values[[model.label_mark[i] == 2 for i in range(data.shape[0])]]
    plt.scatter(cluster_data1[:,0], cluster_data1[:,1],
                marker='o', label='Cluster 3')
    cluster_data1 = data.values[[model.label_mark[i] == 3 for i in range(data.shape[0])]]
    plt.scatter(cluster_data1[:,0], cluster_data1[:,1],
                marker='s', label='Cluster 4')

    plt.xlabel('Average salary per month (USD)', fontweight = 'bold')
    plt.ylabel('Average working hours per day', fontweight = 'bold')
    plt.title('Scatter plot about working hours and salary each country', size = 20, fontweight='bold')
    plt.scatter(centroid[:,0], centroid[:,1], marker="x", c='red', label = 'Centroid')
    plt.legend()
    plt.show()





