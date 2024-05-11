import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
        return self.centroid

    # có centroid, tìm nhãn ban đầu cho các điểm dữ liệu và cập nhật lại tâm
    def find_label(self, x):
        label = np.zeros((x.shape[0], self.k))
        self.label_mark = []
        for i in range(x.shape[0]):
            norm = [np.linalg.norm(x.values[i] - self.centroid[j]) for j in range(self.k)]
            label[i][np.argmin(norm)] = 1
            self.label_mark.append(np.argmin(norm))
        return label

    # Tính hàm mục tiêu
    def sse(self, x):
        label = Kmeans.find_label(self, x)
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
        label = Kmeans.find_label(self, x)
        self.SSE = Kmeans.sse(self,x)
        SSE_update = 0
        while self.SSE < SSE_update:
            #cập nhật tâm
            for i in range(self.k):
                above = np.array([0 for i in range(x.shape[1])])
                below = 0
                for j in range(x.shape[0]):
                    a = x.values[j]*label[j][i]
                    b = label[j][i]
                    above = np.add(above, a)
                    below += b
                mi = above / below
                self.centroid[i] = mi
            #cập nhật label
            label = Kmeans.find_label(self, x)
            #cập nhật hàm mục tiêu
            self.SSE = SSE_update
            SSE_update = Kmeans.sse(self, x)
        return self.SSE, np.array(self.centroid), label, self.label_mark

if __name__ == '__main__':
    data = pd.read_csv('D:/data/Book1.csv', index_col=None)
    model = Kmeans(k=4)
    model.fit(data)
    print(model.label_mark)
    centroid = np.array(model.centroid)
    plt.scatter(data['data1'], data['data2'], c=model.label_mark)
    plt.scatter(centroid[:,0], centroid[:,1], marker="x", c='red', label = 'Tam')
    plt.legend()
    plt.show()





