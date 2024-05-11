import numpy as np


def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2


class LinearRegression:
    def __init__(self, learning_rate= 0.001, n_iters= 50):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for i in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # dw = (1 / n_samples) * np.dot(X.T, y - y_predicted)
            # print("y", y_predicted)
            # print("dw", dw)
            # db = (1 / n_samples) * np.sum(y - y_predicted)
            # print("db", db)

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated


# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)


    X = np.array([[100.0], [200.0], [150.0], [250.0]])
    y = np.array([1000.0, 2500.0, 1700.0, 3000.0])
    X_p = np.array([[185.0]])

    regressor = LinearRegression(learning_rate=0.00001, n_iters=100000)
    regressor.fit(X, y)
    predictions = regressor.predict(X_p)
    print(predictions)

    y_pred_line = regressor.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X, y, color=cmap(0.9), s=10)
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.legend()
    plt.show()

'''
    Đoạn mã lệnh không sai, ở đây chỉ cần chỉnh sửa tham số learning_rate tùy thuộc
vào từng bài toán sao cho hợp lí để tránh thực hiện những bước nhảy quá lớn làm vượt
qua các điểm cực tiểu.
    Bài toán trên với learning_rate = 0.00001 đã cho ra kết quả tương đối chính xác
với tập dữ liệu.
'''