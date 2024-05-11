import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1/(1+np.e**(-z))

X = np.array([[100.0], [200.0], [150.0], [250.0]])
y = np.array([0, 1, 0, 1])
X_p = np.array([[185.0], [123.0]])

class LOgisticRegression:
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
        return np.array([1 if i>=0.5 else 0 for i in sigmoid(y_approximated)])

if __name__ =='__main__':
    logistic = LOgisticRegression(0.0000001, 100    )
    logistic.fit(X, y)
    plt.scatter(X, y)
    plt.scatter(X_p, logistic.predict(X_p), marker='*', s= 20, label="Predict")
    plt.grid(True)
    plt.legend()
    plt.show()
