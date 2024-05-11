import numpy as np
import matplotlib.pyplot as plt
from code01 import LinearRegression as GD
from sklearn.linear_model import LinearRegression

# X = np.array([[100.0], [200.0], [150.0], [250.0]])
# y = np.array([1000.0, 2500.0, 1700.0, 3000.0])
# X_p = 185.0
X = train = np.random.randint(100,1000, [1000,1])
y = (np.multiply(X.T, np.ones(1000)*10.0).reshape(1000)+np.random.randint(100,5000, 1000))


# b= float(input())
# avg = 0.0
# for i in range(X.shape[0]):
#     a = 0.0

#     a = (y[i-1]-b)/X[i-1]
#     avg += a
# avg = avg/X.shape[0]
# print("a=", avg)
# print("y_p=", avg*X_p+b)
#
# X_ = 170.25
# y_ = 2050
# avg_b = 0
#
# for i in range(X.shape[0]):
#     w = 0
#     w = ((X[i-1]-X_)*(y[i-1]-y_))/(X[i-1]-X_)**2
#     b = y_-w*X_
#     avg_b += b
# bb = avg_b/X.shape[0]
# print(bb)


def ptdt(X, w, b0):
    return [w*x + b0 for x in X]

def hsg(X, Y):
    avg_x = np.mean(X)
    avg_y = np.mean(Y)
    a = sum([(x-avg_x)*(y-avg_y) for x, y in zip(X,Y)])
    b = sum([(x-avg_x)**2 for x in X])
    return a/b

def get_b0(X,Y,w):
    avg_x = np.mean(X)
    avg_y = np.mean(Y)
    return avg_y- w*avg_x

if __name__ == '__main__':
    ptdt(X, hsg(X,y), get_b0(X, y, hsg(X,y)))
    gd = GD(learning_rate=0.000005, n_iters=800000)
    gd.fit(X,y)
    sklearn = LinearRegression()
    sklearn.fit(X, y)
    print(gd.bias)
    print(sklearn.intercept_)

    plt.scatter(X, sklearn.predict(X), color = 'black', label="sklearn")
    plt.scatter(X, ptdt(X, hsg(X, y), get_b0(X, y, hsg(X, y))), color='red', label="DONGIAN", s=5)
    plt.scatter(X, gd.predict(X), color='blue', label="GD", s=50)
    plt.scatter(X, y, color = 'green', s = 10)
    plt.legend()
    plt.show()
