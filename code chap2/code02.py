import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from code01 import LinearRegression, r2_score

X = np.array([ [100.0], [200.0], [300.0], [120.0], [210.0], [330.0]])
y = np.array([1030.0, 2800.0, 3670.0,1000.0, 2000.0, 3000.0 ])
X_p = np.array([[224]])



regression = LinearRegression(learning_rate=0.001, n_iters=1000)
regression.fit(X, y)

y_p = regression.predict(X_p)

print(y_p)

plt.scatter(X, y, cmap=9)
plt.scatter(X_p, y_p, cmap=10)
plt.plot(X, regression.predict(X))
plt.show()