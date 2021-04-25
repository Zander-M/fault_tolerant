from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from faultDataset import FaultDataset
import matplotlib.pyplot as plt
import numpy as np

X = []
y = []
d = FaultDataset()

for i in range(len(d)):
    a, b = d[i]
    X.append(a.data.cpu().numpy())
    y.append(b[4].data.cpu().numpy())
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)

pred = regr.predict(X_test)
real = y_test


t = np.argsort(real)
plt.plot(t, pred, "bo")
plt.plot(t, real, "rx")
plt.show()
