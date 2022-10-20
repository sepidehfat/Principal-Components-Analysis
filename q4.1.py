import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% load data
data = pd.read_csv('iris.data', header=None)
data[4] = data[4].replace({"Iris-virginica": 0, "Iris-versicolor": 1, "Iris-setosa": 2})
data = data.sample(frac=1).reset_index(drop=True)
x = data.loc[:, 0:3].to_numpy()
y = data.loc[:, 4].to_numpy()
#%% pre processing
for col in range(x.shape[1]):
    x[:, col] = (x[:, col] - x[:, col].mean()) / x[:, col].std()
#%% compute sigma
cov = np.cov(x.T)
#%% compute eigen vector
u, s, vh = np.linalg.svd(cov, full_matrices=True)
#%% compute Z
u_reduce = -u[:, :2]
Z = np.zeros(shape=(x.shape[0],  2))
Z = np.dot(x, u_reduce)
#%% plot data
plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], c=y )
plt.show()
#%% x approx
x_approx = np.zeros(shape = (x.shape[0], x.shape[1]))
for row in range(len(x)):
    x_approx[row] = np.dot(u_reduce, Z[row])

