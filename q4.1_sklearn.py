from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('iris.data', header=None)
data[4] = data[4].replace({"Iris-virginica": 0, "Iris-versicolor": 1, "Iris-setosa": 2})
data = data.sample(frac=1).reset_index(drop=True)
x = data.loc[:, 0:3].to_numpy()
y = data.loc[:, 4].to_numpy()
for col in range(x.shape[1]):
    x[:, col] = (x[:, col] - x[:, col].mean()) / x[:, col].std()
pca = PCA(n_components=2)
# x_std = StandardScaler().fit_transform(x)
# Z = pca.fit_transform(x_std)
Z = pca.fit_transform(x)
#%% plot data
plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], c=y )
plt.show()