import numpy as np

x0 = np.ones((10,))

x1 = [64.3, 99.6, 145.45, 63.75, 135.46, 92.85, 86.97, 144.76, 59.3, 116.03]

x2 = [2, 3, 4, 2, 3, 4, 2, 4, 1, 3]

y = np.array([62.55, 82.42, 132.62, 73.31, 131.05, 86.57, 85.49, 127.44, 55.25, 104.84])

X = np.stack((x0, x1, x2), axis=1)

Y = y.reshape(10, 1)

X_mat = np.mat(X)

Y_mat = np.mat(Y)

W = ((X_mat.T * X_mat).I * X_mat.T) * Y

print(X)

print(Y)

print(W)

print(W.shape)
