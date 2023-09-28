import numpy as np


# data = np.random.randn(1280, 1000)
data = np.zeros((1280, 1000))
random_array = np.random.rand(1280, 50)*14
data[:, 0:981:20] = data[:, 0:981:20] + random_array + 0

np.savez('data.npz', data=data)