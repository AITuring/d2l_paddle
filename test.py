import math

from chap3 import Timer
import numpy as np
import matplotlib.pylab as pb

n = 10000
a = np.ones(n)
b = np.ones(n)
c = np.zeros(n)
timer = Timer()

for i in range(n):
  c[i] = a[i] + b[i]
print(f'{timer.stop():.5f} sec')


def synthetic_data(w, b, num_examples):  # @save
  """生成 y = Xw + b + 噪声。"""
  X = np.random.normal(0, 1, (num_examples, len(w)))
  y = np.dot(X, w) + b
  y += np.random.normal(0, 0.01, y.shape)
  return X, y.reshape((-1, 1))


true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])

