import matplotlib.pyplot as plt
import numpy as np

np.random.seed(12345)

points = np.random.multivariate_normal([1, 1], ([1, 0.5],[0.5, 1]), 10)
points2 = np.random.multivariate_normal([4, 4], ([1, 0.5],[0.5, 1]), 10)

plt.scatter(*zip(*points), label='Class 1', marker='+', s=150)
plt.scatter(*zip(*points2), label='Class 2', marker='_', s=150)
plt.plot([-x+6 for x in range(0,10)], linestyle='--',
          color='black', label='class boundary')

#plt.text(0,5, '+', fontsize=18)
#plt.text(1.5,5.5, '_', fontsize=18)

r = range(-5,10)
stable = [x for x in r]

plt.plot([1.45 for x in r], stable, linestyle='--',
          color='gray', label='outlier rules')
plt.plot([1.9 for x in r], stable, linestyle='--',
          color='gray')

plt.plot(stable,[0.85 for x in r],  linestyle='--',
          color='gray')
plt.plot(stable,[0.55 for x in r],  linestyle='--',
          color='gray')
plt.xticks([], [])
plt.yticks([], [])



plt.legend()