import matplotlib.pyplot as plt
import numpy as np

np.random.seed(654321)

points = np.random.randint(0, 10, size=(10, 2))
classes = np.random.randint(0, 2, size=(10,))


positives = points[classes == 0]
negatives = points[classes == 1]

plt.scatter(*positives.T, marker='+', s=150)
plt.scatter(*negatives.T, marker='_', s=150)
plt.xticks([], [])
plt.yticks([], [])

plt.plot([1.5 for _ in range(12)], [x for x in range(-1, 11)], linestyle='--', color='black')
plt.plot([4.5 for _ in range(12)], [x for x in range(-1, 11)], linestyle='--', color='black')

plt.plot([x for x in range(-1, 8)], [1.5 for _ in range(9)], linestyle='--', color='black')