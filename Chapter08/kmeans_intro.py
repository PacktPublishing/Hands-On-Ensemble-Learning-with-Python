import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

np.random.seed(87654)

dat = []

t = 0.5

for i in range(20):

#    dat.append(np.random.uniform(size=2))
    c = np.random.randint(3)
    a = np.random.uniform() * 2 * 3.14
    r = t * np.sqrt(np.random.uniform())

    x = r * np.cos(a)
    y = r * np.sin(a)


    dat.append([c/4+x, c/4+y])

plt.figure()
for i in range(1, 5):
    np.random.seed(98765432)

    inits = np.array([[0.95,0.95],[0.95,0.95],[0.95,0.95]

            ])
    km = KMeans(n_clusters=3, init=inits, max_iter=i, n_init=1)
    plt.subplot(2, 2, i)
    plt.xticks([])
    plt.yticks([])
    km.fit(dat)
    km.cluster_centers_ = np.sort(km.cluster_centers_, axis=0)
    c = km.predict(dat)
    plt.scatter(*zip(*dat), c=c)
    c = km.fit_predict(km.cluster_centers_)
    plt.scatter(*zip(*km.cluster_centers_), c='w', marker='*', s=240, edgecolors='r')
    plt.title('Iteration: %d'%i)
    print(km.cluster_centers_)

