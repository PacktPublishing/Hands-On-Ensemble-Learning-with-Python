# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 23:01:53 2019

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

from sklearn.datasets import load_digits, load_breast_cancer, load_diabetes

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA

# =============================================================================
# DATASETS
# =============================================================================
diabetes = load_diabetes()
bc = load_breast_cancer()
digits = load_digits()
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[10:20]):
    plt.subplot(2, 5, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Target: %i' % label)


# =============================================================================
# CLASSIFICATION
# =============================================================================
f = lambda x: 2 * x - 5

pos = []
neg = []

for i in range(30):
    x = np.random.randint(15)
    y = np.random.randint(15)

    if f(x) < y:
        pos.append([x,y])
    else:
        neg.append([x,y])


plt.figure()
plt.xticks([])
plt.yticks([])
plt.scatter(*zip(*pos))
plt.scatter(*zip(*neg))
plt.plot([0,10],[f(0),f(10)], linestyle='--', color='m')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Classification')

# =============================================================================
# REGRESSION
# =============================================================================

dat = []


for i in range(30):
    x = np.random.uniform(10)
    y = f(x) + np.random.uniform(-2.0,2.0)


    dat.append([x,y])


plt.figure()
plt.xticks([])
plt.yticks([])
plt.scatter(*zip(*dat))
plt.plot([0,10],[f(0),f(10)], linestyle='--', color='m')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Regression')

# =============================================================================
# CLUSTERING
# =============================================================================

km = KMeans(n_clusters=3)
dat = []

t = 0.5

for i in range(300):


    c = np.random.randint(3)
    a = np.random.uniform() * 2 * 3.14
    r = t * np.sqrt(np.random.uniform())

    x = r * np.cos(a)
    y = r * np.sin(a)


    dat.append([c+x, c+y])


c = km.fit_predict(dat)
plt.figure()
plt.xticks([])
plt.yticks([])
plt.scatter(*zip(*dat),c=c)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clustering')


# =============================================================================
# PCA
# =============================================================================

from sklearn.datasets import make_circles

pca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
x, y = make_circles(n_samples=400, factor=.3, noise=.05)


pp = pca.fit_transform(x)
plt.figure()
plt.xticks([])
plt.yticks([])
plt.scatter(pp[:,0], pp[:,1], c=y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Clustering')

# =============================================================================
# TSNE
# =============================================================================

from sklearn.manifold import TSNE

tsne = TSNE()

dat = tsne.fit_transform(bc.data)
reds = bc.target == 0
blues = bc.target == 1
plt.scatter(dat[reds,0], dat[reds,1], label='malignant')
plt.scatter(dat[blues,0], dat[blues,1], label='benign')
plt.xlabel('1st Component')
plt.ylabel('2nd Component')
plt.title('Breast Cancer Data')
plt.legend()

# =============================================================================
# ROC
# =============================================================================
import numpy as np
from sklearn import metrics
ax1 = plt.subplot()
ax1.margins(0)
np.random.seed(856522)
y = np.random.choice([1,2], 30)
scores = np.random.choice([i/100 for i in range(0,100)], 30)
fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

x = [i/100 for i in range(0,100)]
y = [i/100 for i in range(0,100)]
plt.plot(x, y, linestyle='-.')
plt.plot(fpr, tpr, label='ROC curve')

plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC')
plt.legend()
