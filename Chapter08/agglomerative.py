from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import numpy as np

a = np.random.uniform(size=10)
Z = hierarchy.linkage(a, 'single')
plt.figure()
dn = hierarchy.dendrogram(Z)
plt.title('Hierarchical Clustering Dendrogram')