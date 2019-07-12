# --- SECTION 1 ---
# Libraries and data loading
import openensembles as oe
import numpy as np
import pandas as pd
import sklearn.metrics

from sklearn.datasets import load_breast_cancer


bc = load_breast_cancer()

# --- SECTION 2 ---
# Create the data object
cluster_data = oe.data(pd.DataFrame(bc.data), bc.feature_names)

np.random.seed(123456)


# --- SECTION 3 ---
# Create the ensembles and calculate the homogeneity score
for K in [2, 3, 4, 5, 6, 7]:
    for ensemble_size in [3, 4, 5]:
        ensemble = oe.cluster(cluster_data)
        for i in range(ensemble_size):
            name = f'kmeans_{ensemble_size}_{i}'
            ensemble.cluster('parent', 'kmeans', name, K)

        preds = ensemble.finish_co_occ_linkage(threshold=0.5)
        print(f'K: {K}, size {ensemble_size}:', end=' ')
        print('%.2f' % sklearn.metrics.homogeneity_score(
                bc.target, preds.labels['co_occ_linkage']))


