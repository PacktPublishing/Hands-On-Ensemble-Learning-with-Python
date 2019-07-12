import matplotlib.pyplot as plt
import numpy as np
import openensembles as oe 
import pandas as pd


from sklearn import metrics


# Load the datasets
data = pd.read_csv('WHR.csv')
regs = pd.read_csv('Regions.csv')

# Use the 2017 data and fill any NaNs
recents = data[data.Year == 2017]
recents = recents.dropna(axis=1, how="all")
recents = recents.fillna(recents.median())


# Use only these specific features
columns = ['Log GDP per capita',
       'Social support', 'Healthy life expectancy at birth',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption','Positive affect', 'Negative affect',
       'Confidence in national government', 'Democratic Quality',
       'Delivery Quality']

# Normalize the features by subtracting the mean
# and dividing by the standard  deviation
normalized = recents[columns]
normalized = normalized - normalized.mean()
normalized = normalized / normalized.std()
# Create the data object
cluster_data = oe.data(recents[columns], columns)


np.random.seed(123456)
results = {'K':[], 'size':[], 'silhouette': []}
# Test different ensemble setups
Ks = [2, 4, 6, 8, 10, 12, 14]
sizes = [5, 10, 20, 50]
for K in Ks:
    for ensemble_size in sizes:
        ensemble = oe.cluster(cluster_data)
        for i in range(ensemble_size):
            name = f'kmeans_{ensemble_size}_{i}'
            ensemble.cluster('parent', 'kmeans', name, K)

        preds = ensemble.finish_co_occ_linkage(threshold=0.5)
        print(f'K: {K}, size {ensemble_size}:', end=' ')
        silhouette = metrics.silhouette_score(recents[columns], 
                                               preds.labels['co_occ_linkage'])
        print('%.2f' % silhouette)
        results['K'].append(K)
        results['size'].append(ensemble_size)
        results['silhouette'].append(silhouette)
        
results_df = pd.DataFrame(results)
cross = pd.crosstab(results_df.K, results_df['size'], 
                    results_df['silhouette'], aggfunc=lambda x: x)

    