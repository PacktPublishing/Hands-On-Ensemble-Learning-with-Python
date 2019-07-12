import pandas as pd

from sklearn import cluster

data = pd.read_csv('WHR.csv')
regs = pd.read_csv('Regions.csv')

recents = data[data.Year == 2017]
recents = recents.dropna(axis=1, how="all")
recents = recents.fillna(recents.median())


def find_region(country):
    return regs[regs['Country name']==country].Region.values[-1]

def find_region_size(region):
    return regs.groupby('Region')['Country name'].count()[region]

km = cluster.KMeans(3)
fits = recents[['Log GDP per capita',
       'Social support', 'Healthy life expectancy at birth',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption','Positive affect', 'Negative affect',
       'Confidence in national government', 'Democratic Quality',
       'Delivery Quality']].values
preds = km.fit_predict(fits)
recents['Cluster'] = preds


grouped = recents.groupby('Cluster')['Country name']
for key, item in grouped:
    countries = grouped.get_group(key).values
    regions = {x: 0 for x in regs.Region.unique()}
    for country in countries:        
        regions[find_region(country)] = regions[find_region(country)]+1
    print(key, countries, regions, "\n\n")
    x, y = [], []
    for k in regions:
        x.append(k)
        y.append(regions[k]/find_region_size(k))
    plt.figure()
    plt.bar(x, y)
    plt.xticks(rotation=90)
    
    
    
    
recents = recents.dropna(axis=1, how="any")
recents = recents.fillna(recents.median())



km = cluster.KMeans(10)
preds = km.fit_predict(recents.drop(['Year', 'Country name'], axis=1).values)
recents['Cluster'] = preds

grouped = recents.groupby('Cluster')['Country name']
for key, item in grouped:
    countries = grouped.get_group(key).values
    regions = {x: 0 for x in regs.Region.unique()}
    for country in countries:        
        regions[find_region(country)] = regions[find_region(country)]+1
    print(key, countries, regions, "\n\n")
    x, y = [], []
    for k in regions:
        x.append(k)
        y.append(regions[k]/find_region_size(k))
    plt.figure()
    plt.bar(x, y)
    plt.xticks(rotation=90)
    

 
 