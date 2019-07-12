import matplotlib.pyplot as plt
import pandas as pd

from matplotlib import cm


data = pd.read_csv('WHR.csv')
regs = pd.read_csv('Regions.csv')

def find_region(country):
    if country in list(regs['Country name'].values):
        return regs[regs['Country name']==country].Region.values[-1]
    return 'None'

recents = data[data.Year == 2018]
recents = recents.dropna(axis=1, how="all")
recents = recents.fillna(recents.median())
recents['Region'] = recents['Country name'].apply(lambda x: find_region(x))




cmap = cm.get_cmap('viridis') 
recents.groupby('Region')['Country name'].count().plot(kind='pie', labels=None, cmap=cmap, autopct='%1.0f%%', textprops={'color':"w"})
plt.ylabel('')
plt.xticks()
plt.legend(labels = recents.groupby('Region')['Country name'].count().index, bbox_to_anchor=(1, 1.05))
    

data[['Year', 'Life Ladder']].set_index('Year').boxplot(by='Year', grid=False)
plt.suptitle("")
plt.title('Life Ladder')
plt.xlabel('Year')

data.groupby('Year')['Life Ladder'].count().plot()
plt.title('Countries per Year')
plt.xlabel('Year')
plt.ylabel('Countries')


def create_scatter(col, nc, nr, index):
    plt.subplot(nc, nr, index)
    render = data.sample(frac=0.3)
    plt.scatter(render[col], render['Life Ladder'])    
    plt.title(str(col)[:20])

i = 1
for key in ['Log GDP per capita',
       'Social support', 'Healthy life expectancy at birth',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption','Positive affect', 'Negative affect',
       'Confidence in national government', 'Democratic Quality',
       'Delivery Quality']:
    create_scatter(key, 4, 3, i)
    i += 1
    
    
t = data[data['Year']==2005].copy()
countries = list(t['Country name'].values)
filtered = data[data['Country name'].isin(countries)]

filtered[['Year', 'Life Ladder']].set_index('Year').boxplot(by='Year', grid=False)
plt.suptitle("")
plt.title('Life Ladder - Same Countries')
plt.xlabel('Year')

from sklearn.manifold import t_sne

t = t_sne.TSNE()
data = data.fillna(data.median())
transformed = t.fit_transform(data[['Log GDP per capita',
       'Social support', 'Healthy life expectancy at birth',
       'Freedom to make life choices', 'Generosity',
       'Perceptions of corruption','Positive affect', 'Negative affect',
       'Confidence in national government', 'Democratic Quality',
       'Delivery Quality']].values)
    
plt.scatter(transformed[:,0], transformed[:,1], c=data['Life Ladder'].values)

regions = {x: 0 for x in regs.Region.unique()}
i = 0
for r in regions:
    regions[r] = i
    i += 1
regions['None'] = i
    
plt.scatter(transformed[:,0], transformed[:,1], c=data['Region'].apply(lambda x: regions[x]).values)
