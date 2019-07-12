import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

data = pd.read_csv('creditcard.csv')

data.Time = (data.Time-data.Time.min())/data.Time.std()
data.Amount = (data.Amount-data.Amount.mean())/data.Amount.std()

plt.figure()
data.groupby('Class').V1.count().plot(kind='bar')
plt.title('0-1 Class distribution')

plt.figure()
ax = data.Amount.hist(grid=False, bins=50)
ax.set_yscale("log", nonposy='clip')
plt.title('Amount')

plt.figure()
data.Time.hist(grid=False, bins=50)
plt.title('Time')

plt.figure()
correlations = data.corr()['Class'].drop('Class')
correlations.sort_values().plot(kind='bar')
plt.title('Correlations to Class')





frauds = data[data.Class == 1]
non_frauds = data[data.Class == 0]

frauds_no = len(frauds)

balanced_data = pd.concat([frauds, non_frauds.sample(frauds_no)])

plt.figure()
balanced_data.groupby('Class').V1.count().plot(kind='bar')
plt.title('0-1 Class distribution (subsampled)')

plt.figure()
ax = balanced_data.Amount.hist(grid=False, bins=50)
ax.set_yscale("log", nonposy='clip')
plt.title('Amount (subsampled)')

plt.figure()
correlations = balanced_data.corr()['Class'].drop('Class')
correlations.sort_values().plot(kind='bar')
plt.title('Correlations to Class (subsampled)')




