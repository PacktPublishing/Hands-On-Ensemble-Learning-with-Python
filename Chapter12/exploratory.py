import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('ratings.csv')

print(data.head())
data.drop('timestamp', axis=1, inplace=True)


data.rating.hist(grid=False)
plt.ylabel('Frequency')
plt.ylabel('Rating')
plt.title('Rating Distribution')

data.describe()