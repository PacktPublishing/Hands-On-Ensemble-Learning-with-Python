import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

# Read the data and assign labels
labels = ['polarity', 'id', 'date', 'query', 'user', 'text']
data = pd.read_csv("sent140.csv", names=labels)

# Plot polarities
data.groupby('polarity').id.count().plot(kind='bar')

# Get most frequent words
data['words'] = data.text.str.split()

words = []
# Get a list of all words
for w in data.words:
    words.extend(w)

# Get the frequencies and plot
freqs = Counter(words).most_common(30)
plt.plot(*zip(*freqs))
plt.xticks(rotation=80)
plt.ylabel('Count')
plt.title('30 most common words.')

