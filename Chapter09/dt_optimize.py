# --- SECTION 1 ---
# Libraries and data loading
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn import metrics




np.random.seed(123456)
data = pd.read_csv('creditcard.csv')
data.Time = (data.Time-data.Time.min())/data.Time.std()
data.Amount = (data.Amount-data.Amount.mean())/data.Amount.std()

# Train-Test slpit of 70%-30%
x_train, x_test, y_train, y_test = train_test_split(
        data.drop('Class', axis=1).values, data.Class.values, test_size=0.3)

# --- SECTION 2 ---
# Base learners evaluation
base_classifiers = [('DT', DecisionTreeClassifier(max_depth=6)),
                ('NB', GaussianNB()),
                ('LR', LogisticRegression())]

raw_f1 = []
raw_recall = []
range_ = [x for x in range(3,12)]
for max_d in range_:
    lr = DecisionTreeClassifier(max_depth=max_d)
    lr.fit(x_train, y_train)

    predictions = lr.predict(x_test)
    raw_f1.append(metrics.f1_score(y_test, predictions))
    raw_recall.append(metrics.recall_score(y_test, predictions))

plt.plot(range_, raw_f1, label='Raw F1')
plt.plot(range_, raw_recall, label='Raw Recall')
print(raw_f1)
print(raw_recall)
# --- SECTION 3 ---
# Filter features according to their correlation to the target
np.random.seed(123456)
threshold = 0.1

correlations = data.corr()['Class'].drop('Class')
fs = list(correlations[(abs(correlations)>threshold)].index.values)
fs.append('Class')
data = data[fs]

x_train, x_test, y_train, y_test = train_test_split(data.drop('Class', axis=1).values, data.Class.values, test_size=0.3)

filter_f1 = []
filter_recall = []
for max_d in range_:
    lr = DecisionTreeClassifier(max_depth=max_d)
    lr.fit(x_train, y_train)

    predictions = lr.predict(x_test)
    filter_f1.append(metrics.f1_score(y_test, predictions))
    filter_recall.append(metrics.recall_score(y_test, predictions))

print(filter_f1)
print(filter_recall)

plt.plot(range_, filter_f1, label='Filtered F1')
plt.plot(range_, filter_recall, label='Filtered Recall')