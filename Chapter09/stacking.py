# --- SECTION 1 ---
# Libraries and data loading
import numpy as np
import pandas as pd

from stacking_classifier import Stacking
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
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
# Ensemble evaluation
base_classifiers = [DecisionTreeClassifier(max_depth=5),
                    GaussianNB(),
                    LogisticRegression(),
                    DecisionTreeClassifier(max_depth=3),
                    DecisionTreeClassifier(max_depth=8)]

l1_classifiers = [DecisionTreeClassifier(max_depth=2), LinearSVC()]

ensemble = Stacking(learner_levels=[base_classifiers, l1_classifiers,
                    [LogisticRegression()]])


ensemble.fit(x_train, y_train)

print('Stacking f1', metrics.f1_score(y_test, ensemble.predict(x_test)))
print('Stacking recall', metrics.recall_score(y_test, ensemble.predict(x_test)))


# --- SECTION 3 ---
# Filter features according to their correlation to the target
np.random.seed(123456)
threshold = 0.1

correlations = data.corr()['Class'].drop('Class')
fs = list(correlations[(abs(correlations) > threshold)].index.values)
fs.append('Class')
data = data[fs]

x_train, x_test, y_train, y_test = train_test_split(
        data.drop('Class', axis=1).values, data.Class.values, test_size=0.3)

ensemble = Stacking(learner_levels=[base_classifiers, l1_classifiers,
                    [LogisticRegression()]])

ensemble.fit(x_train, y_train)

print('Stacking f1', metrics.f1_score(y_test, ensemble.predict(x_test)))
print('Stacking recall', metrics.recall_score(y_test, ensemble.predict(x_test)))
