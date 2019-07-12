# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:20:55 2019

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""
import numpy as np
import pandas as pd


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn import metrics


np.random.seed(123456)
data = pd.read_csv('creditcard.csv')
data.Time = (data.Time-data.Time.min())/data.Time.std()
data.Amount = (data.Amount-data.Amount.mean())/data.Amount.std()

x_train, x_test, y_train, y_test = train_test_split(data.drop('Class', axis=1).values, data.Class.values, test_size=0.3)


X, Y = shuffle(x_train, y_train)

lr = LogisticRegression()
lr.fit(X, Y)

print('f1', metrics.f1_score(y_test, lr.predict(x_test)))
print('recall', metrics.recall_score(y_test, lr.predict(x_test)))


# =============================================================================
# Selected Features
# =============================================================================

np.random.seed(123456)
threshold = 0.1

correlations = data.corr()['Class'].drop('Class')
fs = list(correlations[(abs(correlations)>threshold)].index.values)
fs.append('Class')
data = data[fs]

x_train, x_test, y_train, y_test = train_test_split(data.drop('Class', axis=1).values, data.Class.values, test_size=0.3)

X, Y = shuffle(x_train, y_train)
lr = LogisticRegression()
lr.fit(X, Y)

print('f1', metrics.f1_score(y_test, lr.predict(x_test)))
print('recall', metrics.recall_score(y_test, lr.predict(x_test)))
