# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 19:44:13 2019

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import matplotlib.pyplot as plt
import numpy as np

np.random.seed(123456)

def f(x):
    return np.sin(x)

def sample(size):
    max_v = 20
    step = size/max_v
    x = [x/step for x in range(size)]
    y = [f(x)+np.random.uniform(-0.25,0.25) for x in x]
    return np.array(x).reshape(-1,1), np.array(y).reshape(-1,1)


# =============================================================================
# HIGH BIAS - UNDERFIT
# =============================================================================
from sklearn.linear_model import LinearRegression
x, y = sample(100)


lr = LinearRegression()
lr.fit(x, y)
preds  =  lr.predict(x)
plt.figure()
plt.scatter(x, y, label='data')
plt.plot(x, preds, color='orange', label='model')
plt.title('Biased Model')
plt.legend()

# =============================================================================
# HIGH VARIANCE - OVERFIT
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
x, y = sample(100)

dt = DecisionTreeRegressor()
dt.fit(x, y)
plt.figure()
plt.scatter(x, y, label='training data')
x, y = sample(100)
preds  =  dt.predict(x)
plt.plot(x, preds, color='orange', label='model')
plt.scatter(x, y, label='test data')
plt.title('High Variance Model')
plt.legend()


# =============================================================================
# TRADEOFF
# =============================================================================
def bias(complexity):
    return 100/complexity

def variance(complexity):
    return np.exp(complexity/28)

r = range(5, 100)

variance_ = np.array([variance(x) for x in r])
bias_ =  np.array([bias(x) for x in r])
sum_ = variance_ + bias_
mins = np.argmin(sum_)
min_line = [mins for x in range(0, int(max(sum_)))]


plt.figure()
plt.plot(bias_, label=r'$bias^2$', linestyle='-')
plt.plot(variance_, label='variance', linestyle=':')
plt.plot(sum_, label='error', linestyle='-.')
plt.plot(min_line, [x for x in range(0, int(max(sum_)))], linestyle='--')
plt.title('Minimizing Error')
plt.legend()


# =============================================================================
# BEST MODEL
# =============================================================================
from sklearn.tree import DecisionTreeRegressor
x, y = sample(100)

plt.figure()
plt.scatter(x, y, label='training data')

preds  =  f(x)
plt.plot(x, preds, color='orange', label='model')
plt.title('Perfect Model')
plt.legend()