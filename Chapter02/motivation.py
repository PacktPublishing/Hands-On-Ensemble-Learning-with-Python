# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 23:15:52 2019

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""

import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

# =============================================================================
# ENSEMBLE SIZE - ERROR PLOT
# =============================================================================
def prob(size):
    err = 0.15
    half = int(np.ceil(size/2))
    s = 0
    for i in range(half, size):
        s += binom(size, i)*np.power(err,i)*np.power((1-err),(size-i))
    return s


probs = [15]
rg = range(3,14, 2)
for sz in rg:
    probs.append(prob(sz)*100)
    print(sz, '%.2f'%(prob(sz)*100))

rg = range(1,14, 2)
plt.figure()
plt.bar([x for x in rg], probs)
plt.title('Probability of error for ensemble')
plt.xlabel('Number of base learners')
plt.ylabel('Error %')
plt.xticks([x for x in rg])


# =============================================================================
# VALIDATION CURVES
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import validation_curve
from sklearn.neighbors import KNeighborsClassifier

bc = load_breast_cancer()


# --- SECTION 2 ---
# Create in-sample and out-of-sample scores
x, y = bc.data, bc.target
learner = KNeighborsClassifier()
param_range = [2,3,4,5]
train_scores, test_scores = validation_curve(learner, x, y,
                                             param_name='n_neighbors',
                                             param_range=param_range,
                                             cv=10,
                                             scoring="accuracy")

# --- SECTION 3 ---
# Calculate the average and standard deviation for each hyperparameter
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)


# --- SECTION 4 ---
# Plot the scores
plt.figure()
plt.title('Validation curves')
# Plot the standard deviations
plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="C1")
plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="C0")

# Plot the means
plt.plot(param_range, train_scores_mean, 'o-', color="C1",
         label="Training score")
plt.plot(param_range, test_scores_mean, 'o-', color="C0",
         label="Cross-validation score")

plt.xticks(param_range)
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.legend(loc="best")
plt.show()


# =============================================================================
# LEARNING CURVES
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve
bc = load_breast_cancer()


# --- SECTION 2 ---
# Create in-sample and out-of-sample scores
x, y = bc.data, bc.target
learner = KNeighborsClassifier()
train_sizes = [50, 100, 150, 200, 250, 300]
train_sizes, train_scores, test_scores = learning_curve(learner, x, y,
                                                        train_sizes=train_sizes,
                                                        cv=10)


# --- SECTION 3 ---
# Calculate the average and standard deviation for each hyperparameter
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# --- SECTION 4 ---
# Plot the scores
plt.figure()
plt.title('Learning curves')
# Plot the standard deviations
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                 color="C1")
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color="C0")

# Plot the means
plt.plot(train_sizes, train_scores_mean, 'o-', color="C1",
         label="Training score")
plt.plot(train_sizes, test_scores_mean, 'o-', color="C0",
         label="Cross-validation score")

plt.xticks(train_sizes)
plt.xlabel('Size of training set (instances)')
plt.ylabel('Accuracy')
plt.legend(loc="best")



