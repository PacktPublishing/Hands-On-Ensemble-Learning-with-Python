# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 19:37:48 2019

@author: George Kyriakides
         ge.kyriakides@gmail.com
"""



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import KernelPCA


# =============================================================================
# OLS
# =============================================================================
# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn import metrics
diabetes = load_diabetes()


# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
ols = LinearRegression()
ols.fit(train_x, train_y)
err = metrics.mean_squared_error(test_y, ols.predict(test_x))
r2 = metrics.r2_score(test_y, ols.predict(test_x))

# --- SECTION 4 ---
# Print the model
print('---OLS on diabetes dataset.---')
print('Coefficients:')
print('Intercept (b): %.2f'%ols.intercept_)
for i in range(len(diabetes.feature_names)):
    print(diabetes.feature_names[i]+': %.2f'%ols.coef_[i])
print('-'*30)
print('R-squared: %.2f'%r2, ' MSE: %.2f \n'%err)


# =============================================================================
# LOGIT
# =============================================================================
# --- SECTION 1 ---
# Libraries and data loading
from sklearn.linear_model import  LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
bc = load_breast_cancer()

# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = bc.data[:400], bc.target[:400]
test_x, test_y = bc.data[400:], bc.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
logit = LogisticRegression()
logit.fit(train_x, train_y)
acc = metrics.accuracy_score(test_y, logit.predict(test_x))

# --- SECTION 4 ---
# Print the model
print('---Logistic Regression on breast cancer dataset.---')
print('Coefficients:')
print('Intercept (b): %.2f'%logit.intercept_)
for i in range(len(bc.feature_names)):
    print(bc.feature_names[i]+': %.2f'%logit.coef_[0][i])
print('-'*30)
print('Accuracy: %.2f \n'%acc)
print(metrics.confusion_matrix(test_y, logit.predict(test_x)))

# =============================================================================
# SVM FIGURE
# =============================================================================
f = lambda x: 2 * x - 5
f_upp = lambda x: 2 * x - 5 + 2
f_lower = lambda x: 2 * x - 5 - 2

pos = []
neg = []

np.random.seed(345234)
for i in range(80):
    x = np.random.randint(15)
    y = np.random.randint(15)

    d = np.abs(2*x-y-5)/np.sqrt(2**2+1)
    if f(x) < y and d>=1:
        pos.append([x,y])
    elif f(x) > y and d>=1 :
        neg.append([x,y])

pos.append([4, f_upp(4)])
neg.append([8, f_lower(8)])


plt.figure()
plt.xticks([])
plt.yticks([])
plt.scatter(*zip(*pos))
plt.scatter(*zip(*neg))

plt.plot([0,10],[f(0),f(10)], linestyle='--', color='m')
plt.plot([0,10],[f_upp(0),f_upp(10)], linestyle='--', color='red')
plt.plot([0,10],[f_lower(0),f_lower(10)], linestyle='--', color='red')
plt.plot([4,3],[f_lower(4),f_upp(3)], linestyle='-', color='black')
plt.plot([7,6],[f_lower(7),f_upp(6)], linestyle='-', color='black')
plt.xlabel('x')
plt.ylabel('y')
plt.title('SVM')

# =============================================================================
# SVC
# =============================================================================
# --- SECTION 1 ---
# Libraries and data loading
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn import metrics

# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = bc.data[:400], bc.target[:400]
test_x, test_y = bc.data[400:], bc.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
svc = SVC(kernel='linear')
svc.fit(train_x, train_y)
acc = metrics.accuracy_score(test_y, svc.predict(test_x))

# --- SECTION 4 ---
# Print the model's accuracy
print('---SVM on breast cancer dataset.---')
print('Accuracy: %.2f \n'%acc)
print(metrics.confusion_matrix(test_y, svc.predict(test_x)))

# =============================================================================
# SVR
# =============================================================================
# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_diabetes
from sklearn.svm import SVR
from sklearn import metrics
diabetes = load_diabetes()


# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
svr = SVR(kernel='linear', C=1000)
svr.fit(train_x, train_y)
err = metrics.mean_squared_error(test_y, svr.predict(test_x))
r2 = metrics.r2_score(test_y, svr.predict(test_x))

# --- SECTION 4 ---
# Print the model
print('---SVM on diabetes dataset.---')
print('R-squared: %.2f'%r2, ' MSE: %.2f \n'%err)



# =============================================================================
# MLP REGRESSION
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
diabetes = load_diabetes()


# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
mlpr = MLPRegressor(solver='sgd')
mlpr.fit(train_x, train_y)
err = metrics.mean_squared_error(test_y, mlpr.predict(test_x))
r2 = metrics.r2_score(test_y, mlpr.predict(test_x))

# --- SECTION 4 ---
# Print the model
print('---Neural Networks on diabetes dataset.---')
print('R-squared: %.2f'%r2, ' MSE: %.2f \n'%err)

# =============================================================================
# MLP CLASSIFICATION
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
bc = load_breast_cancer()




# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = bc.data[:400], bc.target[:400]
test_x, test_y = bc.data[400:], bc.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
mlpc = MLPClassifier(solver='lbfgs', random_state=12418)
mlpc.fit(train_x, train_y)
acc = metrics.accuracy_score(test_y, mlpc.predict(test_x))

# --- SECTION 4 ---
# Print the model's accuracy
print('---Neural Networks on breast cancer dataset.---')
print('Accuracy: %.2f \n'%acc)
print(metrics.confusion_matrix(test_y, mlpc.predict(test_x)))

# =============================================================================
# MLP REGRESSION
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn import metrics
diabetes = load_diabetes()


# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
mlpr = MLPRegressor(solver='sgd')
mlpr.fit(train_x, train_y)
err = metrics.mean_squared_error(test_y, mlpr.predict(test_x))
r2 = metrics.r2_score(test_y, mlpr.predict(test_x))

# --- SECTION 4 ---
# Print the model
print('---Neural Networks on diabetes dataset.---')
print('R-squared: %.2f'%r2, ' MSE: %.2f \n'%err)

# =============================================================================
# DTREE REGRESSION
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
diabetes = load_diabetes()


# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
dtr = DecisionTreeRegressor(max_depth=2)
dtr.fit(train_x, train_y)
err = metrics.mean_squared_error(test_y, dtr.predict(test_x))
r2 = metrics.r2_score(test_y, dtr.predict(test_x))

# --- SECTION 4 ---
# Print the model
print('---Neural Networks on diabetes dataset.---')
print('R-squared: %.2f'%r2, ' MSE: %.2f \n'%err)

# =============================================================================
# DTREE CLASSIFICATION
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
bc = load_breast_cancer()



# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = bc.data[:400], bc.target[:400]
test_x, test_y = bc.data[400:], bc.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
dtc = DecisionTreeClassifier(max_depth=2)
dtc.fit(train_x, train_y)
acc = metrics.accuracy_score(test_y, dtc.predict(test_x))

# --- SECTION 4 ---
# Print the model's accuracy
print('---Neural Networks on breast cancer dataset.---')
print('Accuracy: %.2f \n'%acc)
print(metrics.confusion_matrix(test_y, dtc.predict(test_x)))
from sklearn.tree import export_graphviz
export_graphviz(dtc, feature_names=bc.feature_names,
                             class_names=bc.target_names, impurity=False)



# =============================================================================
# KNN REGRESSION
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_diabetes
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
diabetes = load_diabetes()


# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = diabetes.data[:400], diabetes.target[:400]
test_x, test_y = diabetes.data[400:], diabetes.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
knnr = KNeighborsRegressor(n_neighbors=14)
knnr.fit(train_x, train_y)
err = metrics.mean_squared_error(test_y, knnr.predict(test_x))
r2 = metrics.r2_score(test_y, knnr.predict(test_x))

# --- SECTION 4 ---
# Print the model
print('---Neural Networks on diabetes dataset.---')
print('R-squared: %.2f'%r2, ' MSE: %.2f \n'%err)

# =============================================================================
# KNN CLASSIFICATION
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
bc = load_breast_cancer()



# --- SECTION 2 ---
# Split the data into train and test set
train_x, train_y = bc.data[:400], bc.target[:400]
test_x, test_y = bc.data[400:], bc.target[400:]

# --- SECTION 3 ---
# Instantiate, train and evaluate the model
dtc = KNeighborsClassifier(n_neighbors=5)
dtc.fit(train_x, train_y)
acc = metrics.accuracy_score(test_y, dtc.predict(test_x))

# --- SECTION 4 ---
# Print the model's accuracy
print('---Neural Networks on breast cancer dataset.---')
print('Accuracy: %.2f \n'%acc)
print(metrics.confusion_matrix(test_y, dtc.predict(test_x)))


# =============================================================================
# K-MEANS
# =============================================================================

# --- SECTION 1 ---
# Libraries and data loading
from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
bc = load_breast_cancer()


bc.data=bc.data[:,:2]

# --- SECTION 2 ---
# Instantiate and train
km = KMeans(n_clusters=3)
km.fit(bc.data)

# --- SECTION 3 ---
# Create a point mesh to plot cluster areas

# Step size of the mesh.
h = .02

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = bc.data[:, 0].min() - 1, bc.data[:, 0].max() + 1
y_min, y_max = bc.data[:, 1].min() - 1, bc.data[:, 1].max() + 1

# Create the actual mesh and cluster it
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = km.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           aspect='auto', origin='lower')

# --- SECTION 4 ---
# Plot the actual data
c = km.predict(bc.data)

r = c == 0
b = c == 1
g = c == 2


plt.scatter(bc.data[r, 0], bc.data[r, 1], label='cluster 1', color='silver')
plt.scatter(bc.data[b, 0], bc.data[b, 1], label='cluster 2', color='white')
plt.scatter(bc.data[g, 0], bc.data[g, 1], label='cluster 3', color='black')
plt.title('K-means')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.xlabel(bc.feature_names[0])
plt.ylabel(bc.feature_names[1])
plt.show()
plt.legend()